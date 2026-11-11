#!/usr/bin/env python3
"""Cross-instance native proof against a real, isolated Mooncake unit_gds master.

Requires the pinned Mooncake Python client and master. Uses the Metal graph's
CPU oracle so the distributed protocol can be checked without spare GPU VRAM.
Pass an MTP fixture to exercise compact speculative commit and shared restore.
"""
import argparse
import json
import os
from pathlib import Path
import selectors
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
import uuid

def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); return s.getsockname()[1]

class Engine:
    def __init__(self, exe, model, directory, cache=True):
        env = dict(os.environ, MAYA_STATE_CACHE=str(int(cache)), MAYA_STATE_DIR=str(directory),
                   MAYA_STATE_RAM_GB='0.00001',
                   MAYA_STATE_MOONCAKE_HELPER=str(Path(__file__).with_name('state_mooncake.py')))
        self.log = open(str(directory)+'.stderr', 'w')
        self.process = subprocess.Popen([exe, '--cpu', '--threads', '2', '--ubatch', '16', '--pool-slots', '8',
            '--gguf', model, '--serve', '--max-context', '256'], env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.selector = selectors.DefaultSelector(); self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b''
        while not self.line().startswith('READY'): pass
    def line(self):
        end = time.monotonic()+60
        while b'\n' not in self.buffer:
            if not self.selector.select(max(0, end-time.monotonic())): raise TimeoutError('native engine timed out')
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data: raise RuntimeError('native engine exited; see '+self.log.name)
            self.buffer += data
        line, self.buffer = self.buffer.split(b'\n', 1); return line.decode()
    def generate(self, prompt, n=8, spec=0):
        self.process.stdin.write(f"GEN {n} spec={spec} {','.join(map(str,prompt))}\n".encode()); self.process.stdin.flush()
        tokens = []
        while True:
            line = self.line()
            if line.startswith('T '): tokens.append(int(line.split()[1]))
            elif line.startswith('ERR'): raise RuntimeError(line)
            elif line.startswith('DONE '): return tokens, int(line.split()[8])
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(b'QUIT\n'); self.process.stdin.flush()
            try: self.process.wait(timeout=20)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
        self.selector.close(); self.log.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', required=True); parser.add_argument('--engine', required=True)
    parser.add_argument('--model', required=True); args = parser.parse_args()
    rpc, admin = port(), port(); engines = []
    with tempfile.TemporaryDirectory(prefix='maya-live-mooncake-') as directory:
        directory = Path(directory)
        with open(directory/'master.log', 'w') as log:
            master = subprocess.Popen([args.master, f'--rpc_port={rpc}', '--rpc_address=127.0.0.1',
                f'--metrics_port={admin}', '--metrics_host=127.0.0.1', '--eviction_policy=unit_gds',
                '--unit_leaf_last=true', '--unit_anchor_tier=true', '--unit_idle_demote_s=1',
                '--unit_orphan_grace_s=5', '--unit_dead_headroom_ratio=0.05', '--unit_debug=true',
                '--default_kv_lease_ttl=500ms'], stdout=log, stderr=log)
            try:
                url = f'http://127.0.0.1:{admin}'
                end = time.monotonic()+20
                while True:
                    try:
                        with urllib.request.urlopen(url+'/api/v1/units/stats', timeout=1) as r: json.load(r)
                        break
                    except OSError:
                        if master.poll() is not None or time.monotonic() > end: raise RuntimeError('test master failed')
                        time.sleep(.05)
                os.environ.update(MAYA_STATE_BACKEND='mooncake', MAYA_STATE_MOONCAKE_HOSTNAME='127.0.0.1',
                    MAYA_STATE_MOONCAKE_NAMESPACE='maya-test-'+uuid.uuid4().hex,
                    MAYA_STATE_MOONCAKE_MASTER=f'127.0.0.1:{rpc}', MAYA_STATE_MOONCAKE_ADMIN=url)
                from state_mooncake import connect, Reader, manifest_refs
                def prefix_bytes(prefix):
                    def number(n): return struct.pack('<Q', n)
                    def string(s):
                        raw = s.encode(); return number(len(raw))+raw
                    identity, tokens, extra = prefix
                    return string(identity)+number(len(tokens))+b''.join(number(t) for t in tokens)+number(0)
                probe = connect()
                for name, cache in [('reference', False), ('producer', True), ('consumer', True)]:
                    engines.append(Engine(args.engine, args.model, directory/name, cache))
                reference, producer, consumer = engines
                prompt = list(range(1,38))
                expected, _ = reference.generate(prompt, 12, 1)
                answer, _ = producer.generate(prompt, 12, 0)
                assert answer == expected, 'speculation changed the generated tokens'
                # Read the producer's local manifest to obtain the native layout
                # identity, then wait on actual remote completeness, not a timer.
                end = time.monotonic()+15
                while True:
                    ready = False
                    for file in (directory/'producer').glob('unit-*.state'):
                        reader = Reader(file.read_bytes())
                        if reader.string() != 'MAYASTATE3': continue
                        state_prefix = reader.prefix()
                        reader.number(); reader.number(); frontier = reader.number()
                        if frontier and state_prefix[1] == prompt + answer[:-1]:
                            ready = file.name in probe.lookup(prefix_bytes((state_prefix[0], prompt+answer, []))).decode().splitlines()
                            if ready: break
                    if ready: break
                    if time.monotonic() > end: raise RuntimeError('completed state never reached the real store')
                    time.sleep(.05)
                consumer.generate(list(range(80,100)), 4, 1)
                retry, reuse = consumer.generate(prompt, 12, 1)
                assert retry == expected and reuse == len(prompt)-1, (reuse, len(prompt)-1)
                following = prompt + answer
                fresh, _ = reference.generate(following, 8, 1)
                resumed, reuse = consumer.generate(following, 8, 0)
                assert resumed == fresh and reuse == len(following)-1, (reuse, len(following)-1)
                completed_key = file.name
                completed_prefix = state_prefix
                # Lose one member in this dedicated test namespace. A cold
                # engine must select a complete fallback or recompute everything.
                refs = manifest_refs(completed_key, probe.get(completed_key))
                assert refs
                assert probe.store.remove(probe.name(refs[0]), True) == 0
                cold = Engine(args.engine, args.model, directory/'after-loss', True); engines.append(cold)
                repaired, fallback = cold.generate(following, 8, 0)
                assert repaired == fresh and fallback < len(following)-1, (fallback, len(following)-1)
                with urllib.request.urlopen(url+'/api/v1/units/stats?check=1', timeout=2) as r: stats = json.load(r)
                assert stats.get('success') and not stats.get('violations') and not stats.get('errors'), stats
                for engine in engines: engine.close()
                engines.clear(); probe.store.close()
                master.terminate(); master.wait(timeout=10)
                master = subprocess.Popen([args.master, f'--rpc_port={rpc}', '--rpc_address=127.0.0.1',
                    f'--metrics_port={admin}', '--metrics_host=127.0.0.1', '--eviction_policy=unit_gds',
                    '--unit_debug=true', '--default_kv_lease_ttl=500ms'], stdout=log, stderr=log)
                end = time.monotonic()+20
                while True:
                    try:
                        with urllib.request.urlopen(url+'/api/v1/units/stats', timeout=1) as r: json.load(r)
                        break
                    except OSError:
                        if master.poll() is not None or time.monotonic() > end: raise RuntimeError('restarted test master failed')
                        time.sleep(.05)
                probe = connect()
                # Startup repairs the new master's metadata from the producer's
                # complete local SSD units. The consumer has an empty directory.
                restored_producer = Engine(args.engine, args.model, directory/'producer', True)
                engines.append(restored_producer)
                end = time.monotonic()+15
                while completed_key not in probe.lookup(prefix_bytes((completed_prefix[0], following, []))).decode().splitlines():
                    if time.monotonic() > end: raise RuntimeError('SSD units were not republished after master restart')
                    time.sleep(.05)
                restarted_consumer = Engine(args.engine, args.model, directory/'after-restart', True)
                engines.append(restarted_consumer)
                restarted, restart_reuse = restarted_consumer.generate(following, 8, 0)
                assert restarted == fresh and restart_reuse == len(following)-1, restart_reuse
                # Explicit invalidation removes the manifest while preserving
                # shared members; another client's positive memo must reverify it.
                query = prefix_bytes((completed_prefix[0], following, []))
                assert completed_key in probe.lookup(query).decode().splitlines()
                probe.command('X', completed_key, b'')
                assert completed_key not in probe.lookup(query).decode().splitlines()
                assert any(probe.exists([ref]) for ref in manifest_refs(completed_key, (directory/'producer'/completed_key).read_bytes()))
                probe.store.close()
                print(f"real Mooncake: bounded capture beyond RAM, native speculative completion, exact retry, continuation (reuse {reuse}), missing-member fallback and SSD repair after master restart passed")
            except BaseException:
                for file in [directory/'master.log', *directory.glob('*.stderr')]:
                    print(f'--- {file.name} ---\n' + '\n'.join(file.read_text(errors='replace').splitlines()[-25:]))
                raise
            finally:
                for engine in engines: engine.close()
                master.terminate()
                try: master.wait(timeout=10)
                except subprocess.TimeoutExpired: master.kill(); master.wait()

if __name__ == '__main__': main()
