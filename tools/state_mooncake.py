#!/usr/bin/env python3
"""Optional native state-cache transport for the pinned FutureMLS Mooncake store.

Run by state_mooncake.cpp over a private socket. No shell, HTTP data plane or
model tensors in JSON. The only HTTP operations are the master's unit policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import struct
import time
import urllib.request

MAX_FRAME = 64 << 20
PINNED_COMMIT = "59ca03f3c44131cf0ecea299d05bb8ea31f2032f"


class Reader:
    def __init__(self, data):
        self.data, self.at = data, 0

    def raw(self, n):
        if n < 0 or n > len(self.data) - self.at:
            raise ValueError("truncated Mooncake message")
        start = self.at
        self.at += n
        return self.data[start : self.at]

    def number(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def count(self, width=8):
        n = self.number()
        if n > (len(self.data) - self.at) // width:
            raise ValueError("invalid Mooncake message count")
        return n

    def string(self):
        return self.raw(self.count(1)).decode("utf-8")

    def prefix(self):
        identity = self.string()
        tokens = [self.number() for _ in range(self.count())]
        extra = [
            (self.number(), self.number(), self.string()) for _ in range(self.count(24))
        ]
        return identity, tokens, extra

    def end(self):
        if self.at != len(self.data):
            raise ValueError("trailing Mooncake message")


def locators(prefix):
    """At most 4096 deepest 256-token hash units, anchored 16 tokens inside.

    Hash incrementally: a million-token prompt costs one token walk, rather
    than hashing the entire prefix again for every candidate length.
    """
    identity, tokens, extras = prefix
    by_position = {}
    for position, length, key in extras:
        by_position.setdefault(position, []).append((length, key))
    digest = hashlib.sha256(b"maya-locator-v1\0" + identity.encode() + b"\0")
    out = []
    for i, token in enumerate(tokens):
        digest.update(struct.pack("<Q", token))
        values = by_position.get(i, [])
        digest.update(struct.pack("<Q", len(values)))
        for length, key in values:
            raw = key.encode()
            digest.update(struct.pack("<QQ", length, len(raw)) + raw)
        if i + 1 >= 16 and (i + 1 - 16) % 256 == 0:
            out.append("locator-" + digest.hexdigest())
    return out[-4096:]


def manifest_refs(key, data):
    if key != "unit-" + hashlib.sha256(data).hexdigest() + ".state":
        raise ValueError("manifest digest mismatch")
    r = Reader(data)
    if r.string() != "MAYASTATE3":
        raise ValueError("unsupported Maya manifest")
    r.prefix()
    r.raw(32)  # sampling-ready, anchor, completed frontier, minimum prompt length
    refs = []
    for _ in range(r.count(16)):
        r.string()
        for _ in range(r.count(24)):
            name = r.string()
            r.number()
            size = r.number()
            if (
                not name.startswith("chunk-")
                or len(name) != 70
                or not 0 < size <= MAX_FRAME
            ):
                raise ValueError("invalid chunk reference")
            refs.append(name)
    r.number()
    r.end()  # core verifies the manifest checksum again before admission
    return list(dict.fromkeys(refs))


class Connector:
    def __init__(
        self, store, replicate, namespace, admin, timeout=2.0, memo_s=0.25, wait_ms=0
    ):
        if not namespace or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for c in namespace
        ):
            raise ValueError(
                "MAYA_STATE_MOONCAKE_NAMESPACE must be a nonempty identifier"
            )
        self.store, self.replicate = store, replicate
        self.namespace, self.admin = namespace + "/", admin.rstrip("/")
        self.timeout, self.memo_s, self.memo = timeout, memo_s, {}
        self.wait_s = min(1.0, max(0.0, float(wait_ms) / 1000.0))
        self.policy("stats")  # 409 means an incompatible lease-only master: fail closed

    def name(self, key):
        return self.namespace + key

    def policy(self, operation, updates=None):
        path = "/api/v1/units/stats" if operation == "stats" else "/api/v1/units"
        body = (
            None
            if updates is None
            else json.dumps({"updates": updates}, allow_nan=False).encode()
        )
        request = urllib.request.Request(
            self.admin + path, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as reply:
            result = json.loads(reply.read(1 << 20))
        if not isinstance(result, dict) or (
            updates is not None
            and (
                result.get("success") is not True
                or result.get("applied") != len(updates)
                or result.get("errors")
            )
        ):
            raise RuntimeError("Mooncake unit policy rejected the update")
        if operation == "stats" and (
            result.get("success") is not True
            or not isinstance(result.get("units"), int)
            or not isinstance(result.get("clock"), (int, float))
        ):
            raise RuntimeError("Mooncake unit_gds capability response is incompatible")
        return result

    def exists(self, keys):
        for start in range(0, len(keys), 4096):
            batch = keys[start : start + 4096]
            result = self.store.batch_is_exist([self.name(k) for k in batch])
            if len(result) != len(batch) or any(x != 1 for x in result):
                return False
        return True

    def get(self, key):
        size = self.store.get_size(self.name(key))
        if size <= 0 or size > MAX_FRAME:
            raise RuntimeError("Mooncake object missing or too large")
        data = bytes(self.store.get(self.name(key)))
        if len(data) != size:
            raise RuntimeError("incomplete Mooncake read")
        return data

    def put(self, key, data):
        if self.exists([key]):
            return
        config = self.replicate()
        if key.startswith("unit-"):
            config.group_ids = [self.name(key)]
        result = self.store.put(self.name(key), data, config)
        if result != 0 and not self.exists([key]):
            raise RuntimeError("Mooncake put failed")

    def announce(self, key, body):
        r = Reader(body)
        prefix = r.prefix()
        parent = r.string()
        value = float(r.string())
        refs = [r.string() for _ in range(r.count())]
        r.end()
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid recovery cost")
        if not self.exists([key] + refs):
            raise RuntimeError("Mooncake publication incomplete")
        anchors = locators(prefix)
        if not anchors:
            return  # short states remain available in the local tiers
        locator = anchors[-1]
        entries = []
        try:
            entries = json.loads(self.get(locator))
        except (RuntimeError, ValueError):
            pass
        entries = [[len(prefix[1]), key]] + [
            x for x in entries if isinstance(x, list) and len(x) == 2 and x[1] != key
        ]
        entries = sorted(entries, key=lambda x: x[0], reverse=True)[:4]
        if (
            self.store.upsert(
                self.name(locator), json.dumps(entries, separators=(",", ":")).encode()
            )
            != 0
        ):
            raise RuntimeError("Mooncake locator publication failed")
        self.policy(
            "updates",
            [
                {
                    "op": "upsert",
                    "id": self.name(key),
                    "value": value,
                    "aligned": False,
                    "parent": self.name(parent) if parent else "",
                    "members": [self.name(key)],
                    "refs": [self.name(k) for k in refs + [locator]],
                }
            ],
        )
        # Content chunks can be identical at unrelated token positions. Parent
        # hints are explicit; semantic alignment is never inferred from content.
        self.memo.clear()

    def lookup(self, body):
        end = time.monotonic() + self.wait_s
        while True:
            result = self.lookup_once(body)
            remaining = end - time.monotonic()
            if result or remaining <= 0:
                return result
            self.memo.clear()  # a negative probe must not hide a completing write
            time.sleep(min(0.005, remaining))

    def lookup_once(self, body):
        r = Reader(body)
        prefix = r.prefix()
        r.end()
        anchors = locators(prefix)
        memo_key = anchors[-1] if anchors else ""
        now = time.monotonic()
        previous = self.memo.get(memo_key)
        if previous and now - previous[0] < self.memo_s:
            entries = previous[1]
        else:
            entries = []
            for start in range(0, len(anchors), 256):
                keys = anchors[start : start + 256]
                flags = self.store.batch_is_exist([self.name(k) for k in keys])
                present = [k for k, exists in zip(keys, flags) if exists == 1]
                for record in (
                    self.store.get_batch([self.name(k) for k in present])
                    if present
                    else []
                ):
                    try:
                        values = json.loads(record)
                        if isinstance(values, list):
                            entries.extend(values[:4])
                    except (ValueError, TypeError):
                        pass
            if len(self.memo) >= 128:
                self.memo.clear()
            self.memo[memo_key] = (now, entries)
        matches = []
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2:
                continue
            n, key = entry
            if (
                not isinstance(n, int)
                or not 0 < n <= len(prefix[1])
                or not isinstance(key, str)
            ):
                continue
            if not key.startswith("unit-") or len(key) != 75:
                continue
            matches.append((n, key))
        # Always renew the actual members' leases even on a metadata memo hit.
        # No lease TTL is guessed or extended by the client-side memo timer.
        complete = []
        for _, key in sorted(set(matches), reverse=True):
            try:
                data = self.get(key)
                refs = manifest_refs(key, data)
                if self.exists([key] + refs):
                    complete.append(key)
            except (RuntimeError, ValueError):
                pass
            if len(complete) == 8:
                break
        return "\n".join(complete).encode()

    def command(self, operation, key, data):
        if operation == "H":
            return b"MAYA_MOONCAKE1"
        if operation == "P":
            self.put(key, data)
        elif operation == "G":
            if data and not self.exists([data.decode(), key]):
                raise RuntimeError("Mooncake recovery lease expired")
            return self.get(key)
        elif operation == "V":
            if not self.exists([key]):
                raise RuntimeError("Mooncake recovery manifest was revoked")
        elif operation == "B":
            r = Reader(data)
            keys = [r.string() for _ in range(r.count())]
            r.end()
            if not self.exists([key] + keys):
                raise RuntimeError("Mooncake recovery lease expired")
            sizes = [self.store.get_size(self.name(k)) for k in keys]
            if any(n <= 0 for n in sizes) or sum(sizes) + 8 * len(keys) + 1 > MAX_FRAME:
                raise RuntimeError("Mooncake batch exceeds bounded payload")
            payloads = self.store.get_batch([self.name(k) for k in keys])
            if len(payloads) != len(keys):
                raise RuntimeError("incomplete Mooncake batch")
            result = bytearray()
            for payload, size in zip(payloads, sizes):
                if payload is None or len(payload) != size:
                    raise RuntimeError("incomplete Mooncake batch")
                result.extend(struct.pack("<Q", size))
                result.extend(payload)
            return bytes(result)
        elif operation == "X":
            # A numerical/native invalidation revokes future recovery. Ordinary
            # supersession only marks a valid unit dead for capacity reclamation.
            result = self.store.remove(self.name(key), True)
            if result != 0 and self.exists([key]):
                raise RuntimeError("Mooncake invalidation failed")
            self.memo.clear()
            self.policy("updates", [{"op": "dead", "id": self.name(key)}])
        elif operation == "L":
            return self.lookup(data)
        elif operation == "A":
            self.announce(key, data)
        elif operation in ("T", "D"):
            self.policy(
                "updates",
                [{"op": "touch" if operation == "T" else "dead", "id": self.name(key)}],
            )
        else:
            raise ValueError("unknown Mooncake command")
        return b""


def connect():
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    probe = ReplicateConfig()
    if not hasattr(probe, "group_ids"):
        raise RuntimeError(
            "Mooncake group_ids capability missing; build the pinned fork"
        )
    store = MooncakeDistributedStore()
    hostname = os.environ.get("MAYA_STATE_MOONCAKE_HOSTNAME", socket.gethostname())
    result = store.setup(
        hostname,
        os.environ.get("MAYA_STATE_MOONCAKE_METADATA", "P2PHANDSHAKE"),
        int(os.environ.get("MAYA_STATE_MOONCAKE_SEGMENT_MB", "64")) << 20,
        16 << 20,
        os.environ.get("MAYA_STATE_MOONCAKE_PROTOCOL", "tcp"),
        os.environ.get("MAYA_STATE_MOONCAKE_RDMA", ""),
        os.environ["MAYA_STATE_MOONCAKE_MASTER"],
    )
    if result != 0:
        raise RuntimeError("Mooncake client setup failed")
    return Connector(
        store,
        ReplicateConfig,
        os.environ["MAYA_STATE_MOONCAKE_NAMESPACE"],
        os.environ["MAYA_STATE_MOONCAKE_ADMIN"],
        wait_ms=float(os.environ.get("MAYA_STATE_MOONCAKE_WAIT_MS", "0")),
    )


def receive(sock, size):
    out = bytearray()
    while len(out) < size:
        part = sock.recv(size - len(out))
        if not part:
            raise EOFError
        out.extend(part)
    return bytes(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    args = parser.parse_args()
    connector = None
    with socket.socket(fileno=args.fd) as sock:
        while True:
            try:
                size = struct.unpack("<Q", receive(sock, 8))[0]
                if not 9 <= size <= MAX_FRAME:
                    raise ValueError("invalid Mooncake frame length")
                request = Reader(receive(sock, size))
                operation = request.raw(1).decode()
                key = request.string()
                data = request.raw(len(request.data) - request.at)
                if connector is None:
                    connector = connect()
                response = b"\0" + connector.command(operation, key, data)
            except (EOFError, BrokenPipeError):
                return
            except (
                RuntimeError,
                ValueError,
                OSError,
                KeyError,
                ImportError,
                TypeError,
                struct.error,
            ) as error:
                response = b"\1" + str(error).encode()[:2048]
            sock.sendall(struct.pack("<Q", len(response)) + response)


if __name__ == "__main__":
    main()
