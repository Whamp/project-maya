"""Protocol and fault checks, without requiring a Mooncake installation."""

import hashlib
import struct
import unittest
from unittest import mock

from state_mooncake import Connector, Reader, locators, manifest_refs


def number(n):
    return struct.pack("<Q", n)


def string(s):
    raw = s.encode()
    return number(len(raw)) + raw


def prefix_bytes(p):
    identity, tokens, extras = p
    return (
        string(identity)
        + number(len(tokens))
        + b"".join(number(t) for t in tokens)
        + number(len(extras))
        + b"".join(number(at) + number(n) + string(key) for at, n, key in extras)
    )


def manifest(p, chunk):
    data = (
        string("MAYASTATE3")
        + prefix_bytes(p)
        + number(0) * 4
        + number(1)
        + string("recurrent")
        + number(1)
        + string(chunk)
        + number(123)
        + number(8)
        + number(0)
    )
    return "unit-" + hashlib.sha256(data).hexdigest() + ".state", data


class Replicate:
    group_ids = None


class Store:
    def __init__(self, objects):
        self.objects, self.queries, self.groups = objects, [], {}

    def batch_is_exist(self, keys):
        self.queries.extend(keys)
        return [int(k in self.objects) for k in keys]

    def get_size(self, key):
        return len(self.objects[key]) if key in self.objects else -1

    def get(self, key):
        return self.objects.get(key, b"")

    def get_batch(self, keys):
        return [self.get(k) for k in keys]

    def put(self, key, data, config):
        self.objects[key] = data
        self.groups[key] = config.group_ids
        return 0

    def upsert(self, key, data):
        self.objects[key] = data
        return 0

    def remove(self, key, force=False):
        self.objects.pop(key, None)
        return 0


class TestConnector(Connector):
    def policy(self, operation, updates=None):
        if updates:
            self.updates.extend(updates)
        return {"success": True, "applied": len(updates or [])}

    def __init__(self, store):
        self.updates = []
        super().__init__(store, Replicate, "fixture", "http://unused")


class Protocol(unittest.TestCase):
    def setUp(self):
        self.objects = {}
        self.a = TestConnector(Store(self.objects))
        self.b = TestConnector(Store(self.objects))
        self.prefix = ("model-layout", list(range(37)), [(4, 1, "picture-a")])
        self.chunk = "chunk-" + hashlib.sha256(b"12345678").hexdigest()
        self.unit, self.data = manifest(self.prefix, self.chunk)

    def publish(self):
        self.a.put(self.chunk, b"12345678")
        self.a.put(self.unit, self.data)
        self.a.announce(
            self.unit,
            prefix_bytes(self.prefix)
            + string("")
            + string("1.25")
            + number(1)
            + string(self.chunk),
        )

    def test_second_client_loads_only_a_complete_unit_and_renews_leases(self):
        self.publish()
        prompt = (self.prefix[0], self.prefix[1] + [99], self.prefix[2])
        self.assertEqual(self.b.lookup(prefix_bytes(prompt)).decode(), self.unit)
        self.assertEqual(self.b.get(self.chunk), b"12345678")
        self.assertEqual(
            self.a.store.groups[self.a.name(self.unit)], [self.a.name(self.unit)]
        )
        update = self.a.updates[-1]
        self.assertEqual(update["members"], [self.a.name(self.unit)])
        self.assertIn(self.a.name(self.chunk), update["refs"])
        self.assertFalse(update["aligned"])
        queries = len(self.b.store.queries)
        self.assertEqual(self.b.lookup(prefix_bytes(prompt)).decode(), self.unit)
        self.assertIn(self.b.name(self.chunk), self.b.store.queries[queries:])
        del self.objects[self.b.name(self.chunk)]
        self.assertEqual(self.b.lookup(prefix_bytes(prompt)), b"")

    def test_no_locator_is_published_before_all_members_complete(self):
        self.a.put(self.unit, self.data)
        with self.assertRaises(RuntimeError):
            self.a.announce(
                self.unit,
                prefix_bytes(self.prefix)
                + string("")
                + string("1")
                + number(1)
                + string(self.chunk),
            )
        self.assertNotIn(self.a.name(locators(self.prefix)[-1]), self.objects)

    def test_image_identity_and_model_layout_change_the_locator(self):
        self.publish()
        self.assertEqual(
            self.b.lookup(
                prefix_bytes(("different-model", self.prefix[1], self.prefix[2]))
            ),
            b"",
        )
        self.assertEqual(
            self.b.lookup(
                prefix_bytes((self.prefix[0], self.prefix[1], [(4, 1, "picture-b")]))
            ),
            b"",
        )

    def test_missing_manifest_or_corruption_is_a_miss_even_on_memo_hit(self):
        self.publish()
        self.b.lookup(prefix_bytes(self.prefix))
        self.objects[self.a.name(self.unit)] = self.data[:-1]
        self.assertEqual(self.b.lookup(prefix_bytes(self.prefix)), b"")
        with self.assertRaises(ValueError):
            manifest_refs(self.unit, self.data[:-1])

    def test_batch_renews_members_and_invalidation_revokes_memoized_frontier(self):
        self.publish()
        self.b.lookup(prefix_bytes(self.prefix))
        payload = self.b.command("B", self.unit, number(1) + string(self.chunk))
        self.assertEqual(payload, number(8) + b"12345678")
        self.b.command("V", self.unit, b"")
        self.a.command("X", self.unit, b"")
        with self.assertRaises(RuntimeError):
            self.b.command("V", self.unit, b"")
        self.assertEqual(self.b.lookup(prefix_bytes(self.prefix)), b"")
        self.assertIn(self.a.name(self.chunk), self.objects)
        with self.assertRaises(RuntimeError):
            self.b.command("B", self.unit, number(1) + string(self.chunk))

    def test_write_wait_reprobes_without_negative_memo_hiding_completion(self):
        self.b.wait_s = 0.1
        original = self.b.lookup_once
        calls = []

        def probe(body):
            calls.append(1)
            if len(calls) == 2:
                self.publish()
            return original(body)

        with mock.patch.object(self.b, "lookup_once", side_effect=probe):
            self.assertEqual(
                self.b.lookup(prefix_bytes(self.prefix)).decode(), self.unit
            )
        self.assertEqual(len(calls), 2)

    def test_bounded_parser_rejects_untrusted_lengths(self):
        with self.assertRaises(ValueError):
            Reader(number(2**63)).string()
        self.assertEqual(len(locators(("m", list(range(16 + 256 * 4200)), []))), 4096)
