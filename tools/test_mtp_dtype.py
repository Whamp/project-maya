import unittest
import numpy as np
from mtp_rt import dense_tensor

class MixedDtype(unittest.TestCase):
    def test_bf16_bytes_are_preserved_beside_quantized_projection(self):
        source = np.array([[0x3f80, 0x4000, 0xbf00, 0x0000]], dtype='<u2')
        raw, kind, rows, cols = dense_tensor('mlp.gate.weight', 30, source)
        self.assertEqual((raw, kind, rows, cols), (source.tobytes(), 'bf16', 1, 4))
        projection = np.tile(source, (2, 8))
        qraw, qkind, rows, cols = dense_tensor('fc_hidden.weight', 30, projection)
        self.assertEqual((qkind, rows, cols, len(qraw)), ('q8_0', 2, 32, 68))
        self.assertEqual(dense_tensor('mlp.gate.weight', 30, source)[0], source.tobytes())

    def test_norm_conversion_is_explicit_and_unrelated_dtypes_are_rejected(self):
        raw, kind, _, _ = dense_tensor('self_attn.q_norm.weight', 0, np.array([0, .5], dtype='<f4'))
        self.assertEqual(kind, 'f32')
        np.testing.assert_array_equal(np.frombuffer(raw, dtype='<f4'), [1, 1.5])
        with self.assertRaises(ValueError): dense_tensor('mlp.gate.weight', 8, np.zeros((2,34), dtype='u1'))
        with self.assertRaises(ValueError): dense_tensor('mlp.gate.weight', 0, np.zeros((2,32), dtype='<f4'))
