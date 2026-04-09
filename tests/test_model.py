"""Unit tests for Triplet-MHA ViT components."""
import numpy as np
import tensorflow as tf
import unittest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import build_triplet_vit, PatchExtract, PatchEmbedding
from models.triplet_encoder import triple_ViT_block


class TestPatchExtract(unittest.TestCase):
    def test_output_shape(self):
        layer = PatchExtract(patch_size=(12, 12))
        x = tf.random.uniform((2, 224, 224, 3))
        out = layer(x)
        # 224/12 = 18 → 18*18 = 324 patches
        self.assertEqual(out.shape, (2, 324, 432))


class TestPatchEmbedding(unittest.TestCase):
    def test_output_shape(self):
        layer = PatchEmbedding(num_patch=324, embed_dim=64)
        x = tf.random.uniform((2, 324, 432))
        out = layer(x)
        self.assertEqual(out.shape, (2, 324, 64))


class TestTripletBlock(unittest.TestCase):
    def test_output_shape(self):
        x = tf.random.uniform((2, 324, 64))
        out = triple_ViT_block(x, heads=8, dims=64,
                                dense_units=[256, 64], dropout=0.0)
        self.assertEqual(out.shape, x.shape)


class TestFullModel(unittest.TestCase):
    def test_build_and_forward(self):
        model = build_triplet_vit(
            input_shape=(224, 224, 3), num_classes=21,
            patch_size=(12, 12), embed_dim=64, num_heads=8,
            dense_units=[256, 64], mlp_head_units=[128],
            dropout=0.0
        )
        x = np.random.rand(2, 224, 224, 3).astype(np.float32)
        out = model(x, training=False)
        self.assertEqual(out.shape, (2, 21))


if __name__ == "__main__":
    unittest.main()