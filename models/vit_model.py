# =============================================================================
# models/vit_model.py
# Triplet-MHA ViT — Full Model Assembly and Compilation
#
# Assembles the complete Triplet-MHA ViT pipeline:
#   Image → PatchExtract → PatchEmbedding
#        → [Triplet Block × 2]
#        → [Triplet Block × 2]
#        → [Triplet Block × 1]
#        → LayerNorm → GlobalAveragePool → Dense (softmax)
#
# The progressive MaxPooling1D layers reduce the sequence length:
#   324 → 162 → 81 → 40, reducing compute while preserving structure.
# =============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.metrics import Precision, Recall

from .patch_extract import PatchExtract
from .patch_embedding import PatchEmbedding
from .triplet_encoder import triple_ViT_block


def build_triplet_vit(
    input_shape,
    num_classes,
    patch_size,
    embed_dim,
    num_heads,
    dense_units,
    mlp_head_units,
    dropout,
    learning_rate=0.0001,
):
    """Build and compile the full Triplet-MHA ViT model.

    Args:
        input_shape   (tuple): Image shape (H, W, C), e.g. (224, 224, 3).
        num_classes   (int)  : Number of disease/class categories, e.g. 21.
        patch_size    (tuple): Patch dimensions (ph, pw), e.g. (12, 12).
        embed_dim     (int)  : Embedding/model dimension, e.g. 64.
        num_heads     (int)  : Number of attention heads, e.g. 8.
        dense_units (list[int]): MLP hidden units per encoder block,
                                 e.g. [256, 64] (= [dims*4, dims]).
        mlp_head_units (list[int]): Classification head MLP units, e.g. [128].
        dropout       (float): Global dropout rate, e.g. 0.3.
        learning_rate (float): Adam optimizer learning rate. Default: 0.001.

    Returns:
        Compiled keras.Model ready for training.

    Encoder stack layout:
        Block 1
        Block 2
        Block 3
        Block 4
        Block 5
        ↓
        LayerNorm → GlobalAveragePool1D → Dense(num_classes, softmax)
    """
    img_size = input_shape[0]
    num_patches = (img_size // patch_size[0]) ** 2

    ## ── Input ────────────────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape)

    ## ── Patch pipeline ───────────────────────────────────────────────────
    patches = PatchExtract(patch_size)(inputs)
    embedded_patches = PatchEmbedding(num_patches, embed_dim)(patches)

    ## ── Encoder stage 1: 2 blocks + MaxPool ─────────────────────────────
    embedded_patches = triple_ViT_block(
        x=embedded_patches, heads=num_heads, dims=embed_dim,
        dense_units=dense_units, dropout=dropout
    )
    embedded_patches = triple_ViT_block(
        x=embedded_patches, heads=num_heads, dims=embed_dim,
        dense_units=dense_units, dropout=dropout
    )
    # embedded_patches = layers.MaxPooling1D(pool_size=2)(embedded_patches)

    ## ── Encoder stage 2: 2 blocks + MaxPool ─────────────────────────────
    embedded_patches = triple_ViT_block(
        x=embedded_patches, heads=num_heads, dims=embed_dim,
        dense_units=dense_units, dropout=dropout
    )
    embedded_patches = triple_ViT_block(
        x=embedded_patches, heads=num_heads, dims=embed_dim,
        dense_units=dense_units, dropout=dropout
    )

    ## ── Encoder stage 4: final block ────────────────────────────────────
    embedded_patches = triple_ViT_block(
        x=embedded_patches, heads=num_heads, dims=embed_dim,
        dense_units=dense_units, dropout=dropout
    )

    ## ── Classification head ──────────────────────────────────────────────
    x = layers.LayerNormalization(epsilon=1e-6)(embedded_patches)
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    ## ── Model build & compile ────────────────────────────────────────────
    model = keras.Model(inputs, output, name="Triplet_MHA_ViT")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[
            "accuracy",
            Precision(name="Precision"),
            Recall(name="Recall"),
        ],
    )
    return model