# =============================================================================
# models/triplet_encoder.py
# Triplet-MHA ViT — Novel Triplet Multi-Head Attention Encoder Block
#
# CORE NOVEL CONTRIBUTION:
#   Standard ViT uses ONE MHA sublayer per encoder block.
#   This module introduces THREE cascaded MHA sublayers (Triplet-MHA),
#   each with its own LayerNorm + residual Add connection, before the
#   final MLP sublayer. This enables richer, multi-level attention
#   representations critical for fine-grained plant disease features.
#
# Block structure per encoder:
#   Input (x)
#     ├─ LN → MHA-1 → Add(x)    → x2
#     ├─ LN → MHA-2 → Add(x2)   → x4
#     ├─ LN → MHA-3 → Add(x4)   → x6
#     └─ LN → MLP   → Add(x6)   → output
# =============================================================================

import tensorflow as tf
from tensorflow.keras import layers


def mlp(x, hidden_units, dropout_rate):
    """GELU-activated feed-forward MLP sublayer.

    Applies a sequence of Dense → GELU → Dropout transformations.
    Used as the final sublayer in each Triplet-MHA encoder block.

    Args:
        x             : Input tensor of shape (batch, seq_len, d).
        hidden_units (list[int]): Units per Dense layer. e.g. [256, 64].
        dropout_rate (float)    : Dropout probability after each Dense.

    Returns:
        Tensor of shape (batch, seq_len, hidden_units[-1]).
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def triple_ViT_block(x, heads, dims, dense_units, dropout):
    """Triplet Multi-Head Attention (Triplet-MHA) encoder block.

    Replaces the single MHA sublayer of a standard ViT encoder block with
    THREE sequential self-attention sublayers. Each MHA sublayer is preceded
    by LayerNormalization and followed by a residual Add connection, preserving
    gradient flow. The three-stage attention refinement allows the model to
    progressively focus on increasingly discriminative disease features before
    the MLP consolidation step.

    Args:
        x            : Input tensor of shape (batch, seq_len, dims).
        heads  (int) : Number of parallel attention heads (e.g. 8).
        dims   (int) : Key dimension per head (e.g. 64). Must match the
                       last axis of x for residual connections to work.
        dense_units (list[int]): Hidden layer sizes for the MLP sublayer
                                 (e.g. [256, 64]).
        dropout (float): Dropout rate inside MHA and MLP layers (e.g. 0.3).

    Returns:
        Output tensor of same shape as input: (batch, seq_len, dims).

    Detailed architecture:
        ┌──────────────────────────────────────────────────────────┐
        │  x  ──► LN ──► MHA-1(q=x1,k=x1,v=x1) ──► Add(x)  ► x2 │
        │  x2 ──► LN ──► MHA-2(q=x3,k=x3,v=x3) ──► Add(x2) ► x4 │
        │  x4 ──► LN ──► MHA-3(q=x5,k=x5,v=x5) ──► Add(x4) ► x6 │
        │  x6 ──► LN ──► MLP(GELU)             ──► Add(x6) ► out│
        └──────────────────────────────────────────────────────────┘
    """

    ## ── MHA sublayer 1 ──────────────────────────────────────────────────
    x1          = layers.LayerNormalization(epsilon=1e-6)(x)
    attention   = layers.MultiHeadAttention(
                      num_heads=heads, key_dim=dims, dropout=dropout
                  )(x1, x1)
    x2          = layers.Add()([attention, x])
    print(x2.shape)

    ## ── MHA sublayer 2 ──────────────────────────────────────────────────
    x3          = layers.LayerNormalization(epsilon=1e-6)(x2)
    attention_2 = layers.MultiHeadAttention(
                      num_heads=heads, key_dim=dims, dropout=dropout
                  )(x3, x3)
    x4          = layers.Add()([attention_2, x2])

    ## ── MHA sublayer 3 ──────────────────────────────────────────────────
    x5          = layers.LayerNormalization(epsilon=1e-6)(x4)
    attention_3 = layers.MultiHeadAttention(
                      num_heads=heads, key_dim=dims, dropout=dropout
                  )(x5, x5)
    x6          = layers.Add()([attention_3, x4])
    print(x6.shape)

    ## ── MLP sublayer ────────────────────────────────────────────────────
    x7              = layers.LayerNormalization(epsilon=1e-6)(x6)
    x8              = mlp(x7, hidden_units=dense_units, dropout_rate=dropout)
    embedded_patches = layers.Add()([x8, x6])

    return embedded_patches