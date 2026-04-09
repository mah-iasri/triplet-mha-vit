# =============================================================================
# models/patch_embedding.py
# Triplet-MHA ViT — Patch Embedding Layer
#
# Projects each flattened patch into a dense embedding space (linear
# projection) and adds learnable 1D positional embeddings so the model
# retains spatial ordering information across the patch sequence.
# =============================================================================

import tensorflow as tf
from tensorflow.keras import layers


class PatchEmbedding(layers.Layer):
    """Linear projection of patches + learnable positional embeddings.

    Each patch token is projected to `embed_dim` dimensions via a Dense
    layer. A learnable positional embedding (one per position) is added
    element-wise. Dropout is applied to the summed result during training.

    Args:
        num_patch    (int):   Total patches per image, e.g. (224//12)^2 = 324.
        embed_dim    (int):   Embedding dimensionality, e.g. 64.
        dropout_rate (float): Dropout probability after embedding. Default 0.1.
        l2_reg       (float): L2 regularization on positional embeddings.
                              Default 1e-4.

    Input shape:
        3D tensor: (batch_size, num_patch, patch_dim).

    Output shape:
        3D tensor: (batch_size, num_patch, embed_dim).

    Example:
        >>> layer = PatchEmbedding(num_patch=324, embed_dim=64)
        >>> x = tf.random.uniform((4, 324, 432))
        >>> out = layer(x, training=False)
        >>> out.shape   # (4, 324, 64)
    """

    def __init__(self, num_patch, embed_dim, dropout_rate=0.1, l2_reg=1e-4, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch    = num_patch
        self.embed_dim    = embed_dim
        self.dropout_rate = dropout_rate
        self.l2_reg       = l2_reg

        ## linear projection: maps each patch vector → embed_dim
        self.proj = layers.Dense(embed_dim)

        ## learnable position embedding table: one vector per patch position
        self.pos_embed = layers.Embedding(
            input_dim=num_patch,
            output_dim=embed_dim,
            embeddings_regularizer=tf.keras.regularizers.L2(l2_reg),
        )

        ## dropout applied after patch + position sum
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, patch, training=False):
        # integer position indices: [0, 1, 2, ..., num_patch-1]
        pos = tf.range(start=0, limit=self.num_patch, delta=1)

        # project patches and add positional encodings element-wise
        embedded_patches = self.proj(patch) + self.pos_embed(pos)

        # apply dropout only during training
        if training:
            embedded_patches = self.dropout(embedded_patches)

        return embedded_patches

    def get_config(self):
        config = super(PatchEmbedding, self).get_config()
        config.update({
            "num_patch":    self.num_patch,
            "embed_dim":    self.embed_dim,
            "dropout_rate": self.dropout_rate,
            "l2_reg":       self.l2_reg,
        })
        return config