# =============================================================================
# models/patch_extract.py
# Triplet-MHA ViT — Patch Extraction Layer
#
# Splits an input image into non-overlapping fixed-size patches using
# tf.image.extract_patches, then reshapes the result into a sequence
# of flattened patch vectors: (batch, num_patches, patch_dim).
# =============================================================================

import tensorflow as tf
from tensorflow.keras import layers


class PatchExtract(layers.Layer):
    """Extract non-overlapping patches from a batch of images.

    Given an image of shape (H, W, C) and a patch_size of (ph, pw),
    the layer produces (H//ph * W//pw) patches, each of dimension
    ph * pw * C, returned as a 2D sequence tensor.

    Args:
        patch_size (tuple): (patch_height, patch_width). Both values are
            typically equal, e.g. (12, 12).

    Input shape:
        4D tensor: (batch_size, H, W, C).

    Output shape:
        3D tensor: (batch_size, num_patches, patch_dim)
        where num_patches = (H // ph) * (W // pw)
        and   patch_dim   = ph * pw * C.

    Example:
        >>> layer = PatchExtract(patch_size=(12, 12))
        >>> x = tf.random.uniform((4, 224, 224, 3))
        >>> out = layer(x)
        >>> out.shape   # (4, 324, 432)  — 18*18=324 patches, 12*12*3=432 dims
    """

    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size   = patch_size
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]   # square patches assumed

    def call(self, images):
        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )

        patch_dim = patches.shape[-1]   # ph * pw * C
        patch_num = patches.shape[1]    # H // ph

        # reshape: (batch, grid_h, grid_w, patch_dim) → (batch, num_patches, patch_dim)
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))

    def get_config(self):
        config = super(PatchExtract, self).get_config()
        config.update({"patch_size": self.patch_size})
        return config