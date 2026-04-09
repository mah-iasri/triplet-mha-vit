# =============================================================================
# utils/data_loader.py
# Triplet-MHA ViT — Dataset Loading Utilities
#
# Provides:
#   load_images()       — Load and resize all images in a folder to NumPy.
#   create_train_test() — Build train/test/val arrays from a directory tree.
#   create_exp_path()   — Auto-increment experiment output directories.
# =============================================================================

import os
import numpy as np
from PIL import Image
import tensorflow.keras.utils as np_utils


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------

def load_images(path, s1, s2):
    """Load and resize all images from a class sub-directory.

    Reads every file in `path`, opens it as an RGB PIL Image, resizes to
    (s1, s2), converts to a float32 NumPy array normalised to [0, 1], and
    stacks all images into a single array.

    Args:
        path (str): Path to the directory containing image files.
        s1   (int): Target image height in pixels.
        s2   (int): Target image width in pixels.

    Returns:
        np.ndarray: Shape (N, s1, s2, 3), dtype float32, values in [0, 1].
                    Returns an empty array if no images are found.
    """
    images = []
    for filename in sorted(os.listdir(path)):
        img_path = os.path.join(path, filename)
        if not os.path.isfile(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB").resize((s1, s2))
            images.append(np.array(img, dtype=np.float32) / 255.0)
        except Exception as e:
            print(f"[WARN] Skipping corrupted file {img_path}: {e}")

    if len(images) == 0:
        return np.array([])
    return np.array(images)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def create_train_test(imagedata, s1, s2, channel_first=False):
    """Load train / test / val splits from a structured dataset directory.

    Expects each split directory to contain sub-folders named after the
    classes (e.g. apple_scab/, apple_healthy/, ...). Class labels are
    assigned as integers in alphabetical order of class folder names.

    Args:
        imagedata    (list[str]): Ordered list of split directory paths.
                                  Include train, test, and/or val paths.
        s1           (int):  Image height.
        s2           (int):  Image width.
        channel_first (bool): If True, reshape arrays to (N, C, H, W).
                              Default False → (N, H, W, C).

    Returns:
        tuple: (x_train, y_train, x_test, y_test, x_val, y_val, classes)
               - x_* are float32 NumPy arrays.
               - y_* are one-hot encoded arrays from to_categorical().
               - classes is a list of class name strings (from train split).
               Splits not found in imagedata are returned as empty strings.
    """
    data_list = imagedata
    x_train = ''
    y_train = ''
    x_test  = ''
    y_test  = ''
    x_val   = ''
    y_val   = ''
    classes = []

    for item in data_list:
        X    = np.array([])
        Y    = np.array([])
        flag = 0

        for i in sorted(os.listdir(item)):
            cls_path = os.path.join(item, i)
            if not os.path.isdir(cls_path):
                continue

            images = load_images(cls_path, s1, s2)
            if len(images) == 0:
                flag += 1
                continue

            labels = [0] * len(images)
            for ii in range(len(images)):
                labels[ii] = flag

            if flag == 0:
                X = images
                Y = labels
            else:
                X = np.concatenate((X, images), axis=0)
                Y = np.concatenate((Y, labels), axis=0)

            flag += 1
            if item == data_list[0]:
                classes.append(i)

        print(X.shape)
        X = X.reshape((X.shape[0],) + X.shape[1:4])

        if channel_first:
            X = np.moveaxis(X, -1, 1)

        print(f'shape of X: {np.shape(X)}')
        Y = np_utils.to_categorical(Y)
        print(f'shape of y: {np.shape(Y)}')

        v = os.path.split(item)[-1]
        print(v)

        if v == 'train':
            x_train = X
            y_train = Y
            print(f'Training Samples ::: x_train: {len(x_train)}, y_train: {len(y_train)}')
        elif v == 'test':
            x_test = X
            y_test = Y
            print(f'Testing Samples ::: x_test: {len(x_test)}, y_test: {len(y_test)}')
        elif v == 'val':
            x_val = X
            y_val = Y
            print(f'Validation Samples ::: x_val: {len(x_val)}, y_val: {len(y_val)}')

    print(f'Class variables: {classes}')
    print(f'Number of classes: {len(classes)}')

    return x_train, y_train, x_test, y_test, x_val, y_val, classes


# ---------------------------------------------------------------------------
# Experiment directory helper
# ---------------------------------------------------------------------------

def create_exp_path(path):
    """Create an auto-incremented experiment subfolder inside `path`.

    On first call it creates `exp0/`, on second call `exp1/`, and so on.
    This mirrors the original logic from triplet_MHA_ViT.py exactly.

    Args:
        path (str): Root results directory. Created if it does not exist.

    Returns:
        str: Full path to the newly created experiment directory.
    """
    os.makedirs(path, exist_ok=True)
    fol_name = 'exp'
    flag     = 0
    numbers  = []
    new_path = ''

    folders = [f for f in os.listdir(path) if not f.startswith('.')]

    if len(folders) == 0:
        new_path = os.path.join(path, fol_name + str(flag))
        os.makedirs(new_path)
        print(f"Path created at: {new_path}")
    else:
        for f in folders:
            w = (os.path.split(f)[-1])[3:]
            try:
                numbers.append(int(w))
            except ValueError:
                pass
        flag     = max(numbers) + 1
        new_path = os.path.join(path, fol_name + str(flag))
        os.makedirs(new_path)
        print(f"path created at: {new_path}")

    return new_path