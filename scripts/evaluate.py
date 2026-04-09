# =============================================================================
# scripts/evaluate.py
# Triplet-MHA ViT — Model Evaluation & Inference Script
#
# Usage:
#   python scripts/evaluate.py \
#       --config  configs/config.yaml \
#       --weights results/exp0/weights/best_model.hdf5
#
# What it does:
#   1. Loads the saved best_model.hdf5 with custom layer objects
#   2. Loads validation (and optionally test) data
#   3. Runs test_model() → prints accuracy, loss, precision, recall
#   4. Saves confusion matrix + classification report to Excel
# =============================================================================

import os
import sys
import argparse
import yaml
from keras.models import load_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PatchExtract, PatchEmbedding
from utils  import create_train_test, save_results
from utils.metrics import test_model


def main(config_path, weights_path):
    # ── Load config ────────────────────────────────────────────────────
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    paths        = cfg['paths']
    dataset_root = paths['dataset_root']
    img_size     = cfg['data']['img_size']
    batch_size   = cfg['training']['batch_size']

    train_dir = os.path.join(dataset_root, paths['train_dir'])
    test_dir  = os.path.join(dataset_root, paths['test_dir'])
    val_dir   = os.path.join(dataset_root, paths['val_dir'])

    # ── Load data ──────────────────────────────────────────────────────
    data_list = [train_dir, test_dir, val_dir]
    x_train, y_train, x_test, y_test, x_val, y_val, class_val = \
        create_train_test(data_list, img_size, img_size)

    print(f'[INFO] Classes: {class_val}')

    # ── Load model with custom layers ──────────────────────────────────
    custom_objects = {
        'PatchExtract':   PatchExtract,
        'PatchEmbedding': PatchEmbedding,
    }
    model_best = load_model(weights_path, custom_objects=custom_objects)
    print(f'[INFO] Loaded model from: {weights_path}')

    # ── Evaluate on validation set ────────────────────────────────────
    print('\n[INFO] ── Validation Set Evaluation ──')
    acc, loss, cm, reports = test_model(
        model_best, x_val, y_val, batch_size, class_val
    )

    # ── Save results ───────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(weights_path), '..', 'train_logs')
    results_dir = os.path.normpath(results_dir)
    save_results(cm, reports, results_dir)

    # ── (Optional) Evaluate on test set if available ──────────────────
    if isinstance(x_test, str) or len(x_test) == 0:
        print('[INFO] No test split found — skipping test evaluation.')
    else:
        print('\n[INFO] ── Test Set Evaluation ──')
        acc_t, loss_t, cm_t, reports_t = test_model(
            model_best, x_test, y_test, batch_size, class_val
        )
        test_results_dir = os.path.join(
            os.path.dirname(weights_path), '..', 'test_results'
        )
        save_results(cm_t, reports_t, os.path.normpath(test_results_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a saved Triplet-MHA ViT checkpoint.'
    )
    parser.add_argument(
        '--config', type=str, default='configs/config.yaml',
        help='Path to the YAML configuration file.'
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='Path to the .hdf5 model weights file (e.g. results/exp0/weights/best_model.hdf5).'
    )
    args = parser.parse_args()
    main(args.config, args.weights)