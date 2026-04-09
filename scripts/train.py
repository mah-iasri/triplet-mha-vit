# =============================================================================
# scripts/train.py
# Triplet-MHA ViT — Main Training Script
#
# Usage:
#   python scripts/train.py --config configs/config.yaml
#
# What it does:
#   1. Reads all settings from config.yaml
#   2. Loads train / val / test data via create_train_test()
#   3. Builds and compiles the Triplet-MHA ViT via build_triplet_vit()
#   4. Creates an auto-incremented experiment directory (exp0, exp1, ...)
#   5. Trains the model with ModelCheckpoint + CSVLogger + TensorBoard
#   6. Saves the last epoch weights
#   7. Plots and saves training curves
# =============================================================================

import os
import sys
import argparse
import yaml
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

# Make project root importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_triplet_vit
from utils  import create_train_test, create_exp_path, plot_results


def main(config_path):
    # ── Load config ────────────────────────────────────────────────────
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # ── Paths ──────────────────────────────────────────────────────────
    paths        = cfg['paths']
    dataset_root = paths['dataset_root']
    train_dir    = os.path.join(dataset_root, paths['train_dir'])
    test_dir     = os.path.join(dataset_root, paths['test_dir'])
    val_dir      = os.path.join(dataset_root, paths['val_dir'])
    result_folder = paths['results_dir']

    # ── Data config ────────────────────────────────────────────────────
    data_cfg  = cfg['data']
    img_size  = data_cfg['img_size']

    # ── Hyperparameters ────────────────────────────────────────────────
    t_cfg          = cfg['training']
    batch_size     = t_cfg['batch_size']
    epochs         = t_cfg['epochs']
    learning_rate  = t_cfg['learning_rate']

    m_cfg          = cfg['model']
    patch_size     = tuple(m_cfg['patch_size'])
    embed_dim      = m_cfg['embed_dim']
    num_heads      = m_cfg['num_heads']
    dropout        = m_cfg['dropout']
    mlp_head_units = m_cfg['mlp_head_units']
    dense_units    = [embed_dim * mult for mult in m_cfg['dense_units_multipliers']]

    # ── Load dataset ───────────────────────────────────────────────────
    data_list = [train_dir, test_dir, val_dir]
    x_train, y_train, x_test, y_test, x_val, y_val, class_val = \
        create_train_test(data_list, img_size, img_size)

    n_classes = len(class_val)
    print(f'Training data:   {len(x_train)}')
    print(f'Validation data: {len(x_val)}')
    print(f'Test data:       {len(x_test)}')
    print(f'Class variables: {class_val}')
    print(f'Number of classes: {n_classes}')

    input_shape = (img_size, img_size, 3)

    # ── Build model ────────────────────────────────────────────────────
    model = build_triplet_vit(
        input_shape=input_shape,
        num_classes=n_classes,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dense_units=dense_units,
        mlp_head_units=mlp_head_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    model.summary()

    # ── Experiment directories ─────────────────────────────────────────
    exp_path   = create_exp_path(result_folder)
    tb_logs    = os.path.join(exp_path, 'tb_logs');    os.makedirs(tb_logs)
    model_wts  = os.path.join(exp_path, 'weights');    os.makedirs(model_wts)
    train_logs = os.path.join(exp_path, 'train_logs'); os.makedirs(train_logs)

    # ── Callbacks ──────────────────────────────────────────────────────
    cb_cfg = cfg['callbacks']

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_wts, 'best_model.hdf5'),
        monitor=cb_cfg['monitor'],
        verbose=cb_cfg['verbose'],
        save_best_only=cb_cfg['save_best_only'],
        mode=cb_cfg['mode'],
    )
    csv_logger = CSVLogger(
        filename=os.path.join(train_logs, 'train_logs.csv'),
        separator=',',
        append=False,
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tb_logs, histogram_freq=1
    )

    callbacks_list = [checkpoint, csv_logger]
    if cb_cfg.get('tensorboard', True):
        callbacks_list.append(tensorboard_callback)

    # ── Train ──────────────────────────────────────────────────────────
    model_history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list,
    )

    # ── Save last model ────────────────────────────────────────────────
    model.save(os.path.join(model_wts, 'last_model.hdf5'))
    print(f"[INFO] Last model saved to: {os.path.join(model_wts, 'last_model.hdf5')}")

    # ── Plot training curves ───────────────────────────────────────────
    plot_results(model_history, save_dir=train_logs)

    print("[INFO] Training complete.")
    print(f"[INFO] All results saved under: {exp_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the Triplet-MHA ViT for plant disease detection.'
    )
    parser.add_argument(
        '--config', type=str, default='configs/config.yaml',
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()
    main(args.config)