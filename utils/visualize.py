# =============================================================================
# utils/visualize.py
# Triplet-MHA ViT — Training Curve Visualization
#
# Plots four training metrics side by side:
#   Accuracy | Loss | Precision | Recall
# Faithfully replicates the original plot_results() function with added
# support for saving figures to disk.
# =============================================================================

import os
import matplotlib.pyplot as plt


def plot_results(model_history, save_dir=None):
    """Plot training vs. validation curves for all tracked metrics.

    Produces four separate figures:
        1. Training vs Validation Accuracy
        2. Training vs Validation Loss
        3. Training vs Validation Precision
        4. Training vs Validation Recall

    Green lines denote training; red lines denote validation — consistent
    with the colour scheme in the original triplet_MHA_ViT.py.

    Args:
        model_history : Keras History object returned by model.fit().
        save_dir (str, optional): Directory path to save plot PNGs.
                                  If None, plots are displayed interactively.
                                  Directory is created if it does not exist.
    """
    train_acc  = model_history.history['accuracy']
    test_acc   = model_history.history['val_accuracy']
    train_loss = model_history.history['loss']
    test_loss  = model_history.history['val_loss']
    train_pre  = model_history.history['Precision']
    test_pre   = model_history.history['val_Precision']
    train_rcal = model_history.history['Recall']
    test_rcal  = model_history.history['val_Recall']

    epochs = range(1, len(train_acc) + 1)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    ## ── Accuracy ─────────────────────────────────────────────────────────
    plt.figure()
    plt.plot(epochs, train_acc, 'b', color='green', label='Training Accuracy')
    plt.plot(epochs, test_acc,  'b', color='red',   label='Validation Accuracy')
    plt.title('training vs validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    ## ── Loss ─────────────────────────────────────────────────────────────
    plt.figure()
    plt.plot(epochs, train_loss, 'b', color='green', label='Training loss')
    plt.plot(epochs, test_loss,  'b', color='red',   label='Validation loss')
    plt.title('training vs validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    ## ── Precision ────────────────────────────────────────────────────────
    plt.figure()
    plt.plot(epochs, train_pre, 'b', color='green', label='Training precision')
    plt.plot(epochs, test_pre,  'b', color='red',   label='Validation precision')
    plt.title('training vs validation precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'precision_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    ## ── Recall ───────────────────────────────────────────────────────────
    plt.figure()
    plt.plot(epochs, train_rcal, 'b', color='green', label='Training recall')
    plt.plot(epochs, test_rcal,  'b', color='red',   label='Validation recall')
    plt.title('training vs validation recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'recall_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()