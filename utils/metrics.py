# =============================================================================
# utils/metrics.py
# Triplet-MHA ViT — Model Evaluation and Results Persistence
#
# Provides:
#   test_model()   — Evaluate a trained model; print and return metrics.
#   save_results() — Persist confusion matrix + classification report to Excel.
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def test_model(best_model, x_test, y_test, batch_size, classes):
    """Evaluate a trained Triplet-MHA ViT model on a test/validation split.

    Computes accuracy, loss, precision, and recall via model.evaluate(),
    then generates a full confusion matrix and per-class classification
    report using sklearn.

    Args:
        best_model  : Loaded Keras model (best checkpoint or last epoch).
        x_test      (np.ndarray): Test images, shape (N, H, W, C).
        y_test      (np.ndarray): One-hot encoded test labels, shape (N, C).
        batch_size  (int):        Batch size for inference.
        classes     (list[str]):  Ordered list of class name strings.

    Returns:
        tuple:
            test_accuracy (float) : Accuracy in percentage (0–100).
            test_loss     (float) : Cross-entropy loss value.
            cm            (np.ndarray): Confusion matrix of shape (C, C).
            reports       (dict):  Classification report as a dict
                                   (from sklearn output_dict=True).
    """
    model = best_model

    ## ── Scalar metrics ───────────────────────────────────────────────────
    print("[INFO] Calculating model accuracy")
    scores = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

    test_loss     = scores[0]
    test_accuracy = scores[1] * 100
    test_prec     = scores[2] * 100
    test_recall   = scores[3] * 100

    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {test_accuracy:.2f}%")
    print(f"Precision: {test_prec:.2f}%")
    print(f"Recall:    {test_recall:.2f}%")

    ## ── Predictions ──────────────────────────────────────────────────────
    Y_pred    = model.predict(x_test, batch_size=batch_size, verbose=1)
    y_pred    = np.argmax(Y_pred, axis=1)
    y_test_new = np.argmax(y_test, axis=1)

    ## ── Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test_new, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    ## ── Classification report ────────────────────────────────────────────
    print('\nClassification Report')
    target_names = classes
    reports_prnt = classification_report(
        y_test_new, y_pred, target_names=target_names, output_dict=False
    )
    reports = classification_report(
        y_test_new, y_pred, target_names=target_names, output_dict=True
    )
    print(reports_prnt)

    return test_accuracy, test_loss, cm, reports


def save_results(cm, reports, path):
    """Save confusion matrix and classification report to an Excel file.

    Creates `results_best.xlsx` inside `path` with two sheets:
        Sheet 1 — ConfusionMatrix: the N×N confusion matrix.
        Sheet 2 — ClassificationReport: per-class precision/recall/f1.

    Args:
        cm      (np.ndarray): Confusion matrix.
        reports (dict):       Classification report dict (sklearn format).
        path    (str):        Output directory. Created if it does not exist.
    """
    os.makedirs(path, exist_ok=True)
    excel_file = 'results_best.xlsx'

    cm_df     = pd.DataFrame(cm)
    report_df = pd.DataFrame(reports).transpose()

    writer = pd.ExcelWriter(os.path.join(path, excel_file), engine='xlsxwriter')
    cm_df.to_excel(writer,     header=True, sheet_name='ConfusionMatrix')
    report_df.to_excel(writer, header=True, sheet_name='ClassificationReport')
    writer.save()

    print(f"[INFO] Results saved to: {os.path.join(path, excel_file)}")