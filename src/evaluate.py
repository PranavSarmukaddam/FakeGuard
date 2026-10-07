"""
evaluate.py
-----------
Evaluation utilities for the Fake News Detection system.

Generates:
  - Classification metrics (accuracy, precision, recall, F1)
  - Confusion matrix data (for Chart.js rendering)
  - Per-class breakdown

SPPU IR Syllabus Topics Covered:
  - Precision, Recall, F1-score
  - Confusion matrix
  - Model evaluation methodology
"""

import os
import json
import logging

import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS   = os.path.join(BASE_DIR, "models")


def load_evaluation_results() -> dict:
    """
    Load pre-computed evaluation results that were saved during training.
    Returns a dict with 'best_model', 'results' list, and 'trained_at'.
    """
    path = os.path.join(MODELS, "evaluation_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def get_confusion_matrix_data(results: list) -> list:
    """
    Extract confusion matrix data from the evaluation results list
    and format it for Chart.js rendering.

    Returns a list of dicts with keys: model_name, matrix, labels.
    """
    out = []
    for r in results:
        cm = r.get("confusion_matrix", [[0, 0], [0, 0]])
        out.append({
            "model_name": r["model_name"],
            "matrix": cm,
            "labels": ["REAL", "FAKE"],
            "accuracy":  r["accuracy"],
            "precision": r["precision"],
            "recall":    r["recall"],
            "f1_score":  r["f1_score"],
        })
    return out


def format_metrics_table(results: list) -> list:
    """
    Format all model results into a list of rows for HTML table rendering.
    """
    rows = []
    best_f1 = max((r["f1_score"] for r in results), default=0)
    for r in results:
        rows.append({
            "model":     r["model_name"],
            "accuracy":  f"{r['accuracy']:.4f}",
            "precision": f"{r['precision']:.4f}",
            "recall":    f"{r['recall']:.4f}",
            "f1_score":  f"{r['f1_score']:.4f}",
            "is_best":   r["f1_score"] == best_f1,
            "training_time": r.get("training_time_sec", "N/A"),
        })
    return rows


def get_summary_stats(results: list) -> dict:
    """Return summary statistics across all trained models."""
    if not results:
        return {}
    best = max(results, key=lambda r: r["f1_score"])
    return {
        "best_model":     best["model_name"],
        "best_accuracy":  best["accuracy"],
        "best_f1":        best["f1_score"],
        "num_models":     len(results),
    }


if __name__ == "__main__":
    data = load_evaluation_results()
    if data:
        print(f"Best model: {data.get('best_model')}")
        print(f"Trained at: {data.get('trained_at')}")
        for r in data.get("results", []):
            print(
                f"  {r['model_name']:25s} "
                f"Acc={r['accuracy']:.4f}  "
                f"Prec={r['precision']:.4f}  "
                f"Rec={r['recall']:.4f}  "
                f"F1={r['f1_score']:.4f}"
            )
    else:
        print("No evaluation results found. Run train_model.py first.")
