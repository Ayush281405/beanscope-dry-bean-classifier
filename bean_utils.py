"""Shared helpers for BeanScope.

Both train_models.py and app.py import from here, so the feature list,
the dataset loader and the six evaluation metrics are defined in one place
and the numbers in the app match the numbers in the training run.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
UCI_PAGE = "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset"

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRatio", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "Roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]
TARGET = "Class"
BEAN_CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]

DATA_DIR = "data"
RAW_ARFF = os.path.join(DATA_DIR, "Dry_Bean_Dataset.arff")      # original UCI file (raw, with duplicates)
CLEAN_CSV = os.path.join(DATA_DIR, "dry_bean_clean.csv")         # written by train_models.py, never re-loaded
MODEL_DIR = "model"

# display name -> saved file
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbours": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names / label text so every source looks the same."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # The UCI files spell these "AspectRation" and "roundness".
    df = df.rename(columns={"AspectRation": "AspectRatio", "roundness": "Roundness", "class": "Class"})
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[FEATURES + [TARGET]]


def _read_any(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if lower.endswith(".arff"):
        from scipy.io import arff
        records, _ = arff.loadarff(path)
        frame = pd.DataFrame(records)
        frame["Class"] = frame["Class"].str.decode("utf-8")
        return frame
    raise ValueError(f"Unsupported file type: {path}")


def load_dry_bean(local_path: str | None = None) -> pd.DataFrame:
    """Return the ORIGINAL Dry Bean data (duplicates included) as a DataFrame.

    Order of preference:
      1. an explicit file passed in (csv / xlsx / arff),
      2. any Dry_Bean_Dataset.* file in data/ (the UCI download),
      3. download the official zip from UCI and keep the .arff in data/.
    The cleaned copy written by train_models.py is deliberately not used here,
    so every run starts from the raw file and reports the same duplicates.
    """
    if local_path:
        return _tidy(_read_any(local_path))
    if os.path.isdir(DATA_DIR):
        for name in sorted(os.listdir(DATA_DIR)):
            if name.lower().startswith("dry_bean_dataset") and name.lower().endswith((".csv", ".xlsx", ".arff")):
                return _tidy(_read_any(os.path.join(DATA_DIR, name)))

    print(f"Downloading Dry Bean dataset from {UCI_ZIP_URL} ...")
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        arff_name = next(n for n in zf.namelist() if n.lower().endswith(".arff"))
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(RAW_ARFF, "wb") as fh:
            fh.write(zf.read(arff_name))
    return _tidy(_read_any(RAW_ARFF))


# ---------------------------------------------------------------------------
# Validation of an uploaded test file
# ---------------------------------------------------------------------------
def check_test_frame(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty list = file is fine)."""
    problems = []
    df = df.rename(columns=lambda c: str(c).strip())
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        problems.append("Missing column(s): " + ", ".join(missing))
        return problems
    non_numeric = [c for c in FEATURES if pd.to_numeric(df[c], errors="coerce").isna().any()]
    if non_numeric:
        problems.append("Empty or non-numeric values in: " + ", ".join(non_numeric))
    labels = set(df[TARGET].astype(str).str.strip().str.upper())
    unknown = sorted(labels - set(BEAN_CLASSES))
    if unknown:
        problems.append("Unknown bean class label(s): " + ", ".join(unknown))
    if len(df) == 0:
        problems.append("The file has no rows.")
    return problems


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def score_predictions(y_true, y_pred, proba: np.ndarray | None, model_classes) -> dict:
    """The six assignment metrics.

    Precision / Recall / F1 are macro averages (every bean variety counts
    equally). AUC is macro one-vs-rest from predicted probabilities. If the
    test file only holds some of the classes, AUC is computed over the classes
    present; with fewer than two classes it is reported as NaN.
    """
    y_true = np.asarray(y_true)
    present = sorted(set(y_true))
    labels_for_avg = present  # macro average over the varieties actually in the file

    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, labels=labels_for_avg, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, labels=labels_for_avg, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, labels=labels_for_avg, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": float("nan"),
    }

    if proba is not None and len(present) >= 2:
        model_classes = list(model_classes)
        if len(present) == len(model_classes):
            out["AUC"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=model_classes)
        else:
            # one-vs-rest AUC for each class that is actually in the file
            per_class = []
            for cls in present:
                idx = model_classes.index(cls)
                per_class.append(roc_auc_score((y_true == cls).astype(int), proba[:, idx]))
            out["AUC"] = float(np.mean(per_class))
    return {k: out[k] for k in METRIC_COLUMNS}


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rep.pop("accuracy", None)  # already shown as a metric card
    table = pd.DataFrame(rep).T
    table["support"] = table["support"].astype(int)
    return table


def confusion_frame(y_true, y_pred, labels) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)
