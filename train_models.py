"""Train, compare and save the five BeanScope classifiers.

Run from the project root:
    python train_models.py                      # downloads the data from UCI
    python train_models.py --data path/to/Dry_Bean_Dataset.xlsx   # use a local copy

Writes:
    data/dry_bean_clean.csv        dataset after removing duplicates (the raw UCI file stays in data/)
    test_data.csv                  the 20 % hold-out rows (features + Class)
    metrics_summary.csv            six metrics for each model on the hold-out rows
    model/*.joblib                 five fitted pipelines
    model/metadata.json            features, classes, chosen settings, library versions
    README.md                      the comparison table between the RESULTS markers
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from bean_utils import (
    CLEAN_CSV, FEATURES, METRIC_COLUMNS, MODEL_DIR, MODEL_FILES, TARGET,
    UCI_PAGE, load_dry_bean, score_predictions,
)

SEED = 42
TEST_SIZE = 0.20


def candidate_models():
    """Each entry: (pipeline, small hyper-parameter grid searched with 5-fold CV on the training rows only)."""
    return {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
            {"clf__C": [0.1, 1.0, 10.0, 50.0]},
        ),
        "Decision Tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [6, 8, 10, 12, None],
             "clf__min_samples_leaf": [1, 5, 10]},
        ),
        "K-Nearest Neighbours": (
            Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15],
             "clf__weights": ["uniform", "distance"]},
        ),
        "Gaussian Naive Bayes": (
            Pipeline([("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]},
        ),
        "Support Vector Machine": (
            # Search on a plain SVC (fast); probabilities for AUC are added afterwards.
            Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=SEED))]),
            {"clf__C": [1.0, 2.0, 5.0, 10.0, 20.0]},
        ),
    }


def with_probabilities(svm_pipeline: Pipeline) -> Pipeline:
    """Wrap the tuned SVC so it exposes predict_proba (needed for AUC).

    SVC(probability=True) is deprecated from scikit-learn 1.9, so the
    recommended CalibratedClassifierCV(..., ensemble=False) is used instead.
    """
    svc = svm_pipeline.named_steps["clf"]
    calibrated = CalibratedClassifierCV(SVC(kernel="rbf", C=svc.C, gamma=svc.gamma, random_state=SEED),
                                        ensemble=False, cv=5)
    return Pipeline([("scale", StandardScaler()), ("clf", calibrated)])


def write_readme_table(table: pd.DataFrame, path: str = "README.md") -> None:
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if not os.path.exists(path):
        return
    text = open(path, encoding="utf-8").read()
    if start not in text or end not in text:
        return
    lines = ["| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |",
             "|---|---|---|---|---|---|---|"]
    for name, row in table.iterrows():
        lines.append("| " + name + " | " + " | ".join(f"{row[c]:.4f}" for c in METRIC_COLUMNS) + " |")
    block = start + "\n" + "\n".join(lines) + "\n" + end
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(before + block + after)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="optional local csv / xlsx / arff copy of the Dry Bean dataset")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Load and check --------------------------------------------------------
    raw = load_dry_bean(args.data)
    print(f"Loaded {len(raw):,} rows x {len(FEATURES)} features  (source: {UCI_PAGE})")
    n_missing = int(raw.isna().sum().sum())
    n_dupes = int(raw.duplicated().sum())
    print(f"Missing values: {n_missing}   Exact duplicate rows: {n_dupes}")

    # Duplicates are removed before the split so the same bean cannot appear
    # in both the training and the test rows (that would inflate test scores).
    data = raw.dropna().drop_duplicates().reset_index(drop=True)
    print(f"Rows after removing duplicates/missing: {len(data):,}")
    data.to_csv(CLEAN_CSV, index=False)

    X, y = data[FEATURES], data[TARGET]

    # 2. One stratified split shared by all five models ------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"Train rows: {len(X_train):,}   Test rows: {len(X_test):,}")

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv("test_data.csv", index=False)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, settings, cv_scores = {}, {}, {}

    # 3. Tune, fit, evaluate, save --------------------------------------------
    for name, (pipe, grid) in candidate_models().items():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=folds, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        if name == "Support Vector Machine":
            best = with_probabilities(best).fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = best.predict(X_test)
        proba = best.predict_proba(X_test)
        scores = score_predictions(y_test, y_pred, proba, best.classes_)
        results[name] = scores
        settings[name] = {k.replace("clf__", ""): (v if v is not None else "None")
                          for k, v in search.best_params_.items()}
        cv_scores[name] = round(float(search.best_score_), 4)

        joblib.dump(best, os.path.join(MODEL_DIR, MODEL_FILES[name]))
        print(f"{name:<24} best={settings[name]}  cvF1={cv_scores[name]:.4f}  "
              f"testAcc={scores['Accuracy']:.4f}  ({elapsed:.1f}s)")

    # 4. Comparison table --------------------------------------------------
    table = pd.DataFrame(results).T[METRIC_COLUMNS]
    table.index.name = "ML Model Name"
    table.round(4).to_csv("metrics_summary.csv")
    print("\n" + table.round(4).to_string())

    leaders = {m: table[m].idxmax() for m in METRIC_COLUMNS}
    print("\nBest model per metric:")
    for m, who in leaders.items():
        print(f"  {m:<9} {who}  ({table.loc[who, m]:.4f})")

    meta = {
        "dataset": "UCI Dry Bean Dataset (id 602), CC BY 4.0, DOI 10.24432/C50S4B",
        "rows_loaded": len(raw), "duplicates_removed": n_dupes, "rows_used": len(data),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "features": FEATURES, "target": TARGET, "classes": sorted(y.unique().tolist()),
        "class_counts": {"full": data[TARGET].value_counts().to_dict(),
                         "train": y_train.value_counts().to_dict(),
                         "test": y_test.value_counts().to_dict()},
        "split": {"test_size": TEST_SIZE, "stratified": True, "random_state": SEED},
        "tuned_settings": settings, "cv_macro_f1": cv_scores,
        "test_metrics": table.round(4).to_dict(orient="index"),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    write_readme_table(table)
    print("\nSaved five models to model/, test_data.csv, metrics_summary.csv; README table updated.")


if __name__ == "__main__":
    main()
