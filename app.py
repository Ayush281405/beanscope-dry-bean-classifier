"""BeanScope - Streamlit front end for the five saved Dry Bean classifiers.

The app only *loads* the pipelines saved by train_models.py and scores the
uploaded test file. Nothing here calls .fit(), so the uploaded data is never
used for training.
"""

from __future__ import annotations

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from bean_utils import (
    BEAN_CLASSES, FEATURES, METRIC_COLUMNS, MODEL_DIR, MODEL_FILES, TARGET,
    check_test_frame, confusion_frame, per_class_table, score_predictions,
)

BUNDLED_TEST = "test_data.csv"
COMPARE_ALL = "Compare all five models"

MODEL_NOTES = {
    "Logistic Regression": "Linear decision boundaries on standardised shape measurements.",
    "Decision Tree": "Learns if/else thresholds on raw measurements; easy to read, prone to overfitting.",
    "K-Nearest Neighbours": "Labels a bean by the majority vote of its closest training beans (after scaling).",
    "Gaussian Naive Bayes": "Assumes every measurement is independent and normally distributed within a class.",
    "Support Vector Machine": "RBF-kernel SVM with calibrated probabilities, so curved class boundaries are possible.",
}

st.set_page_config(page_title="BeanScope", page_icon="🫘", layout="wide")

st.markdown(
    """
    <style>
      .bean-hero {padding: 1.1rem 1.4rem; border-radius: 14px;
                  background: linear-gradient(120deg, #5b3a29 0%, #8c5a3c 55%, #c89b6d 100%);
                  color: #fff8ef; margin-bottom: 1rem;}
      .bean-hero h1 {margin: 0; font-size: 2.1rem; color: #fff8ef;}
      .bean-hero p  {margin: .35rem 0 0; opacity: .92;}
      .metric-card {border: 1px solid #e6d6c3; border-radius: 12px; padding: .7rem .9rem;
                    background: #fffaf3;}
      .metric-card .lbl {font-size: .75rem; letter-spacing: .06em; color: #8c5a3c; text-transform: uppercase;}
      .metric-card .val {font-size: 1.55rem; font-weight: 700; color: #3b2418;}
      .metric-card .rank {font-size: .72rem; color: #7a6a5c;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading saved models ...")
def load_models() -> dict:
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_data
def load_metadata() -> dict:
    path = os.path.join(MODEL_DIR, "metadata.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


@st.cache_data
def read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


@st.cache_data
def evaluate(_model, model_name: str, data_key: str, X: pd.DataFrame, y: pd.Series):
    """Predict + score. data_key is only there so Streamlit caches per file."""
    y_pred = _model.predict(X)
    proba = _model.predict_proba(X) if hasattr(_model, "predict_proba") else None
    scores = score_predictions(y, y_pred, proba, _model.classes_)
    return y_pred, proba, scores


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------
def fmt(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def metric_cards(scores: dict, ranks: dict | None = None) -> None:
    cols = st.columns(6)
    names = {"Accuracy": "Accuracy", "AUC": "AUC (macro OvR)", "Precision": "Precision (macro)",
             "Recall": "Recall (macro)", "F1": "F1 (macro)", "MCC": "MCC"}
    for col, key in zip(cols, METRIC_COLUMNS):
        rank_txt = f"rank {ranks[key]} of 5" if ranks and key in ranks else "&nbsp;"
        col.markdown(
            f"<div class='metric-card'><div class='lbl'>{names[key]}</div>"
            f"<div class='val'>{fmt(scores[key])}</div><div class='rank'>{rank_txt}</div></div>",
            unsafe_allow_html=True,
        )


def plot_confusion(cm: pd.DataFrame, as_percent: bool):
    values = cm.values.astype(float)
    if as_percent:
        row_sum = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum > 0) * 100
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(values, cmap="YlOrBr")
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)), cm.index)
    ax.set_xlabel("Predicted variety")
    ax.set_ylabel("Actual variety")
    threshold = values.max() * 0.6 if values.max() > 0 else 1
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            txt = f"{values[i, j]:.1f}" if as_percent else f"{int(values[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] > threshold else "#3b2418")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_metric_bars(table: pd.DataFrame, metric: str):
    ordered = table[metric].sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(ordered.index, ordered.values, color="#a06b45")
    bars[-1].set_color("#5b3a29")
    lo = max(0.0, float(np.nanmin(ordered.values)) - 0.05)
    ax.set_xlim(lo, 1.0)
    for b, v in zip(bars, ordered.values):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2, fmt(v), va="center", fontsize=8)
    ax.set_xlabel(metric)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar: data + model choice
# ---------------------------------------------------------------------------
models = load_models()
meta = load_metadata()

st.sidebar.header("1 · Test data")
uploaded = st.sidebar.file_uploader("Upload a test CSV", type=["csv"],
                                    help="Must contain the 16 feature columns and the Class column.")
if os.path.exists(BUNDLED_TEST):
    with open(BUNDLED_TEST, "rb") as fh:
        st.sidebar.download_button("Download sample test_data.csv", fh.read(),
                                   file_name="test_data.csv", mime="text/csv")

with st.sidebar.expander("Expected CSV format"):
    st.write("Feature columns (numeric):")
    st.code(", ".join(FEATURES), language=None)
    st.write(f"Target column: `{TARGET}` with one of: " + ", ".join(BEAN_CLASSES))
    st.caption("Extra columns are ignored; column order does not matter.")

st.sidebar.header("2 · Model")
choice = st.sidebar.selectbox("Choose a classifier", list(MODEL_FILES) + [COMPARE_ALL])
if choice in MODEL_NOTES:
    st.sidebar.info(MODEL_NOTES[choice])
    tuned = meta.get("tuned_settings", {}).get(choice)
    if tuned:
        st.sidebar.caption("Tuned settings: " + ", ".join(f"{k}={v}" for k, v in tuned.items()))

st.sidebar.markdown("---")
st.sidebar.caption("Models are loaded from `model/` and are **never retrained** on the uploaded file.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='bean-hero'><h1>🫘 BeanScope</h1>"
    "<p>Seven dry-bean varieties, sixteen shape measurements from a camera image, five saved classifiers. "
    "Pick a model and BeanScope scores it on your held-out test file.</p></div>",
    unsafe_allow_html=True,
)

if len(models) < len(MODEL_FILES):
    st.error("Some saved models are missing from `model/`. Run `python train_models.py` first, "
             "then commit the model folder.")
    st.stop()

# ---------------------------------------------------------------------------
# Read + validate the test file
# ---------------------------------------------------------------------------
if uploaded is not None:
    try:
        df = read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read that file as CSV: {exc}")
        st.stop()
    source_name, source_kind = uploaded.name, "uploaded"
elif os.path.exists(BUNDLED_TEST):
    df = read_csv(BUNDLED_TEST)
    source_name, source_kind = BUNDLED_TEST, "bundled"
else:
    st.warning("Upload a test CSV in the sidebar to begin.")
    st.stop()

df = df.rename(columns=lambda c: str(c).strip())
problems = check_test_frame(df)
if problems:
    st.error("This file can't be scored:\n\n- " + "\n- ".join(problems))
    st.stop()

X = df[FEATURES].apply(pd.to_numeric)
y = df[TARGET].astype(str).str.strip().str.upper()
data_key = f"{source_name}-{len(df)}-{pd.util.hash_pandas_object(df, index=False).sum()}"

c1, c2, c3 = st.columns(3)
c1.metric(f"Test file ({source_kind})", source_name)
c2.metric("Rows", f"{len(df):,}")
c3.metric("Varieties present", f"{y.nunique()} of {len(BEAN_CLASSES)}")
if y.nunique() < len(BEAN_CLASSES):
    st.caption("Not every variety is in this file, so AUC and macro averages use only the varieties present.")

# Score every model once (cheap, cached) so ranks and the comparison are available.
all_scores, all_preds = {}, {}
for name, mdl in models.items():
    pred, _, sc = evaluate(mdl, name, data_key, X, y)
    all_scores[name], all_preds[name] = sc, pred
table = pd.DataFrame(all_scores).T[METRIC_COLUMNS]
ranks = {m: table[m].rank(ascending=False, method="min") for m in METRIC_COLUMNS}

tab_eval, tab_compare, tab_data = st.tabs(["Evaluation", "Model comparison", "Data"])

# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------
with tab_eval:
    focus = choice if choice != COMPARE_ALL else st.selectbox(
        "Show details for", list(MODEL_FILES), key="detail_pick")
    scores = all_scores[focus]
    y_pred = all_preds[focus]
    correct = int((np.asarray(y_pred) == y.values).sum())
    st.subheader(focus)
    st.write(f"Correctly classified **{correct:,}** of **{len(y):,}** beans "
             f"({correct / len(y):.1%}).")
    metric_cards(scores, {m: int(ranks[m][focus]) for m in METRIC_COLUMNS if not np.isnan(table.loc[focus, m])})

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Confusion matrix**")
        view = st.radio("Cells show", ["Counts", "Row %"], horizontal=True, key="cm_view")
        labels = [c for c in BEAN_CLASSES if c in set(y) | set(y_pred)]
        fig = plot_confusion(confusion_frame(y, y_pred, labels), view == "Row %")
        st.pyplot(fig, clear_figure=True)
    with right:
        st.markdown("**Classification report**")
        rep = per_class_table(y, y_pred)
        st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                       "f1-score": "{:.3f}", "support": "{:d}"}),
                     width="stretch")
        # biggest single mix-up
        cm = confusion_frame(y, y_pred, labels)
        off = cm.values.copy()
        np.fill_diagonal(off, 0)
        if off.max() > 0:
            i, j = np.unravel_index(off.argmax(), off.shape)
            st.caption(f"Most frequent mistake: {cm.index[i]} predicted as {cm.columns[j]} "
                       f"({off[i, j]} beans).")

    out = df.copy()
    out["Predicted"] = y_pred
    out["Correct"] = out["Predicted"] == y.values
    st.download_button("Download predictions as CSV", out.to_csv(index=False).encode(),
                       file_name=f"beanscope_predictions_{focus.replace(' ', '_').lower()}.csv",
                       mime="text/csv")

# ---------------------------------------------------------------------------
# Comparison tab
# ---------------------------------------------------------------------------
with tab_compare:
    st.markdown("**All five models on this test file**")
    st.dataframe(table.style.format("{:.4f}", na_rep="n/a").highlight_max(axis=0, color="#f3dcc0"),
                 width="stretch")
    pick = st.selectbox("Metric to plot", METRIC_COLUMNS, index=METRIC_COLUMNS.index("F1"))
    st.pyplot(plot_metric_bars(table, pick), clear_figure=True)
    best_mcc = table["MCC"].idxmax()
    st.info(f"Highest MCC on this file: **{best_mcc}** ({table.loc[best_mcc, 'MCC']:.4f}). "
            "MCC uses every cell of the confusion matrix, so it is the fairest single number here.")
    st.download_button("Download comparison table", table.round(4).to_csv().encode(),
                       file_name="beanscope_model_comparison.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Data tab
# ---------------------------------------------------------------------------
with tab_data:
    a, b = st.columns([1, 1.4])
    with a:
        st.markdown("**Varieties in this file**")
        counts = y.value_counts().reindex(BEAN_CLASSES).fillna(0).astype(int)
        st.bar_chart(counts, color="#8c5a3c")
    with b:
        st.markdown("**First rows**")
        st.dataframe(df.head(15), width="stretch")
    st.markdown("**Feature summary**")
    st.dataframe(X.describe().T.style.format("{:.4f}"), width="stretch")
    if meta:
        st.caption(f"Models trained on {meta.get('train_rows', 0):,} rows with scikit-learn "
                   f"{meta.get('versions', {}).get('scikit-learn', '?')}; "
                   f"{meta.get('duplicates_removed', 0)} duplicate rows were removed before splitting.")
