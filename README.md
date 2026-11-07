# BeanScope — Dry Bean Variety Classification

Five classifiers trained on the UCI Dry Bean dataset, compared on six metrics, and served through an interactive Streamlit app.

- **GitHub repository:** https://github.com/Ayush281405/beanscope-dry-bean-classifier
- **Live Streamlit app:** https://beanscope-2026ak05601.streamlit.app

---

## a. Problem statement

A computer-vision grading system photographs dry beans and measures each bean's shape: its area, perimeter, axis lengths, roundness, compactness and four shape factors. Sorting beans by hand into the seven registered varieties is slow and inconsistent, so the question is whether those 16 shape measurements are enough to identify the variety automatically.

This is a **multi-class classification** problem. The target is `Class`, one of seven varieties: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA. The project trains Logistic Regression, a Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and a Support Vector Machine on the same stratified split, compares them on Accuracy, AUC, Precision, Recall, F1 and MCC, saves the fitted pipelines, and deploys a Streamlit app (BeanScope) that scores an uploaded test file with any of the saved models without retraining.

## b. Dataset description

| Item | Details |
|---|---|
| Dataset name and source | Dry Bean Dataset, UCI Machine Learning Repository (id 602): https://archive.ics.uci.edu/dataset/602/dry+bean+dataset — Koklu, M. & Ozkan, I.A. (2020). Licence CC BY 4.0, DOI 10.24432/C50S4B |
| Type of problem | Multi-class classification (7 classes) |
| Number of instances | 13,611 beans in the original file; 13,543 used after removing 68 exact duplicates |
| Number of features | 16 numeric features (min. required: 12) |
| Features | 12 dimension features — Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness — and 4 shape factors, ShapeFactor1–4 |
| Target variable and classes | `Class`: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |
| Missing values / duplicates | No missing values. 68 exact duplicate rows were found (all of them HOROZ) and removed **before** the split, so an identical bean cannot sit in both training and test data |
| Preprocessing | All features are numeric, so no encoding is needed. `StandardScaler` is placed inside the pipeline for Logistic Regression, KNN and SVM (features range from ~0.001 for ShapeFactors to >200,000 for Area, which would dominate distances). Decision Tree and Naive Bayes use the raw values. The scaler is fitted on training rows only and is saved inside each model file |
| Train / test split | 80 % / 20 %, stratified on `Class`, `random_state=42`; the same split is used for all five models. 10,834 training rows, 2,709 test rows (saved as `test_data.csv`) |
| Class distribution (after removing duplicates) | DERMASON 3,546 · SIRA 2,636 · SEKER 2,027 · HOROZ 1,860 · CALI 1,630 · BARBUNYA 1,322 · BOMBAY 522. The largest class is about 6.8× the smallest, which is why macro averages and MCC are reported |
| Class distribution (train / test) | Train: DERMASON 2,837 · SIRA 2,109 · SEKER 1,621 · HOROZ 1,488 · CALI 1,304 · BARBUNYA 1,057 · BOMBAY 418. Test: DERMASON 709 · SIRA 527 · SEKER 406 · HOROZ 372 · CALI 326 · BARBUNYA 265 · BOMBAY 104 |

**Metric definitions used.** Precision, Recall and F1 are **macro** averages, so the small BOMBAY class counts as much as DERMASON. AUC is macro **one-vs-rest**, computed from predicted probabilities. MCC is the multi-class Matthews correlation coefficient.

**Hyper-parameters.** Each model gets a small grid search with 5-fold stratified cross-validation on the training rows only (scored by macro F1); the test rows are used once, for the final numbers. Chosen settings: Logistic Regression `C=50` (the largest value in the grid, i.e. very little regularisation was wanted); Decision Tree `max_depth=10`, `min_samples_leaf=10`; KNN `k=11`, distance-weighted votes; Gaussian NB default `var_smoothing=1e-9`; SVM RBF kernel `C=10`, `gamma='scale'`. Best 5-fold CV macro-F1 on the training rows: SVM 0.9437, LR 0.9377, KNN 0.9374, DT 0.9238, NB 0.7632 — and SVM leads on the test rows too, so its win is not a lucky split. The SVM's probabilities come from `CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False)` because `SVC(probability=True)` is deprecated from scikit-learn 1.9.

## c. GitHub repository link

https://github.com/Ayush281405/beanscope-dry-bean-classifier

```
├── app.py                  Streamlit app (loads saved models, never retrains)
├── bean_utils.py           shared feature list, data loader, CSV validation, metric functions
├── train_models.py         split, tune, train, evaluate and save the five models
├── ML_Assignment_2.ipynb   notebook: EDA, the five models, comparison, saved-model check
├── requirements.txt        pinned dependencies
├── README.md
├── test_data.csv           the 20 % hold-out rows (16 features + Class)
├── metrics_summary.csv     comparison table written by train_models.py
├── .streamlit/config.toml  app theme
├── data/Dry_Bean_Dataset.arff  original UCI file
├── data/dry_bean_clean.csv     after removing the 68 duplicates
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metadata.json       features, classes, chosen settings, class counts, library versions
```

**Run locally**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_models.py        # reads data/Dry_Bean_Dataset.arff (downloads it from UCI if missing)
streamlit run app.py
```

## d. Models used

Results on the 2,709 held-out test rows (this table is written automatically by `train_models.py`; best value per column: SVM for Accuracy, Precision, Recall, F1 and MCC, Logistic Regression for AUC):

<!-- RESULTS:START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9203 | 0.9949 | 0.9319 | 0.9297 | 0.9306 | 0.9037 |
| Decision Tree | 0.9044 | 0.9797 | 0.9179 | 0.9166 | 0.9171 | 0.8844 |
| K-Nearest Neighbours | 0.9147 | 0.9888 | 0.9292 | 0.9257 | 0.9271 | 0.8969 |
| Gaussian Naive Bayes | 0.7630 | 0.9670 | 0.7736 | 0.7685 | 0.7669 | 0.7143 |
| Support Vector Machine | 0.9240 | 0.9904 | 0.9360 | 0.9344 | 0.9351 | 0.9080 |
<!-- RESULTS:END -->

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A very good result for a simple linear model. It came second on Accuracy (0.9203), Precision, Recall, F1 and MCC (0.9037), and first on AUC (0.9949), so it ranks the correct variety very high even when its first choice is wrong. The grid search picked C=50, which means it needed almost no regularisation once the features were scaled. Its weakest varieties were SIRA (recall 0.890) and BARBUNYA (recall 0.894), and most of its mistakes were SIRA and DERMASON being confused with each other. |
| Decision Tree | The weakest of the four good models: Accuracy 0.9044, F1 0.9171, MCC 0.8844. A tree splits on one feature at a time, so it draws step-shaped boundaries between varieties whose real boundaries are smooth and diagonal. That hurts it most on SIRA (recall 0.829, the lowest among the four good models). Its AUC (0.9797) is also lower because a leaf gives the same rough probability to every bean in it. Limiting depth to 10 and needing at least 10 beans per leaf stopped it from overfitting, and it is the easiest model to explain. |
| KNN | Close behind Logistic Regression: Accuracy 0.9147, F1 0.9271, MCC 0.8969, with k=11 and closer neighbours given more weight. Scaling was essential here, because without it Area (tens of thousands) would decide every distance and the small shape factors (below 0.01) would be ignored. It classified every BOMBAY bean correctly (recall 1.00) and had the best CALI recall (0.951), but it had the lowest BARBUNYA recall of the good models (0.875) and confused SIRA with DERMASON. It also has to keep all 10,834 training beans to make predictions, so it is the slowest at prediction time. |
| Naive Bayes | Clearly the worst model: Accuracy 0.7630, F1 0.7669, MCC 0.7143. Naive Bayes assumes every feature is independent, but in this dataset many features are almost copies of each other (Area and ConvexArea have a correlation of 1.00, Compactness and ShapeFactor3 0.999), so it counts the same information several times. Because of this it got only 46% of BARBUNYA beans right (108 of 265 were predicted as CALI) and called 85 DERMASON beans SEKER. Its AUC is still 0.9670, so the right answer is usually near the top of its list, just not first. |
| Support Vector Machine | The best model overall: highest Accuracy (0.9240), Precision (0.9360), Recall (0.9344), F1 (0.9351) and MCC (0.9080), and second-best AUC (0.9904). The RBF kernel with C=10 can draw curved boundaries, which suits varieties that overlap in shape. It had the best recall on BARBUNYA (0.913), DERMASON (0.924) and SEKER (0.953). Its remaining mistakes are mostly SIRA and DERMASON, which every model mixes up, so these two varieties really are hard to separate using shape alone. |
| **Overall winner for this dataset** | **Support Vector Machine.** It is best on 5 of the 6 metrics, including MCC and macro F1, which are the fairest measures when the seven classes have very different sizes. It also had the best cross-validation score on the training data (macro F1 0.9437), so its win is not luck from one split. Logistic Regression is a close second (MCC 0.9037) and would be a good choice if a simpler model or the best probability ranking (AUC) mattered more. Naive Bayes is not suitable here because the features are so strongly correlated. |

## e. Streamlit app (BeanScope)

| Required feature | Implemented | How |
|---|---|---|
| Dataset upload (CSV, test data only) | Yes | Sidebar uploader. If nothing is uploaded the bundled `test_data.csv` is used, and a button downloads it as a sample. The file is checked for all 16 features, the `Class` column, numeric values and valid labels, with a clear error otherwise |
| Model selection dropdown | Yes | Sidebar dropdown with the five models plus "Compare all five models"; shows a one-line description and the tuned settings of the chosen model |
| Display of evaluation metrics | Yes | Six metric cards (Accuracy, AUC, Precision, Recall, F1, MCC), each with the model's rank out of five on the current file |
| Confusion matrix / classification report | Yes | Confusion matrix (counts or row %) next to the per-class classification report, plus the single most frequent mix-up |

Extra: a **Model comparison** tab (all six metrics for all five models, best value highlighted, bar chart for any metric), a **Data** tab (variety counts, first rows, feature summary), and downloads for the predictions and the comparison table. If an uploaded file contains only some varieties, macro scores and AUC are averaged over the varieties present, and AUC shows n/a when only one variety is present.

The app only calls `joblib.load` and `predict` / `predict_proba` on the files in `model/`; it never fits a model on the uploaded data.
