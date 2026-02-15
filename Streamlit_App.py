import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)

from models import fetch_pipeline


st.set_page_config(page_title="Breast Cancer Diagnostic Dashboard", layout="wide")
st.title("Breast Cancer Diagnostic Dashboard")


# ============================================================
# LEFT PANE — MODE SELECTION
# ============================================================

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Default Dataset Analysis",
        "Upload Dataset Analysis",
        "Live Predictor"
    ]
)


# ============================================================
# 1. DEFAULT DATASET ANALYSIS (AUTO)
# ============================================================

if mode == "Default Dataset Analysis":

    data_frame = pd.read_csv("Data.csv")
    data_frame.columns = data_frame.columns.str.strip()

    st.markdown("## Default Dataset Sample")
    st.dataframe(data_frame.head(), use_container_width=True)

    feature_data = data_frame.drop(columns=["diagnosis"])
    target_data = data_frame["diagnosis"]

    if target_data.dtype == "object":
        target_data = LabelEncoder().fit_transform(target_data)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, target_data, test_size=0.2, random_state=42
    )

    st.markdown("## Model Comparison on Default Dataset")

    models_list = [
        "Logistic Regression", "Decision Tree", "KNN",
        "Naive Bayes", "Random Forest", "XGBoost"
    ]

    results = []

    for name in models_list:
        model = fetch_pipeline(name, X_train)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted")
        rec = recall_score(y_test, preds, average="weighted")
        f1 = f1_score(y_test, preds, average="weighted")
        mcc = matthews_corrcoef(y_test, preds)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
        else:
            auc = np.nan

        results.append([name, acc, prec, rec, f1, mcc, auc])

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "Precision", "Recall", "F1", "MCC", "ROC AUC"]
    )

    st.dataframe(
        results_df.style.background_gradient(
            cmap="YlOrBr",
            subset=results_df.columns[1:]
        ),
        use_container_width=True
    )


# ============================================================
# 2. UPLOAD DATASET ANALYSIS (SEPARATE FLOW)
# ============================================================

elif mode == "Upload Dataset Analysis":

    uploaded_file = st.file_uploader("Upload dataset", type=["csv"])

    if uploaded_file is not None:

        data_frame = pd.read_csv(uploaded_file)
        data_frame.columns = data_frame.columns.str.strip()

        st.markdown("## Uploaded Dataset Sample")
        st.dataframe(data_frame.head(), use_container_width=True)

        if "diagnosis" not in data_frame.columns:
            st.error("diagnosis column required")
            st.stop()

        feature_data = data_frame.drop(columns=["diagnosis"])
        target_data = data_frame["diagnosis"]

        if target_data.dtype == "object":
            target_data = LabelEncoder().fit_transform(target_data)

        X_train, X_test, y_train, y_test = train_test_split(
            feature_data, target_data, test_size=0.2, random_state=42
        )

        model_choice = st.selectbox(
            "Choose model",
            [
                "Logistic Regression", "Decision Tree", "KNN",
                "Naive Bayes", "Random Forest", "XGBoost"
            ]
        )

        if st.button("Run Model"):

            model = fetch_pipeline(model_choice, X_train)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted")
            rec = recall_score(y_test, preds, average="weighted")
            f1_val = f1_score(y_test, preds, average="weighted")
            mcc_val = matthews_corrcoef(y_test, preds)

            st.markdown("## Evaluation")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("Precision", f"{prec:.4f}")
            c3.metric("Recall", f"{rec:.4f}")
            c4.metric("F1 Score", f"{f1_val:.4f}")
            c5.metric("MCC", f"{mcc_val:.4f}")


# ============================================================
# 3. LIVE PREDICTOR (SEPARATE MODE)
# ============================================================

elif mode == "Live Predictor":

    default_data = pd.read_csv("Data.csv")
    default_data.columns = default_data.columns.str.strip()

    X_live = default_data.drop(columns=["diagnosis"])
    y_live = default_data["diagnosis"]

    if y_live.dtype == "object":
        y_live = LabelEncoder().fit_transform(y_live)

    numeric_cols = X_live.select_dtypes(include=np.number).columns

    st.markdown("## Live Tumor Predictor")

    model_choice = st.selectbox(
        "Select model",
        [
            "Logistic Regression", "Decision Tree", "KNN",
            "Naive Bayes", "Random Forest", "XGBoost"
        ]
    )

    st.markdown("### Enter Measurements")

    input_data = {}
    cols = st.columns(2)

    for i, col in enumerate(numeric_cols[:10]):
        with cols[i % 2]:
            input_data[col] = st.number_input(
                col,
                value=float(X_live[col].median())
            )

    if st.button("Predict"):

        model = fetch_pipeline(model_choice, X_live)
        model.fit(X_live, y_live)

        pred = model.predict(pd.DataFrame([input_data]))[0]
        result = "Malignant (cancerous)" if pred == 1 else "Benign (non-cancerous)"

        st.success(f"Prediction Result: {result}")