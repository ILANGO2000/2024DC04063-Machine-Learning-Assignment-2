import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from models import fetch_pipeline


st.set_page_config(page_title="Breast Cancer Diagnostic Dashboard", layout="wide")
st.title("Breast Cancer Diagnostic Dashboard")


st.markdown("## Dataset Selection")

data_source = st.radio(
    "Choose data source",
    ["Use default data", "Upload your own data"]
)

if "data_frame" not in st.session_state:
    st.session_state["data_frame"] = None

if "prev_source" not in st.session_state:
    st.session_state["prev_source"] = data_source

if st.session_state["prev_source"] != data_source:
    st.session_state["data_frame"] = None
    st.session_state["prev_source"] = data_source


# ---------- DEFAULT DATA AUTO LOAD ----------
if data_source == "Use default data":

    if st.session_state["data_frame"] is None:
        try:
            st.session_state["data_frame"] = pd.read_csv("Data.csv")
        except:
            st.error("Default dataset not found")
            st.stop()


# ---------- UPLOAD DATA ----------
else:

    uploaded_csv = st.file_uploader("Upload your dataset", type=["csv"])

    if uploaded_csv is not None:
        st.session_state["data_frame"] = pd.read_csv(uploaded_csv)


data_frame = st.session_state["data_frame"]


if data_frame is not None:

    data_frame.columns = data_frame.columns.str.strip()

    if "diagnosis" not in data_frame.columns:
        st.error("Target column 'diagnosis' not found")
        st.stop()

    st.markdown("### Dataset Sample")
    st.dataframe(data_frame.head(), use_container_width=True)


    feature_data = data_frame.drop(columns=["diagnosis"])
    target_data = data_frame["diagnosis"]

    if target_data.dtype == "object":
        encoder = LabelEncoder()
        target_data = encoder.fit_transform(target_data)


    X_train, X_test, y_train, y_test = train_test_split(
        feature_data,
        target_data,
        test_size=0.2,
        random_state=42
    )


    # ============================================================
    # DEFAULT DATA → RUN ALL MODELS
    # ============================================================
    if data_source == "Use default data":

        st.markdown("## Model Comparison on Default Dataset")

        models_list = [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
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

        styled = results_df.style.background_gradient(
            cmap="YlOrBr",
            subset=results_df.columns[1:]
        )

        st.dataframe(styled, use_container_width=True)


    # ============================================================
    # UPLOADED DATA → USER SELECTS MODEL
    # ============================================================
    else:

        model_choice = st.selectbox(
            "Choose a classification algorithm",
            [
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Random Forest",
                "XGBoost"
            ]
        )

        if st.button("Run Model"):

            with st.spinner("Model is running..."):

                trained_pipeline = fetch_pipeline(model_choice, X_train)
                trained_pipeline.fit(X_train, y_train)
                predictions = trained_pipeline.predict(X_test)

                if hasattr(trained_pipeline, "predict_proba"):
                    probabilities = trained_pipeline.predict_proba(X_test)[:, 1]
                    auc_value = roc_auc_score(y_test, probabilities)
                else:
                    auc_value = None

                acc = accuracy_score(y_test, predictions)
                prec = precision_score(y_test, predictions, average="weighted")
                rec = recall_score(y_test, predictions, average="weighted")
                f1_val = f1_score(y_test, predictions, average="weighted")
                mcc_val = matthews_corrcoef(y_test, predictions)


            st.markdown("## Model Evaluation")

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("Precision", f"{prec:.4f}")
            c3.metric("Recall", f"{rec:.4f}")
            c4.metric("F1 Score", f"{f1_val:.4f}")
            c5.metric("MCC", f"{mcc_val:.4f}")

            if auc_value is not None:
                st.metric("ROC AUC", f"{auc_value:.4f}")


            st.markdown("## Confusion Matrix")

            matrix = confusion_matrix(y_test, predictions)
            fig, ax = plt.subplots()
            sns.heatmap(matrix, annot=True, fmt="d", cmap="coolwarm", ax=ax)
            st.pyplot(fig)


            st.markdown("## Classification Report")

            report = classification_report(y_test, predictions, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)


            st.markdown("## Insights")

            col1, col2 = st.columns(2)

            with col1:
                fig1, ax1 = plt.subplots()
                data_frame["diagnosis"].value_counts().plot(kind="bar", ax=ax1)
                st.pyplot(fig1)

            with col2:
                numeric_df = data_frame.select_dtypes(include=np.number)
                if numeric_df.shape[1] > 1:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    sns.heatmap(numeric_df.corr(), cmap="viridis", ax=ax2)
                    st.pyplot(fig2)

            col3, col4 = st.columns(2)

            with col3:
                if hasattr(trained_pipeline, "predict_proba"):
                    prob_values = trained_pipeline.predict_proba(X_test)[:, 1]
                    fig3, ax3 = plt.subplots()
                    ax3.hist(prob_values, bins=20)
                    st.pyplot(fig3)

            with col4:
                error_flags = predictions != y_test
                fig4, ax4 = plt.subplots()
                ax4.bar(
                    ["Correct", "Incorrect"],
                    [(~error_flags).sum(), error_flags.sum()]
                )
                st.pyplot(fig4)