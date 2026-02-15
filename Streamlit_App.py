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
    confusion_matrix,
    roc_curve
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


if data_source == "Use default data":

    if st.session_state["data_frame"] is None:
        try:
            st.session_state["data_frame"] = pd.read_csv("Data.csv")
        except:
            st.error("Default dataset not found")
            st.stop()

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
                    fpr, tpr, _ = roc_curve(y_test, probabilities)
                else:
                    auc_value = None
                    fpr, tpr = None, None

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


            st.markdown("## Evaluation Plots")

            left, right = st.columns(2)

            with left:
                st.subheader("Confusion Matrix")
                matrix = confusion_matrix(y_test, predictions)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(matrix, annot=True, fmt="d", cmap="coolwarm", ax=ax_cm)
                ax_cm.set_xlabel("Predicted Label")
                ax_cm.set_ylabel("Actual Label")
                st.pyplot(fig_cm)

            with right:
                st.subheader("ROC Curve")
                if fpr is not None:
                    fig_roc, ax_roc = plt.subplots()
                    ax_roc.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
                    ax_roc.plot([0, 1], [0, 1], linestyle="--")
                    ax_roc.set_xlabel("False Positive Rate")
                    ax_roc.set_ylabel("True Positive Rate")
                    ax_roc.legend()
                    st.pyplot(fig_roc)


            st.markdown("## Classification Report")

            report = classification_report(y_test, predictions, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)


            st.markdown("## Insights")

            col1, col2 = st.columns(2)

            with col1:
                st.container(border=True)
                fig1, ax1 = plt.subplots()
                data_frame["diagnosis"].value_counts().plot(kind="bar", ax=ax1, legend=True)
                ax1.set_title("Tumor Class Distribution")
                ax1.set_xlabel("Class")
                ax1.set_ylabel("Count")
                st.pyplot(fig1)

            with col2:
                st.container(border=True)
                numeric_df = data_frame.select_dtypes(include=np.number)
                if numeric_df.shape[1] > 1:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    sns.heatmap(numeric_df.corr(), cmap="viridis", ax=ax2)
                    ax2.set_title("Feature Correlation Heatmap")
                    st.pyplot(fig2)

            col3, col4 = st.columns(2)

            with col3:
                st.container(border=True)
                if hasattr(trained_pipeline, "predict_proba"):
                    prob_values = trained_pipeline.predict_proba(X_test)[:, 1]
                    fig3, ax3 = plt.subplots()
                    ax3.hist(prob_values, bins=20, label="Probability")
                    ax3.set_title("Prediction Confidence")
                    ax3.set_xlabel("Predicted Probability")
                    ax3.set_ylabel("Frequency")
                    ax3.legend()
                    st.pyplot(fig3)

            with col4:
                st.container(border=True)
                error_flags = predictions != y_test
                fig4, ax4 = plt.subplots()
                ax4.bar(
                    ["Correct", "Incorrect"],
                    [(~error_flags).sum(), error_flags.sum()],
                    label="Counts"
                )
                ax4.set_title("Prediction Error")
                ax4.set_xlabel("Outcome")
                ax4.set_ylabel("Count")
                ax4.legend()
                st.pyplot(fig4)


            # ---------------- MODEL INSIGHT SUMMARY ----------------
            st.markdown("## Model Insight Summary")

            total_cases = len(y_test)
            error_count = (predictions != y_test).sum()
            error_rate = error_count / total_cases

            class_counts = data_frame["diagnosis"].value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()

            if auc_value is not None:
                if auc_value > 0.90:
                    auc_text = "excellent discrimination ability"
                elif auc_value > 0.80:
                    auc_text = "strong discrimination ability"
                elif auc_value > 0.70:
                    auc_text = "moderate discrimination ability"
                else:
                    auc_text = "limited discrimination ability"
            else:
                auc_text = "unknown discrimination ability"

            if imbalance_ratio > 1.5:
                balance_text = "The dataset exhibits class imbalance"
            else:
                balance_text = "The dataset is reasonably balanced"

            st.write(
                f"The selected {model_choice} model achieved an accuracy of {acc:.2%} "
                f"with an error rate of {error_rate:.2%} on the test data."
            )

            st.write(
                f"The ROC analysis indicates {auc_text}, demonstrating the model's "
                f"ability to distinguish between malignant and benign tumor cases."
            )

            st.write(
                f"{balance_text}. Misclassified instances observed in the error analysis "
                f"should be reviewed to ensure reliable clinical interpretation."
            )