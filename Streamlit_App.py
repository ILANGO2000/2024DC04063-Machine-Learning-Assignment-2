import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
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


if data_source == "Use default data":

    if st.button("Load Default Dataset"):
        try:
            st.session_state["data_frame"] = pd.read_csv("Data.csv")
            st.success("Default dataset loaded successfully")
        except:
            st.error("Default dataset not found in repository")

else:

    uploaded_csv = st.file_uploader(
        "Upload your dataset in CSV format",
        type=["csv"]
    )

    if uploaded_csv is not None:
        st.session_state["data_frame"] = pd.read_csv(uploaded_csv)
        st.success("Uploaded dataset loaded successfully")


data_frame = st.session_state["data_frame"]


if data_frame is not None:

    data_frame.columns = data_frame.columns.str.strip()

    if "diagnosis" not in data_frame.columns:
        st.error("Target column 'diagnosis' not found in dataset")
        st.stop()

    if len(data_frame) > 20000:
        data_frame = data_frame.sample(20000, random_state=42)

    st.markdown("### Dataset Sample")
    st.dataframe(data_frame.head(), use_container_width=True)


    feature_data = data_frame.drop(columns=["diagnosis"])
    target_data = data_frame["diagnosis"]

    if target_data.dtype == "object":
        encoder = LabelEncoder()
        target_data = encoder.fit_transform(target_data)


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

    split_ratio = st.slider(
        "Select test data proportion",
        0.1, 0.5, 0.2
    )


    X_train, X_test, y_train, y_test = train_test_split(
        feature_data,
        target_data,
        test_size=split_ratio,
        random_state=42
    )


    if st.button("Refresh Model"):

        st.session_state["pipeline_model"] = fetch_pipeline(
            model_choice,
            X_train
        )

        st.success("Model refreshed successfully")


    if "pipeline_model" in st.session_state:

        if st.button("Apply Model"):

            with st.spinner("Model is running. Please wait..."):

                trained_pipeline = st.session_state["pipeline_model"]
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

                cv_scores = cross_val_score(
                    fetch_pipeline(model_choice, X_train),
                    feature_data,
                    target_data,
                    cv=3,
                    n_jobs=-1
                )


            st.markdown("## Model Evaluation")

            c1, c2, c3, c4, c5, c6 = st.columns(6)

            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("Precision", f"{prec:.4f}")
            c3.metric("Recall", f"{rec:.4f}")
            c4.metric("F1 Score", f"{f1_val:.4f}")
            c5.metric("MCC", f"{mcc_val:.4f}")
            c6.metric("CV Score", f"{cv_scores.mean():.4f}")

            if auc_value is not None:
                st.metric("ROC AUC", f"{auc_value:.4f}")


            st.markdown("## Confusion Matrix")

            matrix = confusion_matrix(y_test, predictions)
            fig_cm, ax_cm = plt.subplots()

            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="coolwarm",
                linewidths=1,
                linecolor="black",
                ax=ax_cm
            )

            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)


            st.markdown("## Detailed Classification Report")

            report_dict = classification_report(
                y_test,
                predictions,
                output_dict=True
            )

            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)


            st.markdown("## Additional Insights")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Tumor Class Distribution")
                fig1, ax1 = plt.subplots()
                data_frame["diagnosis"].value_counts().plot(kind="bar", ax=ax1)
                st.pyplot(fig1)

            with col2:
                st.subheader("Feature Correlation")
                numeric_df = data_frame.select_dtypes(include=np.number)
                if numeric_df.shape[1] > 1:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    sns.heatmap(numeric_df.corr(), cmap="viridis", ax=ax2)
                    st.pyplot(fig2)

            col3, col4 = st.columns(2)

            with col3:
                st.subheader("Prediction Confidence")
                if hasattr(trained_pipeline, "predict_proba"):
                    prob_values = trained_pipeline.predict_proba(X_test)[:, 1]
                    fig3, ax3 = plt.subplots()
                    ax3.hist(prob_values, bins=20)
                    st.pyplot(fig3)

            with col4:
                st.subheader("Prediction Error")
                error_flags = predictions != y_test
                fig4, ax4 = plt.subplots()
                ax4.bar(
                    ["Correct", "Incorrect"],
                    [(~error_flags).sum(), error_flags.sum()]
                )
                st.pyplot(fig4)


            st.markdown("## Model Findings Summary")

            class_counts = data_frame["diagnosis"].value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()

            balance_text = (
                "Dataset shows class imbalance"
                if imbalance_ratio > 1.5
                else "Dataset is relatively balanced"
            )

            error_rate = 1 - acc

            if auc_value is not None:
                if auc_value > 0.85:
                    discrimination_text = "strong discrimination ability"
                elif auc_value > 0.70:
                    discrimination_text = "moderate discrimination ability"
                else:
                    discrimination_text = "limited discrimination ability"
            else:
                discrimination_text = "unknown discrimination ability"

            st.write(
                f"• The model achieved {acc:.2%} accuracy with {error_rate:.2%} error, indicating reliable tumor classification."
            )

            st.write(
                f"• The model shows {discrimination_text} in distinguishing malignant and benign tumors."
            )

            st.write(
                f"• {balance_text}, and misclassified cases should be reviewed for clinical caution."
            )