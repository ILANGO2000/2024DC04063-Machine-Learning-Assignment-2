# -------------------- IMPORTS --------------------
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

# Updated import based on new model file
from models import fetch_pipeline


# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Breast Cancer Diagnostic Dashboard", layout="wide")

st.title("Breast Cancer Diagnostic Dashboard")


# -------------------- DATA UPLOAD --------------------
uploaded_csv = st.file_uploader(
    "Upload your dataset in CSV format",
    type=["csv"]
)

if uploaded_csv is not None:

    data_frame = pd.read_csv(uploaded_csv)
    data_frame.columns = data_frame.columns.str.strip()

    # Handle very large datasets
    if len(data_frame) > 20000:
        data_frame = data_frame.sample(20000, random_state=42)

    st.markdown("### Dataset Sample")
    st.dataframe(data_frame.head(), use_container_width=True)


    # -------------------- TARGET SELECTION --------------------
    target_column = st.selectbox(
        "Choose the target variable",
        data_frame.columns
    )

    if target_column:

        feature_data = data_frame.drop(columns=[target_column])
        target_data = data_frame[target_column]

        # Encode categorical target values
        if target_data.dtype == "object":
            label_encoder = LabelEncoder()
            target_data = label_encoder.fit_transform(target_data)


        # -------------------- MODEL SELECTION --------------------
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


        # -------------------- TRAIN TEST SPLIT --------------------
        X_train, X_test, y_train, y_test = train_test_split(
            feature_data,
            target_data,
            test_size=split_ratio,
            random_state=42
        )


        # ==================== REFRESH MODEL ====================
        if st.button("Refresh Model"):

            st.session_state["pipeline_model"] = fetch_pipeline(
                model_choice,
                X_train
            )

            st.success("Model refreshed successfully")


        # ==================== APPLY MODEL ====================
        if "pipeline_model" in st.session_state:

            if st.button("Apply Model"):

                with st.spinner("Model is running. Please wait..."):

                    trained_pipeline = st.session_state["pipeline_model"]

                    # Train pipeline
                    trained_pipeline.fit(X_train, y_train)

                    predictions = trained_pipeline.predict(X_test)

                    # ROC AUC if probability supported
                    if hasattr(trained_pipeline, "predict_proba"):
                        probabilities = trained_pipeline.predict_proba(X_test)[:, 1]
                        auc_value = roc_auc_score(y_test, probabilities)
                    else:
                        auc_value = None


                    # -------------------- METRICS --------------------
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


                # ==================== DISPLAY METRICS ====================
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


                # ==================== CONFUSION MATRIX ====================
                st.markdown("## Confusion Matrix")

                matrix = confusion_matrix(y_test, predictions)

                fig, ax = plt.subplots()

                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt="d",
                    cmap="coolwarm",
                    linewidths=1,
                    linecolor="black"
                )

                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Actual Label")

                st.pyplot(fig)


                # ==================== CLASSIFICATION REPORT ====================
                st.markdown("## Detailed Classification Report")

                report_dict = classification_report(
                    y_test,
                    predictions,
                    output_dict=True
                )

                report_df = pd.DataFrame(report_dict).transpose()

                st.dataframe(
                    report_df.round(4),
                    use_container_width=True
                )