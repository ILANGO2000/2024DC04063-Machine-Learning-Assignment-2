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

from models import fetch_pipeline


# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Breast Cancer Diagnostic Dashboard", layout="wide")

st.title("Breast Cancer Diagnostic Dashboard")


# ============================================================
# DATA SOURCE SELECTION (SESSION STATE FIX)
# ============================================================

st.markdown("## Dataset Selection")

data_source = st.radio(
    "Choose data source",
    ["Use default data", "Upload your own data"]
)

# Initialize session storage
if "data_frame" not in st.session_state:
    st.session_state["data_frame"] = None

if "prev_source" not in st.session_state:
    st.session_state["prev_source"] = data_source

# Clear dataset if source changes
if st.session_state["prev_source"] != data_source:
    st.session_state["data_frame"] = None
    st.session_state["prev_source"] = data_source


# ---------- DEFAULT DATA ----------
if data_source == "Use default data":

    if st.button("Load Default Dataset"):
        try:
            st.session_state["data_frame"] = pd.read_csv("Data.csv")
            st.success("Default dataset loaded successfully")
        except:
            st.error("Default dataset not found in repository")


# ---------- UPLOAD DATA ----------
else:

    uploaded_csv = st.file_uploader(
        "Upload your dataset in CSV format",
        type=["csv"]
    )

    if uploaded_csv is not None:
        st.session_state["data_frame"] = pd.read_csv(uploaded_csv)
        st.success("Uploaded dataset loaded successfully")


# ============================================================
# USE DATA FROM SESSION STATE
# ============================================================

data_frame = st.session_state["data_frame"]


# ============================================================
# MAIN WORKFLOW
# ============================================================

if data_frame is not None:

    data_frame.columns = data_frame.columns.str.strip()

    # Limit large datasets
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

        # Encode categorical target
        if target_data.dtype == "object":
            encoder = LabelEncoder()
            target_data = encoder.fit_transform(target_data)


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

                    trained_pipeline.fit(X_train, y_train)

                    predictions = trained_pipeline.predict(X_test)

                    # ROC AUC
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


                # ==================== MODEL PERFORMANCE ====================
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

                fig_cm, ax_cm = plt.subplots()

                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt="d",
                    cmap="coolwarm",
                    linewidths=1,
                    linecolor="black"
                )

                ax_cm.set_xlabel("Predicted Label")
                ax_cm.set_ylabel("Actual Label")

                st.pyplot(fig_cm)


                # ==================== CLASSIFICATION REPORT ====================
                st.markdown("## Detailed Classification Report")

                report_dict = classification_report(
                    y_test,
                    predictions,
                    output_dict=True
                )

                report_df = pd.DataFrame(report_dict).transpose()

                st.dataframe(report_df.round(4), use_container_width=True)


                # ============================================================
                # ADDITIONAL INSIGHTS
                # ============================================================
                st.markdown("## 🔎 Additional Insights for Tumor Classification")


                # 1. TARGET DISTRIBUTION
                st.subheader("Tumor Class Distribution")

                fig1, ax1 = plt.subplots()
                data_frame[target_column].value_counts().plot(kind="bar", ax=ax1)
                st.pyplot(fig1)


                # 2. CORRELATION HEATMAP
                st.subheader("Feature Correlation Heatmap")

                numeric_df = data_frame.select_dtypes(include=np.number)

                if numeric_df.shape[1] > 1:
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    sns.heatmap(numeric_df.corr(), cmap="viridis", ax=ax2)
                    st.pyplot(fig2)


                # 3. FEATURE COMPARISON
                st.subheader("Feature Distribution by Tumor Type")

                numeric_cols = numeric_df.columns.tolist()

                if numeric_cols:
                    selected_feature = st.selectbox(
                        "Select a feature to compare",
                        numeric_cols
                    )

                    fig3, ax3 = plt.subplots()
                    sns.boxplot(
                        x=data_frame[target_column],
                        y=data_frame[selected_feature],
                        ax=ax3
                    )
                    st.pyplot(fig3)


                # 4. PREDICTION CONFIDENCE
                st.subheader("Prediction Confidence Distribution")

                if hasattr(trained_pipeline, "predict_proba"):
                    prob_values = trained_pipeline.predict_proba(X_test)[:, 1]
                    fig4, ax4 = plt.subplots()
                    ax4.hist(prob_values, bins=20)
                    st.pyplot(fig4)


                # 5. ERROR ANALYSIS
                st.subheader("Prediction Error Analysis")

                error_flags = predictions != y_test

                fig5, ax5 = plt.subplots()
                ax5.bar(
                    ["Correct Predictions", "Incorrect Predictions"],
                    [(~error_flags).sum(), error_flags.sum()]
                )
                st.pyplot(fig5)