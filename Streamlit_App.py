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


# ==========================================================
# 🔮 LIVE PREDICTOR (PRETRAINED — TOP COMPONENT)
# ==========================================================

st.markdown("## 🔮 Live Tumor Diagnosis Predictor")

try:
    live_data = pd.read_csv("Data.csv")
    live_data.columns = live_data.columns.str.strip()

    if "id" in live_data.columns:
        live_data = live_data.drop(columns=["id"])

    X_live = live_data.drop(columns=["diagnosis"])
    y_live = live_data["diagnosis"]

    if y_live.dtype == "object":
        y_live = LabelEncoder().fit_transform(y_live)

    model_list = [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]

    live_model_name = st.selectbox(
        "Select Model for Live Prediction",
        model_list,
        key="live_model"
    )

    numeric_df = X_live.select_dtypes(include=np.number)

    temp_df = numeric_df.copy()
    temp_df["target"] = y_live

    corr_values = temp_df.corr()["target"].abs().sort_values(ascending=False)
    top_features = corr_values.index[1:6]

    st.markdown("### Enter Tumor Measurements")

    input_data = {}
    cols = st.columns(3)

    for i, feature in enumerate(top_features):
        with cols[i % 3]:
            input_data[feature] = st.slider(
                feature,
                float(X_live[feature].min()),
                float(X_live[feature].max()),
                float(X_live[feature].mean())
            )

    if st.button("Predict Tumor Type"):

        model_live = fetch_pipeline(live_model_name, X_live)
        model_live.fit(X_live, y_live)

        input_df = pd.DataFrame([input_data])

        for col in X_live.columns:
            if col not in input_df.columns:
                input_df[col] = X_live[col].median()

        input_df = input_df[X_live.columns]

        prediction = model_live.predict(input_df)[0]

        if hasattr(model_live, "predict_proba"):
            probability = model_live.predict_proba(input_df)[0][1]
        else:
            probability = None

        c1, c2 = st.columns([2, 1])

        if prediction == 1:
            c1.error("Predicted Diagnosis: Malignant")
        else:
            c1.success("Predicted Diagnosis: Benign")

        if probability is not None:
            c2.metric("Malignancy Probability", f"{probability:.2%}")

except:
    st.warning("Default dataset not available for live prediction")

st.markdown("---")


# ==========================================================
# ORIGINAL DASHBOARD
# ==========================================================

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
        st.session_state["data_frame"] = pd.read_csv("Data.csv")

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
        target_data = LabelEncoder().fit_transform(target_data)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_data, target_data, test_size=0.2, random_state=42
    )


    # ================= DEFAULT DATA =================

    if data_source == "Use default data":

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

            if hasattr(model, "predict_proba"):
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            else:
                auc = np.nan

            results.append([
                name,
                accuracy_score(y_test, preds),
                precision_score(y_test, preds, average="weighted"),
                recall_score(y_test, preds, average="weighted"),
                f1_score(y_test, preds, average="weighted"),
                matthews_corrcoef(y_test, preds),
                auc
            ])

        results_df = pd.DataFrame(
            results,
            columns=["Model", "Accuracy", "Precision", "Recall", "F1", "MCC", "ROC AUC"]
        )

        st.dataframe(
            results_df.style.background_gradient(cmap="YlOrBr", subset=results_df.columns[1:]),
            use_container_width=True
        )


    # ================= UPLOADED DATA =================

    else:

        model_choice = st.selectbox(
            "Choose a classification algorithm",
            [
                "Logistic Regression", "Decision Tree", "KNN",
                "Naive Bayes", "Random Forest", "XGBoost"
            ]
        )

        if st.button("Run Model"):

            with st.spinner("Model is running..."):

                model = fetch_pipeline(model_choice, X_train)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_test)[:, 1]
                    auc_value = roc_auc_score(y_test, probs)
                    fpr, tpr, _ = roc_curve(y_test, probs)
                else:
                    auc_value = None
                    fpr, tpr = None, None

                acc = accuracy_score(y_test, predictions)
                prec = precision_score(y_test, predictions, average="weighted")
                rec = recall_score(y_test, predictions, average="weighted")
                f1_val = f1_score(y_test, predictions, average="weighted")
                mcc_val = matthews_corrcoef(y_test, predictions)


            # ===== METRICS =====
            st.markdown("## Model Evaluation")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("Precision", f"{prec:.4f}")
            c3.metric("Recall", f"{rec:.4f}")
            c4.metric("F1 Score", f"{f1_val:.4f}")
            c5.metric("MCC", f"{mcc_val:.4f}")

            if auc_value is not None:
                st.metric("ROC AUC", f"{auc_value:.4f}")


            # ===== PLOTS =====
            st.markdown("## Evaluation Plots")

            left, right = st.columns(2)

            with left:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, predictions)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with right:
                st.subheader("ROC Curve")
                if fpr is not None:
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC={auc_value:.3f}")
                    ax.plot([0,1],[0,1],'--')
                    ax.legend()
                    st.pyplot(fig)


            # ===== REPORT =====
            st.markdown("## Classification Report")
            report_df = pd.DataFrame(
                classification_report(y_test, predictions, output_dict=True)
            ).transpose()

            st.dataframe(report_df.round(4), use_container_width=True)


            # ===== INSIGHTS =====
            st.markdown("## Insights")

            col1, col2 = st.columns(2)

            with col1:
                fig1, ax1 = plt.subplots()
                data_frame["diagnosis"].value_counts().plot(kind="bar", ax=ax1)
                ax1.set_title("Tumor Class Distribution")
                st.pyplot(fig1)

            with col2:
                numeric_df = data_frame.select_dtypes(include=np.number)
                fig2, ax2 = plt.subplots()
                sns.heatmap(numeric_df.corr(), cmap="viridis", ax=ax2)
                ax2.set_title("Feature Correlation")
                st.pyplot(fig2)

            col3, col4 = st.columns(2)

            with col3:
                if hasattr(model, "predict_proba"):
                    fig3, ax3 = plt.subplots()
                    ax3.hist(probs, bins=20)
                    ax3.set_title("Prediction Confidence")
                    st.pyplot(fig3)

            with col4:
                errors = predictions != y_test
                fig4, ax4 = plt.subplots()
                ax4.bar(["Correct","Incorrect"], [(~errors).sum(), errors.sum()])
                ax4.set_title("Prediction Error")
                st.pyplot(fig4)


            # ===== SUMMARY =====
            st.markdown("## Model Insight Summary")

            error_rate = errors.sum() / len(y_test)
            class_counts = data_frame["diagnosis"].value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()

            if auc_value:
                auc_text = "excellent" if auc_value > 0.9 else "strong"
            else:
                auc_text = "unknown"

            balance_text = (
                "class imbalance present"
                if imbalance_ratio > 1.5
                else "dataset balanced"
            )

            st.write(
                f"{model_choice} achieved {acc:.2%} accuracy "
                f"with {error_rate:.2%} error rate."
            )

            st.write(
                f"ROC indicates {auc_text} discrimination ability."
            )

            st.write(
                f"{balance_text}. Review misclassified cases carefully."
            )