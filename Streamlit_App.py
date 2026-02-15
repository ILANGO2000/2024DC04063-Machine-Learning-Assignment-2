import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)

from models import fetch_pipeline

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(layout="wide")
st.title("🧬 Breast Cancer Diagnosis Dashboard")

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

data_frame = load_data()

# Remove ID if exists
if "id" in data_frame.columns:
    data_frame = data_frame.drop(columns=["id"])

# Encode Target
label_encoder = LabelEncoder()
data_frame["diagnosis"] = label_encoder.fit_transform(data_frame["diagnosis"])

feature_data = data_frame.drop(columns=["diagnosis"])
target_data = data_frame["diagnosis"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    feature_data, target_data,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# 🔮 LIVE PREDICTOR (TOP SECTION)
# ==========================================================
st.markdown("## 🔮 Live Tumor Diagnosis Predictor")

# Select model for live predictor
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
    model_list
)

# Correlation-based feature selection
numeric_df = data_frame.select_dtypes(include=np.number)

temp_df = numeric_df.copy()
temp_df["target"] = target_data

corr_values = (
    temp_df.corr()["target"]
    .abs()
    .sort_values(ascending=False)
)

top_features = corr_values.index[1:6]

st.markdown("### Adjust Feature Values")

input_data = {}
cols = st.columns(3)

for i, feature in enumerate(top_features):
    with cols[i % 3]:
        min_val = float(data_frame[feature].min())
        max_val = float(data_frame[feature].max())
        mean_val = float(data_frame[feature].mean())

        input_data[feature] = st.slider(
            feature,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )

# Train selected model for live prediction
live_model = fetch_pipeline(live_model_name, X_train)
live_model.fit(X_train, y_train)

if st.button("Predict Diagnosis"):

    input_df = pd.DataFrame([input_data])

    # Fill missing columns
    for col in feature_data.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_data.columns]

    prediction = live_model.predict(input_df)[0]

    if hasattr(live_model, "predict_proba"):
        probability = live_model.predict_proba(input_df)[0][1]
    else:
        probability = None

    st.markdown("### 🔎 Prediction Result")

    c1, c2 = st.columns([2,1])

    if prediction == 1:
        c1.error("Predicted Diagnosis: Malignant")
    else:
        c1.success("Predicted Diagnosis: Benign")

    if probability is not None:
        c2.metric("Malignancy Probability", f"{probability:.2%}")

st.divider()

# ==========================================================
# 📊 PRE-TRAINED MODEL METRICS
# ==========================================================
st.markdown("## 📊 Pre-trained Model Performance")

results = []

for model_name in model_list:

    model = fetch_pipeline(model_name, X_train)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, probs)
    else:
        auc = 0

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds),
        "AUC": auc,
        "MCC": matthews_corrcoef(y_test, preds)
    })

results_df = pd.DataFrame(results)

# Blue gradient styling
st.dataframe(
    results_df.style.background_gradient(cmap="Blues"),
    use_container_width=True
)

st.divider()

# ==========================================================
# 📊 TEST DATA EVALUATION
# ==========================================================
st.markdown("## 📂 Upload Test Dataset")

uploaded_file = st.file_uploader("Upload CSV file")

if uploaded_file:

    test_df = pd.read_csv(uploaded_file)

    if "diagnosis" not in test_df.columns:
        st.error("Target column 'diagnosis' missing.")
    else:

        test_df["diagnosis"] = label_encoder.transform(test_df["diagnosis"])

        X_new = test_df.drop(columns=["diagnosis"])
        y_new = test_df["diagnosis"]

        selected_model = fetch_pipeline(live_model_name, X_train)
        selected_model.fit(X_train, y_train)

        preds = selected_model.predict(X_new)
        probs = selected_model.predict_proba(X_new)[:,1]

        acc = accuracy_score(y_new, preds)
        prec = precision_score(y_new, preds)
        rec = recall_score(y_new, preds)
        f1 = f1_score(y_new, preds)
        auc = roc_auc_score(y_new, probs)
        mcc = matthews_corrcoef(y_new, preds)

        st.markdown("### 📊 Test Metrics")

        m1,m2,m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")

        m4,m5,m6 = st.columns(3)
        m4.metric("F1 Score", f"{f1:.3f}")
        m5.metric("AUC", f"{auc:.3f}")
        m6.metric("MCC", f"{mcc:.3f}")

        # Confusion + ROC side by side
        colA, colB = st.columns(2)

        with colA:
            cm = confusion_matrix(y_new, preds)
            fig1, ax1 = plt.subplots(figsize=(3,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
            ax1.set_title("Confusion Matrix")
            st.pyplot(fig1)

        with colB:
            fpr, tpr, _ = roc_curve(y_new, probs)
            fig2, ax2 = plt.subplots(figsize=(3,3))
            ax2.plot(fpr, tpr)
            ax2.plot([0,1],[0,1],'--')
            ax2.set_title("ROC Curve")
            st.pyplot(fig2)

        st.markdown("### 🧠 Test Dataset Summary")

        st.markdown(f"""
        - Accuracy of **{acc:.2%}** shows overall correctness.
        - Precision of **{prec:.2f}** indicates reliability when predicting Malignant.
        - Recall of **{rec:.2f}** shows ability to detect actual Malignant cases.
        - F1 Score balances precision & recall.
        - AUC of **{auc:.2f}** indicates class separability.
        - MCC of **{mcc:.2f}** reflects balanced prediction strength.
        """)

