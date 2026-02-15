# 🩺 Breast Cancer Diagnosis Classification Using Machine Learning

## 📌 1. Problem Statement

The objective of this project is to build machine learning models that accurately classify breast tumors as:

- **Malignant (Cancerous)**
- **Benign (Non-Cancerous)**

Early and reliable classification helps doctors make timely treatment decisions and improves patient survival rates.

This is a **binary classification problem**.

---

## 📊 2. Dataset Description

- **Dataset File:** `Data.csv`
- **Source:** Breast Cancer Wisconsin Dataset (Kaggle / UCI)
- **Total Records:** 569 samples
- **Total Features:** 30 numerical features
- **Target Variable:** `diagnosis`

| Label | Meaning |
|-------|---------|
| M | Malignant |
| B | Benign |

### 🔍 Feature Information

Features are computed from digitized images of a **Fine Needle Aspirate (FNA)** of breast masses.

They describe characteristics of cell nuclei such as:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

Each feature includes:
- Mean
- Standard Error
- Worst (largest) value

---

## 🧹 3. Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary columns (e.g., ID column)
- Converted diagnosis labels (`M → 1`, `B → 0`)
- Checked for missing values
- Feature scaling using `StandardScaler`
- Train–Test split of dataset

---

## 🤖 4. Machine Learning Models Used

Six classification models were implemented:

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors (kNN)
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

---

## 📏 5. Evaluation Metrics

Models were evaluated using:

- Accuracy
- ROC-AUC Score
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

---

## 📈 6. Model Comparison Results

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC |
|--------|----------|-----------|--------|----|-----|----------|
| Logistic Regression | 0.956 | 0.957 | 0.956 | 0.956 | 0.907 | 0.995 |
| Decision Tree | 0.939 | 0.939 | 0.939 | 0.939 | 0.870 | 0.937 |
| kNN | 0.754 | 0.753 | 0.754 | 0.744 | 0.460 | 0.810 |
| Naive Bayes | 0.614 | 0.386 | 0.614 | 0.474 | -0.073 | 0.892 |
| Random Forest | **0.965** | **0.965** | **0.965** | **0.965** | **0.925** | **0.997** |
| XGBoost | 0.956 | 0.956 | 0.956 | 0.956 | 0.906 | 0.993 |

---

## 🔎 7. Observations on Model Performance

### 🔹 Logistic Regression
Strong baseline performance with excellent discrimination ability.

### 🔹 Decision Tree
Captures nonlinear relationships and is easy to interpret but may overfit.

### 🔹 kNN
Sensitive to feature scaling and high dimensionality.

### 🔹 Naive Bayes
Lower performance due to independence assumption among correlated features.

### 🔹 Random Forest ⭐
Best performing model with highest accuracy and robustness.

### 🔹 XGBoost
Strong boosting model with performance close to Random Forest.

---

## 💡 8. Key Insights

- Ensemble models outperform individual models.
- Dataset features are highly informative.
- Random Forest provides the most reliable predictions.
- MCC is useful for balanced evaluation.

---

## 🧠 9. Model Interpretation

- **Logistic Regression** → Linear decision boundary
- **Decision Tree** → Rule-based splits
- **Random Forest** → Aggregated trees reduce variance
- **XGBoost** → Sequential boosting improves errors iteratively

---

## ✅ 10. Conclusion

Machine learning models can effectively classify breast tumors using medical image features.

**Random Forest achieved the best performance**, making it suitable for real-world diagnostic support systems.

---

## 🚀 11. How to Run the Project

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
