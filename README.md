# Lung Cancer Prediction with Explainable AI (XAI) using SHAP

This project focuses on building a machine learning classification model to predict lung cancer, with a strong emphasis on **model interpretability** using **SHAP (SHapley Additive exPlanations)** — a powerful framework for Explainable AI (XAI).

---

## 📁 Project Structure

```
XAI/
├── shap_results_last.ipynb   # Main notebook: EDA, model training, and SHAP analysis
├── lcs_synthetic_20000.csv   # Lung cancer synthetic dataset (20,000 samples)
└── README.md
```

---

## 📊 Dataset

- **Source:** [Lung Cancer Dataset – Kaggle](https://www.kaggle.com/datasets/chandanmsr/lung-cancer-dataset)
- **Size:** 20,000 rows × 16 columns
- **Target Column:** `lung_cancer` (YES / NO)
- **Features include:** `gender`, `age`, `smoking`, `yellow_fingers`, `anxiety`, `peer_pressure`, `chronic_disease`, `fatigue`, `allergy`, `wheezing`, `alcohol_consuming`, `coughing`, `shortness_of_breath`, `swallowing_difficulty`, `chest_pain`

---

## 🧠 Methodology

### 1. Exploratory Data Analysis (EDA)
- Dataset shape, data types, and missing values inspection
- Duplicate row detection and removal (429 duplicates removed)
- Class balance analysis: `YES` (~57%), `NO` (~43%)

### 2. Data Preprocessing
- Label encoding for categorical columns (`gender`)
- Train/test split (80/20)

### 3. Model Training
- Classification model trained on the preprocessed dataset

### 4. Explainability with SHAP
- SHAP values computed to explain individual predictions and global feature importance
- Visualizations include:
  - SHAP Summary Plots
  - SHAP Bar Charts (global feature importance)
  - SHAP Waterfall / Force Plots (local explanations)

---

## 🔍 Key Insights

- Features like `smoking`, `yellow_fingers`, and `coughing` are among the top contributors to lung cancer prediction
- SHAP analysis provides transparent, human-interpretable reasons behind each prediction

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | ML model training & evaluation |
| SHAP | Explainable AI |
| Matplotlib | Visualization |

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/SravyaKatikala/lung-cancer-xai.git
   cd lung-cancer-xai
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn shap matplotlib
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/chandanmsr/lung-cancer-dataset) and place it in the project folder.

4. Open and run the notebook:
   ```bash
   jupyter notebook shap_results_last.ipynb
   ```

---

## 👩‍💻 Author

**Sravya Katikala**  
GitHub: [@SravyaKatikala](https://github.com/SravyaKatikala)
