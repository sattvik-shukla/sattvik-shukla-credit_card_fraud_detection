# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning techniques on a real-world dataset.

With the rapid growth of digital payments, fraud detection has become a critical challenge. This system leverages data-driven models to identify fraudulent transactions with high accuracy and reliability.

---

## 🚀 Key Highlights

* Handles **highly imbalanced dataset (0.17% fraud cases)**
* Trained and compared **3 ML models**
* Uses **advanced evaluation metrics (AUC-ROC, MCC, F1-score)**
* Implements **feature scaling + undersampling**
* Includes a **Flask-based web application** for real-time prediction

---

## 🧠 Models Used

* Logistic Regression (Baseline)
* Random Forest (Ensemble)
* ⭐ Gradient Boosting (**Best Model**)

---

## 📊 Dataset

* Source: ULB Credit Card Fraud Dataset (Kaggle)
* Total Transactions: **284,807**
* Fraud Cases: **492 (0.17%)**
* Features: **31 (V1–V28 PCA transformed)**

👉 Dataset Link:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## ⚙️ Data Preprocessing

* StandardScaler applied to **Time** and **Amount**
* PCA features (V1–V28) already normalized
* **Undersampling** used to balance dataset
* Final dataset: **984 samples (50% fraud, 50% legitimate)**

---

## 🔬 Methodology

1. Data Loading & Cleaning
2. Feature Scaling
3. Handling Class Imbalance
4. Train-Test Split (80:20)
5. Model Training
6. Performance Evaluation
7. Model Comparison
8. Deployment via Flask

---

## 📈 Results & Performance

| Model               | Accuracy   | F1 Score   | AUC-ROC    | MCC        |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 93.40%     | 93.26%     | 0.9834     | 0.8684     |
| Random Forest       | 93.91%     | 93.75%     | 0.9816     | 0.8789     |
| ⭐ Gradient Boosting | **94.42%** | **94.30%** | **0.9878** | **0.8887** |

👉 **Best Model:** Gradient Boosting
Selected using weighted scoring (F1 + AUC-ROC + Recall + MCC)

---

## 🌐 Web Application

A complete Flask-based web app is implemented for real-time fraud detection.

### Features:

* 📊 Live dashboard with metrics
* 📈 ROC curves & confusion matrices
* 📉 Feature importance visualization
* 🔍 Real-time fraud prediction
* 📋 Model comparison table

### Run the app:

```bash
python app.py
```

Then open:

```
http://localhost:5000
```

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Flask
* HTML/CSS (Frontend)

```

---

## 🔮 Future Improvements

* Use **SMOTE** instead of undersampling
* Implement **XGBoost / LightGBM**
* Add **real-time streaming detection (Kafka)**
* Use **SHAP for explainability**
* Deploy on cloud (AWS / Render)

---

## 👨‍💻 Author

**Sattvik Shukla**

* GitHub: https://github.com/sattvik-shukla
* LinkedIn: https://in.linkedin.com/in/sattvik-shukla

---

## ⭐ Show some support

If you found this project useful, give it a ⭐ on GitHub!

---
