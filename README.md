# 🚗 Policy Claim Risk Modelling  

Predicting the likelihood of an insurance claim using the **Porto Seguro Safe Driver Prediction dataset**.  

This project applies **machine learning and ensembling** techniques to tackle a **highly imbalanced binary classification problem** where only ~3.6% of cases are positive claims.  

---

## 📌 Project Overview  

Insurance companies face the challenge of **predicting claim probability** to:  
- Minimize risk  
- Optimize pricing  
- Improve customer profiling  

In this project, I built a **LightGBM-based model with ensembling** to predict whether a driver will file a claim.  

---

## ⚙️ Key Steps  

1. **Data Preprocessing**  
   - Removed identifiers (`id`)  
   - Handled categorical features  
   - Dealt with missing values  

2. **Class Imbalance Handling**  
   - Acknowledged imbalance (~3.6% positives)  
   - Applied **balanced metrics (ROC-AUC, not accuracy)**  

3. **Model Training**  
   - Core model: **LightGBM**  
   - Early stopping & validation to avoid overfitting  

4. **Ensembling**  
   - Blended LightGBM with Logistic Regression (stacked ensemble)  
   - Achieved **ROC-AUC ≈ 0.64** on validation  

---

## 📊 Results  

- **Baseline Random Forest (SMOTE):** ROC-AUC ≈ 0.58  
- **LightGBM (tuned):** ROC-AUC ≈ 0.64  
- **Stacked Ensemble:** ROC-AUC ≈ 0.62  

✔️ Final model selected: **LightGBM** with early stopping and feature selection.  
