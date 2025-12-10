# Breast Cancer Classification Project

## Project Overview

This project focuses on predicting whether a breast tumor is **benign** or **malignant** using machine learning techniques.  
It uses **Random Forest**, **Gradient Boosting**, and their **ensemble**, along with **feature engineering**, **feature selection**, and thorough **evaluation metrics** for performance assessment.

Find [HERE](https://ueve-my.sharepoint.com/:w:/g/personal/20253305_etud_univ-evry_fr/IQCrhjhVBBUnQL3m5FyawRuvATbe32zNdAz_Kb7ZiSMwKTg?e=Myt0xR) the detailed report analysis

## Environment Setup

```bash
python -m venv myvenv
# Activate the virtual environment:
# Linux / Mac
source myvenv/bin/activate
# Windows
myvenv\Scripts\activate
```

## Dataset

- **Source:** Kaggle Hub via `kagglehub` library

> **Note:** Although the UCI ML repository (`ucimlrepo`) is available, its server is sometimes down. Therefore, datasets are loaded from **Kaggle Hub** in this project.

- **Features:** 30 original features including mean, standard error (SE), and worst measurements for:
  - radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension
  - We removed the `id` column and any `Unnamed` columns as they are not useful for modeling.
- **Target:** `y` → 0 = benign, 1 = malignant

## Project Steps

### Preprocessing

- Handle missing values with mean imputation
- Scale features using `StandardScaler`
- Encode target labels using `LabelEncoder`

### Train-Test Split

- 80% training / 20% testing
- Stratified split to preserve class distribution

### Model Training and ML techninques

The following models and techniques were used in this project:

- **Baseline Random Forest**: Default parameters, trained on all features.
- **Tuned Random Forest**: Hyperparameter tuning using grid search to improve performance.
- **Top Features Random Forest**: Trained only on the top 10 most important features selected based on feature importance.
- **Random Forest + Gradient Boosting Soft Voting Ensemble**: Combines Random Forest ( Tuned Model ) and Gradient Boosting using soft voting.
- **Random Forest + AdaBoost Soft Voting Ensemble**: Combines Random Forest (Tuned Model )and AdaBoost using soft voting.
- **SMOTE**: Applied to balance classes in the training set.
- **Feature Selection**: Selected top features based on feature importance to reduce dimensionality and improve model interpretability.

#### Hyperparameter Tuning

- Use `GridSearchCV` or `RandomizedSearchCV`
- Optimize for **recall** due to the importance in medical diagnosis

### Evaluation

- Metrics for both **training** and **testing** sets:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Visualizations:
  - Confusion matrices
  - ROC curves
  - Precision-Recall curves
- Comparative table showing all metrics for every model

## Notes

- All experiments and visualizations can be reproduced by running the Jupyter notebooks in the repository.
- The project emphasizes **interpretability** using feature importances and **model comparison** through consistent metrics and plots.

## Quick Start

```bash
# Clone the repo
git clone git@github.com:LudmilaAmriou/Random-forest.git
cd Random-forest

# Activate virtual environment - The one you created in step 1
source myvenv/bin/activate  # Linux/Mac
myvenv\Scripts\activate     # Windows

# Install dependencies or run the cell that does it by selecting the kernel myvenv
pip install -r requirements.txt

# Run notebooks or scripts
jupyter notebook
```

# Test Set Performance

| Model                | Accuracy | Precision | Recall | F1    | AUC   |
| -------------------- | -------- | --------- | ------ | ----- | ----- |
| Baseline RF          | 0.965    | 1.0       | 0.905  | 0.950 | 0.994 |
| Tuned RF             | 0.974    | 1.0       | 0.929  | 0.963 | 0.998 |
| Top Features RF      | 0.974    | 1.0       | 0.929  | 0.963 | 0.995 |
| GradientBoost RF     | 0.965    | 1.0       | 0.905  | 0.950 | 0.997 |
| AdaBoost RF          | 0.974    | 1.0       | 0.929  | 0.963 | 0.996 |
| SMOTE RF             | 0.974    | 1.0       | 0.929  | 0.963 | 0.999 |
| Feature Selection RF | 0.965    | 1.0       | 0.905  | 0.950 | 0.994 |

**Insights:**

- **Accuracy:** All models maintain high performance (~96–97%), indicating minimal overfitting.
- **Precision:** Perfect (1.0) across all models — very few false positives.
- **Recall:** Slightly lower for Baseline, GradientBoost, and Feature Selection RF; Tuned RF, Top Features RF, AdaBoost, and SMOTE capture more malignant cases.
- **F1-score:** High (0.95–0.96), confirming strong balance between precision and recall.
- **AUC:** Near-perfect (0.994–0.999), showing excellent generalization.

**Key Takeaways:**

- Ensemble models (GradientBoost, AdaBoost) and SMOTE slightly improve recall without sacrificing precision.
- Feature selection reduces dimensionality with minimal impact on generalization.
- Overall, models are robust and generalize well to unseen data.

### Best Performing Model

Based on test set evaluation, the **Random Forest with SMOTE** provides the best balance between high recall, accuracy, and AUC. This is particularly important for breast cancer detection, where minimizing false negatives is critical.

Other models such as Tuned RF, Top Features RF, and RF + GradientBoost/AdaBoost ensembles perform similarly, but SMOTE RF slightly improves AUC and recall, making it the preferred model.
