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

| Model                  | Accuracy | Precision | Recall | F1    | AUC    |
| ---------------------- | -------- | --------- | ------ | ----- | ------ |
| Baseline RF            | 0.956    | 1.0       | 0.881  | 0.937 | 0.9945 |
| Tuned RF               | 0.974    | 1.0       | 0.929  | 0.963 | 0.9983 |
| Top Features RF        | 0.974    | 1.0       | 0.929  | 0.963 | 0.9944 |
| GradientBoost RF       | 0.965    | 1.0       | 0.905  | 0.950 | 0.9970 |
| AdaBoost RF            | 0.974    | 1.0       | 0.929  | 0.963 | 0.9957 |
| SMOTE RF               | 0.974    | 1.0       | 0.929  | 0.963 | 0.9970 |
| Feature Engineering RF | 0.974    | 1.0       | 0.929  | 0.963 | 0.9993 |

## Insights

- **Accuracy:** Most enhanced models reach **0.9737**, improving over the baseline (**0.9561**).
- **Precision:** All models achieve **1.0**, indicating **zero false positives**.
- **Recall:**
  - Highest (0.9286): Tuned RF, Top Features RF, AdaBoost RF, SMOTE RF, Feature Engineering RF
  - Medium (0.9048): GradientBoost RF
  - Lowest (0.8810): Baseline RF
- **F1-score:** Follows the same trend as recall, with enhanced models reaching **0.9630**.
- **AUC:**
  - **Best AUC: Feature Engineering RF (0.999339)**
  - Tuned RF also very strong (0.998347)
  - SMOTE and GradientBoost around **0.9970**
  - Lowest AUCs: Top Features RF (0.994378) and Baseline (0.994544)

## Key Takeaways

- All enhanced models **outperform the baseline**, especially in recall and F1.
- **Perfect precision** across all models: no false positives.
- **Feature Engineering RF** delivers the **best AUC**, suggesting stronger generalization.
- **Tuned RF, Top Features RF, AdaBoost RF, and SMOTE RF** perform almost identically.
- **Top Features RF** achieves competitive performance despite dimensionality reduction.
- **GradientBoost RF** improves over baseline but remains below the top-performing group.

## Best Performing Model

Based on the combined evaluation of accuracy, recall, F1-score, and especially **AUC**,  
the **Feature Engineering RF** model demonstrates the strongest overall performance.

It offers:

- High accuracy (0.973684)
- Perfect precision (1.0)
- Strong recall (0.928571)
- Excellent F1-score (0.962963)
- **Highest AUC (0.999339)**

This makes it the most robust and reliable model for breast cancer classification in this study.
