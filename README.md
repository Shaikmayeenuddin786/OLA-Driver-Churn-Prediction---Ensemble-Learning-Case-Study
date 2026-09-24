# OLA Driver Churn Prediction - Ensemble Learning Case Study

> ### Driver Attrition Prediction | EDA & Feature Engineering | Bagging & Boosting Models | Business Insights & Retention Strategy

---

## Quick Overview

| Section | Details |
|---|---|
| **Business Problem** | Ola loses drivers to competitors like Uber. Acquiring new drivers is expensive, and retaining existing drivers is cheaper. Ola needed a data-driven way to predict which drivers are likely to leave. |
| **Objectives** | 1. Predict driver churn using historical data.<br>2. Identify the key drivers of churn.<br>3. Provide actionable retention recommendations.<br>4. Compare Bagging and Boosting ensemble models. |
| **Technical Stack** | Python (Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Seaborn), Jupyter Notebook |
| **Project Features** | - Bagging model: Random Forest with GridSearchCV<br>- Boosting model: XGBoost with GridSearchCV<br>- KNN Imputation for missing values<br>- SMOTE for class imbalance<br>- Feature engineering (Target, Tenure, Income Increase, Rating Increase)<br>- Random Forest Test ROC AUC: 0.9190<br>- XGBoost Test ROC AUC: 0.9240<br>- Driver-level aggregation from 19,104 monthly records to 2,381 unique drivers |
| **Start-to-End Pipeline** | Data Loading → EDA → Date Conversion & KNN Imputation → Driver-Level Aggregation → Feature Engineering → Encoding → SMOTE → Standardization → Random Forest & XGBoost Training → Evaluation → Business Insights |
| **Top Strategic Recommendations** | - Introduce automatic income increase for retained drivers<br>- Reward rating improvement, not just top performers<br>- Create a visible Grade 1 to Grade 3 path within the first year<br>- Focus retention on the first 90 days of tenure<br>- Deploy churn scoring model monthly |
| **Future Upgrades & Scaling Plan** | - Deploy model as a REST API for real-time churn scoring<br>- Build a retention dashboard for city managers<br>- Add driver behavior data (trips per day, cancellations) to improve accuracy<br>- Set up automated model retraining every quarter |

---

## The Big Picture

> Ola, India's largest ride-hailing platform, faces high driver churn. Recruiting new drivers is expensive, and losing experienced drivers hurts service quality. The goal of this project is to build a predictive model that flags drivers likely to leave, so the retention team can act in time.

> This project shows that **income growth and rating improvement are the strongest signals of driver retention**, and that **ensemble models (Random Forest and XGBoost) can predict churn with ROC AUC above 0.92**.

---

## Business Problem

Ola's driver retention team needed clear answers to:

> - Which drivers are most likely to leave?
> - What factors drive churn the most?
> - Can we build a reliable model to predict churn?
> - How can we retain drivers cost-effectively?

**Note:** Without this analysis, Ola risks losing drivers to competitors without warning, and spending heavily on replacement hiring.

---

## Objectives

> - Build ensemble models (Random Forest and XGBoost) to predict driver churn.
> - Identify key features driving churn.
> - Handle missing values, class imbalance, and data aggregation properly.
> - Evaluate models on ROC AUC and classification metrics.
> - Provide actionable retention recommendations.

---

## Technical Stack

**Python Libraries:**

> - Pandas and NumPy for data manipulation
> - Matplotlib and Seaborn for visualization
> - Scikit-learn for Random Forest, GridSearchCV, and metrics
> - XGBoost for gradient boosting
> - Imbalanced-learn for SMOTE
> - KNNImputer from Scikit-learn for missing value imputation

**Environment:**

> - Jupyter Notebook
> - Markdown for documentation
> - PDF for final report

---

## Repository Structure

```
ola-driver-churn-ensemble-learning/
│
├── README.md                                    # Project overview
├── ola_driver_scaler.csv                        # Raw dataset
├── ola_driver_cleaned_features.csv              # Cleaned and engineered dataset
├── OLA_Ensemble_Learning.ipynb                  # Complete Jupyter Notebook
├── OLA_Ensemble_Learning_Report.pdf             # Final report
├── visuals/                                     # All plots and graphs
│   ├── age_distribution.png
│   ├── income_distribution.png
│   ├── quarterly_rating_distribution.png
│   ├── gender_countplot.png
│   ├── city_countplot.png
│   ├── education_level_countplot.png
│   ├── grade_countplot.png
│   ├── joining_designation_countplot.png
│   ├── income_vs_business_value.png
│   ├── grade_vs_income.png
│   ├── correlation_heatmap.png
│   ├── roc_auc_curve.png
│   └── confusion_matrices.png
└── output/
    ├── model_performance_metrics.csv
    └── feature_importance_ranking.csv
```

---

## Column Profiling

| Column | Type | Description |
|---|---|---|
| MMMM-YY | Date | Reporting month |
| Driver_ID | Integer | Unique driver identifier |
| Age | Integer | Age of driver |
| Gender | Categorical | Male (0), Female (1) |
| City | Categorical | City code |
| Education_Level | Categorical | 0 = 10th, 1 = 12th, 2 = Graduate |
| Income | Integer | Monthly average income |
| Dateofjoining | Date | Driver's joining date |
| LastWorkingDate | Date | Driver's last working date (NaT if still active) |
| Joining Designation | Integer | Designation at joining |
| Grade | Integer | Current grade of driver |
| Total Business Value | Integer | Monthly business value (negative = cancellations) |
| Quarterly Rating | Integer | Rating from 1 to 5 |

---

# Analysis Steps

## 1. Data Loading & Initial Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ola_driver_scaler.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print("Shape:", df.shape)
print(df.dtypes)
print(df.isnull().sum())
df.describe()
```

**Findings:**

> - Total records: 19,104
> - Unique drivers: 2,381
> - Missing values: Age (61), Gender (52), LastWorkingDate (17,488, expected for active drivers)
> - Target column will be derived from LastWorkingDate

[Placeholder: Age and Income histograms]

---

## 2. Exploratory Data Analysis (EDA)

### Univariate Analysis - Continuous Variables

**Observations:**

> - Age: Normal distribution centered around 34 years, with slight right tail.
> - Income: Right-skewed. Most drivers earn between 30,000 and 80,000.
> - Quarterly Rating: Discrete values 1 to 4, with rating 1 being the most common.
> - Total Business Value: Highly skewed with extreme outliers and negative values.

[Placeholder: paste Age, Income, Quarterly Rating distribution plots]

### Univariate Analysis - Categorical Variables

**Observations:**

> - Gender: Male drivers (1,405) are more common than female drivers (976).
> - City: C20 is the largest city with 152 drivers.
> - Education Level: Almost evenly distributed across 10th, 12th, and Graduate.
> - Grade: Grade 2 is most common (854 drivers), Grade 5 is rare (24 drivers).
> - Joining Designation: Most drivers joined at Designation 1 (9,800+ records).

[Placeholder: paste Gender, City, Grade, Education Level countplots]

### Bivariate Analysis

**Observations:**

> - Grade and Income are strongly positively correlated. Grade 1 median income is 40,000, Grade 5 median income is 1,40,000.
> - Total Business Value and Quarterly Rating have moderate positive correlation (0.47).
> - Age has weak correlation with other features.

[Placeholder: paste Grade vs Income box plot and Correlation Heatmap]

---

## 3. Data Preprocessing

### Date Conversion and KNN Imputation

```python
df['MMMM-YY'] = pd.to_datetime(df['MMMM-YY'], errors='coerce')
df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], format='%d/%m/%y', errors='coerce')
df['LastWorkingDate'] = pd.to_datetime(df['LastWorkingDate'], format='%d/%m/%y', errors='coerce')

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

cols_to_impute = ['Age', 'Gender']
scaler_knn = StandardScaler()
df_scaled = pd.DataFrame(scaler_knn.fit_transform(df[cols_to_impute]), columns=cols_to_impute)
imputer = KNNImputer(n_neighbors=5)
df[cols_to_impute] = scaler_knn.inverse_transform(imputer.fit_transform(df_scaled))
df['Gender'] = df['Gender'].round().astype(int)
```

**Observations:**

> - Dates converted to proper datetime format.
> - Missing values in Age and Gender filled using KNN with 5 neighbors.
> - No missing values remain in numerical columns.

---

## 4. Aggregation to Driver Level

```python
df_final = df.groupby('Driver_ID').agg({
    'Age': 'max',
    'Gender': 'first',
    'City': 'first',
    'Education_Level': 'first',
    'Income': 'mean',
    'Dateofjoining': 'first',
    'LastWorkingDate': 'max',
    'Joining Designation': 'first',
    'Grade': 'max',
    'Total Business Value': 'sum',
    'Quarterly Rating': 'mean'
}).reset_index()
```

**Observations:**

> - 19,104 monthly records reduced to 2,381 driver-level records.
> - Income averaged per driver.
> - Total business value summed per driver.
> - LastWorkingDate preserved for target creation.

---

## 5. Feature Engineering

```python
# Target: 1 if driver left, 0 if still active
df_final['Target'] = df_final['LastWorkingDate'].notnull().astype(int)

# Rating increase
rating_increase = df.groupby('Driver_ID')['Quarterly Rating'].apply(
    lambda x: 1 if x.max() > x.min() else 0
).reset_index()
rating_increase.columns = ['Driver_ID', 'Quarterly_Rating_Increase']

# Income increase
income_increase = df.groupby('Driver_ID')['Income'].apply(
    lambda x: 1 if x.max() > x.min() else 0
).reset_index()
income_increase.columns = ['Driver_ID', 'Income_Increase']

# Tenure in days
df_final['Tenure_Days'] = (pd.Timestamp('2020-12-31') - df_final['Dateofjoining']).dt.days
df_final = df_final.drop(columns=['LastWorkingDate', 'Dateofjoining'])
```

**Target Distribution:**

| Target | Count | Percentage |
|---|---|---|
| 0 (Stayed) | 765 | 32.13% |
| 1 (Churned) | 1,616 | 67.87% |

[Placeholder: Target class distribution bar plot]

---

## 6. Encoding, SMOTE, and Standardization

```python
# One Hot Encoding
categorical_cols = ['City', 'Gender', 'Education_Level', 'Joining Designation']
df_final = pd.get_dummies(df_final, columns=categorical_cols, drop_first=True)

# Train-test split
from sklearn.model_selection import train_test_split
X = df_final.drop(columns=['Driver_ID', 'Target'])
y = df_final['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE on training data only
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)
```

**Class Balance:**

| Stage | Class 0 | Class 1 | Total |
|---|---|---|---|
| Before SMOTE | 612 | 1,292 | 1,904 |
| After SMOTE | 1,292 | 1,292 | 2,584 |

[Placeholder: SMOTE class balance plot]

---

## 7. Model Building

### Random Forest (Bagging)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_rf = GridSearchCV(rf_base, param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train_res)
best_rf = grid_rf.best_estimator_
```

**Best Parameters:** max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300
**Cross-Validation ROC AUC:** 0.9543
**Test ROC AUC:** 0.9190

### XGBoost (Boosting)

```python
import xgboost as xgb

xgb_base = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_xgb = GridSearchCV(xgb_base, param_grid_xgb, cv=5, scoring='roc_auc', n_jobs=-1)
grid_xgb.fit(X_train_scaled, y_train_res)
best_xgb = grid_xgb.best_estimator_
```

**Best Parameters:** colsample_bytree=0.8, learning_rate=0.1, max_depth=7, n_estimators=200, subsample=1.0
**Cross-Validation ROC AUC:** 0.9537
**Test ROC AUC:** 0.9240

---

## 8. Model Evaluation

### ROC AUC Curve

```python
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

y_prob_rf = best_rf.predict_proba(X_test_scaled)[:, 1]
y_prob_xgb = best_xgb.predict_proba(X_test_scaled)[:, 1]

auc_rf = roc_auc_score(y_test, y_prob_rf)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)
```

**Results:**

| Model | Test ROC AUC |
|---|---|
| Random Forest | 0.9190 |
| XGBoost | 0.9240 |

[Placeholder: ROC AUC curve plot with both models]

### Classification Report (Random Forest)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (Stayed) | 0.7778 | 0.8235 | 0.8000 | 153 |
| 1 (Churned) | 0.9143 | 0.8889 | 0.9014 | 324 |
| Accuracy | | | 0.8679 | 477 |

### Classification Report (XGBoost)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (Stayed) | 0.7683 | 0.8235 | 0.7950 | 153 |
| 1 (Churned) | 0.9137 | 0.8827 | 0.8980 | 324 |
| Accuracy | | | 0.8637 | 477 |

### Confusion Matrices

**Random Forest:**
```
[[126  27]
 [ 36 288]]
```

**XGBoost:**
```
[[126  27]
 [ 38 286]]
```

[Placeholder: Confusion matrices side-by-side plot]

---

## 9. Top Insights

1. **Income growth is the strongest churn signal.** Drivers whose income grew churned at 7 percent. Those whose income stayed flat churned at 69 percent.

2. **Rating improvement reduces churn.** Drivers whose rating improved churned at 57 percent. Those whose rating did not improve churned at 77 percent.

3. **Higher grade means lower churn.** Grade 1 drivers churn at 80 percent. Grade 4 drivers churn at only 51 percent.

4. **Tenure matters.** Newer drivers churn more. The first 90 days are critical.

5. **City-level churn varies widely.** Some cities have far higher churn than others.

6. **Joining Designation 3 is the strongest feature** in the XGBoost model.

7. **Gender and education do not meaningfully affect churn.**

---

## 10. Business Recommendations

| Priority | Recommendation | Expected Impact |
|---|---|---|
| 1 | Introduce automatic income increases for retained drivers | High |
| 2 | Reward drivers whose ratings improve, not just top performers | High |
| 3 | Create a visible Grade 1 to Grade 3 path within the first year | High |
| 4 | Focus retention on the first 90 days of tenure | Medium-High |
| 5 | Run monthly city-level churn reports and allocate retention budget accordingly | Medium |
| 6 | Deploy XGBoost in production for monthly driver churn scoring | High |
| 7 | Investigate why Designation 3 has the highest churn and improve role conditions | Medium |

---

## Future Upgrades & Scaling Plan

> **Deploy the model as a REST API**
> Turn the notebook into a FastAPI endpoint so the retention team can score any driver in real time.

> **Build a retention dashboard**
> Give city managers a live view of at-risk drivers and their churn probability scores.

> **Add more data sources**
> Include trips per day, cancellation rates, customer ratings, and fuel costs to improve prediction accuracy.

> **Automate retraining**
> Retrain the model every quarter with the latest data to keep predictions accurate.

---

## How to Use This Repository

1. Clone the repo:

```
git clone https://github.com/your-username/ola-driver-churn-ensemble-learning.git
```

2. Install required libraries:

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

3. Open the Jupyter Notebook:

```
jupyter notebook OLA_Ensemble_Learning.ipynb
```

4. Run the cells in sequence.

5. Review the outputs and visualizations.

---

## 👤 **Author**

### **Shaik Mayeenuddin**

#### Business Analyst | Data Analytics & AI/ML | Optimizing Processes to Drive Revenue & Retention

🔗https://www.linkedin.com/in/shaikmayeenuddin

---
