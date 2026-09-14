# Telecom Customer Churn Prediction

A machine learning project to predict customer churn using a telecom dataset (1,000 customers, 13 features), comparing Logistic Regression, Random Forest, and XGBoost classifiers.

## Project Q&A — What was performed and why

**Q1. What problem does this project solve?**
Predicting whether a telecom customer will churn (leave the service) — `Yes`/`No` — based on their demographic, account, and service usage details. This is a binary classification problem.

**Q2. What dataset was used?**
A CSV file with 1,000 customer records and 13 columns: `customer_id`, `age`, `gender`, `tenure_months`, `monthly_charges`, `total_charges`, `contract_type`, `internet_service`, `payment_method`, `tech_support`, `online_security`, `senior_citizen`, and the target column `Churn`.

**Q3. How was the data inspected?**
Used `df.info()`, `df.describe()`, and `df.isnull().sum()` to check data types, summary statistics, and missing values before any cleaning.

**Q4. How was missing data handled?**
- Numeric columns (`age`, `monthly_charges`, `total_charges`) — filled with the **median**.
- Categorical columns (`gender`, `payment_method`, `tech_support`, `online_security`) — filled with the **mode**.
The `customer_id` column was dropped since it carries no predictive information.

**Q5. What did the exploratory data analysis (EDA) show?**
- Overall churn rate across the dataset.
- Churn rate broken down by `contract_type` and `internet_service`.
- Correlation of numeric features (`age`, `tenure_months`, `monthly_charges`, `total_charges`, `senior_citizen`) with churn.
- Class distribution: **66.2% No / 33.8% Yes** — a moderately imbalanced target, which is why accuracy alone isn't a sufficient metric (ROC-AUC was also tracked).

**Q6. How were categorical features encoded?**
One-hot encoding (`pd.get_dummies(..., drop_first=True)`) was applied to all categorical columns to convert them into numeric features usable by the models.

**Q7. How was the data split for training and testing?**
`train_test_split` with `test_size=0.2`, `random_state=42`, and `stratify=target` — stratification ensures the churn class ratio is preserved in both train and test sets.

**Q8. Was feature scaling applied?**
Yes — `StandardScaler` was fit on the training set only and applied to both train and test sets, to avoid data leakage. Scaling was used for Logistic Regression; tree-based models (Random Forest, XGBoost) don't require it.

**Q9. Which models were trained and compared?**
1. **Logistic Regression** — baseline linear model, trained on scaled features.
2. **Random Forest Classifier** (`n_estimators=300`) — ensemble of decision trees.
3. **XGBoost Classifier** (`n_estimators=300`, `max_depth=4`, `learning_rate=0.05`) — gradient-boosted trees.

All three were trained and evaluated through a shared `evaluate()` function for consistency.

**Q10. How were the models evaluated?**
- **Accuracy**
- **ROC-AUC** (important given the class imbalance)
- **Confusion matrix**
- **Classification report** (precision, recall, F1-score)

**Q11. How were the models compared?**
Results were compiled into a single comparison table (`Model`, `Accuracy`, `ROC-AUC`) to easily see which model performed best.

**Q12. What did feature importance analysis reveal?**
Using the Random Forest model's `feature_importances_`, the top features driving churn predictions were identified and ranked.

**Q13. What are the known limitations of this project?**
- Missing-value imputation was done on the full dataset before the train/test split, which introduces minor data leakage (correct practice: split first, then impute using train-set statistics only).
- No explicit class-imbalance handling (e.g. `class_weight="balanced"`) was applied to the models.
- Model evaluation relies on a single train/test split rather than cross-validation.
- No hyperparameter tuning (e.g. `GridSearchCV`) was performed; model parameters were set to reasonable fixed values.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost

## How to Run
1. Install dependencies: `pip install pandas numpy scikit-learn xgboost`
2. Place `telecom_churn_data.csv` in the working directory.
3. Run the notebook cells in order: data loading → cleaning → EDA → encoding → split → scaling → model training → evaluation → comparison → feature importance.
