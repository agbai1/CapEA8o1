# Capstone Project: Fair and Interpretable Credit Score Prediction

## Project Goal

I set out to build a robust, interpretable, and fair machine learning model to predict credit scores using individual financial behavior data. This project explores how credit scoring systems can be improved for transparency and inclusiveness.

## Dataset

The dataset includes anonymized information about individuals’ financial behavior:

- Numerical features: Annual income, Monthly salary, Number of credit inquiries, Delayed payments, etc.
- Categorical features: Credit Mix, Occupation, Spending Habit, Payment Size
- Target: Credit Score labeled as Good, Standard, or Poor
- A synthetic sensitive attribute (Simulated_Race and Age_Group) was created for fairness evaluation.

## Preprocessing

- Missing values handled with appropriate imputation
- Outliers removed from training data using IQR method
- Categorical variables encoded with OneHot and Ordinal encoders
- Numerical features scaled using RobustScaler
- SMOTE applied to balance classes
- Sensitive features like Age_Group extracted but excluded from model training

## Modeling

- Deep Neural Network built with two hidden layers using ReLU activation and Dropout
- Early stopping used with validation accuracy monitoring
- Class imbalance handled using class weights
- Accuracy and loss curves monitored during training

## Evaluation

- Model achieved ~74% accuracy on the validation set using `accuracy_score` as the metric
- Confusion matrices and classification reports generated
- Threshold tuning tested at 0.6 to filter low-confidence predictions

## Interpretability with SHAP

- SHAP values computed using PermutationExplainer
- Summary bar plot shows most influential features:
  - Delay from Due Date
  - Monthly Inhand Salary
  - Num of Delayed Payments
  - Changed Credit Limit
- Waterfall plots illustrate prediction reasoning for individual samples

## Fairness Auditing

Using Fairlearn, I audited predictions across simulated race groups and binned age groups. Group sizes were fairly balanced across the validation set, providing a reliable comparison of fairness metrics:

- **Group-wise accuracy**:
  - White: 0.75, Black: 0.74, Hispanic: 0.73, Asian: 0.73
- **Demographic Parity Difference**: ~0.026 (Low)

This indicates balanced prediction rates across groups, though additional techniques could improve fairness further.

## Deliverables

- `model.keras`, `pipeline.pkl`, `encoder.pkl`
- `classification_report_full.csv`, `classification_report_confident.csv`
- `credit_predictions_test.csv`, SHAP visualizations

## Impact

This project shows how responsible data science practices, including interpretability and fairness auditing, can be applied to build equitable credit risk models. The approach can inform lending decisions in ways that promote financial inclusion.