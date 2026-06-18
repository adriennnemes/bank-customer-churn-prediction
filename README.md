# Bank Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn in the banking sector using supervised machine learning techniques.

Customer churn refers to customers leaving a service or switching to a competitor. For banks, identifying customers at risk of churn can support targeted retention strategies and improve business decision-making.

The project covers the complete machine learning workflow, including exploratory data analysis, data preprocessing, feature engineering, model development, hyperparameter tuning, threshold optimization, model comparison, and feature importance analysis.


## Dataset

The dataset contains customer-level banking information, including demographic, financial, and behavioral attributes.

The target variable indicates whether a customer exited the bank.

### Key Features

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary

Customer identifiers such as customer ID and surname were removed before modeling.


## Exploratory Data Analysis (EDA)

The exploratory analysis focused on understanding churn behavior across different customer segments.

The analysis included:

* Churn distribution
* Churn by gender
* Churn by geography
* Churn by age groups
* Churn by number of products
* Active vs. inactive customers
* Correlation analysis

The dataset showed a clear class imbalance, with significantly fewer churned customers than retained customers.


## Data Preparation

The preprocessing workflow included:

* Removing irrelevant identifier columns
* Encoding categorical variables
* Creating dummy variables
* Train-test split
* Feature scaling where required
* Stratified sampling to preserve class distribution


## Machine Learning Models

Several classification models were trained and evaluated:

### Logistic Regression

Used as a baseline model for binary classification.

### Decision Tree

Provided interpretable decision rules and served as an additional baseline model.

### Random Forest

An ensemble learning method combining multiple decision trees to improve predictive performance and reduce overfitting.

### XGBoost

A gradient boosting algorithm capable of capturing complex patterns and handling class imbalance effectively.


## Handling Class Imbalance

Since only a minority of customers churned, accuracy alone was not sufficient for evaluation.

To address class imbalance, the project used:

* `class_weight='balanced'` for Random Forest
* `scale_pos_weight` for XGBoost

Evaluation focused on:

* Precision
* Recall
* F1-Score
* ROC-AUC

These metrics provide a more meaningful assessment of model performance in churn prediction scenarios.


## Hyperparameter Tuning

Random Forest was optimized using GridSearchCV with 5-fold cross-validation.

Parameters explored included:

* Number of trees (`n_estimators`)
* Maximum tree depth (`max_depth`)
* Minimum samples per split (`min_samples_split`)
* Minimum samples per leaf (`min_samples_leaf`)
* Maximum features (`max_features`)
* Class weighting (`class_weight='balanced'`)

The optimization objective was ROC-AUC, focusing on the model's ability to distinguish churners from non-churners.


## Threshold Optimization

Different classification thresholds were evaluated to analyze the trade-off between:

* Precision
* Recall
* F1-Score

Threshold tuning allows model predictions to be aligned with different business priorities.

For example:

* Higher recall helps identify more customers at risk of churn.
* Higher precision reduces false positive retention actions.


## Model Evaluation

Model performance was compared using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

The evaluation demonstrated that different models offer different strengths depending on business requirements.

Random Forest provided a strong balance between precision and recall, while XGBoost achieved highly competitive performance after threshold optimization.


## Feature Importance Analysis

Feature importance was analyzed to better understand the factors driving customer churn.

Important features included:

* Age
* Number of Products
* Active Membership Status
* Geography
* Account Balance

The analysis helped translate model outputs into actionable business insights.


## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Jupyter Notebook


## Key Learnings

This project provided hands-on experience with end-to-end machine learning workflows and highlighted several important concepts:

* Class imbalance can significantly impact model performance.
* Accuracy alone is often misleading.
* Precision, Recall, F1-Score, and ROC-AUC provide more meaningful evaluation metrics.
* Hyperparameter tuning improves model robustness.
* Threshold optimization can influence business outcomes.
* Feature importance helps bridge the gap between machine learning and business decision-making.
