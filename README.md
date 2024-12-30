# Lending Club Data Analysis - Credit Risk Prediction

Multifinance companies need to improve the accuracy of credit risk assessments to optimize business decisions and reduce losses. We developed a machine learning model using loan data from Lending Club (2007-2014) to predict credit risk, with a focus on business metrics such as losses and net profit margin. The aim of this data analysis is to identify patterns that indicate loans at potential risk, without strong assumptions, to support investment decision-making.

The dataset used in this project is provided by **ID/X Partners**.

## Data Understanding

The dataset consists of **466,285 rows** and **74 columns**. The columns can be categorized into:

- **22 categorical columns**  
- **52 numerical columns**

### Key Observations:

- Several columns contain unique values, such as `id`, `member_id`, and `url`. These columns will be dropped as they do not provide meaningful information for analysis.
- The columns `policy_code` and `application_type` contain only one unique value each, so they will also be dropped.
- There are **22 columns with more than 20% missing data**, which will be removed.
- **18 columns have missing values below 20%**, for which the missing values will be removed (i.e., rows with missing values will be dropped).

### Columns Dropped:
- `id`
- `member_id`
- `url`
- `policy_code`
- `application_type`

## Exploratory Data Analysis (EDA)

The EDA focuses on uncovering patterns, correlations, and key insights from the dataset. Some of the key findings include:

- **Verification Status and Loan Grades**:  
  - The **Verified** verification status shows an increase in loan percentage across grades A to G.
  - The **Not Verified** status shows a decrease in loan percentage from grade A to G.
  - The **Source Verified** status remains stable across all loan grades.

These insights are helpful in understanding how verification status relates to loan grades and the distribution of loans.

## Data Preparation

### Handling Missing Values:

The data preparation steps include handling missing values and ensuring the dataset is ready for model training. The following steps were taken:

1. **Dropping Unnecessary Columns**:
   - Dropped columns such as `id`, `member_id`, `url`, `policy_code`, and `application_type` because they either contained unique values or did not provide meaningful information.
   
2. **Handling Columns with High Missing Data**:
   - Dropped columns with **more than 20% missing values** as they are not reliable for analysis.
   
3. **Handling Columns with Low Missing Data**:
   - Removed rows with missing values in columns that had **less than 20% missing data** to ensure data integrity.

### Handling Outliers:

- Log transformation was applied to all columns that contained outliers in order to reduce the impact of extreme values and ensure a more normal distribution.

### Label Encoding:

- Label encoding was applied to all **categorical columns** to convert categorical data into numerical format, making it suitable for machine learning algorithms.

After handling missing values, outliers, and performing label encoding, the dataset was reduced by **20.19%**, resulting in **372,161 rows** of data.

## Feature Engineering

### Feature Selection:
- 13 columns that exhibited multicollinearity were removed to prevent overfitting and improve model performance.

### Feature Transformation:
- Log transformation was performed on columns that contained outliers.
- The `loan_status` column was transformed to separate the status into three categories: **Failed to Pay**, **Successfully Paid**, and **Ongoing**.

## Data Modeling

### Class Imbalance:

The `loan_status` label suffers from class imbalance, which could negatively affect model performance. Therefore, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to balance the classes in the dataset.

### Model Comparison:

Two algorithms were used for modeling:

- **Logistic Regression**: A classic linear model.
- **XGBoost**: An advanced boosting algorithm that is known for its superior performance in many machine learning tasks.

Both models were optimized using hyperparameters, and **XGBoost** outperformed **Logistic Regression** in terms of predictive accuracy.

## How to Run

To run the analysis and reproduce the results, you will need to have the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
