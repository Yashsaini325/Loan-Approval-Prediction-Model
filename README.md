# Loan Approval Prediction Model

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-red)
![PyTorch](https://img.shields.io/badge/PyTorch-ANN-ee4c2c?logo=pytorch&logoColor=white)

A machine learning project that predicts whether a loan application will be approved based on applicant, financial, employment, and loan-related features. The project includes exploratory data analysis, missing-value handling, categorical encoding, feature scaling, model comparison, and an Artificial Neural Network implementation.

The notebook identifies **XGBoost Classifier** as the best-performing tested model, with a test accuracy of **93%**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Prediction Interpretation](#prediction-interpretation)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Loan approval decisions often depend on a combination of income, credit history, debt obligations, assets, employment details, and applicant demographics. This project explores those relationships and compares several classification algorithms for predicting the `Loan_Approved` target.

The notebook implements:

- Data loading and inspection
- Exploratory Data Analysis (EDA)
- Missing-value analysis and imputation
- Categorical feature encoding
- Duplicate and identifier handling
- Correlation analysis
- Train-test splitting
- Feature standardization
- Classical machine learning model comparison
- A PyTorch Artificial Neural Network
- Precision, recall, F1-score, accuracy, confusion matrix, and classification report evaluation

---

## Dataset

The project uses `loan_approval_data.csv`.

| Property | Value |
|---|---|
| Original rows | 1,000 |
| Columns | 20 |
| Target column | `Loan_Approved` |
| Target values | `Yes` / `No` |
| Known target distribution | 652 No, 298 Yes |
| Missing-value issue | 50 rows contain missing values across all 20 columns |

### Feature Groups

**Numeric features**

- Applicant_ID
- Applicant_Income
- Coapplicant_Income
- Age
- Dependents
- Credit_Score
- Existing_Loans
- DTI_Ratio
- Savings
- Collateral_Value
- Loan_Amount
- Loan_Term

**Categorical features**

- Employment_Status
- Marital_Status
- Loan_Purpose
- Property_Area
- Education_Level
- Gender
- Employer_Category

**Target**

- Loan_Approved

---

## Project Workflow

1. **Load the dataset**
   - Read `loan_approval_data.csv` with pandas.
   - Create a copy of the original data for preprocessing.

2. **Inspect and explore the data**
   - Examine data types, shapes, sample records, descriptive statistics, and missing values.
   - Visualize distributions and relationships using matplotlib and seaborn.
   - Generate a correlation heatmap for numerical variables.

3. **Handle missing values**
   - Use mean imputation for numerical columns.
   - Use most-frequent-value imputation for categorical columns.

4. **Encode categorical variables**
   - Apply label encoding to `Marital_Status`, `Gender`, `Loan_Approved`, and `Education_Level`.
   - Apply one-hot encoding with `drop_first=True` to `Employer_Category`, `Property_Area`, `Loan_Purpose`, and `Employment_Status`.

5. **Prepare features**
   - Remove `Applicant_ID`.
   - Separate features (`X`) from the target (`y`).
   - Split the data with an 80/20 train-test ratio and `random_state=42`.

6. **Scale the data**
   - Fit `StandardScaler` on the training data.
   - Transform both training and test data to prevent data leakage.

7. **Train and compare models**
   - Train classical models and an ANN.
   - Compare training and testing metrics.

---

## Models Evaluated

The notebook evaluates the following classifiers:

- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Classifier
- Decision Tree
- Random Forest
- XGBoost Classifier
- Artificial Neural Network using PyTorch

---

## Results

| Model | Precision | Recall | F1-Score | Training Accuracy | Test Accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7833 | 0.7705 | 0.7769 | 86.50% | 86.50% |
| KNN | 0.6275 | 0.5246 | 0.5714 | 86.00% | 76.00% |
| Gaussian Naive Bayes | 0.8036 | 0.7377 | 0.7692 | 87.75% | 86.50% |
| SVC | 0.8036 | 0.7377 | 0.7692 | 96.38% | 86.50% |
| Decision Tree | 0.8036 | 0.7377 | 0.7692 | 91.00% | 86.50% |
| Random Forest | 0.8036 | 0.7377 | 0.7692 | 70.75% | 86.50% |
| **XGBoost** | **0.8615** | **0.9180** | **0.8889** | **100.00%** | **93.00%** |
| PyTorch ANN | 0.8070 | 0.7541 | 0.7797 | 91.13% | 87.00% |

### Best Model

**XGBoost Classifier**

- Test accuracy: **93%**
- Precision: **0.8615**
- Recall: **0.9180**
- F1-score: **0.8889**

The recorded XGBoost results indicate strong overall performance. However, the 100% training accuracy suggests possible overfitting and should be validated with cross-validation on unseen data.

---

## Project Structure

```text
Loan Approval Prediction Model/
├── Load_Approval_Model.ipynb  # Data analysis and model training notebook
├── loan_approval_data.csv      # Loan application dataset
└── README.md                   # Project documentation
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Loan Approval Prediction Model"
```

### 2. Create a Virtual Environment

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost torch jupyter
```

Key libraries used in the notebook:

- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- PyTorch
- Jupyter Notebook

---

## How to Run

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `Load_Approval_Model.ipynb` and run the cells from top to bottom.

You can also open the notebook in JupyterLab, VS Code, or Google Colab. When opening it outside the project directory, ensure `loan_approval_data.csv` is available in the same working directory or update the data path in the loading cell.

---

## Prediction Interpretation

After preprocessing and encoding:

- `1` represents an approved loan.
- `0` represents a rejected or not-approved loan.

Model quality should be evaluated using more than accuracy alone. Precision, recall, F1-score, and the confusion matrix are especially important if the cost of wrongly approving or rejecting applicants is significant.

---

## Limitations

- The notebook uses a fixed train-test split rather than repeated cross-validation.
- The exact ANN architecture and hyperparameters should be checked and tuned before production use.
- XGBoost shows perfect training accuracy in the recorded results, indicating a risk of overfitting.
- The dataset contains 50 fully missing records.
- Missing categorical and numerical values are filled with simple imputation, which may not capture the true data distribution.
- Encoders and scalers are not currently packaged into a reusable prediction pipeline.
- A model at this stage should not make real-world lending decisions without fairness, legal, and human-review checks.

---

## Future Improvements

- Add stratified cross-validation and hyperparameter tuning
- Use class imbalance techniques if needed
- Build reusable preprocessing and prediction pipelines
- Remove or more carefully analyze fully missing records
- Tune the PyTorch ANN learning rate, epochs, depth, and regularization
- Add feature importance and SHAP explainability analysis
- Add unit tests and model serialization
- Deploy the model with a Flask, FastAPI, or Streamlit interface
- Test model fairness across protected and sensitive groups
- Track experiments and metrics with MLflow or similar tools

---

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Run and verify the notebook.
5. Open a pull request with a clear description.

Please avoid committing large generated outputs or unnecessary checkpoints.

---

## License

No license is currently specified for this repository. A license should be added if the project is intended for reuse or distribution.