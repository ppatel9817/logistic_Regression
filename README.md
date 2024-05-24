# README

## Credit Card Default Prediction

This project demonstrates a basic application of logistic regression for predicting the default of credit card clients. The dataset used is from the UCI Machine Learning Repository, specifically the "Default of Credit Card Clients" dataset.

### Dataset

The dataset contains information on credit card clients and whether they defaulted on their payment the next month. The main steps involved in the analysis are data preprocessing, model training, and evaluation.

### Files

- `default of credit card clients.xls`: The dataset file containing the credit card client information and target variable.

### Dependencies

To run the code, you need to have the following libraries installed:

- pandas
- numpy
- scikit-learn
- openpyxl (for reading .xls files with pandas)

You can install these libraries using pip:

```sh
pip install pandas numpy scikit-learn openpyxl
```

### Code Overview

The script performs the following steps:

1. **Load the dataset**:
    ```python
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    data = pd.read_excel(url, header=1)
    ```

2. **Data Inspection**:
    Print the first few rows of the dataset to understand its structure.

3. **Data Preprocessing**:
    - Drop the ID column.
    - Rename the target variable for clarity.
    - Encode categorical variables if necessary.
    - Normalize the data using `StandardScaler`.

4. **Split the Data**:
    Split the data into training and testing sets using `train_test_split`.

5. **Train the Model**:
    Train a logistic regression model with `max_iter=1000`.

6. **Make Predictions**:
    Use the trained model to make predictions on the test set.

7. **Evaluate the Model**:
    Evaluate the model using accuracy, precision, recall, and confusion matrix.

8. **Interpret Coefficients**:
    Calculate and print the coefficients and odds ratios of the features.

