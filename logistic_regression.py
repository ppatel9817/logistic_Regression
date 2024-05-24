#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
data = pd.read_excel(url, header=1)

# Inspect the first few rows
print(data.head())

# Drop the ID column
data.drop(columns=['ID'], inplace=True)

# Rename target variable for clarity
data.rename(columns={'default payment next month': 'default'}, inplace=True)

# Encode categorical variables if needed
# Example: data['SEX'] = data['SEX'].astype('category').cat.codes

# Normalize the data
scaler = StandardScaler()
features = data.drop('default', axis=1)
target = data['default']
features_scaled = scaler.fit_transform(features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print("Confusion Matrix:")
print(conf_matrix)

# Coefficients interpretation
coefficients = model.coef_[0]
odds_ratios = np.exp(coefficients)

for feature, coef, odds_ratio in zip(features.columns, coefficients, odds_ratios):
    print(f"Feature: {feature}, Coefficient: {coef}, Odds Ratio: {odds_ratio}")





