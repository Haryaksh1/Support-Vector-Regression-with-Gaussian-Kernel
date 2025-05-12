# Support-Vector-Regression-with-Gaussian-Kernel
This repository provides an end-to-end implementation of Support Vector Regression (SVR) with a Gaussian (RBF) kernel for advanced regression tasks. The project demonstrates robust handling of non-linear relationships and strong generalization to unseen data, using systematic hyperparameter optimization and rigorous model evaluation.
## Overview
- Model: Support Vector Regression (SVR) with Gaussian (RBF) kernel

- Goal: Predict a continuous target variable from multiple input features using a model that balances accuracy and generalization

- Key Features:

- Handles non-linear relationships between features and target

- Robust to outliers

- Systematic hyperparameter search (grid search with cross-validation)

- Clear visualizations of model fit and parameter optimization

## Dataset
File: Supervised-modeling-data-Sheet1.csv

Description: Contains 1000 observations, each with 8 input features (x1–x8) and 1 continuous target variable (y)

Features: Represent various process or system parameters with different scales and ranges

Target: A continuous outcome variable to be predicted

## Workflow
Data Preparation

Load the dataset from CSV

Extract features and target variable

Standardize features using z-score normalization

Train-Test Split

Randomly split data into 80% training and 20% testing sets for unbiased evaluation

Hyperparameter Optimization

Define a grid of candidate values for:

C: Regularization parameter (model complexity)

σ (sigma): Kernel width (controls non-linearity)

Perform 5-fold cross-validation across the grid to find the parameter combination with the lowest mean squared error (MSE)

Model Training

Train the final SVR model using the best-found hyperparameters on the full training set

Evaluation

Predict on the test set and compute key metrics:

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

R² Score (Coefficient of Determination)

Visualize results for interpretability




## Results
Performance: The SVR model achieves strong predictive accuracy and generalization, with R² scores typically around 0.86 on unseen data.

Parameter Selection: Grid search with cross-validation ensures optimal balance between model complexity and generalization.

Interpretability: Visualizations provide clear insight into model fit and parameter sensitivity.

## Usage
Clone the repository and place the dataset CSV in the root directory.

Run the provided MATLAB code to reproduce the results and visualizations.

Review the presentation PDF for a detailed explanation of the methodology and findings.


