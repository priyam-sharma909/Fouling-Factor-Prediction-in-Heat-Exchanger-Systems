# Fouling Factor Prediction in Heat Exchanger Systems

This repository contains Python implementations for:

- Data preprocessing and feature analysis  
- Statistical and machine learning models for fouling prediction  
- Deep learning models for nonlinear and time-dependent behavior  

---

## 1. Data Processing and Feature Analysis

**Folder:** `Dataset`

Process data from heat exchanger systems is used to analyze the factors influencing fouling behavior.

**Includes:**

- Data cleaning and preparation  
- Feature correlation analysis  
- Dimensionality reduction using PCA  

These steps help improve model accuracy and reduce redundancy in input variables.

---

## 2. Machine Learning Models

**Folder:** `ALL_CODES`

Builds multiple regression-based models to predict the fouling factor:

- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- PCA-based modeling  
- Auto-Regressive model  

These models provide baseline and regularized approaches for fouling prediction.

---

## 3. Deep Learning Models

**Folder:** `ALL_CODES`

Implements neural network-based approaches to capture nonlinear fouling dynamics:

- Conventional Artificial Neural Network (ANN)  
- LSTM (Long Short-Term Memory) network  

These models are used to learn complex relationships and time-dependent fouling trends.

---

## Results

Model performance is evaluated using:

- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

Comparisons between models highlight the effectiveness of deep learning methods in predicting fouling behavior.

Simulation outputs, performance comparison, and analysis are documented in the attached project presentation.
