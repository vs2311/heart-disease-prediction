# Heart Disease Prediction using Machine Learning

## Project Overview

This project predicts the risk of heart disease using machine learning.

Multiple classification models were trained and evaluated, and K-Nearest Neighbors (KNN) was selected based on its performance.

The trained model is integrated with a Streamlit web application that allows users to enter patient information and receive a prediction.

## Dataset

The dataset contains 918 patient records with 11 input features and 1 target variable.

### Features

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

### Target

`HeartDisease`

- `0` = Lower risk
- `1` = Higher risk

## Machine Learning Models

The following classification models were compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

## Model Performance

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 87.50% | 88.78% |
| KNN | 88.59% | 89.86% |
| Naive Bayes | 86.96% | 87.88% |
| Decision Tree | 77.22% | 79.40% |
| SVM | 86.41% | 88.04% |

### Selected Model

**K-Nearest Neighbors (KNN)**

- Accuracy: **88.59%**
- F1 Score: **89.86%**

## Model Evaluation

The KNN model was further evaluated using:

- Precision
- Recall
- F1 Score
- Confusion Matrix

## Streamlit Application

The trained KNN model is integrated into a Streamlit application.

Users can enter patient information such as age, blood pressure, cholesterol, maximum heart rate, chest pain type, and other clinical features.

The application then predicts the risk of heart disease.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Project Structure

```text
Heart-Disease-Prediction/
│
├── HeartdiseaseFinal.ipynb
├── app.py
├── heart.csv
├── requirements.txt
├── heart_columns.pkl
├── heart_scaler.pkl
└── knn_heart_model.pkl