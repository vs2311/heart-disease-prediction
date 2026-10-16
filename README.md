# ❤️ Heart Disease Risk Prediction

An end-to-end machine learning project that predicts the risk of heart disease from clinical data, deployed as an interactive Streamlit web app.

Five classification models were trained and compared. **K-Nearest Neighbors (KNN)** was selected for its performance and is explained inside the app through similar-patient lookups, what-if curves and dataset comparisons.

### 🚀 [Live Demo: Try the App](https://heart-disease-prediction-y9i9zwzkxpcsxuezzhwk73.streamlit.app/)

> ⚠️ **Disclaimer:** This project is for educational and demonstration purposes only. It is not a medical diagnosis tool.

---

## ✨ App Features

| Feature | Description |
|---|---|
| **Live prediction** | Risk updates in real time as patient values change. |
| **Risk gauge** | Shows a risk score, the share of nearest neighbours voting "heart disease". |
| **Risk markers** | Highlights common clinical warning signs in the entered profile. |
| **Patient vs. population radar** | Compares the patient with the average healthy and heart-disease patient. |
| **What-If Explorer** | Shows how risk changes when one feature (age, cholesterol, max heart rate, etc.) is varied. |
| **Similar Patients** | Lists the closest patients from the dataset and their outcomes, which explains the KNN decision. |
| **Data Insights** | Feature distributions, class balance and correlation heatmap. |
| **Presets & report** | One-click example patients and a downloadable text report. |

---

## 📊 Dataset

918 patient records, 11 input features and 1 target variable.

**Features**

| Feature | Description |
|---|---|
| `Age` | Age of the patient |
| `Sex` | M / F |
| `ChestPainType` | ATA (atypical angina), NAP (non-anginal pain), TA (typical angina), ASY (asymptomatic) |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dL) |
| `FastingBS` | Fasting blood sugar > 120 mg/dL (1 = yes, 0 = no) |
| `RestingECG` | Normal, ST, LVH |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (Y / N) |
| `Oldpeak` | ST depression induced by exercise |
| `ST_Slope` | Slope of the peak-exercise ST segment (Up / Flat / Down) |

**Target:** `HeartDisease`, where `0` = lower risk and `1` = higher risk.

---

## 🤖 Machine Learning Pipeline

1. **Preprocessing:** one-hot encoding of categorical features and feature scaling (essential for distance-based models such as KNN).
2. **Model training:** five classifiers trained and compared.
3. **Evaluation:** accuracy, F1 score, precision, recall and confusion matrix.
4. **Model selection:** KNN chosen as the best performer.
5. **Deployment:** model, scaler and column order saved with Joblib and served through Streamlit.

### Model Comparison

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 87.50% | 88.78% |
| **KNN (selected)** | **88.59%** | **89.86%** |
| Naive Bayes | 86.96% | 87.88% |
| Decision Tree | 77.22% | 79.40% |
| SVM | 86.41% | 88.04% |

### Selected Model: K-Nearest Neighbors

- **Accuracy:** 88.59%
- **F1 Score:** 89.86%

---

## 🛠️ Technologies Used

- **Language:** Python
- **Data & ML:** Pandas, NumPy, Scikit-learn, Joblib
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Web app:** Streamlit
- **Notebook:** Jupyter Notebook

---

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── HeartdiseaseFinal.ipynb   # Data analysis, model training & evaluation
├── app.py                    # Streamlit web application
├── heart.csv                 # Dataset
├── requirements.txt          # Python dependencies
├── heart_columns.pkl         # Feature column order used in training
├── heart_scaler.pkl          # Fitted scaler
└── knn_heart_model.pkl       # Trained KNN model
```

---

## ▶️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/vs2311/heart-disease-prediction.git
cd heart-disease-prediction

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🔮 Future Improvements

- Hyperparameter tuning with cross-validation
- Probability calibration for more reliable risk scores
- Additional models (Random Forest, XGBoost) and ensemble methods
- Model explainability with SHAP
- Evaluation on an external dataset

---

## 👤 Author

**Vijay Saroj**
GitHub: [@vs2311](https://github.com/vs2311)
