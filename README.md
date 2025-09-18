# 📊 Customer Churn Prediction System

A complete machine learning system to predict customer churn with a **FastAPI backend** and an **interactive Streamlit dashboard**.

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://customer-churn-prediction-78oq.onrender.com/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://customer-churn-prediction-xoywtnzmbcegohqflgpe9m.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



---

## 🌐 Live Demo

- **📊 Dashboard (Streamlit Frontend):**  
  👉 [Customer Churn Prediction Dashboard](https://customer-churn-prediction-xoywtnzmbcegohqflgpe9m.streamlit.app)

- **⚡ API (FastAPI on Render):**  
  👉 [Customer Churn Prediction API](https://customer-churn-prediction-78oq.onrender.com)  

  - `/docs` → Interactive Swagger UI  
  - `/health` → Health check  

- **💻 Source Code (GitHub):**  
  👉 [GitHub Repository](https://github.com/deepanshu0110/customer-churn-prediction)

---

## 🌟 Features

- **Machine Learning Models**: Logistic Regression & Random Forest (best model auto-selected)  
- **REST API**: FastAPI backend with automatic Swagger docs  
- **Interactive Dashboard**: Streamlit UI for real-time churn predictions  
- **Batch Processing**: Upload CSVs for multiple customer predictions  
- **Visualization**: Charts & metrics for churn insights  
- **Model Comparison**: Selects best-performing model based on F1 score  

---

## 📁 Project Structure

customer-churn-prediction/
├── data/ # Dataset files
├── models/ # Trained models
├── api/ # FastAPI backend
│ └── main.py
├── app/ # Streamlit dashboard
│ └── dashboard.py
├── data_preparation.py # Data preprocessing
├── model_training.py # Model training
├── run_api.py # API server launcher
├── run_dashboard.py # Dashboard launcher
├── requirements.txt # Dependencies
└── README.md # Documentation

yaml
Copy code

---

## 🚀 Quick Start (Local)

### 1️⃣ Setup Environment
```bash
# Create virtual environment
python -m venv churn_env

# Activate (Windows)
churn_env\Scripts\activate

# Activate (Mac/Linux)
source churn_env/bin/activate

# Install dependencies
pip install -r requirements.txt
2️⃣ Prepare Data & Train Model
bash
Copy code
python data_preparation.py
python model_training.py
3️⃣ Run the System
Option A – Separate services (recommended):

bash
Copy code
# Terminal 1 - API
python run_api.py

# Terminal 2 - Streamlit Dashboard
python run_dashboard.py
Option B – Manual launch:

bash
Copy code
# API
uvicorn api.main:app --reload --port 8000

# Dashboard
streamlit run app/dashboard.py
4️⃣ Access Locally
API Docs → http://127.0.0.1:8000/docs

Dashboard → http://localhost:8501

📊 Usage
🔹 Single Prediction
Go to Dashboard → Single Prediction

Enter customer details

Click Predict Churn

🔹 Batch Prediction
Upload a CSV file with customer data

Dashboard → Batch Prediction

View results & download predictions

📋 Required CSV Format
Column	Type	Example
Gender	str	Male/Female
SeniorCitizen	int	0/1
Partner	str	Yes/No
Dependents	str	Yes/No
tenure	int	12
PhoneService	str	Yes/No
MultipleLines	str	Yes/No/No phone service
InternetService	str	DSL/Fiber optic/No
OnlineSecurity	str	Yes/No/No internet service
OnlineBackup	str	Yes/No/No internet service
DeviceProtection	str	Yes/No/No internet service
TechSupport	str	Yes/No/No internet service
StreamingTV	str	Yes/No/No internet service
StreamingMovies	str	Yes/No/No internet service
Contract	str	Month-to-month/One year/Two year
PaperlessBilling	str	Yes/No
PaymentMethod	str	Electronic check/Mailed check/...
MonthlyCharges	float	29.85
TotalCharges	float	358.20

🔧 API Endpoints
GET / → API status

GET /health → Health check

GET /sample → Sample input row

GET /model_info → Model details

POST /predict → Single prediction

POST /predict_batch → Batch prediction

🎯 Model Performance
Logistic Regression & Random Forest compared

Best model chosen by F1 score

Decision threshold optimized automatically

📦 Dependencies
nginx
Copy code
pandas
numpy
scikit-learn
fastapi
uvicorn
streamlit
requests
plotly
matplotlib
seaborn
📈 Roadmap
Add more models (XGBoost, LightGBM, Neural Nets)

Implement retraining pipeline

Add authentication for dashboard

Enhance visualization with business KPIs

🤝 Contributing
Fork the repository

Create a new branch

Make your changes

Commit & push

Open a Pull Request

## 📄 License

This project is licensed under the [MIT License](LICENSE).
