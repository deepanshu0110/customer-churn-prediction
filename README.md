# 📊 Customer Churn Prediction System

A complete machine learning system to predict customer churn with a **FastAPI backend** and an **interactive Streamlit dashboard**.

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://customer-churn-prediction-78oq.onrender.com/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://customer-churn-prediction-xoywtnzmbcegohqflqpe9m.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌐 Live Demo

- **📊 Dashboard (Streamlit Frontend):**  
  👉 [Customer Churn Prediction Dashboard](https://customer-churn-prediction-xoywtnzmbcegohqflqpe9m.streamlit.app)

- **⚡ API (FastAPI on Render):**  
  👉 [Customer Churn Prediction API](https://customer-churn-prediction-78oq.onrender.com)  

  - `/` → API status  
  - `/health` → Health check  
  - `/docs` → Interactive Swagger UI  

- **💻 Source Code (GitHub):**  
  👉 [GitHub Repository](https://github.com/deepanshu0110/customer-churn-prediction)

---

## 🌟 Features

- **Machine Learning Model**: Random Forest and Logistic Regression comparison  
- **REST API**: FastAPI backend with automatic Swagger documentation  
- **Interactive Dashboard**: Streamlit UI for real-time predictions  
- **Batch Processing**: Upload CSVs for multiple predictions at once  
- **Data Visualization**: Charts and insights for customer churn  
- **Model Comparison**: Automatically selects the best model based on F1 score  

---

## 📁 Project Structure

customer-churn-prediction/
├── data/ # Dataset files
├── models/ # Trained models (saved as .pkl)
├── api/ # FastAPI backend
│ └── main.py
├── app/ # Streamlit dashboard
│ └── dashboard.py
├── data_preparation.py # Data preprocessing
├── model_training.py # Model training script
├── run_api.py # API server launcher
├── run_dashboard.py # Dashboard launcher
├── requirements.txt # Dependencies
└── README.md # Project documentation

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
Option A: Separate services (recommended)

bash
Copy code
# Terminal 1 - API Server
python run_api.py

# Terminal 2 - Streamlit Dashboard
python run_dashboard.py
Option B: Manual

bash
Copy code
# API
uvicorn api.main:app --reload --port 8000

# Dashboard
streamlit run app/dashboard.py
4️⃣ Access Locally
API: http://127.0.0.1:8000/docs

Dashboard: http://localhost:8501

📊 Usage
🔹 Single Prediction
Go to Dashboard → Single Prediction

Enter customer details

Click Predict Churn

🔹 Batch Prediction
Prepare a CSV file with customer data

Go to Dashboard → Batch Prediction

Upload CSV → View results → Download predictions

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
GET / → API root status

GET /health → Health check

GET /sample → Sample customer data

GET /model_info → Model details

POST /predict → Predict single customer

POST /predict_batch → Predict multiple customers

🎯 Model Performance
Random Forest & Logistic Regression evaluated

Best model selected using F1 score

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
matplotlib
seaborn
plotly
🐛 Troubleshooting
API not starting: Ensure models exist in models/

Dashboard error: Make sure API is live (/health)

CSV errors: Column names must match expected schema

📈 Roadmap
 Add more models (XGBoost, LightGBM, Neural Nets)

 Implement monitoring & retraining pipeline

 Add authentication for dashboard

 Improve visualization with more business KPIs

🤝 Contributing
Contributions welcome!

Fork the repo

Create a branch (feature-new)

Commit changes

Open a Pull Request

📄 License
This project is licensed under the MIT License.

yaml
Copy code

---

👉 This new version has:
- Your **Render API link**  
- Your **Streamlit dashboard link**  
- Clickable badges at the top  
- Cleaner structure with live demo section  

Would you also like me to add **sample screenshots** (your dashboard & API docs) to the README so it looks more profe
