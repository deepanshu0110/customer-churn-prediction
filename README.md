# 📊 Customer Churn Prediction System

A complete machine learning system to predict customer churn with a **FastAPI backend** and an **interactive Streamlit dashboard**.

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://customer-churn-prediction-78oq.onrender.com/docs)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://customer-churn-prediction-xoywtnzmbcegohqflqpe9m.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)

---

## 🌐 Live Demo

- **📊 Dashboard (Streamlit Frontend):**  
  👉 [Customer Churn Prediction Dashboard](https://customer-churn-prediction-xoywtnzmbcegohqflqpe9m.streamlit.app)

- **⚡ API (FastAPI on Render):**  
  👉 [Customer Churn Prediction API](https://customer-churn-prediction-78oq.onrender.com)  

  - [/docs](https://customer-churn-prediction-78oq.onrender.com/docs) → Interactive Swagger UI  
  - [/health](https://customer-churn-prediction-78oq.onrender.com/health) → Health check  

- **💻 Source Code (GitHub):**  
  👉 [GitHub Repository](https://github.com/deepanshu0110/customer-churn-prediction)


---

## 🌟 Features

- **Machine Learning Models**: Logistic Regression & Random Forest (best model auto-selected)  
- **REST API**: FastAPI backend with Swagger docs  
- **Dashboard**: Streamlit UI for interactive predictions  
- **Batch Processing**: Upload CSVs for multiple predictions  
- **Visualizations**: Churn rates, probabilities, and confidence plots  

---

## 📁 Project Structure

customer-churn-prediction/
├── data/ # Dataset files

├── models/ # Trained models (.pkl)

├── api/ # FastAPI backend

│ └── main.py

├── app/ # Streamlit dashboard

│ └── dashboard.py

├── data_preparation.py # Data preprocessing

├── model_training.py # Model training script

├── run_api.py # API server launcher

├── run_dashboard.py # Dashboard launcher

├── requirements.txt # Dependencies

└── README.md # Documentation


---

## 🚀 Local Setup

### 1️⃣ Setup Environment
```bash
python -m venv churn_env
churn_env\Scripts\activate    # Windows
source churn_env/bin/activate # Mac/Linux

pip install -r requirements.txt

2️⃣ Train Model
python data_preparation.py
python model_training.py

3️⃣ Run Services
# API
python run_api.py
# or
uvicorn api.main:app --reload --port 8000

# Dashboard
python run_dashboard.py
# or
streamlit run app/dashboard.py


API: http://127.0.0.1:8000/docs

Dashboard: http://localhost:8501

📊 Usage
🔹 Single Prediction

Go to Dashboard → Single Prediction

Review sample or enter customer details

Click Predict Churn

🔹 Batch Prediction

Prepare a CSV file with required columns

Upload in Dashboard → Batch Prediction

View results and download predictions

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

GET / → API root

GET /health → Health check

GET /sample → Sample input row

GET /model_info → Model details

POST /predict → Single prediction

POST /predict_batch → Batch prediction

📦 Dependencies
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

📄 License

MIT License
