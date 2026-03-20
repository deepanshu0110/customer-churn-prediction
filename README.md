# Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?style=flat-square&logo=streamlit)
![F1 Score](https://img.shields.io/badge/F1--Score-Best%20Model-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A complete machine learning system to predict customer churn with a FastAPI backend and an interactive Streamlit dashboard.

---

## Live Demo

- **Dashboard (Streamlit):** [Customer Churn Prediction Dashboard](https://deepanshu0110-customer-churn-prediction.streamlit.app)
- **API (FastAPI on Render):** [Customer Churn Prediction API](https://customer-churn-api.onrender.com)
  - `/docs` — Interactive Swagger UI
  - `/health` — Health check

---

## Features

- **ML Models:** Logistic Regression & Random Forest (best model auto-selected by F1 score)
- **REST API:** FastAPI backend with automatic Swagger docs
- **Interactive Dashboard:** Streamlit UI for real-time churn predictions
- **Batch Processing:** Upload CSVs for multiple customer predictions
- **Visualization:** Charts & metrics for churn insights
- **Model Comparison:** Auto-selects best-performing model based on F1 score

---

## Project Structure

```
customer-churn-prediction/
├── data/                    # Dataset files
├── models/                  # Trained models
├── api/
│   └── main.py              # FastAPI backend
├── app/
│   └── dashboard.py         # Streamlit dashboard
├── data_preparation.py      # Data preprocessing
├── model_training.py        # Model training pipeline
├── run_api.py               # API server launcher
├── run_dashboard.py         # Dashboard launcher
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone & setup
git clone https://github.com/deepanshu0110/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv churn_env
churn_env\Scripts\activate  # Windows
source churn_env/bin/activate  # Mac/Linux
pip install -r requirements.txt

# 2. Prepare data & train model
python data_preparation.py
python model_training.py

# 3. Run (Option A — separate terminals)
# Terminal 1
python run_api.py
# Terminal 2
python run_dashboard.py

# 3. Run (Option B — manual)
uvicorn api.main:app --reload --port 8000
streamlit run app/dashboard.py
```

---

## Access

| Service | URL |
|---|---|
| API Swagger | http://127.0.0.1:8000/docs |
| Dashboard | http://localhost:8501 |

---

## ML Approach

1. **Preprocessing** — handle missing values, encode categoricals, scale features
2. **Training** — Logistic Regression + Random Forest with cross-validation
3. **Evaluation** — F1 score comparison, confusion matrix, ROC-AUC
4. **Auto-selection** — best model persisted as `.pkl` for API inference

---

## Tech Stack

Python · Pandas · Scikit-learn · FastAPI · Streamlit · Uvicorn

---

## License

MIT License