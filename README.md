# Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

End-to-end ML system that identifies customers at risk of churning — with a REST API for integration and a Streamlit dashboard for business users.

**Live Demo:**
- Dashboard: [customer-churn-prediction.streamlit.app](https://deepanshu0110-customer-churn-prediction.streamlit.app)
- API Docs: [customer-churn-api.onrender.com/docs](https://customer-churn-api.onrender.com/docs)

---

## Business Problem

Acquiring a new customer costs 5–7x more than retaining an existing one. This system flags high-risk customers before they leave, so retention teams can act early.

---

## Model Results

| Model | F1 Score | Notes |
|---|---|---|
| Logistic Regression | Baseline | Interpretable, fast |
| **Random Forest** | **Best** | Auto-selected for deployment |

---

## Features

- REST API (FastAPI) with Swagger UI at `/docs`
- Streamlit dashboard for single-record and batch CSV predictions
- Auto model selection based on F1 score
- Visualization: churn distribution, feature importance, confusion matrix

---

## Quickstart

```bash
git clone https://github.com/deepanshu0110/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv env && source env/bin/activate
pip install -r requirements.txt
python data_preparation.py
python model_training.py
python run_api.py        # Terminal 1
python run_dashboard.py  # Terminal 2
```

| Service | URL |
|---|---|
| API Swagger | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

---

## ML Pipeline

1. Preprocessing — missing value handling, encoding, scaling
2. Training — Logistic Regression + Random Forest with cross-validation
3. Evaluation — F1, ROC-AUC, confusion matrix
4. Deployment — best model served via FastAPI

---

## Tech Stack

Python · Pandas · Scikit-learn · FastAPI · Streamlit · Uvicorn

---

## Author

**Deepanshu Garg** — Freelance Data Scientist
- GitHub: [@deepanshu0110](https://github.com/deepanshu0110)
- Hire: [freelancer.com/u/deepanshu0110](https://www.freelancer.com/u/deepanshu0110)

MIT License