# 📊 Customer Churn Prediction System

A complete machine learning system to predict customer churn with FastAPI backend and Streamlit dashboard.

## 🌟 Features

- **Machine Learning Model**: Random Forest and Logistic Regression comparison
- **REST API**: FastAPI backend with automatic documentation
- **Interactive Dashboard**: Streamlit web interface for predictions
- **Batch Processing**: Upload CSV files for multiple predictions
- **Data Visualization**: Charts and graphs for insights
- **Model Comparison**: Automatic selection of best performing model

## 📁 Project Structure

```
customer-churn-prediction/
├── data/                    # Data files
├── models/                  # Trained models
├── notebooks/              # Jupyter notebooks (optional)
├── api/                    # FastAPI backend
│   └── main.py
├── app/                    # Streamlit dashboard
│   └── dashboard.py
├── data_preparation.py     # Data preprocessing
├── model_training.py       # Model training script
├── run_api.py             # API server launcher
├── run_dashboard.py       # Dashboard launcher
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv churn_env

# Activate virtual environment
# Windows:
churn_env\Scripts\activate
# Mac/Linux:
source churn_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data and Train Model

```bash
# Prepare the dataset
python data_preparation.py

# Train the model
python model_training.py
```

### 3. Run the System

**Option A: Run both services separately (Recommended)**

Terminal 1 - API Server:
```bash
python run_api.py
```

Terminal 2 - Dashboard:
```bash
python run_dashboard.py
```

**Option B: Manual launch**

API Server:
```bash
cd api
python main.py
```

Dashboard:
```bash
streamlit run app/dashboard.py
```

### 4. Access the Applications

- **API Documentation**: http://127.0.0.1:8000/docs
- **Dashboard**: http://localhost:8501

## 📊 Usage

### Single Customer Prediction

1. Go to the Dashboard
2. Navigate to "Single Prediction"
3. Fill in customer details
4. Click "Predict Churn"

### Batch Prediction

1. Prepare a CSV file with customer data
2. Go to "Batch Prediction" in the dashboard
3. Upload your CSV file
4. View results and download predictions

### API Usage

**Health Check:**
```bash
curl http://127.0.0.1:8000/health
```

**Single Prediction:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d @sample_customer.json
```

## 📋 Required CSV Format

Your CSV file should contain these columns:

| Column | Type | Example |
|--------|------|---------|
| Gender | string | Male/Female |
| SeniorCitizen | int | 0/1 |
| Partner | string | Yes/No |
| Dependents | string | Yes/No |
| tenure | int | 12 |
| PhoneService | string | Yes/No |
| MultipleLines | string | Yes/No/No phone service |
| InternetService | string | DSL/Fiber optic/No |
| OnlineSecurity | string | Yes/No/No internet service |
| OnlineBackup | string | Yes/No/No internet service |
| DeviceProtection | string | Yes/No/No internet service |
| TechSupport | string | Yes/No/No internet service |
| StreamingTV | string | Yes/No/No internet service |
| StreamingMovies | string | Yes/No/No internet service |
| Contract | string | Month-to-month/One year/Two year |
| PaperlessBilling | string | Yes/No |
| PaymentMethod | string | Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic) |
| MonthlyCharges | float | 29.85 |
| TotalCharges | float | 358.20 |

## 🔧 API Endpoints

- `GET /` - API health check
- `GET /health` - Detailed health status
- `POST /predict` - Single customer prediction
- `POST /predict/batch` - Batch prediction
- `GET /model/info` - Model information
- `GET /sample/customer` - Sample customer data

## 🎯 Model Performance

The system automatically compares multiple models and selects the best performer based on F1 score:

- **Logistic Regression**: With feature scaling
- **Random Forest**: Tree-based ensemble method

The selected model is saved and used for predictions.

## 🔄 Model Retraining

To retrain the model with new data:

1. Replace the data in `data/telco_customer_churn.csv`
2. Run `python model_training.py`
3. Restart the API server

## 🐛 Troubleshooting

### API Not Starting
- Check if port 8000 is available
- Ensure all dependencies are installed
- Check that model files exist in `models/` directory

### Dashboard Connection Error
- Ensure API is running on http://127.0.0.1:8000
- Check API health endpoint
- Verify no firewall blocking connections

### CSV Upload Issues
- Ensure all required columns are present
- Check for proper column names (case-sensitive)
- Verify data types match expected format

## 📦 Dependencies

```
pandas>=1.3.0
scikit-learn>=1.0.0
fastapi>=0.70.0
uvicorn>=0.15.0
streamlit>=1.15.0
requests>=2.25.0
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=5.0.0
```

## 🚀 Deployment

### Local Development
- Use the run scripts provided
- API runs on http://127.0.0.1:8000
- Dashboard runs on http://localhost:8501

### Production Deployment
For production deployment:

1. **Docker**: Create Docker containers for API and dashboard
2. **Cloud**: Deploy to AWS, GCP, or Azure
3. **Environment Variables**: Configure for different environments

## 📈 Next Steps

- [ ] Add more sophisticated models (XGBoost, Neural Networks)
- [ ] Implement A/B testing for model versions
- [ ] Add data drift monitoring
- [ ] Create automated retraining pipeline
- [ ] Add user authentication
- [ ] Implement caching for better performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Scikit-learn for machine learning capabilities
- FastAPI for the robust API framework
- Streamlit for the intuitive dashboard
- Plotly for beautiful visualizations