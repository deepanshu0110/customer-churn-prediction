import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def download_and_prepare_data():
    """
    Download and prepare the Telco Customer Churn dataset
    """
    # For this example, we'll create a sample dataset similar to Telco
    # In real scenario, you'd download from: 
    # https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    
    print("Creating sample customer churn dataset...")
    
    # Create sample data (similar to Telco dataset)
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data
    data = {
        'CustomerID': [f'CUST_{i:04d}' for i in range(1, n_samples + 1)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'tenure': np.random.randint(1, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.random.uniform(18.0, 120.0, n_samples),
        'TotalCharges': np.random.uniform(18.0, 8500.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create churn labels based on some logic (more realistic)
    churn_prob = 0.1  # Base probability
    
    # Increase churn probability based on features
    churn_factors = (
        (df['Contract'] == 'Month-to-month') * 0.4 +
        (df['tenure'] < 12) * 0.3 +
        (df['MonthlyCharges'] > 80) * 0.2 +
        (df['SeniorCitizen'] == 1) * 0.1 +
        (df['PaymentMethod'] == 'Electronic check') * 0.2
    )
    
    final_churn_prob = np.clip(churn_prob + churn_factors, 0, 0.8)
    df['Churn'] = np.random.binomial(1, final_churn_prob).astype(str)
    df['Churn'] = df['Churn'].map({'1': 'Yes', '0': 'No'})
    
    return df

def clean_and_encode_data(df):
    """
    Clean and encode the dataset
    """
    print("Cleaning and encoding data...")
    
    # Handle missing values (convert TotalCharges to numeric)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Create a copy for encoding
    df_encoded = df.copy()
    
    # Remove CustomerID as it's not useful for prediction
    df_encoded = df_encoded.drop('CustomerID', axis=1)
    
    # Encode binary columns (Yes/No)
    binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_columns:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
    
    # Encode categorical columns with more than 2 categories
    categorical_columns = ['Gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                          'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                          'StreamingMovies', 'Contract', 'PaymentMethod']
    
    le_dict = {}
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            le_dict[col] = le
    
    return df_encoded, le_dict

def prepare_features_and_target(df_encoded):
    """
    Split features and target variable
    """
    print("Splitting features and target...")
    
    # Separate features and target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Churn rate in training: {y_train.mean():.2%}")
    
    return X_train, X_test, y_train, y_test

def main():
    """
    Main function to prepare data
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download and prepare data
    df = download_and_prepare_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    
    # Save raw data
    df.to_csv('data/telco_customer_churn.csv', index=False)
    print("Raw data saved to data/telco_customer_churn.csv")
    
    # Clean and encode data
    df_encoded, le_dict = clean_and_encode_data(df)
    
    # Prepare features and target
    X_train, X_test, y_train, y_test = prepare_features_and_target(df_encoded)
    
    # Save processed data
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    # Save label encoders (for later use in API)
    import pickle
    with open('data/label_encoders.pkl', 'wb') as f:
        pickle.dump(le_dict, f)
    
    print("Data preparation completed!")
    print("\nDataset info:")
    print(df_encoded.info())
    print("\nFirst few rows:")
    print(df_encoded.head())

if __name__ == "__main__":
    main()