import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Use non-GUI backend for matplotlib (prevents hanging in VS Code)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: force UTF-8 output (so emojis won't break on Windows)
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_data():
    """
    Load the prepared training and test data
    """
    print("Loading prepared data...")

    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv')['Churn']
    y_test = pd.read_csv('data/y_test.csv')['Churn']

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """
    Train both Logistic Regression and Random Forest models
    """
    print("Training models...")

    models = {}

    # 1. Logistic Regression (with scaling)
    print("Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    t0 = time.time()
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver="liblinear",  # faster & more stable for small/medium datasets
        verbose=1
    )
    lr_model.fit(X_train_scaled, y_train)
    print(f"✅ Logistic Regression training time: {time.time() - t0:.2f} seconds")

    models['logistic_regression'] = {
        'model': lr_model,
        'scaler': scaler,
        'name': 'Logistic Regression'
    }

    # 2. Random Forest (no scaling needed)
    print("\nTraining Random Forest...")
    t0 = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,  # use all CPU cores
        verbose=1
    )
    rf_model.fit(X_train, y_train)
    print(f"✅ Random Forest training time: {time.time() - t0:.2f} seconds")

    models['random_forest'] = {
        'model': rf_model,
        'scaler': None,
        'name': 'Random Forest'
    }

    return models


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Evaluate all models and select the best one
    """
    print("\nEvaluating models...")

    results = {}

    for model_name, model_info in models.items():
        model = model_info['model']
        scaler = model_info['scaler']
        name = model_info['name']

        print(f"\n=== {name} ===")

        # Prepare test data
        if scaler:
            X_test_processed = scaler.transform(X_test)
            X_train_processed = scaler.transform(X_train)
        else:
            X_test_processed = X_test
            X_train_processed = X_train

        # Make predictions
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"F1 Score: {f1:.4f}")
        print(f"AUC Score: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        results[model_name] = {
            'f1_score': f1,
            'auc_score': auc,
            'model_info': model_info
        }

    # Select best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model_info = results[best_model_name]

    print(f"\nBest model: {best_model_info['model_info']['name']}")
    print(f"F1 Score: {best_model_info['f1_score']:.4f}")

    return best_model_info['model_info'], results


def find_best_threshold(model_info, X_test, y_test):
    """
    Find the best probability threshold for predictions
    """
    print("\nFinding best threshold...")

    model = model_info['model']
    scaler = model_info['scaler']

    # Prepare test data
    if scaler:
        X_test_processed = scaler.transform(X_test)
    else:
        X_test_processed = X_test

    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_results = {}

    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_thresh)
        threshold_results[threshold] = f1
        print(f"Threshold {threshold}: F1 Score = {f1:.4f}")

    # Find best threshold
    best_threshold = max(threshold_results, key=threshold_results.get)
    best_f1 = threshold_results[best_threshold]

    print(f"\nBest threshold: {best_threshold} (F1 Score: {best_f1:.4f})")

    return best_threshold


def plot_feature_importance(model_info, feature_names):
    """
    Plot feature importance for the model (if available)
    """
    model = model_info['model']

    if hasattr(model, 'feature_importances_'):
        print("Creating feature importance plot...")

        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10 features

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.title('Top 10 Feature Importances')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()

        # Save plot only
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Feature importance plot saved to models/feature_importance.png")


def save_model(model_info, best_threshold, feature_names):
    """
    Save the trained model and related information
    """
    print("Saving model...")

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Save model
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(model_info, f)

    # Save threshold
    with open('models/best_threshold.pkl', 'wb') as f:
        pickle.dump(best_threshold, f)

    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    print("Model saved to models/best_model.pkl")
    print(f"Best threshold ({best_threshold}) saved to models/best_threshold.pkl")
    print("Feature names saved to models/feature_names.pkl")


def main():
    """
    Main function for model training
    """
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models and select best one
    best_model_info, all_results = evaluate_models(models, X_train, X_test, y_train, y_test)

    # Find best threshold
    best_threshold = find_best_threshold(best_model_info, X_test, y_test)

    # Plot feature importance
    feature_names = list(X_train.columns)
    plot_feature_importance(best_model_info, feature_names)

    # Save model and related files
    save_model(best_model_info, best_threshold, feature_names)

    print("\nModel training completed successfully!")
    print(f"Best model: {best_model_info['name']}")
    print(f"Best threshold: {best_threshold}")

    # Summary
    print("\nSummary:")
    print("- Trained and compared Logistic Regression and Random Forest")
    print("- Selected best model based on F1 score")
    print("- Found optimal prediction threshold")
    print("- Saved model files to models/ directory")
    print("- Ready for API integration!")


if __name__ == "__main__":
    main()
