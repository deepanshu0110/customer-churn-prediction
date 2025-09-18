#!/usr/bin/env python3
"""
Setup script to initialize the customer churn prediction system
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def create_directories():
    """Create required directories"""
    directories = ['data', 'models', 'notebooks', 'api', 'app']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
    return True

def install_requirements():
    """Install Python requirements"""
    requirements = [
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "streamlit>=1.15.0",
        "requests>=2.25.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "pydantic>=1.8.0"
    ]
    
    print("📦 Installing requirements...")
    for req in requirements:
        if not run_command(f"pip install {req}", f"Installing {req.split('>=')[0]}"):
            return False
    
    # Save requirements to file
    run_command("pip freeze > requirements.txt", "Saving requirements")
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Customer Churn Prediction System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please install manually.")
        sys.exit(1)
    
    # Run data preparation
    if not run_command("python data_preparation.py", "Preparing data"):
        print("❌ Data preparation failed!")
        sys.exit(1)
    
    # Run model training
    if not run_command("python model_training.py", "Training models"):
        print("❌ Model training failed!")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the API server: python run_api.py")
    print("2. In another terminal, start the dashboard: python run_dashboard.py")
    print("3. Access the dashboard at: http://localhost:8501")
    print("4. Access the API docs at: http://127.0.0.1:8000/docs")
    print("\n💡 For more information, see README.md")

if __name__ == "__main__":
    main()