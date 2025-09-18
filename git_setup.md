# 🔧 Git Setup and Deployment Guide

## Initial Git Setup

### 1. Initialize Git Repository
```bash
# Navigate to your project folder
cd customer-churn-prediction

# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Customer Churn Prediction System"
```

### 2. Connect to GitHub

**Option A: Create Repository on GitHub First**
1. Go to https://github.com
2. Click "New" repository
3. Name it: `customer-churn-prediction`
4. Don't initialize with README (we already have one)
5. Copy the repository URL

```bash
# Add remote origin
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-prediction.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Option B: Create Repository via GitHub CLI**
```bash
# Install GitHub CLI first (if not installed)
# Windows: winget install --id GitHub.cli
# Mac: brew install gh
# Linux: See https://github.com/cli/cli/blob/trunk/docs/install_linux.md

# Login to GitHub
gh auth login

# Create repository
gh repo create customer-churn-prediction --public --source=. --remote=origin --push
```

## Deployment Options

### 1. Local Development (Current Setup)
```bash
# Terminal 1: Start API
python run_api.py

# Terminal 2: Start Dashboard  
python run_dashboard.py
```

### 2. Heroku Deployment

**Prepare for Heroku:**

Create `Procfile`:
```
web: uvicorn api.main:app --host=0.0.0.0 --port=${PORT:-8000}
dashboard: streamlit run app/dashboard.py --server.port=${PORT:-8501} --server.address=0.0.0.0
```

Create `runtime.txt`:
```
python-3.9.18
```

**Deploy to Heroku:**
```bash
# Install Heroku CLI
# Windows: Download from https://devcenter.heroku.com/articles/heroku-cli
# Mac: brew install heroku/brew/heroku
# Linux: curl https://cli-assets.heroku.com/install.sh | sh

# Login to Heroku
heroku login

# Create Heroku apps (API and Dashboard separately)
heroku create your-churn-api
heroku create your-churn-dashboard

# Deploy API
git subtree push --prefix=api heroku main

# Deploy Dashboard (requires modification for API URL)
git subtree push --prefix=app heroku main
```

### 3. Docker Deployment

**Create Dockerfile for API:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Create Dockerfile for Dashboard:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py", "--server.address", "0.0.0.0"]
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  api:
    build: .
    dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
  
  dashboard:
    build: .
    dockerfile: Dockerfile.dashboard
    ports:
      - "8501:8501"
    depends_on:
      - api
    environment:
      - API_URL=http://api:8000
```

### 4. Cloud Deployment (AWS/GCP/Azure)

**AWS EC2 Example:**
```bash
# Connect to EC2 instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Clone repository
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# Setup environment
python3 -m venv churn_env
source churn_env/bin/activate
pip install -r requirements.txt

# Run setup
python setup.py

# Start services with PM2 (process manager)
npm install -g pm2
pm2 start "python run_api.py" --name churn-api
pm2 start "python run_dashboard.py" --name churn-dashboard
pm2 startup
pm2 save
```

## Environment Configuration

### Development Environment
Create `.env` file:
```
API_HOST=127.0.0.1
API_PORT=8000
DASHBOARD_HOST=localhost
DASHBOARD_PORT=8501
MODEL_PATH=./models/
DATA_PATH=./data/
```

### Production Environment
```
API_HOST=0.0.0.0
API_PORT=8000
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8501
MODEL_PATH=/app/models/
DATA_PATH=/app/data/
```

## Continuous Deployment

### GitHub Actions Workflow
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/
    
    - name: Deploy to Heroku
      uses: akhileshns/heroku-deploy@v3.12.12
      with:
        heroku_api_key: ${{secrets.HEROKU_API_KEY}}
        heroku_app_name: "your-churn-prediction"
        heroku_email: "your-email@example.com"
```

## Monitoring and Maintenance

### Log Files
```bash
# API logs
tail -f logs/api.log

# Dashboard logs  
tail -f logs/dashboard.log

# System logs
journalctl -u churn-prediction
```

### Health Checks
```bash
# API health
curl http://your-domain:8000/health

# Dashboard health
curl http://your-domain:8501
```

### Model Updates
```bash
# Update model with new data
python model_training.py

# Restart services
pm2 restart churn-api
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in configuration
2. **Permission errors**: Check file permissions
3. **Memory issues**: Increase server memory or optimize model
4. **API connection**: Verify firewall and network settings

### Debug Commands
```bash
# Check process status
pm2 status

# View logs
pm2 logs churn-api
pm2 logs churn-dashboard

# Restart services
pm2 restart all

# Check system resources
htop
df -h
```