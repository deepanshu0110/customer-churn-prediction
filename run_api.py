#!/usr/bin/env python3
"""
Script to run the FastAPI server
"""

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Customer Churn Prediction API...")
    print("📍 API will be available at: http://127.0.0.1:8000")
    print("📖 API docs will be available at: http://127.0.0.1:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",   # change to "0.0.0.0" for external access
        port=8000,
        reload=True,
        log_level="info"
    )
