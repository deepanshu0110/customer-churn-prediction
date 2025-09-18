#!/usr/bin/env python3
"""
Script to run the Streamlit dashboard (no private Streamlit imports)
"""
import sys
import os
import runpy

if __name__ == "__main__":
    print("🎨 Starting Customer Churn Prediction Dashboard...")
    print("📍 Dashboard will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)

    dashboard_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "app", "dashboard.py")
    )

    # Build argv exactly like a CLI call: python -m streamlit run ...
    sys.argv = [
        "streamlit",
        "run",
        dashboard_path,
        "--server.port=8501",
        "--server.address=localhost",  # use 0.0.0.0 to expose on LAN
    ]

    # Runs the "streamlit" module as if from the command line
    runpy.run_module("streamlit", run_name="__main__")
