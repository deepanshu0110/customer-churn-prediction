import streamlit as st
import pandas as pd
import os
import requests
import plotly.express as px
import plotly.graph_objects as go
import base64

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "https://customer-churn-prediction-78oq.onrender.com")

)


# --------------------------
# Styles
# --------------------------
st.markdown(
    """
<style>
.main-header { font-size: 2.4rem; text-align: center; margin-bottom: 1rem; }
.metric-card {
  background-color: #f0f2f6; padding: 1rem; border-radius: 10px;
  border-left: 5px solid #1f77b4;
}
.churn-yes { color: #ff4b4b; font-weight: 700; }
.churn-no  { color: #00cc88; font-weight: 700; }
.stButton > button { width: 100%; background:#1f77b4; color:#fff; border:0; border-radius:8px; padding:.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------
# API helpers
# --------------------------
def api_ok() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def get_model_info():
    r = requests.get(f"{API_URL}/model_info")
    r.raise_for_status()
    return r.json()

def get_sample():
    r = requests.get(f"{API_URL}/sample")
    r.raise_for_status()
    return r.json()["sample"]

def predict_single(row_dict):
    # FastAPI single expects {"__root__": {...}}
    r = requests.post(f"{API_URL}/predict", json={"__root__": row_dict})
    r.raise_for_status()
    return r.json()

def predict_batch(records_list):
    # FastAPI batch expects a bare list of dicts
    r = requests.post(f"{API_URL}/predict_batch", json=records_list)
    r.raise_for_status()
    return r.json()

# --------------------------
# Client-side utils
# --------------------------
def bucket_confidence(prob: float, threshold: float) -> str:
    # distance from decision boundary
    d = abs(prob - threshold)
    if d < 0.05:
        return "Low"
    elif d < 0.15:
        return "Medium"
    return "High"

def add_client_fields(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # prediction is 0/1 from API → map to Yes/No
    out = df.copy()
    if "prediction" in out.columns:
        out["Churn Prediction"] = out["prediction"].map({1: "Yes", 0: "No"})
    if "churn_probability" in out.columns:
        out["Churn Probability"] = out["churn_probability"].astype(float)
        out["Confidence"] = out["Churn Probability"].apply(lambda p: bucket_confidence(p, threshold))
    return out

def batch_summary(df: pd.DataFrame):
    total = len(df)
    churn = int(df["prediction"].sum()) if "prediction" in df.columns else 0
    churn_rate = (churn / total) if total else 0.0
    return {
        "total_customers": total,
        "predicted_churn": churn,
        "churn_rate": churn_rate,
        "retention_rate": 1 - churn_rate,
    }

def download_link(df: pd.DataFrame, filename="churn_predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(
        f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Download Results CSV</a>',
        unsafe_allow_html=True,
    )

# --------------------------
# App
# --------------------------
def main():
    st.markdown('<div class="main-header">📊 Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)

    if not api_ok():
        st.error("❌ API Connection failed. Make sure FastAPI is running at http://127.0.0.1:8000")
        st.info("Start it with:  uvicorn api.main:app --reload --port 8000")
        return
    else:
        st.success("✅ API Connection: Healthy")

    # Cache model info (threshold, features)
    if "model_info" not in st.session_state:
        try:
            st.session_state.model_info = get_model_info()
        except Exception as e:
            st.error(f"Failed to fetch model info: {e}")
            return

    info = st.session_state.model_info
    threshold = float(info.get("best_threshold", 0.5))

    # Sidebar nav
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "👤 Single Prediction", "📄 Batch Prediction", "ℹ️ Model Info"],
    )

    # ---------------------- Home
    if page == "🏠 Home":
        st.write("Use the side menu for single or batch predictions. Below is a sample input format.")
        try:
            sample = get_sample()
            st.dataframe(pd.DataFrame([sample]), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load sample: {e}")

    # ---------------------- Single
    elif page == "👤 Single Prediction":
        st.header("Single Customer Churn Prediction")
        # To keep concise, reuse the sample row for a quick try
        try:
            sample = get_sample()
        except Exception as e:
            st.error(f"Could not load sample: {e}")
            return

        st.caption("Using the API-provided sample row. (You can wire a full form later.)")
        st.dataframe(pd.DataFrame([sample]))

        if st.button("🔮 Predict Churn", key="predict_one"):
            try:
                res = predict_single(sample)
            except Exception as e:
                st.error(f"API Error: {e}")
                return

            prob = float(res["churn_probability"])
            pred = "Yes" if int(res["prediction"]) == 1 else "No"
            conf = bucket_confidence(prob, threshold)

            c1, c2, c3 = st.columns(3)
            with c1:
                css = "churn-yes" if pred == "Yes" else "churn-no"
                st.markdown(f'<div class="metric-card"><h4>Prediction</h4><p class="{css}">{pred}</p></div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><h4>Probability</h4><p>{prob:.1%}</p></div>',
                            unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><h4>Confidence</h4><p>{conf}</p></div>',
                            unsafe_allow_html=True)

            # Gauge with threshold marker
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Churn Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, threshold*100], 'color': "lightgray"},
                        {'range': [threshold*100, 100], 'color': "tomato"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.8,
                        'value': threshold * 100,
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------- Batch
    elif page == "📄 Batch Prediction":
        st.header("Batch Customer Churn Prediction")
        up = st.file_uploader("Upload CSV (columns must match the model features)", type="csv")
        if up is not None:
            try:
                df_in = pd.read_csv(up)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                return

            st.subheader("Data Preview")
            st.dataframe(df_in.head(), use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Rows", len(df_in))
            # You can compute domain-specific quick stats if you know columns
            if "tenure" in df_in.columns:
                with c2:
                    st.metric("Average Tenure", f"{df_in['tenure'].mean():.1f} months")
            if "MonthlyCharges" in df_in.columns:
                with c3:
                    st.metric("Avg Monthly Charges", f"${df_in['MonthlyCharges'].mean():.2f}")

            if st.button("🚀 Predict Churn for All Customers", key="predict_batch"):
                try:
                    raw_results = predict_batch(df_in.to_dict("records"))
                except Exception as e:
                    st.error(f"API Error: {e}")
                    return

                results_df = pd.DataFrame(raw_results)
                results_df = add_client_fields(results_df, threshold)
                summ = batch_summary(results_df)

                st.success("✅ Batch prediction completed!")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Customers", summ["total_customers"])
                with c2:
                    st.metric("Predicted Churn", summ["predicted_churn"])
                with c3:
                    st.metric("Churn Rate", f"{summ['churn_rate']:.1%}")
                with c4:
                    st.metric("Retention Rate", f"{summ['retention_rate']:.1%}")

                st.subheader("📊 Results Visualization")
                # Pie (Yes/No)
                pie_counts = results_df["Churn Prediction"].value_counts()
                fig_pie = px.pie(
                    values=pie_counts.values,
                    names=pie_counts.index,
                    title="Churn Prediction Distribution",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Confidence bar
                conf_counts = results_df["Confidence"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
                fig_bar = px.bar(
                    x=conf_counts.index,
                    y=conf_counts.values,
                    labels={"x": "Confidence", "y": "Count"},
                    title="Confidence Level Distribution",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Probability histogram
                fig_hist = px.histogram(
                    results_df,
                    x="Churn Probability",
                    nbins=20,
                    title="Distribution of Churn Probabilities",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                st.subheader("📋 Detailed Results")
                # Optional filters
                f1, f2 = st.columns(2)
                with f1:
                    filt_churn = st.selectbox("Filter by Churn", ["All", "Yes", "No"])
                with f2:
                    filt_conf = st.selectbox("Filter by Confidence", ["All", "High", "Medium", "Low"])

                table_df = results_df.copy()
                if filt_churn != "All":
                    table_df = table_df[table_df["Churn Prediction"] == filt_churn]
                if filt_conf != "All":
                    table_df = table_df[table_df["Confidence"] == filt_conf]

                st.dataframe(table_df, use_container_width=True)

                st.subheader("💾 Download")
                download_link(results_df)

    # ---------------------- Model Info
    elif page == "ℹ️ Model Info":
        st.header("Model Information")
        info = st.session_state.model_info
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Details")
            st.write(f"**Model Type:** {info.get('model_name')}")
            st.write(f"**Decision Threshold:** {info.get('best_threshold')}")
            st.write(f"**Number of Features:** {info.get('n_features')}")
            st.write(f"**Uses Scaler:** {'Yes' if info.get('uses_scaler') else 'No'}")
        with c2:
            st.subheader("First 10 Features")
            feats = info.get("features", [])[:10]
            for i, f in enumerate(feats, 1):
                st.write(f"{i}. {f}")

        st.caption("Endpoints used: /health, /sample, /predict, /predict_batch, /model_info")

if __name__ == "__main__":
    main()
