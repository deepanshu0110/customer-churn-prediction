# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional

# ---------- Pydantic: v2 RootModel with v1 fallback ----------
try:
    from pydantic import RootModel  # Pydantic v2
    class CustomerData(RootModel[Dict[str, Any]]):
        """Single-record wrapper (v2)."""
        pass
    def _extract_record(payload: "CustomerData") -> Dict[str, Any]:
        return payload.root
except Exception:  # pragma: no cover
    from pydantic import BaseModel  # Pydantic v1
    class CustomerData(BaseModel):
        """Single-record wrapper (v1)."""
        __root__: Dict[str, Any]
    def _extract_record(payload: "CustomerData") -> Dict[str, Any]:
        return payload.__root__

# ---------- App ----------
app = FastAPI(title="Customer Churn Prediction API", version="0.1.0", lifespan=lifespan)

# CORS (open for public access — you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all origins (works with Streamlit Cloud)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Globals / paths ----------
model_info: Optional[dict] = None      # {"model": sklearn_estimator, "scaler": Optional[scaler], "name": str}
best_threshold: Optional[float] = None
feature_names: Optional[list] = None

MODELS_DIR = "models"
MODEL_FILE = os.path.join(MODELS_DIR, "best_model.pkl")
THRESH_FILE = os.path.join(MODELS_DIR, "best_threshold.pkl")
FEATURES_FILE = os.path.join(MODELS_DIR, "feature_names.pkl")

# ---------- Helpers ----------
def _load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model_artifacts():
    """Load model, threshold, and feature names from disk."""
    global model_info, best_threshold, feature_names
    model_info = _load_pickle(MODEL_FILE)
    best_threshold = float(_load_pickle(THRESH_FILE))
    feature_names = list(_load_pickle(FEATURES_FILE))

    if not isinstance(model_info, dict) or "model" not in model_info:
        raise ValueError("best_model.pkl is not in the expected format.")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("feature_names.pkl is empty or invalid.")

def ensure_loaded():
    if model_info is None or best_threshold is None or feature_names is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")

def to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a DataFrame aligned to expected feature_names.
    - Adds any missing columns as 0.
    - Drops extra columns.
    - Reorders to feature_names.
    - Coerces non-numeric to numeric (NaN -> 0).
    """
    if not isinstance(records, list) or len(records) == 0:
        raise HTTPException(status_code=400, detail="Empty payload.")
    df = pd.DataFrame(records)

    # Add missing expected columns as zeros
    missing = [c for c in feature_names if c not in df.columns]
    for c in missing:
        df[c] = 0

    # Keep only expected columns (drop extras) and order them
    df = df[feature_names]

    # Coerce to numeric
    for c in df.columns:
        if df[c].dtype.kind not in "biufc":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df

def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """
    Apply scaler (if any) and return churn probabilities.
    """
    mdl = model_info["model"]
    scaler = model_info.get("scaler", None)

    X = scaler.transform(df) if scaler is not None else df.values

    if hasattr(mdl, "predict_proba"):
        return mdl.predict_proba(X)[:, 1]
    if hasattr(mdl, "decision_function"):
        from scipy.special import expit
        return expit(mdl.decision_function(X))
    # Fallback (not ideal): use predictions as pseudo-probabilities
    return mdl.predict(X).astype(float)

# ---------- Startup (modern lifespan pattern) ----------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model_artifacts()
        print("✅ Model artifacts loaded.")
    except Exception as e:
        print(f"⚠️ Failed to load model artifacts: {e}")
    yield

# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "API is running", "docs": "/docs"}

@app.get("/health")
def health():
    ok = (model_info is not None) and (best_threshold is not None) and (feature_names is not None)
    detail = "ok" if ok else "model artifacts not loaded"
    return {"status": "healthy" if ok else "degraded", "detail": detail}

@app.get("/sample")
def sample():
    """Return a skeleton row with all features = 0."""
    ensure_loaded()
    return {"features": feature_names, "sample": {k: 0 for k in feature_names}}

@app.get("/model_info")
def get_model_info():
    """Model metadata for the dashboard."""
    ensure_loaded()
    name = model_info.get("name", "Unknown")
    has_scaler = model_info.get("scaler", None) is not None
    return {
        "model_name": name,
        "uses_scaler": has_scaler,
        "best_threshold": best_threshold,
        "n_features": len(feature_names),
        "features": feature_names,
    }

@app.post("/predict")
def predict_single(payload: CustomerData):
    """Predict for one record (dict of prepared numeric features)."""
    ensure_loaded()
    record = _extract_record(payload)
    df = to_dataframe([record])
    proba = float(predict_proba(df)[0])
    pred = int(proba >= best_threshold)
    return {"churn_probability": proba, "prediction": pred}


@app.post("/explain")
def explain_prediction(payload: CustomerData):
    """
    SHAP-based explanation for a single prediction.
    Returns top feature contributions — shows WHY the model decided churn or no-churn.
    Explainable AI is now expected by clients and regulators alike.
    """
    ensure_loaded()
    try:
        import shap
    except ImportError:
        raise HTTPException(status_code=501,
                            detail="shap not installed. Run: pip install shap")

    record = _extract_record(payload)
    df     = to_dataframe([record])
    mdl    = model_info["model"]
    scaler = model_info.get("scaler")
    X      = scaler.transform(df) if scaler else df.values

    # TreeExplainer for tree-based models, LinearExplainer as fallback
    if hasattr(mdl, "estimators_") or hasattr(mdl, "tree_"):
        explainer   = shap.TreeExplainer(mdl)
        shap_values = explainer.shap_values(X)
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    else:
        explainer   = shap.LinearExplainer(mdl, X)
        shap_values = explainer.shap_values(X)
        sv = shap_values[0]

    contributions = sorted(
        [{"feature": f, "shap_value": float(v)}
         for f, v in zip(feature_names, sv)],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )

    proba = float(predict_proba(df)[0])
    return {
        "churn_probability": proba,
        "prediction":        int(proba >= best_threshold),
        "top_factors":       contributions[:10],
        "note": "Positive SHAP = pushes toward churn. Negative = pushes away.",
    }

@app.post("/predict_batch")
def predict_batch(records: List[Dict[str, Any]]):
    """Predict for a list of records (each is a dict of prepared numeric features)."""
    ensure_loaded()
    df = to_dataframe(records)
    proba = predict_proba(df)
    preds = (proba >= best_threshold).astype(int)

    out = df.copy()
    out["churn_probability"] = proba
    out["prediction"] = preds

    # Convert numpy types to built-ins for JSON
    result = []
    for row in out.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            else:
                clean[k] = v
        result.append(clean)

    return result
