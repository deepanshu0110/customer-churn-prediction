import json
import sys

from tests.helpers import ROOT, run_script


def test_training_selects_model_and_threshold():
    run_script("data_preparation.py")
    output = run_script("model_training.py")
    assert "Best model:" in output
    assert (ROOT / "models" / "best_model.pkl").exists()
    assert (ROOT / "models" / "best_threshold.pkl").exists()


def test_api_predicts_for_sample_customer():
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(ROOT / "api"))
    import main

    sample = json.loads((ROOT / "sample_customers.json").read_text())
    customer = sample[0] if isinstance(sample, list) else sample
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json=customer)
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["prediction"] in (0, 1)
