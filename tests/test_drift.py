import sqlite3

import pytest
from fastapi.testclient import TestClient

from src import api


@pytest.fixture
def isolated_monitoring(tmp_path, monkeypatch):
    """Use temporary databases instead of real monitoring databases."""

    monkeypatch.setattr(
        api, "PREDICTION_LOG_DB", tmp_path / "prediction_logs.db"
    )
    monkeypatch.setattr(
        api, "HTTP_METRICS_DB", tmp_path / "http_metrics.db"
    )

    api.initialise_prediction_log_db()
    api.initialise_drift_tables()
    api.initialise_http_metrics_db()

    return tmp_path


def test_feature_drift_alert(isolated_monitoring, monkeypatch):
    """A simulated distribution shift must trigger a drift warning."""

    with TestClient(api.app):
       pass
    
    features = list(api.model.feature_names_in_)
    train = api.pd.read_parquet(api.TRAIN_FEATURES_PATH)

    # Select a numeric feature with enough variation.
    feature = next(
        name
        for name in features
        if name != "country"
        and api.pd.to_numeric(
            train[name], errors="coerce"
        ).dropna().nunique() > 10
    )

    reference = api.pd.to_numeric(
        train[feature], errors="coerce"
    ).dropna()

    # Generate a strong distribution shift beyond the training range.
    shifted_value = float(reference.max()) + max(
        1000.0, abs(float(reference.max())) * 10
    )

    # Every simulated customer needs a complete feature snapshot.
    baseline = train.iloc[0][features].to_dict()

    with sqlite3.connect(api.PREDICTION_LOG_DB) as db:
        for customer_id in range(900001, 900031):
            snapshot = {}

            for name, value in baseline.items():
                if hasattr(value, "item"):
                    value = value.item()
                snapshot[name] = value

            snapshot[feature] = shifted_value

            db.execute(
                """
                INSERT INTO prediction_feature_snapshots
                (timestamp_utc, customer_id, features_json)
                VALUES (?, ?, ?)
                """,
                (
                    "2026-10-27T12:00:00+00:00",
                    customer_id,
                    api.json.dumps(snapshot, allow_nan=False),
                ),
            )

    result = api.model_monitoring_summary()

    assert result["live"]["distinct_customers_with_feature_snapshots"] == 30
    assert result["live"]["psi_status"] == "available"

    assert result["live"]["psi_by_feature"][feature] >= 0.25

    assert any(
        alert["type"] == "feature_drift"
        for alert in result["alerts"]
    )