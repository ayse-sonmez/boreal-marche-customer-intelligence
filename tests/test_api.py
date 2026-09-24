from fastapi.testclient import TestClient

from src.api import app


# ---------------------------------------------------------
# Test client
# ---------------------------------------------------------


def test_health():
    """
    Health endpoint should confirm that the API is running.
    """

    with TestClient(app) as client:

        response = client.get(
            "/health"
        )

        assert response.status_code == 200

        data = response.json()

        assert data == {
            "status": "ok"
        }


def test_ready():
    """
    Readiness endpoint should confirm that
    model artifacts are loaded.
    """

    with TestClient(app) as client:

        response = client.get(
            "/ready"
        )

        assert response.status_code == 200

        assert response.json() == {
            "status": "ready"
        }


def test_version_contract():
    """
    Version endpoint should expose the expected fields.
    """

    with TestClient(app) as client:

        response = client.get(
            "/version"
        )

        assert response.status_code == 200

        data = response.json()

        assert "api_version" in data
        assert "model_version" in data
        assert "git_commit" in data
        assert "python_version" in data


def test_prediction_contract():
    """
    Prediction endpoint should return
    the expected response structure.
    """

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json={
                "customer_id": 12346
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data[
            "customer_id"
        ] == 12346

        assert isinstance(
            data["return_probability"],
            float,
        )

        assert isinstance(
            data["threshold"],
            float,
        )

        assert isinstance(
            data["at_risk"],
            bool,
        )


def test_golden_prediction_regression():
    """
    Golden regression test.

    Customer 12346 should continue to produce
    the known prediction from model version 1.0.
    """

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json={
                "customer_id": 12346
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data[
            "customer_id"
        ] == 12346

        assert abs(
            data["return_probability"]
            - 0.3030308708729609
        ) < 1e-6

        assert abs(
            data["threshold"]
            - 0.73
        ) < 1e-6

        assert data[
            "at_risk"
        ] is True


def test_batch_prediction_contract():
    """
    Batch prediction should return one prediction
    for each requested customer.
    """

    with TestClient(app) as client:

        response = client.post(
            "/predict/batch",
            json={
                "customer_ids": [
                    12346,
                    12347,
                ]
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert "predictions" in data

        assert len(
            data["predictions"]
        ) == 2

        for prediction in data[
            "predictions"
        ]:

            assert "customer_id" in prediction
            assert "return_probability" in prediction
            assert "threshold" in prediction
            assert "at_risk" in prediction


def test_recommendation_contract():
    """
    Known basket should return the known
    association-rule recommendation.
    """

    with TestClient(app) as client:

        response = client.post(
            "/recommend",
            json={
                "basket": [
                    "22748"
                ],
                "top_k": 5,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data[
            "strategy"
        ] == "association_rules"

        assert len(
            data["recommendations"]
        ) >= 1

        first = data[
            "recommendations"
        ][0]

        assert first[
            "product"
        ] == "22745"

        assert first[
            "because_product"
        ] == "22748"

        assert "confidence" in first
        assert "lift" in first


def test_empty_recommendation_basket():
    """
    Empty basket should be handled without crashing.
    """

    with TestClient(app) as client:

        response = client.post(
            "/recommend",
            json={
                "basket": [],
                "top_k": 5,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data[
            "strategy"
        ] == "no_matching_rule"

        assert data[
            "recommendations"
        ] == []


def test_unknown_customer():
    """
    Unknown customer should return HTTP 404.
    """

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json={
                "customer_id": 999999999
            },
        )

        assert response.status_code == 404


def test_extra_field_is_rejected():
    """
    Pydantic extra='forbid' should reject
    unexpected request fields.
    """

    with TestClient(app) as client:

        response = client.post(
            "/predict",
            json={
                "customer_id": 12346,
                "unexpected_field": "test",
            },
        )

        assert response.status_code == 422