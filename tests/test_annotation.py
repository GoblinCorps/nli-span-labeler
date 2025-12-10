"""
Tests for annotation endpoints.
"""
import pytest
from fastapi.testclient import TestClient


class TestAnnotate:
    """Tests for /api/annotate endpoint."""

    def test_annotate_requires_auth(self, fresh_client: TestClient):
        """Test /api/annotate requires authentication."""
        response = fresh_client.post("/api/annotate", json={
            "example_id": 1,
            "labels": [],
            "complexity_scores": {}
        })
        assert response.status_code == 401

    def test_annotate_missing_example(self, auth_client: tuple[TestClient, dict]):
        """Test annotating nonexistent example fails."""
        client, user = auth_client
        response = client.post("/api/annotate", json={
            "example_id": 99999,
            "labels": [],
            "complexity_scores": {
                "reasoning": 50,
                "creativity": 50,
                "domain_knowledge": 50,
                "contextual": 50,
                "constraints": 50,
                "ambiguity": 50
            }
        })
        assert response.status_code == 404

    def test_annotate_success(self, auth_client: tuple[TestClient, dict], sample_example: dict):
        """Test successful annotation."""
        client, user = auth_client
        response = client.post("/api/annotate", json={
            "example_id": sample_example["id"],
            "labels": [
                {"token_index": 0, "label_name": "reasoning", "is_custom": False}
            ],
            "complexity_scores": {
                "reasoning": 75,
                "creativity": 50,
                "domain_knowledge": 25,
                "contextual": 50,
                "constraints": 50,
                "ambiguity": 50
            }
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["example_id"] == sample_example["id"]

    def test_annotate_with_custom_label(self, auth_client: tuple[TestClient, dict], sample_example: dict):
        """Test annotation with custom label."""
        client, user = auth_client
        response = client.post("/api/annotate", json={
            "example_id": sample_example["id"],
            "labels": [
                {"token_index": 0, "label_name": "my_custom_label", "is_custom": True, "color": "#ff0000"}
            ],
            "complexity_scores": {
                "reasoning": 50,
                "creativity": 50,
                "domain_knowledge": 50,
                "contextual": 50,
                "constraints": 50,
                "ambiguity": 50
            }
        })
        assert response.status_code == 200

    def test_annotate_invalid_complexity_score(self, auth_client: tuple[TestClient, dict], sample_example: dict):
        """Test annotation with out-of-range complexity score fails."""
        client, user = auth_client
        response = client.post("/api/annotate", json={
            "example_id": sample_example["id"],
            "labels": [],
            "complexity_scores": {
                "reasoning": 150,  # Out of range (should be 1-100)
                "creativity": 50,
                "domain_knowledge": 50,
                "contextual": 50,
                "constraints": 50,
                "ambiguity": 50
            }
        })
        assert response.status_code == 422


class TestSkip:
    """Tests for /api/skip endpoint."""

    def test_skip_requires_auth(self, fresh_client: TestClient):
        """Test /api/skip requires authentication."""
        response = fresh_client.post("/api/skip/1")
        assert response.status_code == 401

    def test_skip_success(self, auth_client: tuple[TestClient, dict], sample_example: dict):
        """Test successful skip."""
        client, user = auth_client
        # Need to lock first
        client.post(f"/api/lock/{sample_example['id']}")
        response = client.post(f"/api/skip/{sample_example['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["skipped"] is True

    def test_skip_nonexistent_example(self, auth_client: tuple[TestClient, dict]):
        """Test skipping nonexistent example fails."""
        client, user = auth_client
        response = client.post("/api/skip/99999")
        assert response.status_code == 404


class TestLabels:
    """Tests for /api/labels endpoint."""

    def test_get_labels_requires_auth(self, fresh_client: TestClient):
        """Test /api/labels requires authentication."""
        response = fresh_client.get("/api/labels")
        assert response.status_code == 401

    def test_get_labels_success(self, auth_client: tuple[TestClient, dict]):
        """Test getting label schema."""
        client, user = auth_client
        response = client.get("/api/labels")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert "difficulty" in data["categories"]
        assert "nli" in data["categories"]


class TestAgreement:
    """Tests for agreement endpoints."""

    def test_agreement_stats_requires_auth(self, fresh_client: TestClient):
        """Test /api/agreement/stats requires authentication."""
        response = fresh_client.get("/api/agreement/stats")
        assert response.status_code == 401

    def test_agreement_stats_success(self, auth_client: tuple[TestClient, dict]):
        """Test getting agreement statistics."""
        client, user = auth_client
        response = client.get("/api/agreement/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_examples_with_agreement" in data
        assert "pools" in data

    def test_pools_requires_auth(self, fresh_client: TestClient):
        """Test /api/pools requires authentication."""
        response = fresh_client.get("/api/pools")
        assert response.status_code == 401

    def test_pools_success(self, auth_client: tuple[TestClient, dict]):
        """Test getting pool status."""
        client, user = auth_client
        response = client.get("/api/pools")
        assert response.status_code == 200
        data = response.json()
        assert "zero_entry" in data
        assert "building" in data
        assert "test" in data
