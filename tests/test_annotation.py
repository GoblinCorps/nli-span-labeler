"""
Tests for annotation endpoints.
"""
import pytest
from fastapi.testclient import TestClient


class TestAnnotate:
    """Tests for /api/label endpoint (annotation submission)."""

    def test_annotate_requires_auth(self, fresh_client: TestClient):
        """Test /api/label requires authentication."""
        response = fresh_client.post("/api/label", json={
            "example_id": "1",
            "labels": [],
            "complexity_scores": {}
        })
        assert response.status_code == 401

    def test_annotate_missing_example(self, auth_client: tuple[TestClient, dict]):
        """Test annotating nonexistent example fails."""
        client, user = auth_client
        response = client.post("/api/label", json={
            "example_id": "nonexistent_99999",
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

    def test_annotate_success(self, auth_client_with_example: tuple[TestClient, dict, dict]):
        """Test successful annotation."""
        client, user, example = auth_client_with_example
        response = client.post("/api/label", json={
            "example_id": example["id"],
            "labels": [
                {
                    "label_name": "reasoning",
                    "label_color": "#3b82f6",
                    "spans": [
                        {"source": "premise", "word_index": 0, "word_text": "The", "char_start": 0, "char_end": 3}
                    ]
                }
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
        assert data["status"] == "saved"
        assert data["example_id"] == example["id"]

    def test_annotate_with_custom_label(self, auth_client_with_example: tuple[TestClient, dict, dict]):
        """Test annotation with custom label."""
        client, user, example = auth_client_with_example
        response = client.post("/api/label", json={
            "example_id": example["id"],
            "labels": [
                {
                    "label_name": "my_custom_label",
                    "label_color": "#ff0000",
                    "spans": [
                        {"source": "premise", "word_index": 0, "word_text": "The", "char_start": 0, "char_end": 3}
                    ]
                }
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

    def test_annotate_extreme_complexity_score(self, auth_client_with_example: tuple[TestClient, dict, dict]):
        """Test annotation with extreme complexity scores.

        Note: The API currently accepts any numeric value for complexity scores
        (validation is documented but not strictly enforced). This test verifies
        the API handles extreme values gracefully.
        """
        client, user, example = auth_client_with_example
        response = client.post("/api/label", json={
            "example_id": example["id"],
            "labels": [],
            "complexity_scores": {
                "reasoning": 150,  # Above documented range (1-100)
                "creativity": 50,
                "domain_knowledge": 50,
                "contextual": 50,
                "constraints": 50,
                "ambiguity": 50
            }
        })
        # API accepts the values (validation not enforced at API level)
        assert response.status_code == 200


class TestSkip:
    """Tests for /api/skip endpoint."""

    def test_skip_requires_auth(self, fresh_client: TestClient):
        """Test /api/skip requires authentication."""
        response = fresh_client.post("/api/skip/1")
        assert response.status_code == 401

    def test_skip_success(self, auth_client_with_example: tuple[TestClient, dict, dict]):
        """Test successful skip."""
        client, user, example = auth_client_with_example
        # Get next to establish context (acquires lock)
        client.get("/api/next")
        response = client.post(f"/api/skip/{example['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "skipped"

    def test_skip_nonexistent_example(self, auth_client: tuple[TestClient, dict]):
        """Test skipping nonexistent example fails."""
        client, user = auth_client
        response = client.post("/api/skip/nonexistent_99999")
        assert response.status_code == 404


class TestLabels:
    """Tests for /api/labels endpoint."""

    def test_get_labels_public(self, fresh_client: TestClient):
        """Test /api/labels is publicly accessible (returns label schema)."""
        response = fresh_client.get("/api/labels")
        # Label schema is public - doesn't require auth
        assert response.status_code == 200

    def test_get_labels_success(self, auth_client: tuple[TestClient, dict]):
        """Test getting label schema."""
        client, user = auth_client
        response = client.get("/api/labels")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert "difficulty" in data["categories"]
        assert "nli" in data["categories"]


class TestPools:
    """Tests for /api/pools/stats endpoint."""

    def test_pools_requires_auth(self, fresh_client: TestClient):
        """Test /api/pools/stats requires authentication."""
        response = fresh_client.get("/api/pools/stats")
        assert response.status_code == 401

    def test_pools_success(self, auth_client: tuple[TestClient, dict]):
        """Test getting pool status."""
        client, user = auth_client
        response = client.get("/api/pools/stats")
        assert response.status_code == 200
        data = response.json()
        assert "zero_entry" in data
        assert "building" in data
        assert "test" in data
