"""
Tests for example-related endpoints.
"""
import pytest
from fastapi.testclient import TestClient


class TestGetNext:
    """Tests for /api/next endpoint."""

    def test_next_requires_auth(self, fresh_client: TestClient):
        """Test /api/next requires authentication."""
        response = fresh_client.get("/api/next")
        assert response.status_code == 401

    def test_next_empty_database(self, auth_client: tuple[TestClient, dict]):
        """Test /api/next returns 404 when no examples exist."""
        client, user = auth_client
        response = client.get("/api/next")
        # No examples loaded yet
        assert response.status_code == 404

    def test_next_with_examples(self, auth_client: tuple[TestClient, dict], sample_example: dict):
        """Test /api/next returns an example when available."""
        client, user = auth_client
        response = client.get("/api/next")
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "premise" in data
        assert "hypothesis" in data
        assert "tokens" in data

    def test_next_with_dataset_filter(self, auth_client: tuple[TestClient, dict], sample_example: dict):
        """Test /api/next respects dataset filter."""
        client, user = auth_client
        # Request non-existent dataset
        response = client.get("/api/next", params={"dataset": "nonexistent"})
        assert response.status_code == 404

        # Request correct dataset
        response = client.get("/api/next", params={"dataset": "test_dataset"})
        assert response.status_code == 200


class TestGetExample:
    """Tests for /api/example/{id} endpoint."""

    def test_get_example_requires_auth(self, fresh_client: TestClient):
        """Test /api/example/{id} requires authentication."""
        response = fresh_client.get("/api/example/1")
        assert response.status_code == 401

    def test_get_example_not_found(self, auth_client: tuple[TestClient, dict]):
        """Test /api/example/{id} returns 404 for nonexistent example."""
        client, user = auth_client
        response = client.get("/api/example/99999")
        assert response.status_code == 404

    def test_get_example_success(self, auth_client: tuple[TestClient, dict], sample_example: dict):
        """Test /api/example/{id} returns example details."""
        client, user = auth_client
        response = client.get(f"/api/example/{sample_example['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_example["id"]
        assert data["premise"] == sample_example["premise"]
        assert data["hypothesis"] == sample_example["hypothesis"]
        assert "tokens" in data


class TestStats:
    """Tests for /api/stats endpoint."""

    def test_stats_requires_auth(self, fresh_client: TestClient):
        """Test /api/stats requires authentication."""
        response = fresh_client.get("/api/stats")
        assert response.status_code == 401

    def test_stats_success(self, auth_client: tuple[TestClient, dict]):
        """Test /api/stats returns statistics."""
        client, user = auth_client
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_examples" in data
        assert "annotated_examples" in data
        assert "datasets" in data


class TestDatasets:
    """Tests for /api/datasets endpoint."""

    def test_datasets_requires_auth(self, fresh_client: TestClient):
        """Test /api/datasets requires authentication."""
        response = fresh_client.get("/api/datasets")
        assert response.status_code == 401

    def test_datasets_success(self, auth_client: tuple[TestClient, dict], sample_example: dict):
        """Test /api/datasets returns dataset list."""
        client, user = auth_client
        response = client.get("/api/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert "test_dataset" in data["datasets"]


class TestLocking:
    """Tests for example locking endpoints."""

    def test_lock_acquire_requires_auth(self, fresh_client: TestClient):
        """Test lock acquire requires authentication."""
        response = fresh_client.post("/api/lock/1")
        assert response.status_code == 401

    def test_lock_acquire_success(self, auth_client: tuple[TestClient, dict], sample_example: dict):
        """Test successful lock acquisition."""
        client, user = auth_client
        response = client.post(f"/api/lock/{sample_example['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["locked"] is True
        assert data["example_id"] == sample_example["id"]

    def test_lock_release_success(self, auth_client: tuple[TestClient, dict], sample_example: dict):
        """Test successful lock release."""
        client, user = auth_client
        # Acquire lock
        client.post(f"/api/lock/{sample_example['id']}")
        # Release lock
        response = client.delete(f"/api/lock/{sample_example['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["released"] is True

    def test_lock_extend(self, auth_client: tuple[TestClient, dict], sample_example: dict):
        """Test lock extension."""
        client, user = auth_client
        # Acquire lock
        client.post(f"/api/lock/{sample_example['id']}")
        # Extend lock
        response = client.put(f"/api/lock/{sample_example['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["extended"] is True
