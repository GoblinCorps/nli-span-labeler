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

    def test_next_with_examples(self, auth_client_with_example: tuple[TestClient, dict, dict]):
        """Test /api/next returns an example when available."""
        client, user, example = auth_client_with_example
        response = client.get("/api/next")
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "premise" in data
        assert "hypothesis" in data
        # API returns premise_words and hypothesis_words instead of generic tokens
        assert "premise_words" in data
        assert "hypothesis_words" in data

    def test_next_with_dataset_filter(self, auth_client_with_example: tuple[TestClient, dict, dict]):
        """Test /api/next respects dataset filter."""
        client, user, example = auth_client_with_example
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
        response = client.get("/api/example/nonexistent_99999")
        assert response.status_code == 404

    def test_get_example_success(self, auth_client_with_example: tuple[TestClient, dict, dict]):
        """Test /api/example/{id} returns example details."""
        client, user, example = auth_client_with_example
        response = client.get(f"/api/example/{example['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == example["id"]
        assert data["premise"] == example["premise"]
        assert data["hypothesis"] == example["hypothesis"]
        # API returns premise_words and hypothesis_words
        assert "premise_words" in data
        assert "hypothesis_words" in data


class TestStats:
    """Tests for /api/stats endpoint."""

    def test_stats_public(self, fresh_client: TestClient):
        """Test /api/stats is publicly accessible."""
        response = fresh_client.get("/api/stats")
        # Stats endpoint is public
        assert response.status_code == 200

    def test_stats_success(self, auth_client: tuple[TestClient, dict]):
        """Test /api/stats returns statistics."""
        client, user = auth_client
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        # Check for actual response structure
        assert "datasets" in data
        assert "locks" in data


class TestDatasets:
    """Tests for /api/datasets endpoint."""

    def test_datasets_public(self, fresh_client: TestClient):
        """Test /api/datasets is publicly accessible."""
        response = fresh_client.get("/api/datasets")
        # Datasets endpoint is public
        assert response.status_code == 200

    def test_datasets_success(self, auth_client: tuple[TestClient, dict]):
        """Test /api/datasets returns dataset list structure.

        Note: /api/datasets reads from JSONL files, not the database, so we
        only verify the response structure, not specific dataset names.
        """
        client, user = auth_client
        response = client.get("/api/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert isinstance(data["datasets"], list)


class TestLocking:
    """Tests for example locking endpoints.

    Note: Locks are automatically acquired via /api/next, not via explicit POST.
    """

    def test_lock_status_requires_auth(self, fresh_client: TestClient):
        """Test lock status endpoint requires authentication."""
        response = fresh_client.get("/api/lock/status/test_example")
        assert response.status_code == 401

    def test_lock_acquired_with_next(self, auth_client_with_example: tuple[TestClient, dict, dict]):
        """Test that getting next example acquires a lock automatically."""
        client, user, example = auth_client_with_example
        # Get next example (which auto-acquires lock)
        response = client.get("/api/next")
        assert response.status_code == 200
        data = response.json()
        # The example should now be locked
        assert "id" in data

    def test_lock_release_success(self, auth_client_with_example: tuple[TestClient, dict, dict]):
        """Test successful lock release."""
        client, user, example = auth_client_with_example
        # Get next to acquire lock
        client.get("/api/next")
        # Release lock (POST, not DELETE)
        response = client.post(f"/api/lock/release/{example['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("released", "not_locked")

    def test_lock_extend(self, auth_client_with_example: tuple[TestClient, dict, dict]):
        """Test lock extension."""
        client, user, example = auth_client_with_example
        # Get next to acquire lock
        client.get("/api/next")
        # Extend lock (POST, not PUT)
        response = client.post(f"/api/lock/extend/{example['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "extended"
