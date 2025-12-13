"""
Tests for admin endpoints.
"""
import pytest
from fastapi.testclient import TestClient


class TestAdminAccess:
    """Tests for admin access control."""

    def test_admin_requires_auth(self, fresh_client: TestClient):
        """Test admin endpoints require authentication."""
        response = fresh_client.get("/api/admin/users")
        assert response.status_code == 401

    def test_admin_requires_admin_role(self, auth_client: tuple[TestClient, dict]):
        """Test admin endpoints require admin role."""
        client, user = auth_client
        # Regular user should be denied
        if user["role"] != "admin":
            response = client.get("/api/admin/users")
            assert response.status_code == 403


class TestAdminUsers:
    """Tests for /api/admin/users endpoint."""

    def test_list_users(self, admin_client: tuple[TestClient, dict]):
        """Test listing users as admin."""
        client, admin = admin_client
        response = client.get("/api/admin/users")
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert isinstance(data["users"], list)

    def test_update_user_role(self, admin_client: tuple[TestClient, dict]):
        """Test updating user role as admin."""
        client, admin = admin_client

        # Create a user to update
        client.post("/api/auth/register", json={
            "username": "role_test_user",
            "password": "password123",
            "display_name": "Role Test"
        })

        # Get user list to find the user ID
        users_response = client.get("/api/admin/users")
        users = users_response.json()["users"]
        target_user = next((u for u in users if u["username"] == "role_test_user"), None)

        if target_user:
            response = client.put(f"/api/admin/users/{target_user['id']}/role", json={
                "role": "admin"
            })
            assert response.status_code == 200
            data = response.json()
            assert data["new_role"] == "admin"


class TestAdminDashboard:
    """Tests for /api/admin/dashboard endpoint."""

    def test_dashboard_requires_admin(self, auth_client: tuple[TestClient, dict]):
        """Test dashboard requires admin role."""
        client, user = auth_client
        if user["role"] != "admin":
            response = client.get("/api/admin/dashboard")
            assert response.status_code == 403

    def test_dashboard_success(self, admin_client: tuple[TestClient, dict]):
        """Test getting dashboard metrics."""
        client, admin = admin_client
        response = client.get("/api/admin/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "overview" in data
        assert "activity" in data
        assert "datasets" in data
        assert "annotators" in data
        assert "quality" in data
        assert "pools" in data

    def test_dashboard_with_filters(self, admin_client: tuple[TestClient, dict]):
        """Test dashboard with query parameters."""
        client, admin = admin_client
        response = client.get("/api/admin/dashboard", params={
            "dataset": "test_dataset",
            "days": 7
        })
        assert response.status_code == 200
        data = response.json()
        assert data["filters"]["dataset"] == "test_dataset"
        assert data["filters"]["days"] == 7


class TestAdminCalibration:
    """Tests for /api/admin/calibration endpoint."""

    def test_calibration_requires_admin(self, auth_client: tuple[TestClient, dict]):
        """Test calibration endpoint requires admin role."""
        client, user = auth_client
        if user["role"] != "admin":
            response = client.get("/api/admin/calibration")
            assert response.status_code == 403

    def test_calibration_success(self, admin_client: tuple[TestClient, dict]):
        """Test getting calibration config."""
        client, admin = admin_client
        response = client.get("/api/admin/calibration")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "test_pool_size" in data
        assert "annotator_counts" in data


class TestExport:
    """Tests for /api/export endpoint."""

    def test_export_requires_admin(self, auth_client: tuple[TestClient, dict]):
        """Test export requires admin role."""
        client, user = auth_client
        if user["role"] != "admin":
            response = client.get("/api/export")
            assert response.status_code == 403

    def test_export_success(self, admin_client: tuple[TestClient, dict]):
        """Test exporting data."""
        client, admin = admin_client
        response = client.get("/api/export")
        assert response.status_code == 200
        # Export returns JSON data
        data = response.json()
        assert "examples" in data or isinstance(data, list)

    def test_export_with_format(self, admin_client: tuple[TestClient, dict]):
        """Test export with format parameter."""
        client, admin = admin_client
        response = client.get("/api/export", params={"format": "json"})
        assert response.status_code == 200
