"""
Tests for authentication endpoints.
"""
import pytest
from fastapi.testclient import TestClient


class TestRegistration:
    """Tests for /api/auth/register endpoint."""

    def test_register_success(self, fresh_client: TestClient):
        """Test successful user registration."""
        response = fresh_client.post("/api/auth/register", json={
            "username": "newuser",
            "password": "password123",
            "display_name": "New User"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "newuser"
        assert data["display_name"] == "New User"
        assert data["role"] == "annotator"

    def test_register_duplicate_username(self, fresh_client: TestClient):
        """Test registration with duplicate username fails."""
        # First registration
        fresh_client.post("/api/auth/register", json={
            "username": "dupuser",
            "password": "password123",
            "display_name": "First User"
        })
        # Second registration with same username
        response = fresh_client.post("/api/auth/register", json={
            "username": "dupuser",
            "password": "differentpass",
            "display_name": "Second User"
        })
        assert response.status_code == 400
        assert "already taken" in response.json()["detail"].lower()

    def test_register_short_password(self, fresh_client: TestClient):
        """Test registration with short password fails."""
        response = fresh_client.post("/api/auth/register", json={
            "username": "shortpass",
            "password": "abc",
            "display_name": "Short Pass"
        })
        assert response.status_code == 422  # Validation error

    def test_register_empty_username(self, fresh_client: TestClient):
        """Test registration with empty username fails."""
        response = fresh_client.post("/api/auth/register", json={
            "username": "",
            "password": "password123",
            "display_name": "Empty User"
        })
        assert response.status_code == 422

    def test_register_admin_user(self, fresh_client: TestClient):
        """Test that ADMIN_USER env var grants admin role."""
        response = fresh_client.post("/api/auth/register", json={
            "username": "test_admin",  # Matches ADMIN_USER env var
            "password": "adminpass",
            "display_name": "Admin"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "admin"


class TestLogin:
    """Tests for /api/auth/login endpoint."""

    def test_login_success(self, fresh_client: TestClient):
        """Test successful login."""
        # Register first
        fresh_client.post("/api/auth/register", json={
            "username": "logintest",
            "password": "password123",
            "display_name": "Login Test"
        })
        # Clear cookies
        fresh_client.cookies.clear()
        # Login
        response = fresh_client.post("/api/auth/login", json={
            "username": "logintest",
            "password": "password123"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "logintest"

    def test_login_wrong_password(self, fresh_client: TestClient):
        """Test login with wrong password fails."""
        # Register
        fresh_client.post("/api/auth/register", json={
            "username": "wrongpass",
            "password": "correctpass",
            "display_name": "Wrong Pass"
        })
        fresh_client.cookies.clear()
        # Login with wrong password
        response = fresh_client.post("/api/auth/login", json={
            "username": "wrongpass",
            "password": "wrongpassword"
        })
        assert response.status_code == 401

    def test_login_nonexistent_user(self, fresh_client: TestClient):
        """Test login with nonexistent user fails."""
        response = fresh_client.post("/api/auth/login", json={
            "username": "ghostuser",
            "password": "password123"
        })
        assert response.status_code == 401


class TestLogout:
    """Tests for /api/auth/logout endpoint."""

    def test_logout_success(self, auth_client: tuple[TestClient, dict]):
        """Test successful logout."""
        client, user = auth_client
        response = client.post("/api/auth/logout")
        assert response.status_code == 200
        assert response.json()["status"] == "logged_out"

    def test_logout_clears_session(self, auth_client: tuple[TestClient, dict]):
        """Test that logout clears the session."""
        client, user = auth_client
        # Verify we're logged in
        me_response = client.get("/api/me")
        assert me_response.status_code == 200

        # Logout
        client.post("/api/auth/logout")

        # Clear cookies (simulate browser behavior)
        client.cookies.clear()

        # Verify we're logged out
        me_response = client.get("/api/me")
        assert me_response.status_code == 401


class TestMe:
    """Tests for /api/me endpoint."""

    def test_me_authenticated(self, auth_client: tuple[TestClient, dict]):
        """Test /api/me returns user info when authenticated."""
        client, user = auth_client
        response = client.get("/api/me")
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == user["username"]
        assert data["display_name"] == user["display_name"]

    def test_me_unauthenticated(self, fresh_client: TestClient):
        """Test /api/me returns 401 when not authenticated."""
        response = fresh_client.get("/api/me")
        assert response.status_code == 401
