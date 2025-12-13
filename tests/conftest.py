"""
Pytest configuration and fixtures for NLI Span Labeler tests.
"""
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment before importing app
os.environ["ANONYMOUS_MODE"] = "0"
os.environ["ADMIN_USER"] = "test_admin"


@pytest.fixture(scope="session")
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file for the test session."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup after all tests
    if db_path.exists():
        db_path.unlink()


@pytest.fixture(scope="session")
def app(temp_db_path: Path):
    """Create FastAPI app with test database."""
    import app as app_module

    # Override database path
    app_module.DB_PATH = temp_db_path

    # Initialize database
    app_module.init_db()

    return app_module.app


@pytest.fixture(scope="session")
def client(app) -> Generator[TestClient, None, None]:
    """Create test client for the session."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def fresh_client(temp_db_path: Path) -> Generator[TestClient, None, None]:
    """Create a fresh test client with clean database for each test."""
    import importlib
    import app as app_module

    # Create new temp db for this test
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db = Path(f.name)

    app_module.DB_PATH = test_db
    app_module.init_db()

    with TestClient(app_module.app) as c:
        yield c

    if test_db.exists():
        test_db.unlink()


@pytest.fixture
def auth_client(client: TestClient) -> Generator[tuple[TestClient, dict], None, None]:
    """Create authenticated test client with a registered user."""
    # Register a test user
    response = client.post("/api/auth/register", json={
        "username": f"testuser_{os.urandom(4).hex()}",
        "password": "testpass123",
        "display_name": "Test User"
    })
    assert response.status_code == 200
    user_data = response.json()

    # Client now has session cookie from registration
    yield client, user_data


@pytest.fixture
def admin_client(client: TestClient) -> Generator[tuple[TestClient, dict], None, None]:
    """Create authenticated test client with admin user."""
    # Register as the admin user (ADMIN_USER env var)
    response = client.post("/api/auth/register", json={
        "username": "test_admin",
        "password": "adminpass123",
        "display_name": "Test Admin"
    })

    if response.status_code == 400:
        # Already registered, login instead
        response = client.post("/api/auth/login", json={
            "username": "test_admin",
            "password": "adminpass123"
        })

    assert response.status_code == 200
    user_data = response.json()

    yield client, user_data


@pytest.fixture
def sample_example(admin_client: tuple[TestClient, dict]) -> dict:
    """Create a sample example in the database for testing."""
    client, admin = admin_client

    # We need to insert an example directly since there's no API for it
    import app as app_module

    with app_module.get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO examples (dataset, premise, hypothesis, label, idx)
            VALUES (?, ?, ?, ?, ?)
        """, ("test_dataset", "The cat sat on the mat.", "The cat is sitting.", "entailment", 0))
        example_id = cursor.lastrowid
        conn.commit()

    return {
        "id": example_id,
        "dataset": "test_dataset",
        "premise": "The cat sat on the mat.",
        "hypothesis": "The cat is sitting.",
        "label": "entailment"
    }
