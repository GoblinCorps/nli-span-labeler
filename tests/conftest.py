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
os.environ["RATE_LIMIT_ENABLED"] = "0"  # Disable rate limiting in tests


@pytest.fixture
def fresh_client() -> Generator[TestClient, None, None]:
    """Create a fresh test client with clean database for each test."""
    import app as app_module

    # Create new temp db for this test
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db = Path(f.name)

    # Save original DB_PATH
    original_db_path = app_module.DB_PATH

    app_module.DB_PATH = test_db
    app_module.init_db()

    with TestClient(app_module.app) as c:
        yield c

    # Restore original path
    app_module.DB_PATH = original_db_path

    if test_db.exists():
        test_db.unlink()


@pytest.fixture
def auth_client() -> Generator[tuple[TestClient, dict], None, None]:
    """Create authenticated test client with a registered user (empty examples table)."""
    import app as app_module

    # Create new temp db for this test
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db = Path(f.name)

    # Save original DB_PATH
    original_db_path = app_module.DB_PATH

    app_module.DB_PATH = test_db
    app_module.init_db()

    with TestClient(app_module.app) as client:
        # Clear examples loaded by startup event (for tests that need empty DB)
        with app_module.get_db() as conn:
            conn.execute("DELETE FROM examples")
            conn.commit()

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

    # Restore original path
    app_module.DB_PATH = original_db_path

    if test_db.exists():
        test_db.unlink()


@pytest.fixture
def admin_client() -> Generator[tuple[TestClient, dict], None, None]:
    """Create authenticated test client with admin user."""
    import app as app_module

    # Create new temp db for this test
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db = Path(f.name)

    # Save original DB_PATH
    original_db_path = app_module.DB_PATH

    app_module.DB_PATH = test_db
    app_module.init_db()

    with TestClient(app_module.app) as client:
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

    # Restore original path
    app_module.DB_PATH = original_db_path

    if test_db.exists():
        test_db.unlink()


@pytest.fixture
def auth_client_with_example() -> Generator[tuple[TestClient, dict, dict], None, None]:
    """Create authenticated test client with a sample example in the database.

    Returns: (client, user_data, example_data)
    """
    import app as app_module

    # Create new temp db for this test
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db = Path(f.name)

    # Save original DB_PATH
    original_db_path = app_module.DB_PATH

    app_module.DB_PATH = test_db
    app_module.init_db()

    with TestClient(app_module.app) as client:
        # Clear examples loaded by startup event (we want only our test example)
        with app_module.get_db() as conn:
            conn.execute("DELETE FROM examples")
            conn.commit()

        # Register a test user
        response = client.post("/api/auth/register", json={
            "username": f"testuser_{os.urandom(4).hex()}",
            "password": "testpass123",
            "display_name": "Test User"
        })
        assert response.status_code == 200
        user_data = response.json()

        # Insert sample example
        example_id = f"test_example_{os.urandom(4).hex()}"
        with app_module.get_db() as conn:
            conn.execute("""
                INSERT INTO examples (id, dataset, premise, hypothesis, gold_label, gold_label_text)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (example_id, "test_dataset", "The cat sat on the mat.", "The cat is sitting.", 0, "entailment"))
            conn.commit()

        example_data = {
            "id": example_id,
            "dataset": "test_dataset",
            "premise": "The cat sat on the mat.",
            "hypothesis": "The cat is sitting.",
            "gold_label": 0,
            "gold_label_text": "entailment"
        }

        yield client, user_data, example_data

    # Restore original path
    app_module.DB_PATH = original_db_path

    if test_db.exists():
        test_db.unlink()


@pytest.fixture
def admin_client_with_example() -> Generator[tuple[TestClient, dict, dict], None, None]:
    """Create admin test client with a sample example in the database.

    Returns: (client, admin_data, example_data)
    """
    import app as app_module

    # Create new temp db for this test
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db = Path(f.name)

    # Save original DB_PATH
    original_db_path = app_module.DB_PATH

    app_module.DB_PATH = test_db
    app_module.init_db()

    with TestClient(app_module.app) as client:
        # Clear examples loaded by startup event (we want only our test example)
        with app_module.get_db() as conn:
            conn.execute("DELETE FROM examples")
            conn.commit()

        # Register as admin
        response = client.post("/api/auth/register", json={
            "username": "test_admin",
            "password": "adminpass123",
            "display_name": "Test Admin"
        })
        assert response.status_code == 200
        admin_data = response.json()

        # Insert sample example
        example_id = f"test_example_{os.urandom(4).hex()}"
        with app_module.get_db() as conn:
            conn.execute("""
                INSERT INTO examples (id, dataset, premise, hypothesis, gold_label, gold_label_text)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (example_id, "test_dataset", "The cat sat on the mat.", "The cat is sitting.", 0, "entailment"))
            conn.commit()

        example_data = {
            "id": example_id,
            "dataset": "test_dataset",
            "premise": "The cat sat on the mat.",
            "hypothesis": "The cat is sitting.",
            "gold_label": 0,
            "gold_label_text": "entailment"
        }

        yield client, admin_data, example_data

    # Restore original path
    app_module.DB_PATH = original_db_path

    if test_db.exists():
        test_db.unlink()
