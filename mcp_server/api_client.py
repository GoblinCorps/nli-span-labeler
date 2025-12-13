"""
API client for the NLI Span Labeler backend.

Wraps all HTTP calls to the FastAPI backend, handling authentication
and response parsing.
"""

import httpx
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class APIConfig:
    """Configuration for the API client."""
    base_url: str = "http://127.0.0.1:8000"
    username: str = "mcp-annotator"
    password: str = "mcp-annotator-password"
    timeout: float = 30.0


@dataclass
class SpanSelection:
    """A single span selection."""
    source: str  # "premise" or "hypothesis"
    word_index: int
    word_text: str
    char_start: int
    char_end: int


@dataclass
class LabelSubmission:
    """A label with its spans."""
    label_name: str
    label_color: str
    spans: list[SpanSelection] = field(default_factory=list)


@dataclass
class AnnotationSubmission:
    """Complete annotation submission."""
    example_id: str
    labels: list[LabelSubmission] = field(default_factory=list)
    complexity_scores: Optional[dict] = None


class NLIApiClient:
    """
    Client for the NLI Span Labeler API.

    Handles authentication via session cookies and provides typed methods
    for all annotation-related endpoints.
    """

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self._client = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            follow_redirects=True,
        )
        self._authenticated = False

    def _ensure_auth(self) -> None:
        """Ensure we have a valid session."""
        if self._authenticated:
            return

        # Try to register first (will fail if user exists)
        try:
            response = self._client.post(
                "/api/auth/register",
                json={
                    "username": self.config.username,
                    "password": self.config.password,
                }
            )
            if response.status_code == 200:
                self._authenticated = True
                return
        except Exception:
            pass

        # Fall back to login
        response = self._client.post(
            "/api/auth/login",
            json={
                "username": self.config.username,
                "password": self.config.password,
            }
        )
        response.raise_for_status()
        self._authenticated = True

    def get_auth_status(self) -> dict:
        """Get authentication status and mode."""
        response = self._client.get("/api/auth/status")
        response.raise_for_status()
        return response.json()

    def get_me(self) -> dict:
        """Get current user info."""
        self._ensure_auth()
        response = self._client.get("/api/me")
        response.raise_for_status()
        return response.json()

    def get_next_example(
        self,
        dataset: Optional[str] = None,
        gold_label: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Get the next example to annotate.

        Args:
            dataset: Optional dataset filter (snli, mnli, anli)
            gold_label: Optional label filter (entailment, neutral, contradiction)

        Returns None if no examples are available.
        """
        self._ensure_auth()
        params = {}
        if dataset:
            params["dataset"] = dataset
        if gold_label:
            params["gold_label"] = gold_label

        response = self._client.get("/api/next", params=params)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def get_example(self, example_id: str) -> Optional[dict]:
        """Get a specific example by ID."""
        self._ensure_auth()
        response = self._client.get(f"/api/example/{example_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def get_auto_spans(self, example_id: str) -> Optional[dict]:
        """Get auto-generated spans for an example."""
        self._ensure_auth()
        response = self._client.get(f"/api/auto-spans/{example_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def submit_annotation(self, submission: AnnotationSubmission) -> dict:
        """Submit an annotation for an example."""
        self._ensure_auth()

        # Convert dataclass to dict
        payload = {
            "example_id": submission.example_id,
            "labels": [
                {
                    "label_name": label.label_name,
                    "label_color": label.label_color,
                    "spans": [
                        {
                            "source": span.source,
                            "word_index": span.word_index,
                            "word_text": span.word_text,
                            "char_start": span.char_start,
                            "char_end": span.char_end,
                        }
                        for span in label.spans
                    ]
                }
                for label in submission.labels
            ],
        }
        if submission.complexity_scores:
            payload["complexity_scores"] = submission.complexity_scores

        response = self._client.post("/api/label", json=payload)
        response.raise_for_status()
        return response.json()

    def skip_example(self, example_id: str) -> dict:
        """Skip an example."""
        self._ensure_auth()
        response = self._client.post(f"/api/skip/{example_id}")
        response.raise_for_status()
        return response.json()

    def extend_lock(self, example_id: str) -> dict:
        """Extend the lock on an example."""
        self._ensure_auth()
        response = self._client.post(f"/api/lock/extend/{example_id}")
        response.raise_for_status()
        return response.json()

    def release_lock(self, example_id: str) -> dict:
        """Release the lock on an example."""
        self._ensure_auth()
        response = self._client.post(f"/api/lock/release/{example_id}")
        response.raise_for_status()
        return response.json()

    def get_stats(self) -> dict:
        """Get annotation statistics."""
        response = self._client.get("/api/stats")
        response.raise_for_status()
        return response.json()

    def get_datasets(self) -> list[str]:
        """Get list of available datasets."""
        response = self._client.get("/api/datasets")
        response.raise_for_status()
        return response.json().get("datasets", [])

    def get_labels(self) -> dict:
        """Get label schema configuration."""
        response = self._client.get("/api/labels")
        response.raise_for_status()
        return response.json()

    def get_reliability_me(self) -> dict:
        """Get current user's reliability score."""
        self._ensure_auth()
        response = self._client.get("/api/reliability/me")
        response.raise_for_status()
        return response.json()

    def get_leaderboard(self) -> dict:
        """Get the reliability leaderboard."""
        self._ensure_auth()
        response = self._client.get("/api/reliability/leaderboard")
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
