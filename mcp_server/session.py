"""
Session state management for annotation sessions.

Tracks the current example being discussed and annotation progress.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class AnnotationProgress:
    """Tracks spans and scores being built up during discussion."""
    labels: dict[str, list[dict]] = field(default_factory=dict)  # label_name -> spans
    complexity_scores: dict[str, int] = field(default_factory=dict)
    edge_case_flags: list[str] = field(default_factory=list)
    discussion_notes: list[str] = field(default_factory=list)


@dataclass
class SessionState:
    """
    Current annotation session state.

    Persisted to disk so sessions survive restarts.
    """
    # Current example being discussed
    current_example_id: Optional[str] = None
    current_example: Optional[dict] = None

    # Annotation being built
    progress: AnnotationProgress = field(default_factory=AnnotationProgress)

    # Session stats
    examples_completed: int = 0
    examples_skipped: int = 0
    session_started: Optional[str] = None

    # Dataset filter (optional)
    active_dataset: Optional[str] = None


class SessionManager:
    """
    Manages persistent session state.

    State is saved to a JSON file so it survives server restarts.
    """

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or Path("session_state.json")
        self._state: Optional[SessionState] = None

    @property
    def state(self) -> SessionState:
        """Get current state, loading from disk if needed."""
        if self._state is None:
            self._state = self._load_state()
        return self._state

    def _load_state(self) -> SessionState:
        """Load state from disk, or create new."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                # Reconstruct nested dataclass
                progress_data = data.pop("progress", {})
                state = SessionState(**data)
                state.progress = AnnotationProgress(**progress_data)
                return state
            except Exception:
                pass
        return SessionState(session_started=datetime.now().isoformat())

    def _save_state(self) -> None:
        """Save state to disk."""
        if self._state is None:
            return
        data = asdict(self._state)
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def set_current_example(self, example: dict) -> None:
        """Set the current example being discussed."""
        self.state.current_example_id = example.get("id")
        self.state.current_example = example
        self.state.progress = AnnotationProgress()  # Reset progress
        self._save_state()

    def clear_current_example(self) -> None:
        """Clear the current example (after submit/skip)."""
        self.state.current_example_id = None
        self.state.current_example = None
        self.state.progress = AnnotationProgress()
        self._save_state()

    def add_span(self, label_name: str, span: dict) -> None:
        """Add a span to the current annotation progress."""
        if label_name not in self.state.progress.labels:
            self.state.progress.labels[label_name] = []
        self.state.progress.labels[label_name].append(span)
        self._save_state()

    def remove_span(self, label_name: str, word_index: int, source: str) -> bool:
        """Remove a span from the current annotation progress."""
        if label_name not in self.state.progress.labels:
            return False
        spans = self.state.progress.labels[label_name]
        for i, span in enumerate(spans):
            if span.get("word_index") == word_index and span.get("source") == source:
                spans.pop(i)
                self._save_state()
                return True
        return False

    def set_complexity_score(self, dimension: str, score: int) -> None:
        """Set a complexity score."""
        self.state.progress.complexity_scores[dimension] = score
        self._save_state()

    def add_edge_case_flag(self, flag: str) -> None:
        """Add an edge case flag."""
        if flag not in self.state.progress.edge_case_flags:
            self.state.progress.edge_case_flags.append(flag)
            self._save_state()

    def add_discussion_note(self, note: str) -> None:
        """Add a discussion note."""
        self.state.progress.discussion_notes.append(note)
        self._save_state()

    def mark_completed(self) -> None:
        """Mark current example as completed."""
        self.state.examples_completed += 1
        self.clear_current_example()

    def mark_skipped(self) -> None:
        """Mark current example as skipped."""
        self.state.examples_skipped += 1
        self.clear_current_example()

    def set_dataset_filter(self, dataset: Optional[str]) -> None:
        """Set the active dataset filter."""
        self.state.active_dataset = dataset
        self._save_state()

    def get_session_summary(self) -> dict:
        """Get a summary of the current session."""
        return {
            "current_example_id": self.state.current_example_id,
            "has_current_example": self.state.current_example is not None,
            "pending_labels": list(self.state.progress.labels.keys()),
            "pending_scores": list(self.state.progress.complexity_scores.keys()),
            "examples_completed": self.state.examples_completed,
            "examples_skipped": self.state.examples_skipped,
            "session_started": self.state.session_started,
            "active_dataset": self.state.active_dataset,
        }

    def reset_session(self) -> None:
        """Reset the session state entirely."""
        self._state = SessionState(session_started=datetime.now().isoformat())
        self._save_state()
