#!/usr/bin/env python3
"""
NLI Span Labeler - A web-based tool for annotating NLI examples with span-level labels.

Features:
- Word-level span selection with position tracking
- Pre-filled labels for difficulty dimensions and NLI relations
- Custom label creation with unique colors
- Multiple labels per token visualization
- Complexity scoring (1-100 scale)
- SQLite persistence
- Stats dashboard and export

Usage:
    cd nli-span-labeler
    uvicorn app:app --reload --port 8000
    # Then open http://localhost:8000
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Paths - relative to this file's directory
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data" / "nli"
DB_PATH = APP_DIR / "labels.db"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NLI Span Labeler")

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# ============================================================================
# Database Setup
# ============================================================================

def init_db():
    """Initialize SQLite database."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS examples (
                id TEXT PRIMARY KEY,
                dataset TEXT NOT NULL,
                premise TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                gold_label INTEGER,
                gold_label_text TEXT,
                source_file TEXT,
                loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                example_id TEXT NOT NULL,
                label_name TEXT NOT NULL,
                label_color TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (example_id) REFERENCES examples(id)
            );

            CREATE TABLE IF NOT EXISTS span_selections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label_id INTEGER NOT NULL,
                source TEXT NOT NULL,  -- 'premise' or 'hypothesis'
                word_index INTEGER NOT NULL,
                word_text TEXT NOT NULL,
                char_start INTEGER,
                char_end INTEGER,
                FOREIGN KEY (label_id) REFERENCES labels(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS complexity_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                example_id TEXT NOT NULL UNIQUE,
                reasoning INTEGER,
                creativity INTEGER,
                domain_knowledge INTEGER,
                contextual INTEGER,
                constraints INTEGER,
                ambiguity INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (example_id) REFERENCES examples(id)
            );

            CREATE TABLE IF NOT EXISTS skipped (
                example_id TEXT PRIMARY KEY,
                skipped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (example_id) REFERENCES examples(id)
            );

            CREATE INDEX IF NOT EXISTS idx_labels_example ON labels(example_id);
            CREATE INDEX IF NOT EXISTS idx_spans_label ON span_selections(label_id);
            CREATE INDEX IF NOT EXISTS idx_examples_dataset ON examples(dataset);
        """)


@contextmanager
def get_db():
    """Database connection context manager."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ============================================================================
# Pydantic Models
# ============================================================================

class SpanSelection(BaseModel):
    source: str  # 'premise' or 'hypothesis'
    word_index: int
    word_text: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class LabelSubmission(BaseModel):
    label_name: str
    label_color: str
    spans: list[SpanSelection]


class AnnotationSubmission(BaseModel):
    example_id: str
    labels: list[LabelSubmission]
    complexity_scores: Optional[dict] = None


class ExampleResponse(BaseModel):
    id: str
    dataset: str
    premise: str
    hypothesis: str
    gold_label: Optional[int]
    gold_label_text: Optional[str]
    premise_words: list[dict]
    hypothesis_words: list[dict]
    existing_labels: list[dict]
    existing_scores: Optional[dict]


# ============================================================================
# Data Loading
# ============================================================================

def load_examples_from_file(filepath: Path, limit: int = 1000) -> list[dict]:
    """Load examples from a JSONL file."""
    examples = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            if line.strip():
                ex = json.loads(line)
                examples.append(ex)
    return examples


def ensure_examples_loaded(dataset: str, limit: int = 500):
    """Ensure examples from a dataset are in the database."""
    with get_db() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM examples WHERE dataset = ?",
            (dataset,)
        ).fetchone()[0]

        if count > 0:
            return count

        # Find and load the dataset file
        patterns = [f"{dataset}_*.jsonl", f"{dataset}.jsonl"]
        files = []
        for pattern in patterns:
            files.extend(DATA_DIR.glob(pattern))

        if not files:
            return 0

        loaded = 0
        for filepath in files:
            examples = load_examples_from_file(filepath, limit=limit)
            for ex in examples:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO examples
                        (id, dataset, premise, hypothesis, gold_label, gold_label_text, source_file)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ex.get("id", f"{dataset}_{loaded}"),
                        dataset,
                        ex.get("premise", ""),
                        ex.get("hypothesis", ""),
                        ex.get("label"),
                        ex.get("label_text"),
                        filepath.name
                    ))
                    loaded += 1
                except Exception as e:
                    print(f"Error loading example: {e}")

        return loaded


def tokenize_text(text: str) -> list[dict]:
    """Simple word tokenization with character offsets."""
    words = []
    current_pos = 0

    for i, word in enumerate(text.split()):
        # Find the actual position in the original text
        start = text.find(word, current_pos)
        if start == -1:
            start = current_pos
        end = start + len(word)

        words.append({
            "index": i,
            "text": word,
            "char_start": start,
            "char_end": end
        })
        current_pos = end

    return words


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    init_db()
    # Pre-load some datasets
    for dataset in ["snli", "mnli", "anli"]:
        ensure_examples_loaded(dataset, limit=200)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main labeling interface."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/datasets")
async def list_datasets():
    """List available datasets."""
    files = list(DATA_DIR.glob("*.jsonl"))
    datasets = set()
    for f in files:
        # Extract dataset name from filename
        name = f.stem.split("_")[0]
        datasets.add(name)
    return {"datasets": sorted(datasets)}


@app.get("/api/example/{dataset}/{row_id}")
async def get_example(dataset: str, row_id: str):
    """Get a specific example by dataset and row ID."""
    ensure_examples_loaded(dataset)

    with get_db() as conn:
        # Try exact match first
        row = conn.execute(
            "SELECT * FROM examples WHERE id = ?",
            (row_id,)
        ).fetchone()

        # If not found, try with dataset prefix
        if not row:
            row = conn.execute(
                "SELECT * FROM examples WHERE id = ? OR id = ?",
                (row_id, f"{dataset}_{row_id}")
            ).fetchone()

        if not row:
            raise HTTPException(404, f"Example not found: {dataset}/{row_id}")

        # Get existing labels
        labels = conn.execute("""
            SELECT l.id, l.label_name, l.label_color,
                   GROUP_CONCAT(s.source || ':' || s.word_index || ':' || s.word_text, '|') as spans
            FROM labels l
            LEFT JOIN span_selections s ON s.label_id = l.id
            WHERE l.example_id = ?
            GROUP BY l.id
        """, (row["id"],)).fetchall()

        existing_labels = []
        for label in labels:
            spans = []
            if label["spans"]:
                for span_str in label["spans"].split("|"):
                    parts = span_str.split(":")
                    if len(parts) >= 3:
                        spans.append({
                            "source": parts[0],
                            "word_index": int(parts[1]),
                            "word_text": parts[2]
                        })
            existing_labels.append({
                "id": label["id"],
                "label_name": label["label_name"],
                "label_color": label["label_color"],
                "spans": spans
            })

        # Get existing complexity scores
        scores_row = conn.execute(
            "SELECT * FROM complexity_scores WHERE example_id = ?",
            (row["id"],)
        ).fetchone()

        existing_scores = None
        if scores_row:
            existing_scores = {
                "reasoning": scores_row["reasoning"],
                "creativity": scores_row["creativity"],
                "domain_knowledge": scores_row["domain_knowledge"],
                "contextual": scores_row["contextual"],
                "constraints": scores_row["constraints"],
                "ambiguity": scores_row["ambiguity"],
            }

        return ExampleResponse(
            id=row["id"],
            dataset=row["dataset"],
            premise=row["premise"],
            hypothesis=row["hypothesis"],
            gold_label=row["gold_label"],
            gold_label_text=row["gold_label_text"],
            premise_words=tokenize_text(row["premise"]),
            hypothesis_words=tokenize_text(row["hypothesis"]),
            existing_labels=existing_labels,
            existing_scores=existing_scores
        )


@app.get("/api/next")
async def get_next_example(dataset: Optional[str] = None):
    """Get the next unlabeled example."""
    with get_db() as conn:
        # Build query based on whether dataset is specified
        if dataset:
            ensure_examples_loaded(dataset)
            row = conn.execute("""
                SELECT e.* FROM examples e
                LEFT JOIN complexity_scores c ON e.id = c.example_id
                LEFT JOIN skipped s ON e.id = s.example_id
                WHERE e.dataset = ? AND c.example_id IS NULL AND s.example_id IS NULL
                ORDER BY RANDOM()
                LIMIT 1
            """, (dataset,)).fetchone()
        else:
            row = conn.execute("""
                SELECT e.* FROM examples e
                LEFT JOIN complexity_scores c ON e.id = c.example_id
                LEFT JOIN skipped s ON e.id = s.example_id
                WHERE c.example_id IS NULL AND s.example_id IS NULL
                ORDER BY RANDOM()
                LIMIT 1
            """).fetchone()

        if not row:
            return {"message": "No more examples to label!", "complete": True}

        return ExampleResponse(
            id=row["id"],
            dataset=row["dataset"],
            premise=row["premise"],
            hypothesis=row["hypothesis"],
            gold_label=row["gold_label"],
            gold_label_text=row["gold_label_text"],
            premise_words=tokenize_text(row["premise"]),
            hypothesis_words=tokenize_text(row["hypothesis"]),
            existing_labels=[],
            existing_scores=None
        )


@app.post("/api/annotate")
async def save_annotation(submission: AnnotationSubmission):
    """Save span labels and complexity scores."""
    with get_db() as conn:
        # Verify example exists
        example = conn.execute(
            "SELECT id FROM examples WHERE id = ?",
            (submission.example_id,)
        ).fetchone()

        if not example:
            raise HTTPException(404, f"Example not found: {submission.example_id}")

        # Delete existing labels for this example (to allow re-annotation)
        conn.execute("DELETE FROM labels WHERE example_id = ?", (submission.example_id,))

        # Save new labels
        for label in submission.labels:
            if not label.spans:
                continue  # Skip labels with no spans

            cursor = conn.execute("""
                INSERT INTO labels (example_id, label_name, label_color)
                VALUES (?, ?, ?)
            """, (submission.example_id, label.label_name, label.label_color))
            label_id = cursor.lastrowid

            # Save span selections
            for span in label.spans:
                conn.execute("""
                    INSERT INTO span_selections
                    (label_id, source, word_index, word_text, char_start, char_end)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    label_id,
                    span.source,
                    span.word_index,
                    span.word_text,
                    span.char_start,
                    span.char_end
                ))

        # Save complexity scores if provided
        if submission.complexity_scores:
            conn.execute("""
                INSERT OR REPLACE INTO complexity_scores
                (example_id, reasoning, creativity, domain_knowledge, contextual, constraints, ambiguity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                submission.example_id,
                submission.complexity_scores.get("reasoning"),
                submission.complexity_scores.get("creativity"),
                submission.complexity_scores.get("domain_knowledge"),
                submission.complexity_scores.get("contextual"),
                submission.complexity_scores.get("constraints"),
                submission.complexity_scores.get("ambiguity"),
            ))

        return {"status": "saved", "example_id": submission.example_id}


@app.post("/api/skip/{example_id}")
async def skip_example(example_id: str):
    """Mark an example as skipped."""
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO skipped (example_id) VALUES (?)",
            (example_id,)
        )
        return {"status": "skipped", "example_id": example_id}


@app.get("/api/stats")
async def get_stats():
    """Get labeling statistics."""
    with get_db() as conn:
        # Total examples per dataset
        dataset_counts = conn.execute("""
            SELECT dataset, COUNT(*) as total FROM examples GROUP BY dataset
        """).fetchall()

        # Labeled examples per dataset
        labeled_counts = conn.execute("""
            SELECT e.dataset, COUNT(DISTINCT c.example_id) as labeled
            FROM examples e
            LEFT JOIN complexity_scores c ON e.id = c.example_id
            WHERE c.example_id IS NOT NULL
            GROUP BY e.dataset
        """).fetchall()

        # Skipped examples per dataset
        skipped_counts = conn.execute("""
            SELECT e.dataset, COUNT(DISTINCT s.example_id) as skipped
            FROM examples e
            LEFT JOIN skipped s ON e.id = s.example_id
            WHERE s.example_id IS NOT NULL
            GROUP BY e.dataset
        """).fetchall()

        # Total span labels
        span_stats = conn.execute("""
            SELECT
                COUNT(DISTINCT l.id) as total_labels,
                COUNT(s.id) as total_spans,
                COUNT(DISTINCT l.example_id) as examples_with_spans
            FROM labels l
            LEFT JOIN span_selections s ON s.label_id = l.id
        """).fetchone()

        # Label distribution
        label_dist = conn.execute("""
            SELECT label_name, COUNT(*) as count
            FROM labels
            GROUP BY label_name
            ORDER BY count DESC
        """).fetchall()

        # Complexity score averages
        score_avgs = conn.execute("""
            SELECT
                AVG(reasoning) as reasoning,
                AVG(creativity) as creativity,
                AVG(domain_knowledge) as domain_knowledge,
                AVG(contextual) as contextual,
                AVG(constraints) as constraints,
                AVG(ambiguity) as ambiguity,
                COUNT(*) as total
            FROM complexity_scores
        """).fetchone()

        return {
            "datasets": {
                row["dataset"]: {"total": row["total"]}
                for row in dataset_counts
            },
            "labeled": {
                row["dataset"]: row["labeled"]
                for row in labeled_counts
            },
            "skipped": {
                row["dataset"]: row["skipped"]
                for row in skipped_counts
            },
            "spans": {
                "total_labels": span_stats["total_labels"],
                "total_spans": span_stats["total_spans"],
                "examples_with_spans": span_stats["examples_with_spans"]
            },
            "label_distribution": {
                row["label_name"]: row["count"]
                for row in label_dist
            },
            "complexity_averages": {
                "reasoning": score_avgs["reasoning"],
                "creativity": score_avgs["creativity"],
                "domain_knowledge": score_avgs["domain_knowledge"],
                "contextual": score_avgs["contextual"],
                "constraints": score_avgs["constraints"],
                "ambiguity": score_avgs["ambiguity"],
            } if score_avgs["total"] > 0 else None,
            "total_labeled": score_avgs["total"]
        }


@app.get("/api/export")
async def export_labels():
    """Export all labels as JSONL."""
    with get_db() as conn:
        examples = conn.execute("""
            SELECT e.*, c.reasoning, c.creativity, c.domain_knowledge,
                   c.contextual, c.constraints, c.ambiguity
            FROM examples e
            JOIN complexity_scores c ON e.id = c.example_id
        """).fetchall()

        results = []
        for ex in examples:
            # Get labels for this example
            labels = conn.execute("""
                SELECT l.label_name, l.label_color,
                       s.source, s.word_index, s.word_text, s.char_start, s.char_end
                FROM labels l
                JOIN span_selections s ON s.label_id = l.id
                WHERE l.example_id = ?
            """, (ex["id"],)).fetchall()

            # Group spans by label
            label_spans = {}
            for row in labels:
                name = row["label_name"]
                if name not in label_spans:
                    label_spans[name] = {
                        "label_name": name,
                        "label_color": row["label_color"],
                        "spans": []
                    }
                label_spans[name]["spans"].append({
                    "source": row["source"],
                    "word_index": row["word_index"],
                    "word_text": row["word_text"],
                    "char_start": row["char_start"],
                    "char_end": row["char_end"]
                })

            results.append({
                "id": ex["id"],
                "dataset": ex["dataset"],
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "gold_label": ex["gold_label"],
                "gold_label_text": ex["gold_label_text"],
                "complexity_scores": {
                    "reasoning": ex["reasoning"],
                    "creativity": ex["creativity"],
                    "domain_knowledge": ex["domain_knowledge"],
                    "contextual": ex["contextual"],
                    "constraints": ex["constraints"],
                    "ambiguity": ex["ambiguity"],
                },
                "span_labels": list(label_spans.values())
            })

        return {"count": len(results), "data": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
