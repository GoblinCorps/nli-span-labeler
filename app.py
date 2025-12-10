#!/usr/bin/env python3
"""
NLI Span Labeler - A web-based tool for annotating NLI examples with span-level labels.

Features:
- WordPiece tokenization matching ModernBERT training target
- Word-level span selection with position tracking
- Pre-filled labels for difficulty dimensions and NLI relations
- Custom label creation with unique colors
- Multiple labels per token visualization
- Complexity scoring (1-100 scale)
- SQLite persistence
- Stats dashboard and export
- Multi-user support with session-based authentication
- Annotator tracking for all labels and scores
- Label schema enforcement with custom label tracking
- Example locking for concurrent multi-user annotation
- Role-based admin mode with protected endpoints
- Inter-annotator agreement metrics and question pools

Usage:
    cd nli-span-labeler
    uvicorn app:app --reload --port 8000
    # Then open http://localhost:8000

Environment Variables:
    ANONYMOUS_MODE=1  - Disable auth, use anonymous user (for local single-user)
    TOKENIZER_MODEL=answerdotai/ModernBERT-base  - HuggingFace model for tokenizer
    LOCK_TIMEOUT_MINUTES=30  - How long example locks last (default: 30)
    ADMIN_USER=username  - Bootstrap admin user (gets admin role on startup)
    CONSENSUS_THRESHOLD=10  - Annotations needed before consensus calculation (default: 10)
    AGREEMENT_HIGH_THRESHOLD=0.8  - Agreement score to promote to test pool (default: 0.8)
"""

import json
import sqlite3
import secrets
import hashlib
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from contextlib import contextmanager
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query, Request, Response, Depends, Cookie
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# Tokenizer - using the actual training target model
from transformers import AutoTokenizer

# Paths - relative to this file's directory
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data" / "nli"
DB_PATH = APP_DIR / "labels.db"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
ANONYMOUS_MODE = os.environ.get("ANONYMOUS_MODE", "0") == "1"
SESSION_EXPIRY_DAYS = 30
ANONYMOUS_USER_ID = 1
LEGACY_USER_ID = 2
TOKENIZER_MODEL = os.environ.get("TOKENIZER_MODEL", "answerdotai/ModernBERT-base")
LOCK_TIMEOUT_MINUTES = int(os.environ.get("LOCK_TIMEOUT_MINUTES", "30"))
ADMIN_USER = os.environ.get("ADMIN_USER", "")  # Username to bootstrap as admin

# Role constants
ROLE_ADMIN = "admin"
ROLE_ANNOTATOR = "annotator"

# Agreement configuration
CONSENSUS_THRESHOLD = int(os.environ.get("CONSENSUS_THRESHOLD", "10"))
AGREEMENT_HIGH_THRESHOLD = float(os.environ.get("AGREEMENT_HIGH_THRESHOLD", "0.8"))

# ============================================================================
# Label Schema Configuration
# ============================================================================

# System-defined labels organized by category
# These are the "official" labels that get is_custom=False
# Any label not in this list will be marked is_custom=True
SYSTEM_LABELS = {
    "difficulty": {
        "description": "Difficulty dimension labels for complexity analysis",
        "labels": [
            {"name": "reasoning", "color": "#3b82f6", "description": "Requires logical inference"},
            {"name": "creativity", "color": "#8b5cf6", "description": "Requires imaginative interpretation"},
            {"name": "domain_knowledge", "color": "#06b6d4", "description": "Requires specialized expertise"},
            {"name": "contextual", "color": "#22c55e", "description": "Depends on implicit context"},
            {"name": "constraints", "color": "#eab308", "description": "Multiple conditions to track"},
            {"name": "ambiguity", "color": "#f97316", "description": "Answer is debatable"},
        ]
    },
    "nli": {
        "description": "Natural Language Inference relation labels",
        "labels": [
            {"name": "entailment", "color": "#4ade80", "description": "Marks tokens supporting entailment"},
            {"name": "neutral", "color": "#fbbf24", "description": "Marks tokens indicating neutrality"},
            {"name": "contradiction", "color": "#f87171", "description": "Marks tokens showing contradiction"},
        ]
    }
}

# Flattened set of all system label names for quick lookup
SYSTEM_LABEL_NAMES = set()
for category in SYSTEM_LABELS.values():
    for label in category["labels"]:
        SYSTEM_LABEL_NAMES.add(label["name"])


def is_system_label(label_name: str) -> bool:
    """Check if a label name is a system-defined label."""
    return label_name.lower() in SYSTEM_LABEL_NAMES


def get_label_schema() -> dict:
    """Get the complete label schema with metadata."""
    return {
        "categories": SYSTEM_LABELS,
        "system_labels": sorted(SYSTEM_LABEL_NAMES),
        "allow_custom": True,
        "custom_tracking": True,
    }


# Global tokenizer - initialized at startup
_tokenizer = None

# ============================================================================
# API Documentation
# ============================================================================

API_DESCRIPTION = """
# NLI Span Labeler API

A multi-user annotation tool for Natural Language Inference (NLI) examples with span-level labeling.

## Features

- **Span-level annotation**: Select specific tokens in premise/hypothesis pairs
- **Multiple labels per token**: Annotate tokens with multiple semantic labels
- **Complexity scoring**: Rate examples on 6 difficulty dimensions (1-100 scale)
- **Multi-user support**: Session-based authentication with per-user annotation tracking
- **Role-based access**: Admin and annotator roles with protected endpoints
- **WordPiece tokenization**: Uses ModernBERT tokenizer for model-aligned annotations
- **Label schema enforcement**: System labels tracked separately from custom labels
- **Concurrent annotation**: Example locking prevents duplicate work
- **Inter-annotator agreement**: Track consensus and annotator reliability

## Authentication

Most endpoints require authentication via session cookie. Use the `/api/auth/login` or 
`/api/auth/register` endpoints to obtain a session.

Set `ANONYMOUS_MODE=1` environment variable to disable authentication for local single-user usage.

## Admin Mode

Set `ADMIN_USER=username` environment variable to bootstrap a user as admin on startup.
Admins have access to additional endpoints under `/api/admin/` for user management and
system-wide operations.

## Example Locking

When you request an example via `/api/next`, it's automatically locked to you for 30 minutes.
Other users won't receive the same example. Locks are released when you save or skip, or
you can explicitly release via `/api/lock/release/{example_id}`.

## Question Pools

Examples are categorized into pools:
- **test**: High-consensus examples used for calibrating new annotators
- **building**: Being actively annotated, need more responses
- **zero_entry**: Unannotated examples

## Label Schema

The system defines official labels in two categories:
- **Difficulty**: reasoning, creativity, domain_knowledge, contextual, constraints, ambiguity
- **NLI**: entailment, neutral, contradiction

Custom labels are allowed but tracked with `is_custom=True` for data quality analysis.

## Datasets

Place JSONL files in `data/nli/` directory. Expected format:
```json
{"id": "example_1", "premise": "...", "hypothesis": "...", "label": 0, "label_text": "entailment"}
```
"""

TAGS_METADATA = [
    {
        "name": "Authentication",
        "description": "User registration, login, logout, and session management.",
    },
    {
        "name": "Admin",
        "description": "Admin-only endpoints for user management and system configuration. Requires admin role.",
    },
    {
        "name": "Examples",
        "description": "Retrieve NLI examples for annotation. Examples are served per-user to avoid duplicate annotations.",
    },
    {
        "name": "Annotation",
        "description": "Submit span labels and complexity scores for examples.",
    },
    {
        "name": "Locking",
        "description": "Manage example locks for concurrent annotation.",
    },
    {
        "name": "Agreement",
        "description": "Inter-annotator agreement metrics and question pool management.",
    },
    {
        "name": "Statistics",
        "description": "View annotation statistics, export data, and manage annotators.",
    },
    {
        "name": "System",
        "description": "System information endpoints (tokenizer, datasets, labels, health).",
    },
]

app = FastAPI(
    title="NLI Span Labeler API",
    description=API_DESCRIPTION,
    version="1.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=TAGS_METADATA,
    contact={
        "name": "GoblinCorps",
        "url": "https://github.com/GoblinCorps/nli-span-labeler",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# ============================================================================
# Tokenization
# ============================================================================

def get_tokenizer():
    """Get the global tokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        print(f"Loading tokenizer: {TOKENIZER_MODEL}")
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        print(f"Tokenizer loaded: {type(_tokenizer).__name__}")
    return _tokenizer


def tokenize_text(text: str) -> list[dict]:
    """
    Tokenize text using the target model's tokenizer.
    
    Returns tokens with character offsets that map back to the original text.
    WordPiece subword tokens (##xxx) are marked with is_subword=True.
    
    This ensures annotators see exactly what the model will see during training.
    """
    tokenizer = get_tokenizer()
    
    # Tokenize with offset mapping
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,  # Don't add [CLS], [SEP] for display
    )
    
    tokens = encoding.tokens()
    offsets = encoding.offset_mapping
    
    words = []
    for i, (token, (char_start, char_end)) in enumerate(zip(tokens, offsets)):
        # Detect WordPiece subword tokens
        is_subword = token.startswith("##")
        display_text = token[2:] if is_subword else token  # Strip ## for display
        
        words.append({
            "index": i,
            "text": display_text,
            "token": token,  # Original token including ## prefix
            "char_start": char_start,
            "char_end": char_end,
            "is_subword": is_subword,
        })
    
    return words


# ============================================================================
# Database Setup
# ============================================================================

def init_db():
    """Initialize SQLite database with user support."""
    with get_db() as conn:
        # Check if we need to migrate (do users table exist?)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        users_exist = cursor.fetchone() is not None
        
        # Check if labels table exists (for migration)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='labels'"
        )
        labels_exist = cursor.fetchone() is not None
        
        # Check if annotator_id column exists in labels
        needs_user_migration = False
        needs_custom_migration = False
        needs_role_migration = False
        if labels_exist:
            cursor = conn.execute("PRAGMA table_info(labels)")
            columns = [row[1] for row in cursor.fetchall()]
            needs_user_migration = 'annotator_id' not in columns
            needs_custom_migration = 'is_custom' not in columns
        
        # Check if role column exists in users
        if users_exist:
            cursor = conn.execute("PRAGMA table_info(users)")
            columns = [row[1] for row in cursor.fetchall()]
            needs_role_migration = 'role' not in columns

        # Check if examples table needs pool_status column
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='examples'"
        )
        examples_exist = cursor.fetchone() is not None
        needs_pool_migration = False
        if examples_exist:
            cursor = conn.execute("PRAGMA table_info(examples)")
            columns = [row[1] for row in cursor.fetchall()]
            needs_pool_migration = 'pool_status' not in columns

        # Create users table first (needed for foreign keys)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                display_name TEXT,
                role TEXT NOT NULL DEFAULT 'annotator',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        """)

        # Handle role migration for existing databases
        if needs_role_migration:
            migrate_add_role(conn)

        # Create system users if they don't exist
        conn.execute("""
            INSERT OR IGNORE INTO users (id, username, display_name, role)
            VALUES (?, 'anonymous', 'Anonymous User', 'annotator')
        """, (ANONYMOUS_USER_ID,))
        
        conn.execute("""
            INSERT OR IGNORE INTO users (id, username, display_name, role)
            VALUES (?, 'legacy', 'Legacy Annotations', 'annotator')
        """, (LEGACY_USER_ID,))

        # Bootstrap admin user if configured
        if ADMIN_USER:
            bootstrap_admin_user(conn)

        # Create examples table
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS examples (
                id TEXT PRIMARY KEY,
                dataset TEXT NOT NULL,
                premise TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                gold_label INTEGER,
                gold_label_text TEXT,
                source_file TEXT,
                pool_status TEXT DEFAULT 'zero_entry',
                loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_examples_dataset ON examples(dataset);
            CREATE INDEX IF NOT EXISTS idx_examples_pool ON examples(pool_status);
        """)

        # Add pool_status to existing examples table if needed
        if needs_pool_migration:
            migrate_add_pool_status(conn)

        # Create example_locks table for concurrency handling
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS example_locks (
                example_id TEXT PRIMARY KEY,
                locked_by INTEGER NOT NULL,
                locked_until TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (example_id) REFERENCES examples(id) ON DELETE CASCADE,
                FOREIGN KEY (locked_by) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_locks_user ON example_locks(locked_by);
            CREATE INDEX IF NOT EXISTS idx_locks_until ON example_locks(locked_until);
        """)

        # Create agreement tracking tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS example_agreement (
                example_id TEXT PRIMARY KEY,
                annotation_count INTEGER DEFAULT 0,
                agreement_score REAL,
                complexity_agreement REAL,
                span_agreement REAL,
                last_calculated TIMESTAMP,
                FOREIGN KEY (example_id) REFERENCES examples(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS annotator_agreement (
                user_id INTEGER PRIMARY KEY,
                total_annotations INTEGER DEFAULT 0,
                agreement_with_consensus REAL,
                complexity_agreement REAL,
                span_agreement REAL,
                last_calculated TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_example_agreement_score ON example_agreement(agreement_score);
            CREATE INDEX IF NOT EXISTS idx_annotator_agreement_score ON annotator_agreement(agreement_with_consensus);
        """)

        # Handle migration for existing tables
        if needs_user_migration and labels_exist:
            # Migrate existing data (add annotator_id)
            migrate_existing_data(conn)
        elif needs_custom_migration and labels_exist:
            # Add is_custom column to existing table
            migrate_add_is_custom(conn)
        else:
            # Create new tables with all columns
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    example_id TEXT NOT NULL,
                    annotator_id INTEGER NOT NULL DEFAULT 1,
                    label_name TEXT NOT NULL,
                    label_color TEXT,
                    is_custom BOOLEAN NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (example_id) REFERENCES examples(id),
                    FOREIGN KEY (annotator_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS span_selections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    word_index INTEGER NOT NULL,
                    word_text TEXT NOT NULL,
                    char_start INTEGER,
                    char_end INTEGER,
                    FOREIGN KEY (label_id) REFERENCES labels(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS complexity_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    example_id TEXT NOT NULL,
                    annotator_id INTEGER NOT NULL DEFAULT 1,
                    reasoning INTEGER,
                    creativity INTEGER,
                    domain_knowledge INTEGER,
                    contextual INTEGER,
                    constraints INTEGER,
                    ambiguity INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (example_id) REFERENCES examples(id),
                    FOREIGN KEY (annotator_id) REFERENCES users(id),
                    UNIQUE(example_id, annotator_id)
                );

                CREATE TABLE IF NOT EXISTS skipped (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    example_id TEXT NOT NULL,
                    annotator_id INTEGER NOT NULL DEFAULT 1,
                    skipped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (example_id) REFERENCES examples(id),
                    FOREIGN KEY (annotator_id) REFERENCES users(id),
                    UNIQUE(example_id, annotator_id)
                );

                CREATE INDEX IF NOT EXISTS idx_labels_example ON labels(example_id);
                CREATE INDEX IF NOT EXISTS idx_labels_annotator ON labels(annotator_id);
                CREATE INDEX IF NOT EXISTS idx_labels_custom ON labels(is_custom);
                CREATE INDEX IF NOT EXISTS idx_spans_label ON span_selections(label_id);
                CREATE INDEX IF NOT EXISTS idx_complexity_annotator ON complexity_scores(annotator_id);
                CREATE INDEX IF NOT EXISTS idx_skipped_annotator ON skipped(annotator_id);
            """)


def migrate_add_role(conn):
    """Add role column to existing users table."""
    print("Adding role column to users table...")
    
    # Add column with default value
    conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'annotator'")
    
    print("Role column added. All existing users set to 'annotator' role.")


def bootstrap_admin_user(conn):
    """Bootstrap the configured admin user."""
    if not ADMIN_USER:
        return
    
    # Check if user exists
    row = conn.execute(
        "SELECT id, role FROM users WHERE username = ?",
        (ADMIN_USER,)
    ).fetchone()
    
    if row:
        # User exists - update role to admin if not already
        if row["role"] != ROLE_ADMIN:
            conn.execute(
                "UPDATE users SET role = ? WHERE id = ?",
                (ROLE_ADMIN, row["id"])
            )
            print(f"User '{ADMIN_USER}' promoted to admin role.")
        else:
            print(f"User '{ADMIN_USER}' already has admin role.")
    else:
        # User doesn't exist yet - they'll get admin role when they register
        print(f"Admin user '{ADMIN_USER}' not found. Will be granted admin role on registration.")


def migrate_existing_data(conn):
    """Migrate existing tables to include annotator_id."""
    print("Migrating existing database to multi-user schema...")
    
    # Rename old tables
    conn.executescript("""
        ALTER TABLE labels RENAME TO labels_old;
        ALTER TABLE complexity_scores RENAME TO complexity_scores_old;
        ALTER TABLE skipped RENAME TO skipped_old;
    """)
    
    # Create new tables with annotator_id and is_custom
    conn.executescript("""
        CREATE TABLE labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            example_id TEXT NOT NULL,
            annotator_id INTEGER NOT NULL DEFAULT 2,
            label_name TEXT NOT NULL,
            label_color TEXT,
            is_custom BOOLEAN NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (example_id) REFERENCES examples(id),
            FOREIGN KEY (annotator_id) REFERENCES users(id)
        );

        CREATE TABLE complexity_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            example_id TEXT NOT NULL,
            annotator_id INTEGER NOT NULL DEFAULT 2,
            reasoning INTEGER,
            creativity INTEGER,
            domain_knowledge INTEGER,
            contextual INTEGER,
            constraints INTEGER,
            ambiguity INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (example_id) REFERENCES examples(id),
            FOREIGN KEY (annotator_id) REFERENCES users(id),
            UNIQUE(example_id, annotator_id)
        );

        CREATE TABLE skipped (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            example_id TEXT NOT NULL,
            annotator_id INTEGER NOT NULL DEFAULT 2,
            skipped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (example_id) REFERENCES examples(id),
            FOREIGN KEY (annotator_id) REFERENCES users(id),
            UNIQUE(example_id, annotator_id)
        );
    """)
    
    # Copy data with legacy user ID, determining is_custom for each label
    old_labels = conn.execute("SELECT * FROM labels_old").fetchall()
    for label in old_labels:
        is_custom = 0 if is_system_label(label["label_name"]) else 1
        conn.execute("""
            INSERT INTO labels (id, example_id, annotator_id, label_name, label_color, is_custom, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (label["id"], label["example_id"], LEGACY_USER_ID, 
              label["label_name"], label["label_color"], is_custom, label["created_at"]))
    
    conn.execute("""
        INSERT INTO complexity_scores (example_id, annotator_id, reasoning, creativity, 
                                       domain_knowledge, contextual, constraints, ambiguity, created_at)
        SELECT example_id, ?, reasoning, creativity, domain_knowledge, contextual, 
               constraints, ambiguity, created_at
        FROM complexity_scores_old
    """, (LEGACY_USER_ID,))
    
    conn.execute("""
        INSERT INTO skipped (example_id, annotator_id, skipped_at)
        SELECT example_id, ?, skipped_at
        FROM skipped_old
    """, (LEGACY_USER_ID,))
    
    # Drop old tables
    conn.executescript("""
        DROP TABLE labels_old;
        DROP TABLE complexity_scores_old;
        DROP TABLE skipped_old;
    """)
    
    # Recreate indexes
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_labels_example ON labels(example_id);
        CREATE INDEX IF NOT EXISTS idx_labels_annotator ON labels(annotator_id);
        CREATE INDEX IF NOT EXISTS idx_labels_custom ON labels(is_custom);
        CREATE INDEX IF NOT EXISTS idx_spans_label ON span_selections(label_id);
        CREATE INDEX IF NOT EXISTS idx_complexity_annotator ON complexity_scores(annotator_id);
        CREATE INDEX IF NOT EXISTS idx_skipped_annotator ON skipped(annotator_id);
    """)
    
    print("Migration complete. Existing annotations assigned to 'legacy' user.")


def migrate_add_is_custom(conn):
    """Add is_custom column to existing labels table."""
    print("Adding is_custom column to labels table...")
    
    # Add column
    conn.execute("ALTER TABLE labels ADD COLUMN is_custom BOOLEAN NOT NULL DEFAULT 0")
    
    # Update existing labels based on SYSTEM_LABELS
    # Mark labels not in system labels as custom
    all_labels = conn.execute("SELECT id, label_name FROM labels").fetchall()
    for label in all_labels:
        is_custom = 0 if is_system_label(label["label_name"]) else 1
        conn.execute("UPDATE labels SET is_custom = ? WHERE id = ?", (is_custom, label["id"]))
    
    # Add index
    conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_custom ON labels(is_custom)")
    
    print(f"Migration complete. Updated {len(all_labels)} labels.")


def migrate_add_pool_status(conn):
    """Add pool_status column to existing examples table."""
    print("Adding pool_status column to examples table...")
    
    # Add column with default value
    conn.execute("ALTER TABLE examples ADD COLUMN pool_status TEXT DEFAULT 'zero_entry'")
    
    # Update existing examples based on annotation count
    # Examples with annotations go to 'building' pool
    conn.execute("""
        UPDATE examples SET pool_status = 'building'
        WHERE id IN (SELECT DISTINCT example_id FROM complexity_scores)
    """)
    
    # Add index
    conn.execute("CREATE INDEX IF NOT EXISTS idx_examples_pool ON examples(pool_status)")
    
    print("Migration complete. Existing annotated examples moved to 'building' pool.")


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
# Agreement Calculation Helpers
# ============================================================================

def calculate_complexity_agreement(scores: list[dict]) -> float:
    """
    Calculate agreement for complexity scores across annotators.
    
    Uses normalized mean absolute deviation - lower deviation = higher agreement.
    Returns a score from 0 (no agreement) to 1 (perfect agreement).
    """
    if len(scores) < 2:
        return None
    
    dimensions = ['reasoning', 'creativity', 'domain_knowledge', 'contextual', 'constraints', 'ambiguity']
    total_agreement = 0
    valid_dimensions = 0
    
    for dim in dimensions:
        values = [s[dim] for s in scores if s.get(dim) is not None]
        if len(values) < 2:
            continue
        
        # Calculate mean and mean absolute deviation
        mean_val = sum(values) / len(values)
        mad = sum(abs(v - mean_val) for v in values) / len(values)
        
        # Normalize to 0-1 scale (max deviation is 50 on a 1-100 scale)
        # Agreement = 1 - (MAD / 50)
        dimension_agreement = max(0, 1 - (mad / 50))
        total_agreement += dimension_agreement
        valid_dimensions += 1
    
    if valid_dimensions == 0:
        return None
    
    return total_agreement / valid_dimensions


def calculate_span_agreement(annotations: list[dict]) -> float:
    """
    Calculate agreement for span selections across annotators.
    
    Uses Jaccard similarity - intersection over union of selected spans.
    Returns a score from 0 (no overlap) to 1 (perfect agreement).
    """
    if len(annotations) < 2:
        return None
    
    # Extract span sets per annotator, per label
    # Format: {annotator_id: {label_name: set((source, word_index), ...)}}
    annotator_spans = defaultdict(lambda: defaultdict(set))
    
    for ann in annotations:
        annotator_id = ann['annotator_id']
        for label in ann.get('labels', []):
            label_name = label['label_name']
            for span in label.get('spans', []):
                annotator_spans[annotator_id][label_name].add(
                    (span['source'], span['word_index'])
                )
    
    if len(annotator_spans) < 2:
        return None
    
    # Calculate pairwise Jaccard similarity for each label
    annotator_ids = list(annotator_spans.keys())
    all_labels = set()
    for spans in annotator_spans.values():
        all_labels.update(spans.keys())
    
    if not all_labels:
        return 1.0  # No spans = perfect agreement (both empty)
    
    total_similarity = 0
    comparisons = 0
    
    for i, aid1 in enumerate(annotator_ids):
        for aid2 in annotator_ids[i+1:]:
            label_similarities = []
            for label in all_labels:
                set1 = annotator_spans[aid1].get(label, set())
                set2 = annotator_spans[aid2].get(label, set())
                
                if not set1 and not set2:
                    # Both empty = perfect agreement for this label
                    label_similarities.append(1.0)
                elif not set1 or not set2:
                    # One empty, one not = no agreement
                    label_similarities.append(0.0)
                else:
                    # Jaccard: |intersection| / |union|
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    label_similarities.append(intersection / union)
            
            if label_similarities:
                total_similarity += sum(label_similarities) / len(label_similarities)
                comparisons += 1
    
    if comparisons == 0:
        return None
    
    return total_similarity / comparisons


def calculate_example_agreement(conn, example_id: str) -> dict:
    """
    Calculate comprehensive agreement metrics for an example.
    
    Returns dict with:
    - annotation_count: number of annotators
    - agreement_score: combined agreement (0-1)
    - complexity_agreement: agreement on complexity scores
    - span_agreement: agreement on span selections
    """
    # Get all complexity scores for this example
    scores = conn.execute("""
        SELECT annotator_id, reasoning, creativity, domain_knowledge,
               contextual, constraints, ambiguity
        FROM complexity_scores
        WHERE example_id = ?
    """, (example_id,)).fetchall()
    
    scores = [dict(s) for s in scores]
    annotation_count = len(scores)
    
    if annotation_count < 2:
        return {
            'annotation_count': annotation_count,
            'agreement_score': None,
            'complexity_agreement': None,
            'span_agreement': None,
        }
    
    # Calculate complexity agreement
    complexity_agreement = calculate_complexity_agreement(scores)
    
    # Get all span annotations for this example
    annotations = []
    for score in scores:
        annotator_id = score['annotator_id']
        labels = conn.execute("""
            SELECT l.label_name, l.label_color,
                   s.source, s.word_index, s.word_text
            FROM labels l
            LEFT JOIN span_selections s ON s.label_id = l.id
            WHERE l.example_id = ? AND l.annotator_id = ?
        """, (example_id, annotator_id)).fetchall()
        
        # Group by label
        label_dict = defaultdict(lambda: {'label_name': '', 'spans': []})
        for row in labels:
            label_name = row['label_name']
            label_dict[label_name]['label_name'] = label_name
            if row['source']:  # Has span data
                label_dict[label_name]['spans'].append({
                    'source': row['source'],
                    'word_index': row['word_index'],
                })
        
        annotations.append({
            'annotator_id': annotator_id,
            'labels': list(label_dict.values())
        })
    
    # Calculate span agreement
    span_agreement = calculate_span_agreement(annotations)
    
    # Combined agreement: weighted average (complexity more important for NLI)
    weights = {'complexity': 0.6, 'span': 0.4}
    
    combined_parts = []
    if complexity_agreement is not None:
        combined_parts.append(('complexity', complexity_agreement))
    if span_agreement is not None:
        combined_parts.append(('span', span_agreement))
    
    if combined_parts:
        total_weight = sum(weights[p[0]] for p in combined_parts)
        agreement_score = sum(weights[p[0]] * p[1] for p in combined_parts) / total_weight
    else:
        agreement_score = None
    
    return {
        'annotation_count': annotation_count,
        'agreement_score': agreement_score,
        'complexity_agreement': complexity_agreement,
        'span_agreement': span_agreement,
    }


def update_example_agreement(conn, example_id: str):
    """
    Recalculate and store agreement metrics for an example.
    Also updates pool status based on agreement.
    """
    metrics = calculate_example_agreement(conn, example_id)
    
    # Store in example_agreement table
    conn.execute("""
        INSERT INTO example_agreement (example_id, annotation_count, agreement_score,
                                       complexity_agreement, span_agreement, last_calculated)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(example_id) DO UPDATE SET
            annotation_count = excluded.annotation_count,
            agreement_score = excluded.agreement_score,
            complexity_agreement = excluded.complexity_agreement,
            span_agreement = excluded.span_agreement,
            last_calculated = CURRENT_TIMESTAMP
    """, (example_id, metrics['annotation_count'], metrics['agreement_score'],
          metrics['complexity_agreement'], metrics['span_agreement']))
    
    # Update pool status
    update_example_pool_status(conn, example_id, metrics)
    
    return metrics


def update_example_pool_status(conn, example_id: str, metrics: dict):
    """
    Update pool status based on annotation count and agreement.
    
    Pool transitions:
    - zero_entry -> building: First annotation
    - building -> test: Reaches threshold AND high agreement
    - test -> building: If agreement drops (shouldn't happen often)
    """
    annotation_count = metrics['annotation_count']
    agreement_score = metrics['agreement_score']
    
    if annotation_count == 0:
        new_status = 'zero_entry'
    elif annotation_count < CONSENSUS_THRESHOLD:
        new_status = 'building'
    elif agreement_score is not None and agreement_score >= AGREEMENT_HIGH_THRESHOLD:
        new_status = 'test'
    else:
        # Has enough annotations but agreement is low or unknown
        new_status = 'building'
    
    conn.execute("""
        UPDATE examples SET pool_status = ? WHERE id = ?
    """, (new_status, example_id))


def calculate_annotator_agreement(conn, user_id: int) -> dict:
    """
    Calculate agreement metrics for a specific annotator.
    
    Measures how often they agree with consensus on examples
    that have reached the consensus threshold.
    """
    # Get all examples this user has annotated that have consensus
    user_examples = conn.execute("""
        SELECT c.example_id, ea.agreement_score, ea.annotation_count
        FROM complexity_scores c
        JOIN example_agreement ea ON c.example_id = ea.example_id
        WHERE c.annotator_id = ? AND ea.annotation_count >= ?
    """, (user_id, CONSENSUS_THRESHOLD)).fetchall()
    
    if not user_examples:
        return {
            'total_annotations': 0,
            'agreement_with_consensus': None,
            'complexity_agreement': None,
            'span_agreement': None,
        }
    
    total_annotations = len(user_examples)
    
    # For each example, compare this user's annotation to the consensus
    complexity_agreements = []
    span_agreements = []
    
    for row in user_examples:
        example_id = row['example_id']
        
        # Get this user's scores
        user_scores = conn.execute("""
            SELECT reasoning, creativity, domain_knowledge,
                   contextual, constraints, ambiguity
            FROM complexity_scores
            WHERE example_id = ? AND annotator_id = ?
        """, (example_id, user_id)).fetchone()
        
        if not user_scores:
            continue
        
        # Get average scores (consensus) from other annotators
        other_scores = conn.execute("""
            SELECT AVG(reasoning) as reasoning, AVG(creativity) as creativity,
                   AVG(domain_knowledge) as domain_knowledge, AVG(contextual) as contextual,
                   AVG(constraints) as constraints, AVG(ambiguity) as ambiguity
            FROM complexity_scores
            WHERE example_id = ? AND annotator_id != ?
        """, (example_id, user_id)).fetchone()
        
        if other_scores and other_scores['reasoning'] is not None:
            # Calculate agreement with consensus for complexity
            dimensions = ['reasoning', 'creativity', 'domain_knowledge', 'contextual', 'constraints', 'ambiguity']
            dim_agreements = []
            for dim in dimensions:
                user_val = user_scores[dim]
                consensus_val = other_scores[dim]
                if user_val is not None and consensus_val is not None:
                    # Agreement = 1 - normalized difference
                    diff = abs(user_val - consensus_val) / 100
                    dim_agreements.append(1 - diff)
            
            if dim_agreements:
                complexity_agreements.append(sum(dim_agreements) / len(dim_agreements))
        
        # Get this user's span selections
        user_spans = conn.execute("""
            SELECT l.label_name, s.source, s.word_index
            FROM labels l
            LEFT JOIN span_selections s ON s.label_id = l.id
            WHERE l.example_id = ? AND l.annotator_id = ?
        """, (example_id, user_id)).fetchall()
        
        # Get other annotators' span selections
        other_spans = conn.execute("""
            SELECT l.label_name, s.source, s.word_index, l.annotator_id
            FROM labels l
            LEFT JOIN span_selections s ON s.label_id = l.id
            WHERE l.example_id = ? AND l.annotator_id != ?
        """, (example_id, user_id)).fetchall()
        
        if other_spans:
            # Build user's span set
            user_span_set = defaultdict(set)
            for row in user_spans:
                if row['source']:
                    user_span_set[row['label_name']].add((row['source'], row['word_index']))
            
            # Build consensus span set (majority vote)
            span_votes = defaultdict(lambda: defaultdict(int))
            other_annotator_count = len(set(r['annotator_id'] for r in other_spans))
            
            for row in other_spans:
                if row['source']:
                    span_votes[row['label_name']][(row['source'], row['word_index'])] += 1
            
            # Consensus = spans selected by majority
            consensus_spans = defaultdict(set)
            majority_threshold = other_annotator_count / 2
            for label, spans in span_votes.items():
                for span, count in spans.items():
                    if count >= majority_threshold:
                        consensus_spans[label].add(span)
            
            # Calculate Jaccard with consensus
            all_labels = set(user_span_set.keys()) | set(consensus_spans.keys())
            if all_labels:
                label_agreements = []
                for label in all_labels:
                    user_set = user_span_set.get(label, set())
                    consensus_set = consensus_spans.get(label, set())
                    
                    if not user_set and not consensus_set:
                        label_agreements.append(1.0)
                    elif not user_set or not consensus_set:
                        label_agreements.append(0.0)
                    else:
                        intersection = len(user_set & consensus_set)
                        union = len(user_set | consensus_set)
                        label_agreements.append(intersection / union)
                
                span_agreements.append(sum(label_agreements) / len(label_agreements))
    
    # Aggregate results
    complexity_agreement = sum(complexity_agreements) / len(complexity_agreements) if complexity_agreements else None
    span_agreement = sum(span_agreements) / len(span_agreements) if span_agreements else None
    
    # Combined agreement
    if complexity_agreement is not None and span_agreement is not None:
        agreement_with_consensus = 0.6 * complexity_agreement + 0.4 * span_agreement
    elif complexity_agreement is not None:
        agreement_with_consensus = complexity_agreement
    elif span_agreement is not None:
        agreement_with_consensus = span_agreement
    else:
        agreement_with_consensus = None
    
    return {
        'total_annotations': total_annotations,
        'agreement_with_consensus': agreement_with_consensus,
        'complexity_agreement': complexity_agreement,
        'span_agreement': span_agreement,
    }


def update_annotator_agreement(conn, user_id: int):
    """Recalculate and store agreement metrics for an annotator."""
    metrics = calculate_annotator_agreement(conn, user_id)
    
    conn.execute("""
        INSERT INTO annotator_agreement (user_id, total_annotations, agreement_with_consensus,
                                         complexity_agreement, span_agreement, last_calculated)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET
            total_annotations = excluded.total_annotations,
            agreement_with_consensus = excluded.agreement_with_consensus,
            complexity_agreement = excluded.complexity_agreement,
            span_agreement = excluded.span_agreement,
            last_calculated = CURRENT_TIMESTAMP
    """, (user_id, metrics['total_annotations'], metrics['agreement_with_consensus'],
          metrics['complexity_agreement'], metrics['span_agreement']))
    
    return metrics


# ============================================================================
# Locking Helpers
# ============================================================================

def acquire_lock(conn, example_id: str, user_id: int) -> Optional[datetime]:
    """
    Acquire a lock on an example for a user.
    
    Returns the lock expiry time if successful, None if already locked by another user.
    If already locked by the same user, extends the lock.
    """
    now = datetime.now()
    lock_until = now + timedelta(minutes=LOCK_TIMEOUT_MINUTES)
    
    # Check existing lock
    existing = conn.execute("""
        SELECT locked_by, locked_until FROM example_locks WHERE example_id = ?
    """, (example_id,)).fetchone()
    
    if existing:
        # Lock exists - check if it's expired or owned by this user
        existing_until = datetime.fromisoformat(existing["locked_until"])
        if existing["locked_by"] == user_id:
            # User's own lock - extend it
            conn.execute("""
                UPDATE example_locks SET locked_until = ? WHERE example_id = ?
            """, (lock_until.isoformat(), example_id))
            return lock_until
        elif existing_until > now:
            # Someone else's active lock
            return None
        else:
            # Expired lock - take it over
            conn.execute("""
                UPDATE example_locks SET locked_by = ?, locked_until = ?, created_at = ?
                WHERE example_id = ?
            """, (user_id, lock_until.isoformat(), now.isoformat(), example_id))
            return lock_until
    else:
        # No lock - create one
        conn.execute("""
            INSERT INTO example_locks (example_id, locked_by, locked_until)
            VALUES (?, ?, ?)
        """, (example_id, user_id, lock_until.isoformat()))
        return lock_until


def release_lock(conn, example_id: str, user_id: int) -> bool:
    """
    Release a lock on an example.
    
    Returns True if lock was released, False if user didn't own the lock.
    """
    result = conn.execute("""
        DELETE FROM example_locks WHERE example_id = ? AND locked_by = ?
    """, (example_id, user_id))
    return result.rowcount > 0


def get_lock_status(conn, example_id: str) -> Optional[dict]:
    """Get the current lock status for an example."""
    row = conn.execute("""
        SELECT l.locked_by, l.locked_until, u.username, u.display_name
        FROM example_locks l
        JOIN users u ON l.locked_by = u.id
        WHERE l.example_id = ?
    """, (example_id,)).fetchone()
    
    if not row:
        return None
    
    locked_until = datetime.fromisoformat(row["locked_until"])
    now = datetime.now()
    
    if locked_until <= now:
        # Expired - clean it up
        conn.execute("DELETE FROM example_locks WHERE example_id = ?", (example_id,))
        return None
    
    return {
        "locked_by": row["locked_by"],
        "locked_by_username": row["username"],
        "locked_by_display_name": row["display_name"],
        "locked_until": row["locked_until"],
        "expires_in_seconds": int((locked_until - now).total_seconds())
    }


def cleanup_expired_locks(conn):
    """Remove all expired locks."""
    now = datetime.now().isoformat()
    result = conn.execute("""
        DELETE FROM example_locks WHERE locked_until < ?
    """, (now,))
    return result.rowcount


# ============================================================================
# Authentication Helpers
# ============================================================================

def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    if not password_hash or ':' not in password_hash:
        return False
    salt, hashed = password_hash.split(':', 1)
    return hashlib.sha256((salt + password).encode()).hexdigest() == hashed


def create_session(user_id: int) -> str:
    """Create a new session for a user."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(days=SESSION_EXPIRY_DAYS)
    
    with get_db() as conn:
        conn.execute("""
            INSERT INTO sessions (token, user_id, expires_at)
            VALUES (?, ?, ?)
        """, (token, user_id, expires_at))
        
        # Update last_seen
        conn.execute("""
            UPDATE users SET last_seen = CURRENT_TIMESTAMP WHERE id = ?
        """, (user_id,))
    
    return token


def get_user_from_session(token: str) -> Optional[dict]:
    """Get user from session token."""
    if not token:
        return None
        
    with get_db() as conn:
        row = conn.execute("""
            SELECT u.id, u.username, u.display_name, u.role, u.created_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ? AND s.expires_at > CURRENT_TIMESTAMP
        """, (token,)).fetchone()
        
        if row:
            # Update last_seen
            conn.execute("""
                UPDATE users SET last_seen = CURRENT_TIMESTAMP WHERE id = ?
            """, (row["id"],))
            
            return {
                "id": row["id"],
                "username": row["username"],
                "display_name": row["display_name"],
                "role": row["role"],
                "created_at": row["created_at"]
            }
    
    return None


def delete_session(token: str):
    """Delete a session."""
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


async def get_current_user(request: Request) -> dict:
    """Dependency to get current user from session cookie."""
    if ANONYMOUS_MODE:
        return {
            "id": ANONYMOUS_USER_ID,
            "username": "anonymous",
            "display_name": "Anonymous User",
            "role": ROLE_ANNOTATOR,
            "is_anonymous": True
        }
    
    token = request.cookies.get("session")
    user = get_user_from_session(token)
    
    if not user:
        raise HTTPException(401, "Not authenticated. Please log in.")
    
    return user


async def get_optional_user(request: Request) -> Optional[dict]:
    """Dependency to get current user, or None if not authenticated."""
    if ANONYMOUS_MODE:
        return {
            "id": ANONYMOUS_USER_ID,
            "username": "anonymous",
            "display_name": "Anonymous User",
            "role": ROLE_ANNOTATOR,
            "is_anonymous": True
        }
    
    token = request.cookies.get("session")
    return get_user_from_session(token)


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Dependency to require admin role."""
    if user.get("role") != ROLE_ADMIN:
        raise HTTPException(403, "Admin access required.")
    return user


# ============================================================================
# Pydantic Models
# ============================================================================

class SpanSelection(BaseModel):
    """A single span selection within an example."""
    source: str = Field(..., description="Text source: 'premise' or 'hypothesis'", example="premise")
    word_index: int = Field(..., description="Token index within the source text", example=3)
    word_text: str = Field(..., description="The selected token text", example="running")
    char_start: Optional[int] = Field(None, description="Character start offset in original text")
    char_end: Optional[int] = Field(None, description="Character end offset in original text")


class LabelSubmission(BaseModel):
    """A label with its associated span selections."""
    label_name: str = Field(..., description="Name of the label", example="reasoning")
    label_color: str = Field(..., description="Hex color for the label", example="#3b82f6")
    spans: list[SpanSelection] = Field(..., description="List of selected spans for this label")


class AnnotationSubmission(BaseModel):
    """Complete annotation submission for an example."""
    example_id: str = Field(..., description="ID of the example being annotated", example="snli_12345")
    labels: list[LabelSubmission] = Field(..., description="List of labels with span selections")
    complexity_scores: Optional[dict] = Field(
        None,
        description="Complexity scores (1-100) for dimensions: reasoning, creativity, domain_knowledge, contextual, constraints, ambiguity",
        example={"reasoning": 75, "creativity": 40, "domain_knowledge": 50, "contextual": 60, "constraints": 30, "ambiguity": 45}
    )


class ExampleResponse(BaseModel):
    """Response containing an NLI example with tokenized text."""
    id: str = Field(..., description="Unique example identifier")
    dataset: str = Field(..., description="Source dataset name", example="snli")
    premise: str = Field(..., description="The premise text")
    hypothesis: str = Field(..., description="The hypothesis text")
    gold_label: Optional[int] = Field(None, description="Gold label as integer (0=entailment, 1=neutral, 2=contradiction)")
    gold_label_text: Optional[str] = Field(None, description="Gold label as text", example="entailment")
    premise_words: list[dict] = Field(..., description="Tokenized premise with character offsets")
    hypothesis_words: list[dict] = Field(..., description="Tokenized hypothesis with character offsets")
    existing_labels: list[dict] = Field(..., description="Previously saved labels for this example (by current user)")
    existing_scores: Optional[dict] = Field(None, description="Previously saved complexity scores (by current user)")
    lock_until: Optional[str] = Field(None, description="ISO timestamp when the lock on this example expires")
    pool_status: Optional[str] = Field(None, description="Question pool status: test, building, or zero_entry")


class UserCreate(BaseModel):
    """Request body for user registration."""
    username: str = Field(..., min_length=3, description="Username (min 3 characters)", example="annotator1")
    password: str = Field(..., min_length=6, description="Password (min 6 characters)")
    display_name: Optional[str] = Field(None, description="Display name (defaults to username)", example="Alice")


class UserLogin(BaseModel):
    """Request body for user login."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class UserResponse(BaseModel):
    """Response containing user information."""
    id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    display_name: Optional[str] = Field(None, description="Display name")
    role: str = Field(ROLE_ANNOTATOR, description="User role (admin or annotator)")
    is_anonymous: bool = Field(False, description="True if running in anonymous mode")


class UserRoleUpdate(BaseModel):
    """Request body for updating a user's role."""
    role: str = Field(..., description="New role (admin or annotator)", example="admin")


class LabelSchemaResponse(BaseModel):
    """Response containing the label schema configuration."""
    categories: dict = Field(..., description="Label categories with their labels")
    system_labels: list[str] = Field(..., description="List of all system-defined label names")
    allow_custom: bool = Field(..., description="Whether custom labels are allowed")
    custom_tracking: bool = Field(..., description="Whether custom labels are tracked separately")


class LockStatusResponse(BaseModel):
    """Response containing lock status for an example."""
    example_id: str = Field(..., description="Example ID")
    locked: bool = Field(..., description="Whether the example is currently locked")
    locked_by: Optional[int] = Field(None, description="User ID of lock owner")
    locked_by_username: Optional[str] = Field(None, description="Username of lock owner")
    locked_until: Optional[str] = Field(None, description="ISO timestamp when lock expires")
    expires_in_seconds: Optional[int] = Field(None, description="Seconds until lock expires")
    is_own_lock: bool = Field(False, description="Whether current user owns the lock")


class ExampleAgreementResponse(BaseModel):
    """Response containing agreement metrics for an example."""
    example_id: str = Field(..., description="Example ID")
    annotation_count: int = Field(..., description="Number of annotators")
    agreement_score: Optional[float] = Field(None, description="Combined agreement score (0-1)")
    complexity_agreement: Optional[float] = Field(None, description="Complexity score agreement (0-1)")
    span_agreement: Optional[float] = Field(None, description="Span selection agreement (0-1)")
    pool_status: str = Field(..., description="Current pool status")
    consensus_threshold: int = Field(..., description="Annotations needed for consensus")
    needs_more_annotations: bool = Field(..., description="Whether more annotations are needed")


class AnnotatorAgreementResponse(BaseModel):
    """Response containing agreement metrics for an annotator."""
    user_id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    total_annotations: int = Field(..., description="Total annotations by this user")
    agreement_with_consensus: Optional[float] = Field(None, description="Agreement with consensus (0-1)")
    complexity_agreement: Optional[float] = Field(None, description="Complexity agreement (0-1)")
    span_agreement: Optional[float] = Field(None, description="Span agreement (0-1)")
    last_calculated: Optional[str] = Field(None, description="When metrics were last calculated")


class PoolStatsResponse(BaseModel):
    """Response containing question pool statistics."""
    test: int = Field(..., description="High-consensus calibration examples")
    building: int = Field(..., description="Examples being actively annotated")
    zero_entry: int = Field(..., description="Unannotated examples")
    total: int = Field(..., description="Total examples")
    consensus_threshold: int = Field(..., description="Annotations needed for consensus")
    agreement_threshold: float = Field(..., description="Agreement score for test pool")


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
                        (id, dataset, premise, hypothesis, gold_label, gold_label_text, source_file, pool_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 'zero_entry')
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


# ============================================================================
# API Endpoints - Root
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize database and tokenizer on startup."""
    init_db()
    # Pre-load tokenizer (downloads on first run, then cached)
    get_tokenizer()
    # Pre-load some datasets
    for dataset in ["snli", "mnli", "anli"]:
        ensure_examples_loaded(dataset, limit=200)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the main labeling interface."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


# ============================================================================
# API Endpoints - Authentication
# ============================================================================

@app.post(
    "/api/auth/register",
    tags=["Authentication"],
    summary="Register a new user",
    description="Create a new user account. Returns a session cookie on success. Disabled when ANONYMOUS_MODE=1. If ADMIN_USER env var matches the username, user gets admin role.",
)
async def register(user: UserCreate, response: Response):
    """Register a new user."""
    if ANONYMOUS_MODE:
        raise HTTPException(400, "Registration disabled in anonymous mode")
    
    if len(user.username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(user.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    
    with get_db() as conn:
        # Check if username exists
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (user.username,)
        ).fetchone()
        
        if existing:
            raise HTTPException(400, "Username already taken")
        
        # Determine role - admin if matches ADMIN_USER env var
        role = ROLE_ADMIN if user.username == ADMIN_USER else ROLE_ANNOTATOR
        
        # Create user
        password_hash = hash_password(user.password)
        cursor = conn.execute("""
            INSERT INTO users (username, password_hash, display_name, role)
            VALUES (?, ?, ?, ?)
        """, (user.username, password_hash, user.display_name or user.username, role))
        
        user_id = cursor.lastrowid
    
    # Create session
    token = create_session(user_id)
    response.set_cookie(
        key="session",
        value=token,
        httponly=True,
        max_age=SESSION_EXPIRY_DAYS * 24 * 60 * 60,
        samesite="lax"
    )
    
    return {"status": "registered", "user_id": user_id, "username": user.username, "role": role}


@app.post(
    "/api/auth/login",
    tags=["Authentication"],
    summary="Log in",
    description="Authenticate with username and password. Returns a session cookie on success.",
)
async def login(user: UserLogin, response: Response):
    """Log in a user."""
    if ANONYMOUS_MODE:
        raise HTTPException(400, "Login disabled in anonymous mode")
    
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, password_hash, role FROM users WHERE username = ?",
            (user.username,)
        ).fetchone()
        
        if not row or not verify_password(user.password, row["password_hash"]):
            raise HTTPException(401, "Invalid username or password")
        
        user_id = row["id"]
        role = row["role"]
    
    # Create session
    token = create_session(user_id)
    response.set_cookie(
        key="session",
        value=token,
        httponly=True,
        max_age=SESSION_EXPIRY_DAYS * 24 * 60 * 60,
        samesite="lax"
    )
    
    return {"status": "logged_in", "user_id": user_id, "username": user.username, "role": role}


@app.post(
    "/api/auth/logout",
    tags=["Authentication"],
    summary="Log out",
    description="Invalidate the current session and clear the session cookie.",
)
async def logout(request: Request, response: Response):
    """Log out the current user."""
    token = request.cookies.get("session")
    if token:
        delete_session(token)
    response.delete_cookie("session")
    return {"status": "logged_out"}


@app.get(
    "/api/me",
    tags=["Authentication"],
    summary="Get current user",
    description="Get information about the currently authenticated user, including their role.",
    response_model=UserResponse,
)
async def get_me(user: dict = Depends(get_optional_user)):
    """Get current user info."""
    if user:
        return UserResponse(
            id=user["id"],
            username=user["username"],
            display_name=user.get("display_name"),
            role=user.get("role", ROLE_ANNOTATOR),
            is_anonymous=user.get("is_anonymous", False)
        )
    return {"authenticated": False, "anonymous_mode": ANONYMOUS_MODE}


@app.get(
    "/api/auth/status",
    tags=["Authentication"],
    summary="Get auth status",
    description="Check if the server is running in anonymous mode and whether registration is enabled.",
)
async def auth_status():
    """Get authentication status and mode."""
    return {
        "anonymous_mode": ANONYMOUS_MODE,
        "registration_enabled": not ANONYMOUS_MODE,
        "admin_bootstrap_configured": bool(ADMIN_USER)
    }


# ============================================================================
# API Endpoints - Admin
# ============================================================================

@app.get(
    "/api/admin/users",
    tags=["Admin"],
    summary="List all users (admin)",
    description="List all users with detailed information including roles. Admin only.",
)
async def admin_list_users(admin: dict = Depends(require_admin)):
    """List all users with admin details."""
    with get_db() as conn:
        users = conn.execute("""
            SELECT u.id, u.username, u.display_name, u.role, u.created_at, u.last_seen,
                   COUNT(DISTINCT c.example_id) as annotations
            FROM users u
            LEFT JOIN complexity_scores c ON u.id = c.annotator_id
            GROUP BY u.id
            ORDER BY u.id
        """).fetchall()
        
        return {
            "users": [
                {
                    "id": row["id"],
                    "username": row["username"],
                    "display_name": row["display_name"],
                    "role": row["role"],
                    "created_at": row["created_at"],
                    "last_seen": row["last_seen"],
                    "annotations": row["annotations"],
                    "is_system_user": row["id"] in (ANONYMOUS_USER_ID, LEGACY_USER_ID)
                }
                for row in users
            ],
            "total": len(users)
        }


@app.get(
    "/api/admin/user/{user_id}",
    tags=["Admin"],
    summary="Get user details (admin)",
    description="Get detailed information about a specific user. Admin only.",
)
async def admin_get_user(user_id: int, admin: dict = Depends(require_admin)):
    """Get detailed user information."""
    with get_db() as conn:
        user = conn.execute("""
            SELECT id, username, display_name, role, created_at, last_seen
            FROM users WHERE id = ?
        """, (user_id,)).fetchone()
        
        if not user:
            raise HTTPException(404, f"User not found: {user_id}")
        
        # Get annotation stats
        stats = conn.execute("""
            SELECT 
                COUNT(DISTINCT c.example_id) as annotations,
                COUNT(DISTINCT s.example_id) as skipped,
                COUNT(DISTINCT CASE WHEN l.is_custom = 1 THEN l.id END) as custom_labels
            FROM users u
            LEFT JOIN complexity_scores c ON u.id = c.annotator_id
            LEFT JOIN skipped s ON u.id = s.annotator_id
            LEFT JOIN labels l ON u.id = l.annotator_id
            WHERE u.id = ?
        """, (user_id,)).fetchone()
        
        # Get agreement metrics
        agreement = conn.execute("""
            SELECT agreement_with_consensus, complexity_agreement, span_agreement
            FROM annotator_agreement WHERE user_id = ?
        """, (user_id,)).fetchone()
        
        return {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
            "role": user["role"],
            "created_at": user["created_at"],
            "last_seen": user["last_seen"],
            "is_system_user": user["id"] in (ANONYMOUS_USER_ID, LEGACY_USER_ID),
            "stats": {
                "annotations": stats["annotations"],
                "skipped": stats["skipped"],
                "custom_labels": stats["custom_labels"]
            },
            "agreement": {
                "agreement_with_consensus": agreement["agreement_with_consensus"] if agreement else None,
                "complexity_agreement": agreement["complexity_agreement"] if agreement else None,
                "span_agreement": agreement["span_agreement"] if agreement else None,
            } if agreement else None
        }


@app.put(
    "/api/admin/user/{user_id}/role",
    tags=["Admin"],
    summary="Update user role (admin)",
    description="Change a user's role. Cannot modify system users. Admin only.",
)
async def admin_update_user_role(
    user_id: int,
    role_update: UserRoleUpdate,
    admin: dict = Depends(require_admin)
):
    """Update a user's role."""
    # Validate role
    if role_update.role not in (ROLE_ADMIN, ROLE_ANNOTATOR):
        raise HTTPException(400, f"Invalid role: {role_update.role}. Must be '{ROLE_ADMIN}' or '{ROLE_ANNOTATOR}'.")
    
    # Prevent modifying system users
    if user_id in (ANONYMOUS_USER_ID, LEGACY_USER_ID):
        raise HTTPException(400, "Cannot modify system user roles.")
    
    # Prevent admin from demoting themselves
    if user_id == admin["id"] and role_update.role != ROLE_ADMIN:
        raise HTTPException(400, "Cannot demote yourself. Ask another admin.")
    
    with get_db() as conn:
        user = conn.execute("SELECT id, username, role FROM users WHERE id = ?", (user_id,)).fetchone()
        
        if not user:
            raise HTTPException(404, f"User not found: {user_id}")
        
        old_role = user["role"]
        
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (role_update.role, user_id))
        
        return {
            "status": "updated",
            "user_id": user_id,
            "username": user["username"],
            "old_role": old_role,
            "new_role": role_update.role
        }


# ============================================================================
# API Endpoints - System
# ============================================================================

@app.get(
    "/api/tokenizer/info",
    tags=["System"],
    summary="Get tokenizer info",
    description="Get information about the tokenizer used for text segmentation. The tokenizer matches the target model (ModernBERT by default).",
)
async def tokenizer_info():
    """Get information about the tokenizer being used."""
    tokenizer = get_tokenizer()
    return {
        "model": TOKENIZER_MODEL,
        "type": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
    }


@app.get(
    "/api/labels",
    tags=["System"],
    summary="Get label schema",
    description="Get the label schema configuration including system-defined labels and custom label policy.",
    response_model=LabelSchemaResponse,
)
async def get_labels():
    """Get the label schema configuration."""
    schema = get_label_schema()
    return LabelSchemaResponse(
        categories=schema["categories"],
        system_labels=schema["system_labels"],
        allow_custom=schema["allow_custom"],
        custom_tracking=schema["custom_tracking"],
    )


@app.get(
    "/api/datasets",
    tags=["System"],
    summary="List available datasets",
    description="List all available NLI datasets. Datasets are detected from JSONL files in the data/nli directory.",
)
async def list_datasets():
    """List available datasets."""
    files = list(DATA_DIR.glob("*.jsonl"))
    datasets = set()
    for f in files:
        # Extract dataset name from filename
        name = f.stem.split("_")[0]
        datasets.add(name)
    return {"datasets": sorted(datasets)}


# ============================================================================
# API Endpoints - Locking
# ============================================================================

@app.get(
    "/api/lock/status/{example_id}",
    tags=["Locking"],
    summary="Get lock status",
    description="Check if an example is currently locked and by whom.",
    response_model=LockStatusResponse,
)
async def get_example_lock_status(
    example_id: str,
    user: dict = Depends(get_current_user)
):
    """Get the lock status for an example."""
    user_id = user["id"]
    
    with get_db() as conn:
        status = get_lock_status(conn, example_id)
        
        if status:
            return LockStatusResponse(
                example_id=example_id,
                locked=True,
                locked_by=status["locked_by"],
                locked_by_username=status["locked_by_username"],
                locked_until=status["locked_until"],
                expires_in_seconds=status["expires_in_seconds"],
                is_own_lock=status["locked_by"] == user_id
            )
        else:
            return LockStatusResponse(
                example_id=example_id,
                locked=False,
                is_own_lock=False
            )


@app.post(
    "/api/lock/release/{example_id}",
    tags=["Locking"],
    summary="Release lock",
    description="Release your lock on an example. Only the lock owner can release it.",
)
async def release_example_lock(
    example_id: str,
    user: dict = Depends(get_current_user)
):
    """Release a lock on an example."""
    user_id = user["id"]
    
    with get_db() as conn:
        released = release_lock(conn, example_id, user_id)
        
        if released:
            return {"status": "released", "example_id": example_id}
        else:
            # Check if it's locked by someone else
            status = get_lock_status(conn, example_id)
            if status:
                raise HTTPException(403, f"Example is locked by {status['locked_by_username']}")
            else:
                return {"status": "not_locked", "example_id": example_id}


@app.post(
    "/api/lock/extend/{example_id}",
    tags=["Locking"],
    summary="Extend lock",
    description="Extend your lock on an example by another 30 minutes. Only the lock owner can extend.",
)
async def extend_example_lock(
    example_id: str,
    user: dict = Depends(get_current_user)
):
    """Extend a lock on an example."""
    user_id = user["id"]
    
    with get_db() as conn:
        # Check current lock
        status = get_lock_status(conn, example_id)
        
        if not status:
            raise HTTPException(404, "No lock exists for this example")
        
        if status["locked_by"] != user_id:
            raise HTTPException(403, f"Example is locked by {status['locked_by_username']}")
        
        # Extend the lock
        lock_until = acquire_lock(conn, example_id, user_id)
        
        return {
            "status": "extended",
            "example_id": example_id,
            "lock_until": lock_until.isoformat(),
            "expires_in_seconds": LOCK_TIMEOUT_MINUTES * 60
        }


@app.get(
    "/api/lock/mine",
    tags=["Locking"],
    summary="List my locks",
    description="List all examples currently locked by the current user.",
)
async def list_my_locks(user: dict = Depends(get_current_user)):
    """List all examples locked by the current user."""
    user_id = user["id"]
    now = datetime.now()
    
    with get_db() as conn:
        # Clean up expired locks first
        cleanup_expired_locks(conn)
        
        locks = conn.execute("""
            SELECT l.example_id, l.locked_until, e.dataset, e.premise, e.hypothesis
            FROM example_locks l
            JOIN examples e ON l.example_id = e.id
            WHERE l.locked_by = ?
            ORDER BY l.locked_until DESC
        """, (user_id,)).fetchall()
        
        return {
            "locks": [
                {
                    "example_id": row["example_id"],
                    "dataset": row["dataset"],
                    "premise": row["premise"][:100] + "..." if len(row["premise"]) > 100 else row["premise"],
                    "locked_until": row["locked_until"],
                    "expires_in_seconds": int((datetime.fromisoformat(row["locked_until"]) - now).total_seconds())
                }
                for row in locks
            ],
            "count": len(locks)
        }


# ============================================================================
# API Endpoints - Agreement
# ============================================================================

@app.get(
    "/api/agreement/example/{example_id}",
    tags=["Agreement"],
    summary="Get example agreement",
    description="Get inter-annotator agreement metrics for a specific example.",
    response_model=ExampleAgreementResponse,
)
async def get_example_agreement(
    example_id: str,
    recalculate: bool = Query(False, description="Force recalculation of metrics"),
    user: dict = Depends(get_current_user)
):
    """Get agreement metrics for an example."""
    with get_db() as conn:
        # Check example exists
        example = conn.execute(
            "SELECT id, pool_status FROM examples WHERE id = ?",
            (example_id,)
        ).fetchone()
        
        if not example:
            raise HTTPException(404, f"Example not found: {example_id}")
        
        if recalculate:
            metrics = update_example_agreement(conn, example_id)
            # Re-fetch pool status after update
            example = conn.execute(
                "SELECT pool_status FROM examples WHERE id = ?",
                (example_id,)
            ).fetchone()
        else:
            # Try to get cached metrics
            cached = conn.execute("""
                SELECT annotation_count, agreement_score, complexity_agreement, span_agreement
                FROM example_agreement WHERE example_id = ?
            """, (example_id,)).fetchone()
            
            if cached:
                metrics = dict(cached)
            else:
                metrics = update_example_agreement(conn, example_id)
                example = conn.execute(
                    "SELECT pool_status FROM examples WHERE id = ?",
                    (example_id,)
                ).fetchone()
        
        return ExampleAgreementResponse(
            example_id=example_id,
            annotation_count=metrics['annotation_count'],
            agreement_score=metrics['agreement_score'],
            complexity_agreement=metrics['complexity_agreement'],
            span_agreement=metrics['span_agreement'],
            pool_status=example['pool_status'],
            consensus_threshold=CONSENSUS_THRESHOLD,
            needs_more_annotations=metrics['annotation_count'] < CONSENSUS_THRESHOLD
        )


@app.get(
    "/api/agreement/user/{user_id}",
    tags=["Agreement"],
    summary="Get annotator agreement",
    description="Get agreement metrics for a specific annotator.",
    response_model=AnnotatorAgreementResponse,
)
async def get_annotator_agreement(
    user_id: int,
    recalculate: bool = Query(False, description="Force recalculation of metrics"),
    current_user: dict = Depends(get_current_user)
):
    """Get agreement metrics for an annotator."""
    with get_db() as conn:
        # Check user exists
        user = conn.execute(
            "SELECT id, username FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        
        if not user:
            raise HTTPException(404, f"User not found: {user_id}")
        
        if recalculate:
            metrics = update_annotator_agreement(conn, user_id)
        else:
            # Try to get cached metrics
            cached = conn.execute("""
                SELECT total_annotations, agreement_with_consensus, 
                       complexity_agreement, span_agreement, last_calculated
                FROM annotator_agreement WHERE user_id = ?
            """, (user_id,)).fetchone()
            
            if cached:
                return AnnotatorAgreementResponse(
                    user_id=user_id,
                    username=user['username'],
                    total_annotations=cached['total_annotations'],
                    agreement_with_consensus=cached['agreement_with_consensus'],
                    complexity_agreement=cached['complexity_agreement'],
                    span_agreement=cached['span_agreement'],
                    last_calculated=cached['last_calculated']
                )
            else:
                metrics = update_annotator_agreement(conn, user_id)
        
        return AnnotatorAgreementResponse(
            user_id=user_id,
            username=user['username'],
            total_annotations=metrics['total_annotations'],
            agreement_with_consensus=metrics['agreement_with_consensus'],
            complexity_agreement=metrics['complexity_agreement'],
            span_agreement=metrics['span_agreement'],
            last_calculated=datetime.now().isoformat()
        )


@app.get(
    "/api/agreement/stats",
    tags=["Agreement"],
    summary="Get agreement statistics",
    description="Get global agreement statistics across all examples and annotators.",
)
async def get_agreement_stats(user: dict = Depends(get_optional_user)):
    """Get global agreement statistics."""
    with get_db() as conn:
        # Overall agreement stats
        example_stats = conn.execute("""
            SELECT 
                COUNT(*) as total_with_metrics,
                AVG(agreement_score) as avg_agreement,
                AVG(complexity_agreement) as avg_complexity_agreement,
                AVG(span_agreement) as avg_span_agreement,
                MIN(agreement_score) as min_agreement,
                MAX(agreement_score) as max_agreement
            FROM example_agreement
            WHERE agreement_score IS NOT NULL
        """).fetchone()
        
        # Distribution of agreement scores
        distribution = conn.execute("""
            SELECT 
                CASE 
                    WHEN agreement_score >= 0.9 THEN 'excellent'
                    WHEN agreement_score >= 0.7 THEN 'good'
                    WHEN agreement_score >= 0.5 THEN 'moderate'
                    ELSE 'low'
                END as category,
                COUNT(*) as count
            FROM example_agreement
            WHERE agreement_score IS NOT NULL
            GROUP BY category
        """).fetchall()
        
        # Annotator agreement stats
        annotator_stats = conn.execute("""
            SELECT 
                COUNT(*) as annotators_with_metrics,
                AVG(agreement_with_consensus) as avg_annotator_agreement,
                MIN(agreement_with_consensus) as min_annotator_agreement,
                MAX(agreement_with_consensus) as max_annotator_agreement
            FROM annotator_agreement
            WHERE agreement_with_consensus IS NOT NULL
        """).fetchone()
        
        # Examples needing more annotations
        needs_annotations = conn.execute("""
            SELECT COUNT(*) as count FROM example_agreement
            WHERE annotation_count > 0 AND annotation_count < ?
        """, (CONSENSUS_THRESHOLD,)).fetchone()
        
        return {
            "example_agreement": {
                "total_with_metrics": example_stats['total_with_metrics'],
                "average": example_stats['avg_agreement'],
                "complexity_average": example_stats['avg_complexity_agreement'],
                "span_average": example_stats['avg_span_agreement'],
                "min": example_stats['min_agreement'],
                "max": example_stats['max_agreement'],
            },
            "distribution": {
                row['category']: row['count'] for row in distribution
            },
            "annotator_agreement": {
                "annotators_with_metrics": annotator_stats['annotators_with_metrics'],
                "average": annotator_stats['avg_annotator_agreement'],
                "min": annotator_stats['min_annotator_agreement'],
                "max": annotator_stats['max_annotator_agreement'],
            },
            "needs_more_annotations": needs_annotations['count'],
            "consensus_threshold": CONSENSUS_THRESHOLD,
            "agreement_high_threshold": AGREEMENT_HIGH_THRESHOLD,
        }


@app.get(
    "/api/pools",
    tags=["Agreement"],
    summary="Get pool statistics",
    description="Get question pool distribution.",
    response_model=PoolStatsResponse,
)
async def get_pool_stats(user: dict = Depends(get_optional_user)):
    """Get question pool statistics."""
    with get_db() as conn:
        pools = conn.execute("""
            SELECT pool_status, COUNT(*) as count
            FROM examples
            GROUP BY pool_status
        """).fetchall()
        
        pool_counts = {row['pool_status']: row['count'] for row in pools}
        total = sum(pool_counts.values())
        
        return PoolStatsResponse(
            test=pool_counts.get('test', 0),
            building=pool_counts.get('building', 0),
            zero_entry=pool_counts.get('zero_entry', 0),
            total=total,
            consensus_threshold=CONSENSUS_THRESHOLD,
            agreement_threshold=AGREEMENT_HIGH_THRESHOLD
        )


@app.post(
    "/api/pools/recalculate",
    tags=["Agreement"],
    summary="Recalculate all pools",
    description="Trigger recalculation of agreement metrics and pool status for all examples with annotations.",
)
async def recalculate_pools(user: dict = Depends(get_current_user)):
    """Recalculate pool status for all annotated examples."""
    with get_db() as conn:
        # Get all examples with annotations
        examples = conn.execute("""
            SELECT DISTINCT example_id FROM complexity_scores
        """).fetchall()
        
        updated = 0
        for row in examples:
            update_example_agreement(conn, row['example_id'])
            updated += 1
        
        # Get new pool counts
        pools = conn.execute("""
            SELECT pool_status, COUNT(*) as count
            FROM examples
            GROUP BY pool_status
        """).fetchall()
        
        pool_counts = {row['pool_status']: row['count'] for row in pools}
        
        return {
            "status": "recalculated",
            "examples_updated": updated,
            "pools": pool_counts
        }


# ============================================================================
# API Endpoints - Examples
# ============================================================================

@app.get(
    "/api/example/{dataset}/{row_id}",
    tags=["Examples"],
    summary="Get specific example",
    description="Retrieve a specific example by dataset and ID. Returns the tokenized text and any existing annotations by the current user. Does NOT acquire a lock.",
    response_model=ExampleResponse,
)
async def get_example(
    dataset: str, 
    row_id: str,
    user: dict = Depends(get_current_user)
):
    """Get a specific example by dataset and row ID."""
    ensure_examples_loaded(dataset)
    user_id = user["id"]

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

        # Get existing labels for this user
        labels = conn.execute("""
            SELECT l.id, l.label_name, l.label_color, l.is_custom,
                   GROUP_CONCAT(s.source || ':' || s.word_index || ':' || s.word_text, '|') as spans
            FROM labels l
            LEFT JOIN span_selections s ON s.label_id = l.id
            WHERE l.example_id = ? AND l.annotator_id = ?
            GROUP BY l.id
        """, (row["id"], user_id)).fetchall()

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
                "is_custom": bool(label["is_custom"]),
                "spans": spans
            })

        # Get existing complexity scores for this user
        scores_row = conn.execute(
            "SELECT * FROM complexity_scores WHERE example_id = ? AND annotator_id = ?",
            (row["id"], user_id)
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

        # Check lock status
        lock_status = get_lock_status(conn, row["id"])
        lock_until = None
        if lock_status and lock_status["locked_by"] == user_id:
            lock_until = lock_status["locked_until"]

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
            existing_scores=existing_scores,
            lock_until=lock_until,
            pool_status=row["pool_status"]
        )


@app.get(
    "/api/next",
    tags=["Examples"],
    summary="Get next unlabeled example",
    description="Get a random unlabeled example for the current user. Optionally filter by dataset. Returns complete=true when all examples have been labeled. Automatically acquires a 30-minute lock on the returned example.",
    response_model=ExampleResponse,
)
async def get_next_example(
    dataset: Optional[str] = Query(None, description="Filter by dataset name"),
    user: dict = Depends(get_current_user)
):
    """Get the next unlabeled example for the current user."""
    user_id = user["id"]
    
    with get_db() as conn:
        # Clean up expired locks periodically
        cleanup_expired_locks(conn)
        
        now = datetime.now().isoformat()
        
        # Build query based on whether dataset is specified
        # Only show examples this user hasn't labeled or skipped
        # Also exclude examples locked by OTHER users (but include user's own locks)
        if dataset:
            ensure_examples_loaded(dataset)
            row = conn.execute("""
                SELECT e.* FROM examples e
                LEFT JOIN complexity_scores c ON e.id = c.example_id AND c.annotator_id = ?
                LEFT JOIN skipped s ON e.id = s.example_id AND s.annotator_id = ?
                LEFT JOIN example_locks l ON e.id = l.example_id AND l.locked_until > ? AND l.locked_by != ?
                WHERE e.dataset = ? 
                  AND c.example_id IS NULL 
                  AND s.example_id IS NULL
                  AND l.example_id IS NULL
                ORDER BY RANDOM()
                LIMIT 1
            """, (user_id, user_id, now, user_id, dataset)).fetchone()
        else:
            row = conn.execute("""
                SELECT e.* FROM examples e
                LEFT JOIN complexity_scores c ON e.id = c.example_id AND c.annotator_id = ?
                LEFT JOIN skipped s ON e.id = s.example_id AND s.annotator_id = ?
                LEFT JOIN example_locks l ON e.id = l.example_id AND l.locked_until > ? AND l.locked_by != ?
                WHERE c.example_id IS NULL 
                  AND s.example_id IS NULL
                  AND l.example_id IS NULL
                ORDER BY RANDOM()
                LIMIT 1
            """, (user_id, user_id, now, user_id)).fetchone()

        if not row:
            return {"message": "No more examples to label!", "complete": True}

        # Acquire lock on this example
        lock_until = acquire_lock(conn, row["id"], user_id)
        
        if not lock_until:
            # Race condition - someone else got it first, try again
            # This shouldn't happen often, but handle it gracefully
            raise HTTPException(409, "Example was locked by another user. Please try again.")

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
            existing_scores=None,
            lock_until=lock_until.isoformat(),
            pool_status=row["pool_status"]
        )


# ============================================================================
# API Endpoints - Annotation
# ============================================================================

@app.post(
    "/api/annotate",
    tags=["Annotation"],
    summary="Save annotation",
    description="Save span labels and complexity scores for an example. Replaces any existing annotation by the current user for this example. Labels are validated against the schema - custom labels are allowed but marked with is_custom=true. Releases any lock on the example. Triggers agreement recalculation.",
)
async def save_annotation(
    submission: AnnotationSubmission,
    user: dict = Depends(get_current_user)
):
    """Save span labels and complexity scores for the current user."""
    user_id = user["id"]
    
    # Track which labels are custom for the response
    custom_labels_used = []
    
    with get_db() as conn:
        # Verify example exists
        example = conn.execute(
            "SELECT id FROM examples WHERE id = ?",
            (submission.example_id,)
        ).fetchone()

        if not example:
            raise HTTPException(404, f"Example not found: {submission.example_id}")

        # Delete existing labels for this example BY THIS USER
        conn.execute(
            "DELETE FROM labels WHERE example_id = ? AND annotator_id = ?",
            (submission.example_id, user_id)
        )

        # Save new labels with is_custom flag
        for label in submission.labels:
            if not label.spans:
                continue  # Skip labels with no spans

            # Check if this is a custom label
            is_custom = not is_system_label(label.label_name)
            if is_custom:
                custom_labels_used.append(label.label_name)

            cursor = conn.execute("""
                INSERT INTO labels (example_id, annotator_id, label_name, label_color, is_custom)
                VALUES (?, ?, ?, ?, ?)
            """, (submission.example_id, user_id, label.label_name, label.label_color, is_custom))
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
                (example_id, annotator_id, reasoning, creativity, domain_knowledge, 
                 contextual, constraints, ambiguity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                submission.example_id,
                user_id,
                submission.complexity_scores.get("reasoning"),
                submission.complexity_scores.get("creativity"),
                submission.complexity_scores.get("domain_knowledge"),
                submission.complexity_scores.get("contextual"),
                submission.complexity_scores.get("constraints"),
                submission.complexity_scores.get("ambiguity"),
            ))

        # Release the lock (if any)
        release_lock(conn, submission.example_id, user_id)
        
        # Update agreement metrics for this example
        agreement_metrics = update_example_agreement(conn, submission.example_id)

        response = {
            "status": "saved", 
            "example_id": submission.example_id, 
            "annotator_id": user_id,
            "lock_released": True,
            "agreement": {
                "annotation_count": agreement_metrics['annotation_count'],
                "agreement_score": agreement_metrics['agreement_score'],
            }
        }
        
        if custom_labels_used:
            response["custom_labels"] = custom_labels_used
            response["note"] = f"Custom labels used: {', '.join(custom_labels_used)}. These are tracked separately from system labels."
        
        return response


@app.post(
    "/api/skip/{example_id}",
    tags=["Annotation"],
    summary="Skip example",
    description="Mark an example as skipped. Skipped examples won't appear in /api/next for the current user. Releases any lock on the example.",
)
async def skip_example(
    example_id: str,
    user: dict = Depends(get_current_user)
):
    """Mark an example as skipped by the current user."""
    user_id = user["id"]
    
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO skipped (example_id, annotator_id)
            VALUES (?, ?)
        """, (example_id, user_id))
        
        # Release the lock (if any)
        release_lock(conn, example_id, user_id)
        
        return {"status": "skipped", "example_id": example_id, "annotator_id": user_id, "lock_released": True}


# ============================================================================
# API Endpoints - Statistics
# ============================================================================

@app.get(
    "/api/stats",
    tags=["Statistics"],
    summary="Get statistics",
    description="Get annotation statistics including per-dataset counts, label distribution, complexity score averages, per-annotator stats, custom label usage, active locks, and agreement metrics.",
)
async def get_stats(user: dict = Depends(get_optional_user)):
    """Get labeling statistics."""
    user_id = user["id"] if user else None
    
    with get_db() as conn:
        # Clean up expired locks
        cleanup_expired_locks(conn)
        
        # Total examples per dataset
        dataset_counts = conn.execute("""
            SELECT dataset, COUNT(*) as total FROM examples GROUP BY dataset
        """).fetchall()

        # Labeled examples per dataset (all users)
        labeled_counts = conn.execute("""
            SELECT e.dataset, COUNT(DISTINCT c.example_id) as labeled
            FROM examples e
            LEFT JOIN complexity_scores c ON e.id = c.example_id
            WHERE c.example_id IS NOT NULL
            GROUP BY e.dataset
        """).fetchall()

        # Skipped examples per dataset (all users)
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

        # Label distribution (with custom flag)
        label_dist = conn.execute("""
            SELECT label_name, is_custom, COUNT(*) as count
            FROM labels
            GROUP BY label_name, is_custom
            ORDER BY count DESC
        """).fetchall()

        # Custom label stats
        custom_stats = conn.execute("""
            SELECT 
                SUM(CASE WHEN is_custom = 1 THEN 1 ELSE 0 END) as custom_count,
                SUM(CASE WHEN is_custom = 0 THEN 1 ELSE 0 END) as system_count,
                COUNT(DISTINCT CASE WHEN is_custom = 1 THEN label_name END) as unique_custom_labels
            FROM labels
        """).fetchone()

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

        # Annotator stats
        annotator_stats = conn.execute("""
            SELECT u.username, u.display_name, u.role, COUNT(DISTINCT c.example_id) as labeled
            FROM users u
            LEFT JOIN complexity_scores c ON u.id = c.annotator_id
            GROUP BY u.id
            HAVING labeled > 0
            ORDER BY labeled DESC
        """).fetchall()

        # Active locks stats
        lock_stats = conn.execute("""
            SELECT COUNT(*) as active_locks,
                   COUNT(DISTINCT locked_by) as users_with_locks
            FROM example_locks
        """).fetchone()

        # Pool stats
        pool_stats = conn.execute("""
            SELECT pool_status, COUNT(*) as count
            FROM examples
            GROUP BY pool_status
        """).fetchall()

        # Agreement stats
        agreement_stats = conn.execute("""
            SELECT 
                AVG(agreement_score) as avg_agreement,
                COUNT(*) as examples_with_agreement
            FROM example_agreement
            WHERE agreement_score IS NOT NULL
        """).fetchone()

        # User-specific stats
        user_stats = None
        if user_id:
            user_labeled = conn.execute("""
                SELECT COUNT(*) as count FROM complexity_scores WHERE annotator_id = ?
            """, (user_id,)).fetchone()
            user_skipped = conn.execute("""
                SELECT COUNT(*) as count FROM skipped WHERE annotator_id = ?
            """, (user_id,)).fetchone()
            user_custom = conn.execute("""
                SELECT COUNT(*) as count FROM labels WHERE annotator_id = ? AND is_custom = 1
            """, (user_id,)).fetchone()
            user_locks = conn.execute("""
                SELECT COUNT(*) as count FROM example_locks WHERE locked_by = ?
            """, (user_id,)).fetchone()
            user_agreement = conn.execute("""
                SELECT agreement_with_consensus FROM annotator_agreement WHERE user_id = ?
            """, (user_id,)).fetchone()
            
            user_stats = {
                "labeled": user_labeled["count"],
                "skipped": user_skipped["count"],
                "custom_labels_used": user_custom["count"],
                "active_locks": user_locks["count"],
                "agreement_with_consensus": user_agreement["agreement_with_consensus"] if user_agreement else None
            }

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
                row["label_name"]: {
                    "count": row["count"],
                    "is_custom": bool(row["is_custom"])
                }
                for row in label_dist
            },
            "custom_label_stats": {
                "custom_count": custom_stats["custom_count"] or 0,
                "system_count": custom_stats["system_count"] or 0,
                "unique_custom_labels": custom_stats["unique_custom_labels"] or 0,
            },
            "lock_stats": {
                "active_locks": lock_stats["active_locks"],
                "users_with_locks": lock_stats["users_with_locks"],
                "lock_timeout_minutes": LOCK_TIMEOUT_MINUTES,
            },
            "pool_stats": {
                row["pool_status"]: row["count"]
                for row in pool_stats
            },
            "agreement_stats": {
                "average_agreement": agreement_stats["avg_agreement"],
                "examples_with_agreement": agreement_stats["examples_with_agreement"],
                "consensus_threshold": CONSENSUS_THRESHOLD,
            },
            "complexity_averages": {
                "reasoning": score_avgs["reasoning"],
                "creativity": score_avgs["creativity"],
                "domain_knowledge": score_avgs["domain_knowledge"],
                "contextual": score_avgs["contextual"],
                "constraints": score_avgs["constraints"],
                "ambiguity": score_avgs["ambiguity"],
            } if score_avgs["total"] > 0 else None,
            "total_labeled": score_avgs["total"],
            "annotators": [
                {
                    "username": row["username"],
                    "display_name": row["display_name"],
                    "role": row["role"],
                    "labeled": row["labeled"]
                }
                for row in annotator_stats
            ],
            "user_stats": user_stats,
            "tokenizer": TOKENIZER_MODEL,
        }


@app.get(
    "/api/export",
    tags=["Statistics"],
    summary="Export annotations",
    description="Export all annotations as JSONL format. Includes annotator information, complexity scores, span labels with is_custom flag, and agreement metrics.",
)
async def export_labels():
    """Export all labels as JSONL with annotator information."""
    with get_db() as conn:
        examples = conn.execute("""
            SELECT DISTINCT e.*, u.username as annotator, u.display_name as annotator_name,
                   c.reasoning, c.creativity, c.domain_knowledge,
                   c.contextual, c.constraints, c.ambiguity, c.annotator_id
            FROM examples e
            JOIN complexity_scores c ON e.id = c.example_id
            JOIN users u ON c.annotator_id = u.id
        """).fetchall()

        results = []
        for ex in examples:
            # Get labels for this example by this annotator
            labels = conn.execute("""
                SELECT l.label_name, l.label_color, l.is_custom,
                       s.source, s.word_index, s.word_text, s.char_start, s.char_end
                FROM labels l
                JOIN span_selections s ON s.label_id = l.id
                WHERE l.example_id = ? AND l.annotator_id = ?
            """, (ex["id"], ex["annotator_id"])).fetchall()

            # Group spans by label
            label_spans = {}
            for row in labels:
                name = row["label_name"]
                if name not in label_spans:
                    label_spans[name] = {
                        "label_name": name,
                        "label_color": row["label_color"],
                        "is_custom": bool(row["is_custom"]),
                        "spans": []
                    }
                label_spans[name]["spans"].append({
                    "source": row["source"],
                    "word_index": row["word_index"],
                    "word_text": row["word_text"],
                    "char_start": row["char_start"],
                    "char_end": row["char_end"]
                })

            # Get agreement for this example
            agreement = conn.execute("""
                SELECT agreement_score, complexity_agreement, span_agreement
                FROM example_agreement WHERE example_id = ?
            """, (ex["id"],)).fetchone()

            results.append({
                "id": ex["id"],
                "dataset": ex["dataset"],
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "gold_label": ex["gold_label"],
                "gold_label_text": ex["gold_label_text"],
                "pool_status": ex["pool_status"],
                "annotator": ex["annotator"],
                "annotator_name": ex["annotator_name"],
                "tokenizer": TOKENIZER_MODEL,
                "complexity_scores": {
                    "reasoning": ex["reasoning"],
                    "creativity": ex["creativity"],
                    "domain_knowledge": ex["domain_knowledge"],
                    "contextual": ex["contextual"],
                    "constraints": ex["constraints"],
                    "ambiguity": ex["ambiguity"],
                },
                "span_labels": list(label_spans.values()),
                "agreement": {
                    "agreement_score": agreement["agreement_score"] if agreement else None,
                    "complexity_agreement": agreement["complexity_agreement"] if agreement else None,
                    "span_agreement": agreement["span_agreement"] if agreement else None,
                } if agreement else None
            })

        return {"count": len(results), "data": results, "tokenizer": TOKENIZER_MODEL}


@app.get(
    "/api/users",
    tags=["Statistics"],
    summary="List annotators",
    description="List all registered annotators with their annotation counts. Excludes system users (anonymous, legacy).",
)
async def list_users(user: dict = Depends(get_current_user)):
    """List all annotators (for admin purposes)."""
    with get_db() as conn:
        users = conn.execute("""
            SELECT u.id, u.username, u.display_name, u.role, u.created_at, u.last_seen,
                   COUNT(DISTINCT c.example_id) as annotations
            FROM users u
            LEFT JOIN complexity_scores c ON u.id = c.annotator_id
            WHERE u.id > 2  -- Exclude system users (anonymous, legacy)
            GROUP BY u.id
            ORDER BY annotations DESC
        """).fetchall()
        
        return {
            "users": [
                {
                    "id": row["id"],
                    "username": row["username"],
                    "display_name": row["display_name"],
                    "role": row["role"],
                    "created_at": row["created_at"],
                    "last_seen": row["last_seen"],
                    "annotations": row["annotations"]
                }
                for row in users
            ]
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
