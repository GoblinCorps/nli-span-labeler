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

Usage:
    cd nli-span-labeler
    uvicorn app:app --reload --port 8000
    # Then open http://localhost:8000

Environment Variables:
    ANONYMOUS_MODE=1  - Disable auth, use anonymous user (for local single-user)
    TOKENIZER_MODEL=answerdotai/ModernBERT-base  - HuggingFace model for tokenizer
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

from fastapi import FastAPI, HTTPException, Query, Request, Response, Depends, Cookie
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

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

# Global tokenizer - initialized at startup
_tokenizer = None

app = FastAPI(title="NLI Span Labeler")

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
        needs_migration = False
        if labels_exist:
            cursor = conn.execute("PRAGMA table_info(labels)")
            columns = [row[1] for row in cursor.fetchall()]
            needs_migration = 'annotator_id' not in columns

        # Create users table first (needed for foreign keys)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                display_name TEXT,
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

        # Create system users if they don't exist
        conn.execute("""
            INSERT OR IGNORE INTO users (id, username, display_name)
            VALUES (?, 'anonymous', 'Anonymous User')
        """, (ANONYMOUS_USER_ID,))
        
        conn.execute("""
            INSERT OR IGNORE INTO users (id, username, display_name)
            VALUES (?, 'legacy', 'Legacy Annotations')
        """, (LEGACY_USER_ID,))

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
                loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_examples_dataset ON examples(dataset);
        """)

        # Handle migration for existing tables
        if needs_migration and labels_exist:
            # Migrate existing data
            migrate_existing_data(conn)
        else:
            # Create new tables with annotator_id
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    example_id TEXT NOT NULL,
                    annotator_id INTEGER NOT NULL DEFAULT 1,
                    label_name TEXT NOT NULL,
                    label_color TEXT,
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
                CREATE INDEX IF NOT EXISTS idx_spans_label ON span_selections(label_id);
                CREATE INDEX IF NOT EXISTS idx_complexity_annotator ON complexity_scores(annotator_id);
                CREATE INDEX IF NOT EXISTS idx_skipped_annotator ON skipped(annotator_id);
            """)


def migrate_existing_data(conn):
    """Migrate existing tables to include annotator_id."""
    print("Migrating existing database to multi-user schema...")
    
    # Rename old tables
    conn.executescript("""
        ALTER TABLE labels RENAME TO labels_old;
        ALTER TABLE complexity_scores RENAME TO complexity_scores_old;
        ALTER TABLE skipped RENAME TO skipped_old;
    """)
    
    # Create new tables with annotator_id
    conn.executescript("""
        CREATE TABLE labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            example_id TEXT NOT NULL,
            annotator_id INTEGER NOT NULL DEFAULT 2,
            label_name TEXT NOT NULL,
            label_color TEXT,
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
    
    # Copy data with legacy user ID
    conn.execute("""
        INSERT INTO labels (id, example_id, annotator_id, label_name, label_color, created_at)
        SELECT id, example_id, ?, label_name, label_color, created_at
        FROM labels_old
    """, (LEGACY_USER_ID,))
    
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
        CREATE INDEX IF NOT EXISTS idx_spans_label ON span_selections(label_id);
        CREATE INDEX IF NOT EXISTS idx_complexity_annotator ON complexity_scores(annotator_id);
        CREATE INDEX IF NOT EXISTS idx_skipped_annotator ON skipped(annotator_id);
    """)
    
    print("Migration complete. Existing annotations assigned to 'legacy' user.")


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
            SELECT u.id, u.username, u.display_name, u.created_at
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
            "is_anonymous": True
        }
    
    token = request.cookies.get("session")
    return get_user_from_session(token)


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


class UserCreate(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    display_name: Optional[str]
    is_anonymous: bool = False


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


# ============================================================================
# API Endpoints - Authentication
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


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main labeling interface."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/auth/register")
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
        
        # Create user
        password_hash = hash_password(user.password)
        cursor = conn.execute("""
            INSERT INTO users (username, password_hash, display_name)
            VALUES (?, ?, ?)
        """, (user.username, password_hash, user.display_name or user.username))
        
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
    
    return {"status": "registered", "user_id": user_id, "username": user.username}


@app.post("/api/auth/login")
async def login(user: UserLogin, response: Response):
    """Log in a user."""
    if ANONYMOUS_MODE:
        raise HTTPException(400, "Login disabled in anonymous mode")
    
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = ?",
            (user.username,)
        ).fetchone()
        
        if not row or not verify_password(user.password, row["password_hash"]):
            raise HTTPException(401, "Invalid username or password")
        
        user_id = row["id"]
    
    # Create session
    token = create_session(user_id)
    response.set_cookie(
        key="session",
        value=token,
        httponly=True,
        max_age=SESSION_EXPIRY_DAYS * 24 * 60 * 60,
        samesite="lax"
    )
    
    return {"status": "logged_in", "user_id": user_id, "username": user.username}


@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    """Log out the current user."""
    token = request.cookies.get("session")
    if token:
        delete_session(token)
    response.delete_cookie("session")
    return {"status": "logged_out"}


@app.get("/api/me")
async def get_me(user: dict = Depends(get_optional_user)):
    """Get current user info."""
    if user:
        return UserResponse(
            id=user["id"],
            username=user["username"],
            display_name=user.get("display_name"),
            is_anonymous=user.get("is_anonymous", False)
        )
    return {"authenticated": False, "anonymous_mode": ANONYMOUS_MODE}


@app.get("/api/auth/status")
async def auth_status():
    """Get authentication status and mode."""
    return {
        "anonymous_mode": ANONYMOUS_MODE,
        "registration_enabled": not ANONYMOUS_MODE
    }


@app.get("/api/tokenizer/info")
async def tokenizer_info():
    """Get information about the tokenizer being used."""
    tokenizer = get_tokenizer()
    return {
        "model": TOKENIZER_MODEL,
        "type": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
    }


# ============================================================================
# API Endpoints - Datasets and Examples
# ============================================================================

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
            SELECT l.id, l.label_name, l.label_color,
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
async def get_next_example(
    dataset: Optional[str] = None,
    user: dict = Depends(get_current_user)
):
    """Get the next unlabeled example for the current user."""
    user_id = user["id"]
    
    with get_db() as conn:
        # Build query based on whether dataset is specified
        # Only show examples this user hasn't labeled or skipped
        if dataset:
            ensure_examples_loaded(dataset)
            row = conn.execute("""
                SELECT e.* FROM examples e
                LEFT JOIN complexity_scores c ON e.id = c.example_id AND c.annotator_id = ?
                LEFT JOIN skipped s ON e.id = s.example_id AND s.annotator_id = ?
                WHERE e.dataset = ? AND c.example_id IS NULL AND s.example_id IS NULL
                ORDER BY RANDOM()
                LIMIT 1
            """, (user_id, user_id, dataset)).fetchone()
        else:
            row = conn.execute("""
                SELECT e.* FROM examples e
                LEFT JOIN complexity_scores c ON e.id = c.example_id AND c.annotator_id = ?
                LEFT JOIN skipped s ON e.id = s.example_id AND s.annotator_id = ?
                WHERE c.example_id IS NULL AND s.example_id IS NULL
                ORDER BY RANDOM()
                LIMIT 1
            """, (user_id, user_id)).fetchone()

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
async def save_annotation(
    submission: AnnotationSubmission,
    user: dict = Depends(get_current_user)
):
    """Save span labels and complexity scores for the current user."""
    user_id = user["id"]
    
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

        # Save new labels
        for label in submission.labels:
            if not label.spans:
                continue  # Skip labels with no spans

            cursor = conn.execute("""
                INSERT INTO labels (example_id, annotator_id, label_name, label_color)
                VALUES (?, ?, ?, ?)
            """, (submission.example_id, user_id, label.label_name, label.label_color))
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

        return {"status": "saved", "example_id": submission.example_id, "annotator_id": user_id}


@app.post("/api/skip/{example_id}")
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
        return {"status": "skipped", "example_id": example_id, "annotator_id": user_id}


@app.get("/api/stats")
async def get_stats(user: dict = Depends(get_optional_user)):
    """Get labeling statistics."""
    user_id = user["id"] if user else None
    
    with get_db() as conn:
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

        # Annotator stats
        annotator_stats = conn.execute("""
            SELECT u.username, u.display_name, COUNT(DISTINCT c.example_id) as labeled
            FROM users u
            LEFT JOIN complexity_scores c ON u.id = c.annotator_id
            GROUP BY u.id
            HAVING labeled > 0
            ORDER BY labeled DESC
        """).fetchall()

        # User-specific stats
        user_stats = None
        if user_id:
            user_labeled = conn.execute("""
                SELECT COUNT(*) as count FROM complexity_scores WHERE annotator_id = ?
            """, (user_id,)).fetchone()
            user_skipped = conn.execute("""
                SELECT COUNT(*) as count FROM skipped WHERE annotator_id = ?
            """, (user_id,)).fetchone()
            user_stats = {
                "labeled": user_labeled["count"],
                "skipped": user_skipped["count"]
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
            "total_labeled": score_avgs["total"],
            "annotators": [
                {"username": row["username"], "display_name": row["display_name"], "labeled": row["labeled"]}
                for row in annotator_stats
            ],
            "user_stats": user_stats,
            "tokenizer": TOKENIZER_MODEL,
        }


@app.get("/api/export")
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
                SELECT l.label_name, l.label_color,
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
                "span_labels": list(label_spans.values())
            })

        return {"count": len(results), "data": results, "tokenizer": TOKENIZER_MODEL}


@app.get("/api/users")
async def list_users(user: dict = Depends(get_current_user)):
    """List all annotators (for admin purposes)."""
    with get_db() as conn:
        users = conn.execute("""
            SELECT u.id, u.username, u.display_name, u.created_at, u.last_seen,
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
