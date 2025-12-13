# NLI Span Labeler

A web-based annotation tool for Natural Language Inference (NLI) examples with span-level labeling support.

## Features

- **Word-level span selection** - Click words to toggle selection; positions are tracked (not just text)
- **Multiple labels per token** - Words can belong to multiple labels, shown as colored dots
- **Pre-filled labels** for:
  - Difficulty dimensions: `reasoning`, `creativity`, `domain_knowledge`, `contextual`, `constraints`, `ambiguity`
  - NLI relations: `entailment`, `neutral`, `contradiction`
- **Custom labels** - Add unlimited additional labels with unique colors
- **Complexity scoring** - 1-100 scale with both slider and numeric input
- **SQLite persistence** - All labels saved locally
- **Stats dashboard** - Track labeling progress by dataset
- **Export** - Download all labels as JSONL

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/nli-span-labeler.git
cd nli-span-labeler

# Install dependencies
pip install -r requirements.txt

# Download NLI datasets (optional - see Data Acquisition below)
python scripts/download_data.py
```

## Quick Start

```bash
# Start the server
./run.sh
# Or: uvicorn app:app --reload --port 8000

# Open http://localhost:8000 in your browser
```

## Data Acquisition

The labeler expects JSONL files in `data/nli/` with this format:

```json
{"id": "snli_train_0", "premise": "A person on a horse...", "hypothesis": "A person is training...", "label": 1, "label_text": "neutral"}
```

### Option 1: Use the download script

```bash
python scripts/download_data.py
```

This downloads and converts:
- **SNLI** - Stanford NLI (~570K examples)
- **MNLI** - Multi-Genre NLI (~433K examples)
- **ANLI** - Adversarial NLI (~170K examples)

### Option 2: Manual download from Hugging Face

```python
from datasets import load_dataset
import json

# Download SNLI
snli = load_dataset("stanfordnlp/snli")

# Convert to JSONL
with open("data/nli/snli_train.jsonl", "w") as f:
    for i, ex in enumerate(snli["train"]):
        if ex["label"] == -1:  # Skip examples without gold labels
            continue
        record = {
            "id": f"snli_train_{i}",
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "label": ex["label"],
            "label_text": ["entailment", "neutral", "contradiction"][ex["label"]],
        }
        f.write(json.dumps(record) + "\n")
```

### Option 3: Use your own data

Create JSONL files in `data/nli/` with the required fields:
- `id` (string) - Unique identifier
- `premise` (string) - The premise text
- `hypothesis` (string) - The hypothesis text
- `label` (int, optional) - 0=entailment, 1=neutral, 2=contradiction
- `label_text` (string, optional) - Human-readable label

## Usage

1. **Select a dataset** from the dropdown (or "Any" for all)
2. **Click a label** in the Span Labels section to activate it
3. **Click words** in the premise/hypothesis to toggle them for that label
4. **Set complexity scores** using sliders or type values directly
5. Click **Save & Next** or **Skip**

### Keyboard shortcuts
- Words show colored dots for all their labels
- The active label's dot is larger and the word gets a colored border
- Hover over dots to see label names

## Output Format

Exported JSONL includes:

```json
{
  "id": "snli_train_123",
  "dataset": "snli",
  "premise": "A person on a horse jumps over a broken down airplane.",
  "hypothesis": "A person is outdoors, on a horse.",
  "gold_label": 0,
  "gold_label_text": "entailment",
  "complexity_scores": {
    "reasoning": 35,
    "creativity": 20,
    "domain_knowledge": 15,
    "contextual": 40,
    "constraints": 10,
    "ambiguity": 25
  },
  "span_labels": [
    {
      "label_name": "entailment",
      "label_color": "#4ade80",
      "spans": [
        {"source": "premise", "word_index": 1, "word_text": "person", "char_start": 2, "char_end": 8},
        {"source": "hypothesis", "word_index": 1, "word_text": "person", "char_start": 2, "char_end": 8}
      ]
    }
  ]
}
```

## API Endpoints

### Authentication
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Register new user |
| `/api/auth/login` | POST | Login with credentials |
| `/api/auth/logout` | POST | Logout current session |
| `/api/me` | GET | Get current user info |

### Examples & Annotation
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main labeling interface |
| `/api/datasets` | GET | List available datasets |
| `/api/next` | GET | Get next unlabeled example (auto-locks) |
| `/api/example/{id}` | GET | Get specific example by ID |
| `/api/label` | POST | Save labels and complexity scores |
| `/api/skip/{id}` | POST | Skip an example |
| `/api/stats` | GET | Get labeling statistics |
| `/api/labels` | GET | Get label schema (difficulty + NLI labels) |

### Locking
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/lock/status/{id}` | GET | Check lock status |
| `/api/lock/release/{id}` | POST | Release a lock |
| `/api/lock/extend/{id}` | POST | Extend lock by 30 minutes |
| `/api/lock/mine` | GET | List your current locks |

### Admin (requires admin role)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/users` | GET | List all users |
| `/api/admin/user/{id}/role` | PUT | Update user role |
| `/api/admin/dashboard` | GET | Admin metrics dashboard |
| `/api/admin/calibration` | GET | Calibration configuration |
| `/api/export` | GET | Export all annotations as JSONL |

### Agreement & Reliability
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/pools/stats` | GET | Get question pool statistics |
| `/api/agreement/example/{id}` | GET | Get example agreement metrics |
| `/api/agreement/annotator/{id}` | GET | Get annotator reliability |

See `/docs` for interactive API documentation (Swagger UI).

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANONYMOUS_MODE` | `0` | Set to `1` for single-user mode (no auth) |
| `ADMIN_USER` | - | Username to auto-grant admin role on registration |
| `TOKENIZER_MODEL` | `answerdotai/ModernBERT-base` | HuggingFace model for tokenization |
| `LOCK_TIMEOUT_MINUTES` | `30` | How long example locks persist |
| `TRAINING_ENABLED` | `1` | Enable training mode for new users |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=term-missing
```

## Project Structure

```
nli-span-labeler/
├── app.py              # FastAPI application
├── run.sh              # Startup script
├── requirements.txt    # Python dependencies
├── pytest.ini          # Pytest configuration
├── static/
│   └── index.html      # Frontend (single-page app)
├── data/
│   └── nli/            # JSONL data files
├── tests/              # Pytest test suite
├── scripts/
│   └── download_data.py
└── labels.db           # SQLite database (created on first run)
```

## License

MIT License
