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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main labeling interface |
| `/api/datasets` | GET | List available datasets |
| `/api/next` | GET | Get next unlabeled example |
| `/api/example/{dataset}/{id}` | GET | Get specific example |
| `/api/annotate` | POST | Save labels and scores |
| `/api/skip/{id}` | POST | Skip an example |
| `/api/stats` | GET | Get labeling statistics |
| `/api/export` | GET | Export all labels as JSONL |

## Project Structure

```
nli-span-labeler/
├── app.py              # FastAPI application
├── run.sh              # Startup script
├── requirements.txt    # Python dependencies
├── static/
│   └── index.html      # Frontend (single-page app)
├── data/
│   └── nli/            # JSONL data files
├── scripts/
│   └── download_data.py
└── labels.db           # SQLite database (created on first run)
```

## License

MIT License
