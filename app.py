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

# PLACEHOLDER - This is a test push. The actual 113KB file content will need to be handled differently.
print('Test push successful')
