#!/usr/bin/env python3
"""
NLI Span Labeler MCP Server

Provides tools for collaborative NLI annotation via chat interfaces.
Uses the FastMCP framework for MCP protocol implementation.
"""

import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .api_client import APIConfig, NLIApiClient
from .session import SessionManager
from . import formatting as fmt


# Configuration from environment
API_BASE_URL = os.environ.get("NLI_API_URL", "http://127.0.0.1:8000")
API_USERNAME = os.environ.get("NLI_API_USERNAME", "mcp-annotator")
API_PASSWORD = os.environ.get("NLI_API_PASSWORD", "mcp-annotator-password")
STATE_FILE = Path(os.environ.get("NLI_STATE_FILE", "nli_session_state.json"))


# Initialize MCP server
mcp = FastMCP(
    name="nli-span-labeler",
    instructions="""
    NLI Span Labeler - Collaborative annotation tools for Natural Language Inference.

    Use these tools to:
    1. Fetch examples to annotate (get_next_example)
    2. View current example details (get_current_example)
    3. Build up annotation through discussion (add_span, set_difficulty)
    4. Submit completed annotations (submit_annotation)
    5. Track progress (get_session_stats, get_leaderboard)

    Typical workflow:
    1. Call get_next_example to fetch a new example
    2. Discuss with channel participants which spans support the label
    3. Use add_span to record agreed-upon spans
    4. Use set_difficulty to record complexity scores
    5. Call submit_annotation when consensus is reached
    """
)


# Shared state
_api_client: Optional[NLIApiClient] = None
_session_manager: Optional[SessionManager] = None


def get_api_client() -> NLIApiClient:
    """Get or create the API client."""
    global _api_client
    if _api_client is None:
        config = APIConfig(
            base_url=API_BASE_URL,
            username=API_USERNAME,
            password=API_PASSWORD,
        )
        _api_client = NLIApiClient(config)
    return _api_client


def get_session() -> SessionManager:
    """Get or create the session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(STATE_FILE)
    return _session_manager


# =============================================================================
# MCP Tools - Example Fetching
# =============================================================================

@mcp.tool()
def get_next_example(dataset: Optional[str] = None) -> dict:
    """
    Fetch the next example to annotate.

    Gets an unannotated example from the NLI labeler, automatically acquiring
    a lock. The example includes premise, hypothesis, gold label, and auto-spans.

    Args:
        dataset: Optional dataset filter (snli, mnli, anli)

    Returns:
        Example data with id, premise, hypothesis, tokens, and auto-spans,
        or error message if no examples available.
    """
    client = get_api_client()
    session = get_session()

    # Use session's dataset filter if none specified
    if dataset is None:
        dataset = session.state.active_dataset

    example = client.get_next_example(dataset=dataset)
    if example is None:
        return {"error": "No examples available", "dataset_filter": dataset}

    # Store in session
    session.set_current_example(example)

    return {
        "id": example["id"],
        "dataset": example.get("dataset"),
        "premise": example["premise"],
        "hypothesis": example["hypothesis"],
        "gold_label": example.get("gold_label_text", "unknown"),
        "premise_tokens": _format_tokens(example.get("premise_words", [])),
        "hypothesis_tokens": _format_tokens(example.get("hypothesis_words", [])),
        "auto_spans": example.get("auto_spans", {}),
        "lock_until": example.get("lock_until"),
        "display": fmt.format_example_display(example),
    }


@mcp.tool()
def get_current_example() -> dict:
    """
    Get the current example being discussed.

    Returns the example that was fetched with get_next_example, along with
    any annotation progress built up during discussion.

    Returns:
        Current example data and annotation progress, or error if none active.
    """
    session = get_session()

    if session.state.current_example is None:
        return {"error": "No current example. Use get_next_example first."}

    example = session.state.current_example
    progress = session.state.progress

    return {
        "id": example["id"],
        "premise": example["premise"],
        "hypothesis": example["hypothesis"],
        "gold_label": example.get("gold_label_text", "unknown"),
        "premise_tokens": _format_tokens(example.get("premise_words", [])),
        "hypothesis_tokens": _format_tokens(example.get("hypothesis_words", [])),
        "auto_spans": example.get("auto_spans", {}),
        "pending_labels": progress.labels,
        "pending_scores": progress.complexity_scores,
        "edge_case_flags": progress.edge_case_flags,
        "display": fmt.format_progress_display(
            example,
            progress.labels,
            progress.complexity_scores,
            progress.edge_case_flags,
        ),
    }


@mcp.tool()
def get_example_by_id(example_id: str) -> dict:
    """
    Get a specific example by ID.

    Fetches example details without changing the current session example.
    Useful for reviewing previously seen examples.

    Args:
        example_id: The example ID to fetch

    Returns:
        Example data or error if not found.
    """
    client = get_api_client()
    example = client.get_example(example_id)

    if example is None:
        return {"error": f"Example not found: {example_id}"}

    return {
        "id": example["id"],
        "premise": example["premise"],
        "hypothesis": example["hypothesis"],
        "gold_label": example.get("gold_label_text", "unknown"),
        "premise_tokens": _format_tokens(example.get("premise_words", [])),
        "hypothesis_tokens": _format_tokens(example.get("hypothesis_words", [])),
        "auto_spans": example.get("auto_spans", {}),
        "display": fmt.format_example_display(example),
    }


# =============================================================================
# MCP Tools - Annotation Building
# =============================================================================

@mcp.tool()
def add_span(
    label_name: str,
    source: str,
    word_indices: list[int],
) -> dict:
    """
    Add spans to the current annotation.

    Records that specific tokens should be labeled. Call multiple times
    to build up the annotation through discussion.

    Args:
        label_name: Label to apply (e.g., "novel_info", "aligned_tokens")
        source: Either "premise" or "hypothesis"
        word_indices: List of token indices to label

    Returns:
        Updated pending labels or error.
    """
    session = get_session()

    if session.state.current_example is None:
        return {"error": "No current example. Use get_next_example first."}

    if source not in ("premise", "hypothesis"):
        return {"error": "Source must be 'premise' or 'hypothesis'"}

    example = session.state.current_example
    words_key = f"{source}_words"
    words = example.get(words_key, [])

    # Validate indices and build spans
    added = []
    errors = []
    for idx in word_indices:
        if idx < 0 or idx >= len(words):
            errors.append(f"Invalid index {idx} (max: {len(words) - 1})")
            continue

        word = words[idx]
        span = {
            "source": source,
            "word_index": idx,
            "word_text": word.get("text", "").strip(),
            "char_start": word.get("char_start", 0),
            "char_end": word.get("char_end", 0),
        }
        session.add_span(label_name, span)
        added.append(idx)

    return {
        "label": label_name,
        "source": source,
        "added_indices": added,
        "errors": errors if errors else None,
        "pending_labels": session.state.progress.labels,
        "display": fmt.format_span_result(label_name, source, added, words, errors),
    }


@mcp.tool()
def remove_span(label_name: str, source: str, word_index: int) -> dict:
    """
    Remove a span from the current annotation.

    Args:
        label_name: Label to remove from
        source: Either "premise" or "hypothesis"
        word_index: Token index to remove

    Returns:
        Updated pending labels or error.
    """
    session = get_session()

    if session.state.current_example is None:
        return {"error": "No current example. Use get_next_example first."}

    removed = session.remove_span(label_name, word_index, source)

    return {
        "removed": removed,
        "label": label_name,
        "source": source,
        "word_index": word_index,
        "pending_labels": session.state.progress.labels,
    }


@mcp.tool()
def clear_labels(label_name: Optional[str] = None) -> dict:
    """
    Clear pending labels.

    Args:
        label_name: Specific label to clear, or None to clear all

    Returns:
        Updated pending labels.
    """
    session = get_session()

    if label_name:
        session.state.progress.labels.pop(label_name, None)
    else:
        session.state.progress.labels.clear()

    session._save_state()
    return {"pending_labels": session.state.progress.labels}


@mcp.tool()
def set_difficulty(
    reasoning: Optional[int] = None,
    creativity: Optional[int] = None,
    domain_knowledge: Optional[int] = None,
    contextual: Optional[int] = None,
    constraints: Optional[int] = None,
    ambiguity: Optional[int] = None,
) -> dict:
    """
    Set difficulty/complexity scores for the current example.

    Scores are 0-10 where higher means more difficult. Only provided
    scores are updated; omit to keep existing values.

    Args:
        reasoning: Reasoning complexity (0-10)
        creativity: Creative thinking required (0-10)
        domain_knowledge: Domain expertise needed (0-10)
        contextual: Context dependency (0-10)
        constraints: Constraint complexity (0-10)
        ambiguity: Ambiguity level (0-10)

    Returns:
        Updated pending scores.
    """
    session = get_session()

    if session.state.current_example is None:
        return {"error": "No current example. Use get_next_example first."}

    scores = {
        "reasoning": reasoning,
        "creativity": creativity,
        "domain_knowledge": domain_knowledge,
        "contextual": contextual,
        "constraints": constraints,
        "ambiguity": ambiguity,
    }

    errors = []
    for dim, score in scores.items():
        if score is not None:
            if not 0 <= score <= 10:
                errors.append(f"{dim}: must be 0-10, got {score}")
            else:
                session.set_complexity_score(dim, score)

    return {
        "pending_scores": session.state.progress.complexity_scores,
        "errors": errors if errors else None,
        "display": fmt.format_difficulty_scores(session.state.progress.complexity_scores),
    }


@mcp.tool()
def add_edge_case_flag(flag: str) -> dict:
    """
    Flag the current example as an edge case.

    Common flags: ambiguous, multi-interpretation, annotator-disagreement,
    requires-world-knowledge, linguistic-edge-case

    Args:
        flag: The edge case flag to add

    Returns:
        Updated edge case flags.
    """
    session = get_session()

    if session.state.current_example is None:
        return {"error": "No current example. Use get_next_example first."}

    session.add_edge_case_flag(flag)

    return {"edge_case_flags": session.state.progress.edge_case_flags}


# =============================================================================
# MCP Tools - Submission
# =============================================================================

@mcp.tool()
def submit_annotation() -> dict:
    """
    Submit the current annotation.

    Submits all pending labels and complexity scores for the current example.
    Clears the current example and progress on success.

    Returns:
        Submission result with agreement metrics, or error.
    """
    from .api_client import AnnotationSubmission, LabelSubmission, SpanSelection

    session = get_session()
    client = get_api_client()

    if session.state.current_example is None:
        return {"error": "No current example. Use get_next_example first."}

    example_id = session.state.current_example_id
    progress = session.state.progress

    # Build submission
    labels = []
    # Default colors for common labels
    label_colors = {
        "novel_info": "#4CAF50",
        "aligned_tokens": "#2196F3",
        "contradiction": "#F44336",
        "premise_only": "#FF9800",
        "hypothesis_only": "#9C27B0",
    }

    for label_name, spans in progress.labels.items():
        color = label_colors.get(label_name, "#607D8B")
        label_spans = [
            SpanSelection(
                source=s["source"],
                word_index=s["word_index"],
                word_text=s["word_text"],
                char_start=s["char_start"],
                char_end=s["char_end"],
            )
            for s in spans
        ]
        labels.append(LabelSubmission(
            label_name=label_name,
            label_color=color,
            spans=label_spans,
        ))

    submission = AnnotationSubmission(
        example_id=example_id,
        labels=labels,
        complexity_scores=progress.complexity_scores if progress.complexity_scores else None,
    )

    result = client.submit_annotation(submission)
    session.mark_completed()

    return {
        "status": "submitted",
        "example_id": example_id,
        "labels_submitted": len(labels),
        "agreement": result.get("agreement"),
        "session_completed": session.state.examples_completed,
        "display": fmt.format_submission_result(
            example_id,
            len(labels),
            result.get("agreement"),
            session.state.examples_completed,
        ),
    }


@mcp.tool()
def skip_example(reason: Optional[str] = None) -> dict:
    """
    Skip the current example.

    Marks the example as skipped and clears the current session.
    Optionally record a reason for skipping.

    Args:
        reason: Optional reason for skipping

    Returns:
        Skip confirmation.
    """
    session = get_session()
    client = get_api_client()

    if session.state.current_example is None:
        return {"error": "No current example. Use get_next_example first."}

    example_id = session.state.current_example_id

    if reason:
        session.add_discussion_note(f"Skipped: {reason}")

    client.skip_example(example_id)
    session.mark_skipped()

    return {
        "status": "skipped",
        "example_id": example_id,
        "reason": reason,
        "session_skipped": session.state.examples_skipped,
    }


# =============================================================================
# MCP Tools - Statistics & Progress
# =============================================================================

@mcp.tool()
def get_session_stats() -> dict:
    """
    Get current session statistics.

    Returns progress for this annotation session.
    """
    session = get_session()
    stats = session.get_session_summary()
    stats["display"] = fmt.format_session_stats(stats)
    return stats


@mcp.tool()
def get_global_stats() -> dict:
    """
    Get global annotation statistics.

    Returns overall stats from the NLI labeler including total examples,
    annotations, label distribution, etc.
    """
    client = get_api_client()
    return client.get_stats()


@mcp.tool()
def get_leaderboard() -> dict:
    """
    Get the annotator reliability leaderboard.

    Shows ranking by reliability score for all annotators.
    """
    client = get_api_client()
    data = client.get_leaderboard()
    data["display"] = fmt.format_leaderboard(data)
    return data


@mcp.tool()
def get_my_reliability() -> dict:
    """
    Get the current user's reliability score.

    Shows calibration status and scored annotation count.
    """
    client = get_api_client()
    return client.get_reliability_me()


@mcp.tool()
def get_available_datasets() -> dict:
    """
    Get list of available datasets.

    Returns datasets that can be filtered on.
    """
    client = get_api_client()
    datasets = client.get_datasets()
    return {"datasets": datasets}


@mcp.tool()
def set_dataset_filter(dataset: Optional[str] = None) -> dict:
    """
    Set the dataset filter for get_next_example.

    Args:
        dataset: Dataset to filter by, or None to clear filter

    Returns:
        Current filter setting.
    """
    session = get_session()
    session.set_dataset_filter(dataset)
    return {"active_dataset": dataset}


@mcp.tool()
def get_label_schema() -> dict:
    """
    Get the label schema configuration.

    Returns available system labels and custom label policy.
    """
    client = get_api_client()
    schema = client.get_labels()
    schema["display"] = fmt.format_label_schema(schema)
    return schema


# =============================================================================
# MCP Tools - Discussion Helpers
# =============================================================================

@mcp.tool()
def show_tokens(source: str = "both") -> dict:
    """
    Show tokens of the current example with indices.

    Useful during discussion to reference specific words.
    Use subscript indices to discuss: "I think ₃walks and ₂enters are aligned"

    Args:
        source: "premise", "hypothesis", or "both" (default)

    Returns:
        Formatted token display.
    """
    session = get_session()

    if session.state.current_example is None:
        return {"error": "No current example. Use get_next_example first."}

    example = session.state.current_example
    lines = []

    if source in ("premise", "both"):
        premise_words = example.get("premise_words", [])
        lines.append("📖 **PREMISE tokens:**")
        lines.append(f"   {fmt.format_indexed_tokens(premise_words)}")

    if source in ("hypothesis", "both"):
        hyp_words = example.get("hypothesis_words", [])
        if lines:
            lines.append("")
        lines.append("💭 **HYPOTHESIS tokens:**")
        lines.append(f"   {fmt.format_indexed_tokens(hyp_words)}")

    return {
        "source": source,
        "display": "\n".join(lines),
    }


@mcp.tool()
def find_token(text: str, source: str = "both") -> dict:
    """
    Find tokens containing the given text.

    Searches premise and/or hypothesis for tokens matching the search text.
    Returns indices that can be used with add_span.

    Args:
        text: Text to search for (case-insensitive)
        source: "premise", "hypothesis", or "both" (default)

    Returns:
        Matching token indices and words.
    """
    session = get_session()

    if session.state.current_example is None:
        return {"error": "No current example. Use get_next_example first."}

    example = session.state.current_example
    text_lower = text.lower()

    matches = {"premise": [], "hypothesis": []}

    if source in ("premise", "both"):
        for i, w in enumerate(example.get("premise_words", [])):
            word_text = w.get("text", "").strip().lower()
            if text_lower in word_text:
                matches["premise"].append({
                    "index": i,
                    "text": w.get("text", "").strip(),
                })

    if source in ("hypothesis", "both"):
        for i, w in enumerate(example.get("hypothesis_words", [])):
            word_text = w.get("text", "").strip().lower()
            if text_lower in word_text:
                matches["hypothesis"].append({
                    "index": i,
                    "text": w.get("text", "").strip(),
                })

    # Format display
    lines = [f"🔍 Search results for \"{text}\":"]

    if matches["premise"]:
        premise_strs = [f"{fmt.subscript_number(m['index'])}{m['text']}" for m in matches["premise"]]
        lines.append(f"   📖 Premise: {' '.join(premise_strs)}")
    elif source in ("premise", "both"):
        lines.append("   📖 Premise: (no matches)")

    if matches["hypothesis"]:
        hyp_strs = [f"{fmt.subscript_number(m['index'])}{m['text']}" for m in matches["hypothesis"]]
        lines.append(f"   💭 Hypothesis: {' '.join(hyp_strs)}")
    elif source in ("hypothesis", "both"):
        lines.append("   💭 Hypothesis: (no matches)")

    return {
        "search_text": text,
        "matches": matches,
        "display": "\n".join(lines),
    }


@mcp.tool()
def suggest_alignment() -> dict:
    """
    Suggest potential token alignments between premise and hypothesis.

    Uses simple word matching to suggest which tokens might be aligned.
    These are suggestions for discussion - not authoritative annotations.

    Returns:
        List of potential alignments with indices.
    """
    session = get_session()

    if session.state.current_example is None:
        return {"error": "No current example. Use get_next_example first."}

    example = session.state.current_example
    premise_words = example.get("premise_words", [])
    hyp_words = example.get("hypothesis_words", [])

    # Build word -> indices map for both (normalized lowercase)
    premise_map: dict[str, list[int]] = {}
    for i, w in enumerate(premise_words):
        text = w.get("text", "").strip().lower()
        if text and len(text) > 2:  # Skip short words
            premise_map.setdefault(text, []).append(i)

    hyp_map: dict[str, list[int]] = {}
    for i, w in enumerate(hyp_words):
        text = w.get("text", "").strip().lower()
        if text and len(text) > 2:  # Skip short words
            hyp_map.setdefault(text, []).append(i)

    # Find overlapping words
    alignments = []
    common_words = set(premise_map.keys()) & set(hyp_map.keys())

    for word in common_words:
        p_indices = premise_map[word]
        h_indices = hyp_map[word]
        alignments.append({
            "word": word,
            "premise_indices": p_indices,
            "hypothesis_indices": h_indices,
        })

    # Format display
    lines = ["🔗 **Suggested Alignments** (exact word matches):"]

    if alignments:
        for a in alignments[:10]:  # Limit to top 10
            p_strs = [f"P{fmt.subscript_number(i)}" for i in a["premise_indices"]]
            h_strs = [f"H{fmt.subscript_number(i)}" for i in a["hypothesis_indices"]]
            lines.append(f"   • \"{a['word']}\": {', '.join(p_strs)} ↔ {', '.join(h_strs)}")
    else:
        lines.append("   (no exact word matches found)")

    lines.append("")
    lines.append("💡 Use `add_span('aligned_tokens', 'premise', [indices])` to mark these")

    return {
        "alignments": alignments,
        "display": "\n".join(lines),
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _format_tokens(words: list[dict]) -> list[dict]:
    """Format token list for display."""
    return [
        {
            "index": w.get("index", i),
            "text": w.get("text", "").strip(),
        }
        for i, w in enumerate(words)
    ]


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
