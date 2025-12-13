"""
Rich text formatting for MCP tool outputs.

Transforms raw API responses into human-readable chat-friendly formats.
Uses subscript numbers for token indices and box drawing for structure.
"""

# Subscript digit mapping
SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# Label emoji mapping
LABEL_EMOJI = {
    "entailment": "✅",
    "neutral": "➖",
    "contradiction": "❌",
    "unknown": "❓",
}

# Label color bar mapping (for visual distinction)
LABEL_COLOR_BAR = {
    "entailment": "🟢",
    "neutral": "🟡",
    "contradiction": "🔴",
}


def subscript_number(n: int) -> str:
    """Convert a number to subscript digits."""
    return str(n).translate(SUBSCRIPT_DIGITS)


def format_indexed_tokens(words: list[dict], highlight_indices: set[int] = None) -> str:
    """
    Format tokens with subscript indices for display.

    Args:
        words: List of word dicts with 'text' and optionally 'index'
        highlight_indices: Set of indices to highlight with brackets

    Returns:
        Formatted string like "₀The ₁cat ₂sat ₃on [₄the] ₅mat"
    """
    if not words:
        return "(empty)"

    highlight_indices = highlight_indices or set()
    parts = []

    for i, word in enumerate(words):
        idx = word.get("index", i)
        text = word.get("text", "").strip()
        subscript = subscript_number(idx)

        if idx in highlight_indices:
            parts.append(f"[{subscript}{text}]")
        else:
            parts.append(f"{subscript}{text}")

    return " ".join(parts)


def format_plain_sentence(words: list[dict]) -> str:
    """Format tokens as a plain sentence without indices."""
    if not words:
        return "(empty)"
    return " ".join(w.get("text", "").strip() for w in words)


def format_example_display(example: dict, include_auto_spans: bool = True) -> str:
    """
    Format an example for chat display.

    Creates a visually structured block showing premise, hypothesis,
    and gold label with indexed tokens.
    """
    example_id = example.get("id", "unknown")
    dataset = example.get("dataset", "")
    gold_label = example.get("gold_label", example.get("gold_label_text", "unknown")).lower()

    premise_words = example.get("premise_words", [])
    hyp_words = example.get("hypothesis_words", [])

    # Build the display
    lines = []

    # Header
    label_emoji = LABEL_EMOJI.get(gold_label, "❓")
    header = f"📋 Example {example_id}"
    if dataset:
        header += f" ({dataset})"
    lines.append(header)
    lines.append("═" * 50)

    # Premise
    lines.append("")
    lines.append(f"📖 **PREMISE**")
    lines.append(f"   {format_plain_sentence(premise_words)}")
    lines.append(f"   {format_indexed_tokens(premise_words)}")

    # Hypothesis
    lines.append("")
    lines.append(f"💭 **HYPOTHESIS**")
    lines.append(f"   {format_plain_sentence(hyp_words)}")
    lines.append(f"   {format_indexed_tokens(hyp_words)}")

    # Gold label
    lines.append("")
    lines.append(f"🏷️ Gold Label: {label_emoji} {gold_label.upper()}")

    # Auto-spans summary if present
    auto_spans = example.get("auto_spans", {})
    if include_auto_spans and auto_spans:
        lines.append("")
        lines.append("🤖 **Auto-detected spans:**")
        for label_name, spans in auto_spans.items():
            if spans:
                indices = [s.get("word_index", "?") for s in spans]
                source = spans[0].get("source", "?") if spans else "?"
                lines.append(f"   • {label_name}: {source} [{', '.join(map(str, indices))}]")

    lines.append("═" * 50)

    return "\n".join(lines)


def format_example_compact(example: dict) -> str:
    """
    Format an example in compact single-line form.

    Useful for lists or when space is limited.
    """
    example_id = example.get("id", "unknown")
    gold_label = example.get("gold_label", example.get("gold_label_text", "unknown")).lower()
    premise = example.get("premise", "")[:50]

    label_emoji = LABEL_EMOJI.get(gold_label, "❓")

    if len(premise) >= 50:
        premise += "..."

    return f"{label_emoji} **{example_id}**: {premise}"


def format_progress_display(
    example: dict,
    pending_labels: dict,
    pending_scores: dict,
    edge_case_flags: list,
) -> str:
    """
    Format current annotation progress.

    Shows the example with pending selections highlighted.
    """
    lines = []

    example_id = example.get("id", "unknown")
    gold_label = example.get("gold_label", example.get("gold_label_text", "unknown")).lower()

    premise_words = example.get("premise_words", [])
    hyp_words = example.get("hypothesis_words", [])

    # Collect highlighted indices per source
    premise_highlights = set()
    hyp_highlights = set()

    for label_name, spans in pending_labels.items():
        for span in spans:
            source = span.get("source", "")
            idx = span.get("word_index", -1)
            if source == "premise":
                premise_highlights.add(idx)
            elif source == "hypothesis":
                hyp_highlights.add(idx)

    # Header
    label_emoji = LABEL_EMOJI.get(gold_label, "❓")
    lines.append(f"📝 Annotation Progress - {example_id}")
    lines.append("─" * 50)

    # Premise with highlights
    lines.append(f"📖 PREMISE: {format_indexed_tokens(premise_words, premise_highlights)}")

    # Hypothesis with highlights
    lines.append(f"💭 HYPOTHESIS: {format_indexed_tokens(hyp_words, hyp_highlights)}")

    lines.append("")

    # Pending labels summary
    if pending_labels:
        lines.append("🏷️ **Pending Labels:**")
        for label_name, spans in pending_labels.items():
            premise_spans = [s for s in spans if s.get("source") == "premise"]
            hyp_spans = [s for s in spans if s.get("source") == "hypothesis"]

            parts = []
            if premise_spans:
                indices = [str(s.get("word_index", "?")) for s in premise_spans]
                parts.append(f"P[{','.join(indices)}]")
            if hyp_spans:
                indices = [str(s.get("word_index", "?")) for s in hyp_spans]
                parts.append(f"H[{','.join(indices)}]")

            lines.append(f"   • {label_name}: {' '.join(parts)}")
    else:
        lines.append("🏷️ No labels added yet")

    # Pending scores
    if pending_scores:
        lines.append("")
        lines.append("📊 **Difficulty Scores:**")
        score_strs = [f"{dim}={score}" for dim, score in pending_scores.items()]
        lines.append(f"   {', '.join(score_strs)}")

    # Edge case flags
    if edge_case_flags:
        lines.append("")
        lines.append(f"⚠️ **Flags:** {', '.join(edge_case_flags)}")

    lines.append("─" * 50)

    return "\n".join(lines)


def format_span_result(
    label_name: str,
    source: str,
    added_indices: list[int],
    words: list[dict],
    errors: list[str] = None,
) -> str:
    """Format the result of adding spans."""
    lines = []

    if added_indices:
        added_words = []
        for idx in added_indices:
            if 0 <= idx < len(words):
                word_text = words[idx].get("text", "?").strip()
                added_words.append(f"{subscript_number(idx)}{word_text}")

        lines.append(f"✅ Added to **{label_name}** ({source}):")
        lines.append(f"   {' '.join(added_words)}")

    if errors:
        lines.append(f"⚠️ Errors: {', '.join(errors)}")

    return "\n".join(lines) if lines else "No changes made."


def format_submission_result(
    example_id: str,
    labels_submitted: int,
    agreement: dict = None,
    session_completed: int = 0,
) -> str:
    """Format annotation submission result."""
    lines = []

    lines.append(f"✅ **Annotation Submitted** - {example_id}")
    lines.append(f"   Labels: {labels_submitted}")

    if agreement:
        agreement_pct = agreement.get("percentage", 0)
        status = agreement.get("status", "unknown")
        if agreement_pct >= 80:
            lines.append(f"   Agreement: 🟢 {agreement_pct:.0f}% ({status})")
        elif agreement_pct >= 50:
            lines.append(f"   Agreement: 🟡 {agreement_pct:.0f}% ({status})")
        else:
            lines.append(f"   Agreement: 🔴 {agreement_pct:.0f}% ({status})")

    lines.append(f"   Session total: {session_completed} completed")

    return "\n".join(lines)


def format_session_stats(stats: dict) -> str:
    """Format session statistics."""
    lines = []

    completed = stats.get("examples_completed", 0)
    skipped = stats.get("examples_skipped", 0)
    current_id = stats.get("current_example_id")
    active_dataset = stats.get("active_dataset")
    started = stats.get("session_started", "unknown")

    lines.append("📊 **Session Statistics**")
    lines.append("─" * 30)
    lines.append(f"✅ Completed: {completed}")
    lines.append(f"⏭️ Skipped: {skipped}")

    if current_id:
        lines.append(f"📝 Current: {current_id}")
    else:
        lines.append("📝 Current: (none)")

    if active_dataset:
        lines.append(f"📂 Dataset filter: {active_dataset}")

    lines.append(f"🕐 Started: {started}")

    return "\n".join(lines)


def format_leaderboard(data: dict) -> str:
    """Format the reliability leaderboard."""
    lines = []

    lines.append("🏆 **Reliability Leaderboard**")
    lines.append("═" * 40)

    rankings = data.get("rankings", [])

    if not rankings:
        lines.append("No rankings yet.")
        return "\n".join(lines)

    medals = ["🥇", "🥈", "🥉"]

    for i, entry in enumerate(rankings[:10]):  # Top 10
        username = entry.get("username", "unknown")
        score = entry.get("score", 0)
        count = entry.get("annotation_count", 0)

        medal = medals[i] if i < 3 else f"{i+1}."
        bar_length = int(score * 10)  # 0-10 scale
        bar = "█" * bar_length + "░" * (10 - bar_length)

        lines.append(f"{medal} **{username}**")
        lines.append(f"   {bar} {score:.2f} ({count} annotations)")

    return "\n".join(lines)


def format_difficulty_scores(scores: dict) -> str:
    """Format difficulty score update confirmation."""
    if not scores:
        return "📊 No difficulty scores set."

    lines = ["📊 **Difficulty Scores Updated:**"]

    # Visual bars for each dimension
    for dim, score in scores.items():
        bar_length = score
        bar = "█" * bar_length + "░" * (10 - bar_length)
        lines.append(f"   {dim}: {bar} {score}/10")

    return "\n".join(lines)


def format_error(message: str) -> str:
    """Format an error message."""
    return f"❌ **Error:** {message}"


def format_label_schema(schema: dict) -> str:
    """Format label schema for display."""
    lines = []

    lines.append("🏷️ **Available Labels**")
    lines.append("─" * 30)

    system_labels = schema.get("system_labels", [])
    custom_allowed = schema.get("custom_labels_allowed", False)

    for label in system_labels:
        name = label.get("name", "unknown")
        color = label.get("color", "#888")
        desc = label.get("description", "")
        lines.append(f"• **{name}** {color}")
        if desc:
            lines.append(f"  {desc}")

    lines.append("")
    if custom_allowed:
        lines.append("✅ Custom labels are allowed")
    else:
        lines.append("❌ Custom labels are disabled")

    return "\n".join(lines)
