"""Extra LangChain tools used by sample named flows (persona-style agents)."""

from __future__ import annotations

from langchain.tools import tool


@tool
def text_metrics(text: str) -> str:
    """Return word count, character count, and non-empty line count for the given text."""
    words = len(text.split())
    chars = len(text)
    lines = len([ln for ln in text.splitlines() if ln.strip()])
    return f"words={words}, characters={chars}, non_empty_lines={lines}"


@tool
def reverse_text(text: str) -> str:
    """Return the input string with character order reversed."""
    return text[::-1]


@tool
def bulletize_text(text: str) -> str:
    """Turn non-empty lines into a Markdown bullet list (prefix each with '- ')."""
    bullets = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            bullets.append(f"- {stripped}")
    return "\n".join(bullets) if bullets else "(no non-empty lines)"


SAMPLE_AGENT_TOOLS = [text_metrics, reverse_text, bulletize_text]
