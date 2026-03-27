from langchain.tools import tool


@tool
def echo_text(text: str) -> str:
    """Return the same text back. Use to verify tool wiring or echo user snippets."""
    return text


@tool
def word_count(text: str) -> int:
    """Count whitespace-separated words in the given text."""
    return len(text.split())


TOOLS = [echo_text, word_count]
