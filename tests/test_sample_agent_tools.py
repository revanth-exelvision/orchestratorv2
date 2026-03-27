from orchestrator.tools.sample_agents import bulletize_text, reverse_text, text_metrics


def test_text_metrics():
    out = text_metrics.invoke({"text": "a b\nc\n"})
    assert "words=3" in out
    assert "characters=" in out
    assert "non_empty_lines=2" in out


def test_reverse_text():
    assert reverse_text.invoke({"text": "abc"}) == "cba"


def test_bulletize_text():
    assert bulletize_text.invoke({"text": "one\n\ntwo"}) == "- one\n- two"
    assert bulletize_text.invoke({"text": "   \n"}) == "(no non-empty lines)"
