from langchain_openai import ChatOpenAI

from orchestrator.config import get_settings


def get_chat_model(model: str | None = None) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        api_key=s.openai_api_key or None,
        model=model or s.openai_model,
    )
