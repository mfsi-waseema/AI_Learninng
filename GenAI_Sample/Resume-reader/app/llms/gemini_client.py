from typing import Sequence

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.settings import settings

llm = ChatGoogleGenerativeAI(
    model=settings.gemini_chat_model,
    google_api_key=settings.google_api_key,
    temperature=0.2,
    timeout=120,
)


def _message_chunk_to_text(chunk: BaseMessageChunk) -> str:
    """Gemini/LC may use str or list-of-blocks for message chunk content."""
    c = chunk.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        return "".join(parts)
    return ""


async def stream_llm(messages: Sequence[BaseMessage]):
    try:
        async for chunk in llm.astream(messages):
            text = _message_chunk_to_text(chunk)
            if text:
                yield text
    except ValueError as e:
        if "No generation chunks were returned" in str(e):
            yield (
                "[Model returned no tokens. Check GOOGLE_API_KEY, GEMINI_CHAT_MODEL, "
                "and account quotas.]\n"
            )
            return
        raise
