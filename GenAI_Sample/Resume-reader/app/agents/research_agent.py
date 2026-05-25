from app.tools.retriever import retrieve_docs
from app.llms.gemini_client import stream_llm
from app.prompts.templates import ANSWER_PROMPT
from app.config.settings import settings
from app.config.logging import get_logger, log_event

logger = get_logger("agents.agent")


async def run_agent(query: str):
    docs = await retrieve_docs(query)
    context = "\n\n".join(docs) if docs else "No matching context found."
    messages = ANSWER_PROMPT.format_messages(
        system_prompt=settings.system_prompt,
        context=context,
        question=query,
    )
    log_event(logger, "agent_prompt_built", context_chars=len(context), query=query)

    async for token in stream_llm(messages):
        yield token
