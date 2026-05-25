"""
Streamlit Agentic RAG demo — standalone (does not depend on stages/).

Sidebar: paste URLs and click Ingest -> chunks the pages and indexes them in Qdrant.
Main:    chat with the agent. Each turn:
         1. Long memory recall: search Qdrant 'chat_memory' for relevant past turns.
         2. Build agent executor with those recalled memories injected.
         3. Run agent (it decides whether to call search_docs / calculator / current_time).
         4. Save (user msg, agent reply) into 'chat_memory' for next time.

Run:
  ./run.sh                      # quiet mode (silences gRPC + EOL warnings)
  # or:
  streamlit run app.py
"""
from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import settings
from core import (
    QdrantChatMemory,
    bootstrap_qdrant,
    build_agent_executor,
    build_plain_retriever,
    build_reranked_retriever,
    get_sql_chat_history,
    get_vectorstore,
    ingest_urls,
    is_reranker_enabled,
    list_chat_sessions,
    load_session_messages,
    set_reranker_enabled,
)


# ---------- one-time setup ---------------------------------------------------

@st.cache_resource
def _bootstrap() -> bool:
    bootstrap_qdrant()
    return True


@st.cache_resource
def get_long_memory() -> QdrantChatMemory:
    return QdrantChatMemory()


_bootstrap()


# ---------- agent invocation with both memories -----------------------------

def answer(question: str, session_id: str) -> dict:
    memory = get_long_memory()
    recalled = memory.recall(question, k=3)
    extras = [memory.format_for_prompt(recalled)] if recalled else None
    executor = build_agent_executor(extra_system_messages=extras, verbose=False)

    chained = RunnableWithMessageHistory(
        executor,
        get_sql_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )
    result = chained.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    output = result["output"]
    memory.save_turn(session_id, question, output)

    # Always show both retrievals in the UI so the user can see the difference,
    # but the agent's search_docs tool used whichever the toggle selected.
    reranked = build_reranked_retriever(top_k_initial=20, top_n_final=5).invoke(question)
    plain = build_plain_retriever(k=5).invoke(question)
    return {
        "answer": output,
        "recalled": recalled,
        "reranked": reranked,
        "plain": plain,
        "used_reranker": is_reranker_enabled(),
    }


# ---------- UI --------------------------------------------------------------

st.set_page_config(page_title="Agentic RAG Demo", page_icon="🧠", layout="wide")
st.title("Agentic RAG")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"streamlit-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    # ---- Session picker ----
    st.header("Session")
    sessions = list_chat_sessions()
    options = ["[ new session ]"] + [
        f"{s['session_id']} ({s['message_count']})" for s in sessions
    ]
    pick = st.selectbox(
        "Resume past chat or start new",
        options=options,
        index=0,
        help="(N) = number of messages stored. Picking a past session loads its history.",
    )
    if st.button("Load", use_container_width=True):
        if pick == "[ new session ]":
            st.session_state["session_id"] = f"streamlit-{uuid.uuid4().hex[:8]}"
            st.session_state["messages"] = []
        else:
            sid = pick.rsplit(" (", 1)[0]
            st.session_state["session_id"] = sid
            st.session_state["messages"] = load_session_messages(sid)
        st.rerun()
    st.caption(f"Active session: `{st.session_state['session_id']}`")

    st.divider()

    # ---- Reranker toggle ----
    st.header("Retrieval")
    use_reranker = st.toggle(
        "Use cross-encoder reranker",
        value=is_reranker_enabled(),
        help="OFF = plain vector similarity. ON = vector top-20 → reranker top-N. "
             "Toggle and re-ask the same question to compare.",
    )
    set_reranker_enabled(use_reranker)
    st.caption(f"Reranker: {'**ON** (' + settings.reranker_model.split('/')[-1] + ')' if use_reranker else '**OFF** (plain similarity)'}")

    st.divider()

    # ---- Ingestion ----
    st.header("Ingest URLs")
    urls_text = st.text_area(
        "One URL per line",
        value="https://en.wikipedia.org/wiki/Retrieval-augmented_generation\nhttps://en.wikipedia.org/wiki/Vector_database",
        height=120,
    )
    if st.button("Ingest", type="primary", use_container_width=True):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if urls:
            with st.spinner(f"Ingesting {len(urls)} URL(s)..."):
                n = ingest_urls(urls)
            st.success(f"Indexed {n} chunks into '{settings.docs_collection}'")
        else:
            st.warning("No URLs given.")

    st.divider()

    # ---- Stats ----
    info = get_vectorstore().client.get_collection(settings.docs_collection)
    st.metric("Docs collection points", info.points_count)
    info_mem = get_vectorstore().client.get_collection(settings.chat_memory_collection)
    st.metric("Chat-memory points", info_mem.points_count)


def _render_chunk_list(docs: list) -> None:
    if not docs:
        st.caption("No chunks.")
        return
    for d in docs:
        st.markdown(f"- **{d.metadata.get('source', '')}**")
        st.caption(d.page_content[:300] + "...")


def _render_panels(
    recalled: list,
    reranked: list,
    plain: list,
    used_reranker: bool,
) -> None:
    with st.expander("Recalled long-term memories"):
        if recalled:
            for r in recalled:
                st.markdown(
                    f"- _score {r['score']:.3f}_ **({r['role']} @ {r['timestamp']})** {r['text'][:300]}"
                )
        else:
            st.caption("No prior memories matched.")
    label_used = "(used by agent)"
    rerank_label = f"Reranked (top 5) {label_used if used_reranker else ''}"
    plain_label = f"Plain similarity (top 5) {label_used if not used_reranker else ''}"
    col1, col2 = st.columns(2)
    with col1:
        with st.expander(rerank_label, expanded=False):
            _render_chunk_list(reranked)
    with col2:
        with st.expander(plain_label, expanded=False):
            _render_chunk_list(plain)


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            _render_panels(
                msg.get("recalled", []),
                msg.get("reranked", []),
                msg.get("plain", []),
                msg.get("used_reranker", True),
            )

prompt = st.chat_input("Ask something...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer(prompt, st.session_state["session_id"])
        st.write(result["answer"])
        _render_panels(
            result["recalled"],
            result["reranked"],
            result["plain"],
            result["used_reranker"],
        )
    st.session_state["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "recalled": result["recalled"],
        "reranked": result["reranked"],
        "plain": result["plain"],
        "used_reranker": result["used_reranker"],
    })
