"""Streamlit chat UI for the KNUST Admissions Assistant.

Talks ONLY to the FastAPI backend (main.py) via HTTP — never imports
retrieval_pipeline directly. Run the backend first, then:

    streamlit run app.py
"""
import os

import requests
import streamlit as st

# Config
DEFAULT_API_URL = os.getenv("KNUST_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 30  # seconds — retrieval + generation can take a few seconds

st.set_page_config(
    page_title="KNUST Admissions Assistant",
    page_icon="🎓",
    layout="centered",
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL


# Backend helpers
def check_health(api_url: str) -> bool:
    try:
        resp = requests.get(f"{api_url}/health", timeout=3)
        return resp.ok and resp.json().get("status") == "ok"
    except requests.RequestException:
        return False


def ask(api_url: str, question: str, history: list) -> dict:
    """Call POST /query. Raises requests.RequestException on failure."""
    resp = requests.post(
        f"{api_url}/query",
        json={"question": question, "history": history},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def history_payload() -> list:
    """Prior turns as {"role", "content"} dicts, in order — matches the
    backend's ChatMessage schema. Sources are stripped out; the backend
    only needs role + content to resolve follow-up references.
    """
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]


# Sidebar
with st.sidebar:
    st.title("🎓 KNUST Admissions")
    st.caption("Ask about programmes, entry requirements, and cut-off aggregates.")

    st.session_state.api_url = st.text_input(
        "Backend URL", value=st.session_state.api_url,
        help="Where the FastAPI backend (main.py) is running.",
    )

    healthy = check_health(st.session_state.api_url)
    if healthy:
        st.success("Backend connected")
    else:
        st.error("Backend unreachable")
        st.caption(f"Checked: {st.session_state.api_url}/health")

    st.divider()

    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Answers are grounded in KNUST's official admissions document. "
        "If something isn't in the source material, the assistant will say so "
        "rather than guess."
    )


# Main chat area
st.title("KNUST Admissions Assistant")

if not st.session_state.messages:
    st.info(
        "👋 Ask me about entry requirements, cut-off aggregates, application "
        "steps, or specific programmes at KNUST.\n\n"
        "*Example: \"What are the WASSCE requirements for BSc. Computer Science?\"*"
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s) cited"):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(
                        f"**[{i}] {src.get('title') or 'Untitled section'}** "
                        f"· score: `{src.get('score', 0):.3f}`"
                    )
                    st.caption(src.get("text", "")[:400] + ("…" if len(src.get("text", "")) > 400 else ""))
                    if i < len(msg["sources"]):
                        st.markdown("---")

# Chat input
question = st.chat_input(
    "Ask about entry requirements, cut-offs, or how to apply...",
    disabled=not healthy,
)

if not healthy:
    st.caption("⚠️ Connect to the backend (see sidebar) before asking a question.")

if question:
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get and show the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Searching admissions requirements..."):
            try:
                # Snapshot history BEFORE appending this turn's messages —
                # the backend should only see prior turns, not this one.
                result = ask(st.session_state.api_url, question, history_payload())
                answer = result["answer"]
                sources = result.get("sources", [])
            except requests.Timeout:
                answer = "The request timed out. The backend may be under load — please try again."
                sources = []
            except requests.RequestException as exc:
                answer = f"Sorry, I couldn't reach the backend: {exc}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander(f"📚 {len(sources)} source(s) cited"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(
                        f"**[{i}] {src.get('title') or 'Untitled section'}** "
                        f"· score: `{src.get('score', 0):.3f}`"
                    )
                    st.caption(src.get("text", "")[:400] + ("…" if len(src.get("text", "")) > 400 else ""))
                    if i < len(sources):
                        st.markdown("---")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )