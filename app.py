"""Streamlit chat UI for the resume-interview FastAPI backend."""
import json
import os

import requests
import streamlit as st

API = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
MAX_TURNS = 50  # keep in sync with MAX_TURNS_PER_CHAT in backend/main.py
MAX_CHARS = 500  # keep in sync with MAX_PROMPT_CHARS in backend/main.py

STARTERS = [
    "Walk me through your work experience",
    "What are your strongest technical skills?",
    "Tell me about the projects you are proudest of",
    "What is your educational background?",
]

st.set_page_config(page_title="Ask Arpit", page_icon="💬", layout="centered")


# ---------- start the FastAPI backend inside this app (single free deployment) ----------
@st.cache_resource(show_spinner="Starting up (reading the resume)...")
def start_backend():
    """Launch uvicorn once per server process and wait until it answers."""
    if os.getenv("BACKEND_URL"):  # an external backend is configured, don't start one
        return None
    import subprocess
    import sys
    import time

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"],
        cwd=BACKEND_DIR,
    )
    for _ in range(90):
        try:
            if requests.get(f"{API}/", timeout=2).ok:
                return proc
        except requests.RequestException:
            pass
        if proc.poll() is not None:
            st.error("Backend failed to start. Check that GROQ_API_KEY is set in the app secrets.")
            st.stop()
        time.sleep(1)
    st.error("Backend took too long to start.")
    st.stop()


start_backend()


class ApiError(Exception):
    pass


# ---------- state ----------
st.session_state.setdefault("session_id", None)
st.session_state.setdefault("messages", [])  # [{"role", "content"}]
st.session_state.setdefault("turns", 0)


def new_chat():
    st.session_state.update(session_id=None, messages=[], turns=0)


def open_chat(sid: str):
    try:
        r = requests.get(f"{API}/chat/{sid}", timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        st.session_state.flash = "Couldn't load that chat."
        return
    msgs = [{"role": m["role"], "content": m["content"]} for m in r.json()["messages"]]
    st.session_state.update(
        session_id=sid,
        messages=msgs,
        turns=sum(m["role"] == "user" for m in msgs),
    )


def delete_chat(sid: str):
    try:
        requests.delete(f"{API}/chat/{sid}", timeout=10)
    except requests.RequestException:
        st.session_state.flash = "Couldn't delete that chat."
        return
    if st.session_state.session_id == sid:
        new_chat()


def use_starter(text: str):
    st.session_state.pending = text


def fetch_chats():
    try:
        r = requests.get(f"{API}/chats", timeout=5)
        r.raise_for_status()
        return r.json()["chats"]
    except requests.RequestException:
        return None


# ---------- streaming ----------
def stream_answer(question: str):
    """Yield answer tokens from /chat/stream (Server-Sent Events)."""
    try:
        r = requests.post(
            f"{API}/chat/stream",
            json={"question": question, "session_id": st.session_state.session_id},
            stream=True,
            timeout=120,
        )
    except requests.RequestException:
        raise ApiError(f"Can't reach the backend at {API}. Make sure uvicorn is running.")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = None
        raise ApiError(detail if isinstance(detail, str) else f"Request failed ({r.status_code}).")
    r.encoding = "utf-8"
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        ev = json.loads(line[5:])
        kind = ev.get("type")
        if kind == "meta":
            st.session_state.session_id = ev["session_id"]
        elif kind == "token":
            yield ev["content"]
        elif kind == "error":
            raise ApiError(ev.get("detail", "Generation failed. Please try again."))
        elif kind == "done":
            st.session_state.turns = ev["turns_used"]


# ---------- main ----------
st.title("Ask Arpit")
st.caption("A first-round screen, on demand. Every answer comes straight from his resume.")

if "flash" in st.session_state:
    st.warning(st.session_state.pop("flash"))

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if not st.session_state.messages:
    st.write("")
    for s in STARTERS:
        st.button(s, key=f"starter-{s}", on_click=use_starter, args=(s,), use_container_width=True)

full = st.session_state.turns >= MAX_TURNS
if full:
    st.info(f"This chat reached its {MAX_TURNS}-question limit. Start a new chat to keep asking.")

typed = st.chat_input("Ask about experience, skills, projects…", max_chars=MAX_CHARS, disabled=full)
question = typed or st.session_state.pop("pending", None)

if question and not full:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(question))
        except ApiError as e:
            st.error(str(e))
        else:
            if answer:
                st.session_state.messages += [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]

# ---------- sidebar (drawn last so a new chat shows up right away) ----------
with st.sidebar:
    st.header("Chats")
    st.button("New chat", type="primary", on_click=new_chat, use_container_width=True)
    if st.session_state.turns:
        left = MAX_TURNS - st.session_state.turns
        st.progress(st.session_state.turns / MAX_TURNS, text=f"{left} questions left in this chat")

    chats = fetch_chats()
    if chats is None:
        st.error(f"Backend unreachable at {API}.")
    elif not chats:
        st.caption("No saved chats yet. Ask a question to start one.")
    else:
        term = st.text_input("Search chats", placeholder="Search chats", label_visibility="collapsed").strip().lower()
        for c in chats:
            title = c.get("first_question") or "Untitled chat"
            if term and term not in title.lower():
                continue
            sid = c["session_id"]
            label = title if len(title) <= 34 else title[:33] + "…"
            a, b = st.columns([5, 1])
            a.button(
                label,
                key=f"open-{sid}",
                on_click=open_chat,
                args=(sid,),
                use_container_width=True,
                type="primary" if sid == st.session_state.session_id else "secondary",
                help=f"{c['message_count'] // 2} questions",
            )
            b.button("✕", key=f"del-{sid}", on_click=delete_chat, args=(sid,), help="Delete chat")
