"""Streamlit chat UI for the resume-interview FastAPI backend."""
import html
import json
import os

import requests
import streamlit as st

RESUME_FILE = "Arpit_Kandari_Resume_Updated.pdf"

API = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
MAX_TURNS = 15  # keep in sync with MAX_TURNS_PER_CHAT in backend/main.py
MAX_CHARS = 500  # keep in sync with MAX_PROMPT_CHARS in backend/main.py

STARTERS = [
    "Walk me through your work experience",
    "What are your strongest technical skills?",
    "Tell me about the projects you are proudest of",
    "What is your educational background?",
]

# Social links. Leave a value empty ("") to hide that button.
# LinkedIn and email come from the resume; fill in the other three with your usernames.
LINKS = [
    ("LinkedIn", "https://www.linkedin.com/in/arpit-kandari-71b726293"),
    ("GitHub", "https://github.com/Arpit0034"),
    ("GeeksforGeeks", "https://www.geeksforgeeks.org/profile/arpitk995i?tab=activity"),
    ("HackerRank", "https://www.hackerrank.com/profile/Arpit0987"),
    ("LeetCode", "https://leetcode.com/u/Myth001/"),
    ("Email", "mailto:arpitk1105@gmail.com"),
    ("Phone", "tel:+917668115878"),
]
# Certifications: (name, issuer and year, credential link or "").
# Example: ("Java Programming Masterclass", "Udemy · 2025", "https://www.udemy.com/certificate/UC-xxxx/")
CERTIFICATIONS = [
    ("Certicate of Accomplishment Problem Solving (Basic)", "HackerRank · 2024", "https://www.hackerrank.com/certificates/02ca6263258c"),
    ("Java Online Course", "GeeksforGeeks· 2025", "https://media.geeksforgeeks.org/courses/certificates/b7ebdcef809bc1a79f67cfa55890b5bc.pdf"),
    ("Python Essential 1", "Cisco Networking Academy · 2025", "https://www.credly.com/badges/c07ecf2d-2ce9-4a41-9072-5c618b078261/public_url"),
    ("Spring Boot Course", "Scaler · 2025", "https://moonshot.scaler.com/s/sl/0vzlxrFU4t"),
    ("Spring Boot 0 - 1 - Beginner", "Coding Shuttle · 2025", "https://drive.google.com/file/d/1AJpA0gh2NItimcaoip_f3cVCGAU7O_2J/view?usp=drive_link"),
    ("Spring Boot 1 - 100 - Expert", "Coding Shuttle · 2026", "https://drive.google.com/file/d/1PKSHdRm3FGJPU8jeI9DsJpSXZM23X7EV/view?usp=drive_link"),
]

st.set_page_config(
    page_title="Ask Arpit",
    page_icon="💬",
    layout="centered"
)


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
st.session_state.setdefault("effort", "Low")  # answer mode: "Low" or "Medium"


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
            json={
                "question": question,
                "session_id": st.session_state.session_id,
                "effort": st.session_state.effort.lower(),
            },
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


# ---------- projects ----------
PROJECTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects.json")


@st.cache_data
def load_projects():
    try:
        with open(PROJECTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def readme_url(github_url: str):
    """Repo URL -> README anchor. Profile URLs (no repo) have no README."""
    parts = [p for p in github_url.replace("https://github.com/", "").split("/") if p]
    return f"{github_url.rstrip('/')}#readme" if len(parts) >= 2 else None


ICONS = {
    'LinkedIn': ('0A66C2', 'M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z'),
    'GitHub': ('181717', 'M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12'),
    'LeetCode': ('FFA116', 'M13.483 0a1.374 1.374 0 0 0-.961.438L7.116 6.226l-3.854 4.126a5.266 5.266 0 0 0-1.209 2.104 5.35 5.35 0 0 0-.125.513 5.527 5.527 0 0 0 .062 2.362 5.83 5.83 0 0 0 .349 1.017 5.938 5.938 0 0 0 1.271 1.818l4.277 4.193.039.038c2.248 2.165 5.852 2.133 8.063-.074l2.396-2.392c.54-.54.54-1.414.003-1.955a1.378 1.378 0 0 0-1.951-.003l-2.396 2.392a3.021 3.021 0 0 1-4.205.038l-.02-.019-4.276-4.193c-.652-.64-.972-1.469-.948-2.263a2.68 2.68 0 0 1 .066-.523 2.545 2.545 0 0 1 .619-1.164L9.13 8.114c1.058-1.134 3.204-1.27 4.43-.278l3.501 2.831c.593.48 1.461.387 1.94-.207a1.384 1.384 0 0 0-.207-1.943l-3.5-2.831c-.8-.647-1.766-1.045-2.774-1.202l2.015-2.158A1.384 1.384 0 0 0 13.483 0zm-2.866 12.815a1.38 1.38 0 0 0-1.38 1.382 1.38 1.38 0 0 0 1.38 1.382H20.79a1.38 1.38 0 0 0 1.38-1.382 1.38 1.38 0 0 0-1.38-1.382z'),
    'GeeksforGeeks': ('2F8D46', 'M21.45 14.315c-.143.28-.334.532-.565.745a3.691 3.691 0 0 1-1.104.695 4.51 4.51 0 0 1-3.116-.016 3.79 3.79 0 0 1-2.135-2.078 3.571 3.571 0 0 1-.13-.353h7.418a4.26 4.26 0 0 1-.368 1.008zm-11.99-.654a3.793 3.793 0 0 1-2.134 2.078 4.51 4.51 0 0 1-3.117.016 3.7 3.7 0 0 1-1.104-.695 2.652 2.652 0 0 1-.564-.745 4.221 4.221 0 0 1-.368-1.006H9.59c-.038.12-.08.238-.13.352zm14.501-1.758a3.849 3.849 0 0 0-.082-.475l-9.634-.008a3.932 3.932 0 0 1 1.143-2.348c.363-.35.79-.625 1.26-.809a3.97 3.97 0 0 1 4.484.957l1.521-1.49a5.7 5.7 0 0 0-1.922-1.357 6.283 6.283 0 0 0-2.544-.49 6.35 6.35 0 0 0-2.405.457 6.007 6.007 0 0 0-1.963 1.276 6.142 6.142 0 0 0-1.325 1.94 5.862 5.862 0 0 0-.466 1.864h-.063a5.857 5.857 0 0 0-.467-1.865 6.13 6.13 0 0 0-1.325-1.939A6 6 0 0 0 8.21 6.34a6.698 6.698 0 0 0-4.949.031A5.708 5.708 0 0 0 1.34 7.73l1.52 1.49a4.166 4.166 0 0 1 4.484-.958c.47.184.898.46 1.26.81.368.36.66.792.859 1.268.146.344.242.708.285 1.08l-9.635.008A4.714 4.714 0 0 0 0 12.457a6.493 6.493 0 0 0 .345 2.127 4.927 4.927 0 0 0 1.08 1.783c.528.56 1.17 1 1.88 1.293a6.454 6.454 0 0 0 2.504.457c.824.005 1.64-.15 2.404-.457a5.986 5.986 0 0 0 1.964-1.277 6.116 6.116 0 0 0 1.686-3.076h.273a6.13 6.13 0 0 0 1.686 3.077 5.99 5.99 0 0 0 1.964 1.276 6.345 6.345 0 0 0 2.405.457 6.45 6.45 0 0 0 2.502-.457 5.42 5.42 0 0 0 1.882-1.293 4.928 4.928 0 0 0 1.08-1.783A6.52 6.52 0 0 0 24 12.457a4.757 4.757 0 0 0-.039-.554z'),
    'HackerRank': ('00EA64', 'M0 0v24h24V0zm9.95 8.002h1.805c.061 0 .111.05.111.111v7.767c0 .061-.05.111-.11.111H9.95c-.061 0-.111-.05-.111-.11v-2.87H7.894v2.87c0 .06-.05.11-.11.11H5.976a.11.11 0 01-.11-.11V8.112c0-.06.05-.11.11-.11h1.806c.061 0 .11.05.11.11v2.869H9.84v-2.87c0-.06.05-.11.11-.11zm2.999 0h5.778c.061 0 .111.05.111.11v7.767a.11.11 0 01-.11.112h-5.78a.11.11 0 01-.11-.11V8.111c0-.06.05-.11.11-.11z'),
    'Email': ('EA4335', 'M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-2.023 2.309-3.178 3.927-1.964L5.455 4.64 12 9.548l6.545-4.91 1.528-1.145C21.69 2.28 24 3.434 24 5.457z'),
}
GLYPHS = {  # 24x24 Material-style glyphs for things without a brand logo
    "Phone": "M6.62 10.79c1.44 2.83 3.76 5.14 6.59 6.59l2.2-2.2c.27-.27.67-.36 1.02-.24 1.12.37 2.33.57 3.57.57.55 0 1 .45 1 1V20c0 .55-.45 1-1 1-9.39 0-17-7.61-17-17 0-.55.45-1 1-1h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 1.02l-2.2 2.2z",
    "README": "M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z",
    "Live": "M19 19H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z",
}
GLYPH_COLORS = {"Phone": "var(--aa-accent)", "README": "currentColor", "Live": "currentColor"}

THEMES = {
    # name: (label, dark?, bg, sidebar, surface, text, muted, border, accent, text-on-accent)
    "Teal":     ("🟢 Teal",     False, "#ffffff", "#f6f8f7", "#f6f8f7", "#1f2933", "#6b7780", "#d9e0de", "#0a7a6f", "#ffffff"),
    "Ocean":    ("🔵 Ocean",    False, "#f5f9ff", "#e8f0fe", "#ffffff", "#1b2a41", "#64748b", "#cfdcf2", "#2563eb", "#ffffff"),
    "Sunset":   ("🟠 Sunset",   False, "#fff8f2", "#ffece0", "#ffffff", "#3a2418", "#8a6a58", "#f3d6c3", "#e8590c", "#ffffff"),
    "Rose":     ("🌸 Rose",     False, "#fff6f8", "#ffe7ee", "#ffffff", "#3b1f2b", "#8f6676", "#f4cdd9", "#d6336c", "#ffffff"),
    "Forest":   ("🌲 Forest",   False, "#f5faf5", "#e6f2e6", "#ffffff", "#1d3123", "#5f7a67", "#cfe3d1", "#2f9e44", "#ffffff"),
    "Midnight": ("🌙 Midnight", True,  "#0e1117", "#151a23", "#171c26", "#e6e9ef", "#98a2b3", "#2a3140", "#38bdf8", "#04121c"),
    "Violet":   ("🟣 Violet",   True,  "#17132a", "#1f1a38", "#1e1938", "#ece8ff", "#a79fcb", "#332c57", "#a78bfa", "#140d2b"),
    "Amber":    ("🟡 Amber",    True,  "#141210", "#1c1916", "#1e1a16", "#f3ece2", "#a89c8c", "#332d25", "#f59e0b", "#1a1100"),
}
DEFAULT_THEME = "Violet"


def theme_css(name: str) -> str:
    _, _, bg, side, surface, text, muted, border, accent, on_accent = THEMES.get(name, THEMES[DEFAULT_THEME])
    root = (
        f":root{{--aa-bg:{bg};--aa-side:{side};--aa-surface:{surface};--aa-text:{text};"
        f"--aa-muted:{muted};--aa-border:{border};--aa-accent:{accent};--aa-on:{on_accent}}}"
    )
    return "<style>" + root + STATIC_CSS + "</style>"


STATIC_CSS = """
.stApp{background-color:var(--aa-bg);background-image:radial-gradient(ellipse 60% 40% at 15% 0%,color-mix(in srgb,var(--aa-accent) 22%,transparent),transparent 70%),radial-gradient(ellipse 50% 35% at 90% 10%,color-mix(in srgb,var(--aa-accent) 12%,transparent),transparent 70%),radial-gradient(color-mix(in srgb,var(--aa-text) 9%,transparent) 1px,transparent 1px);background-size:auto,auto,26px 26px;background-attachment:fixed}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:var(--aa-side);border-right:1px solid var(--aa-border)}
.stApp :is(p,li,label,span,h1,h2,h3,h4,h5,h6){color:var(--aa-text)}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *{color:var(--aa-muted)!important;opacity:1!important}
[data-testid="stAlert"] [data-testid="stAlertContainer"]{background:var(--aa-surface)!important;border:1px solid var(--aa-border)}
[data-testid="stAlert"] *{color:var(--aa-text)!important}
[data-testid="stExpander"] summary{background:var(--aa-surface)!important}
[data-testid="stExpander"] details{background:var(--aa-surface);border:1px solid var(--aa-border)!important;border-radius:12px}
.stApp h1{width:fit-content;background:linear-gradient(90deg,var(--aa-text) 30%,var(--aa-accent));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;color:transparent!important;font-weight:800}
/* tabs (new react-aria markup + older baseweb markup) */
[data-testid="stTabs"] [data-orientation="horizontal"]{border-bottom:none!important;box-shadow:none!important}
[data-testid="stTabs"] [role="tablist"]{gap:6px;padding:6px;background:var(--aa-surface);border:1px solid var(--aa-border)!important;border-radius:16px}
[role="tab"]{flex:1;justify-content:center;height:auto;padding:9px 14px;border-radius:11px;transition:background .15s}
[role="tab"] p{font-weight:600;font-size:.95rem}
[role="tab"]:hover{background:color-mix(in srgb,var(--aa-accent) 14%,transparent)}
[role="tab"][aria-selected="true"]{background:var(--aa-accent)!important}
[role="tab"][aria-selected="true"] :is(p,span,div){color:var(--aa-on)!important}
.react-aria-SelectionIndicator,[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important}
[role="tabpanel"]{padding-top:18px}
/* buttons */
button[kind="primary"],[data-testid="stBaseButton-primary"]{background:var(--aa-accent)!important;border:1px solid var(--aa-accent)!important}
button[kind="primary"] *,[data-testid="stBaseButton-primary"] *{color:var(--aa-on)!important}
button[kind="secondary"],[data-testid="stBaseButton-secondary"]{background:var(--aa-surface)!important;border:1px solid var(--aa-border)!important;transition:border-color .15s}
button[kind="secondary"]:hover,[data-testid="stBaseButton-secondary"]:hover{border-color:var(--aa-accent)!important}
/* chat */
[data-testid="stChatInput"]{background:var(--aa-surface);border:1px solid var(--aa-border);border-radius:14px}
[data-testid="stChatInput"],[data-testid="stChatInput"]>div,[data-testid="stChatInput"] [data-baseweb]{background:var(--aa-surface)!important}
[data-testid="stChatInput"] textarea{background:transparent!important;color:var(--aa-text)!important;-webkit-text-fill-color:var(--aa-text)}
[data-testid="stChatInput"] textarea::placeholder{color:var(--aa-muted)!important;-webkit-text-fill-color:var(--aa-muted);opacity:1}
[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:transparent}
[data-testid="stChatMessage"]{background:var(--aa-surface);border:1px solid var(--aa-border);border-radius:14px}
[data-baseweb="input"],[data-baseweb="base-input"]{background:var(--aa-surface)!important}
[data-baseweb="input"] input{color:var(--aa-text)!important;-webkit-text-fill-color:var(--aa-text)}
/* links inside chat answers follow the active theme */
[data-testid="stChatMessage"] a,
[data-testid="stChatMessageContent"] a,
[data-testid="stChatMessage"] a:visited{color:var(--aa-accent)!important;font-weight:600;text-decoration:underline!important;text-decoration-color:color-mix(in srgb,var(--aa-accent) 45%,transparent)!important;text-underline-offset:3px;transition:text-decoration-color .15s,opacity .15s}
[data-testid="stChatMessage"] a:hover{text-decoration-color:var(--aa-accent)!important;opacity:.85}
/* cards */
.aa-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px;margin:4px 0 18px}
.aa-card,.aa-link{border:1px solid var(--aa-border);border-radius:14px;background:var(--aa-surface);transition:transform .15s,box-shadow .15s,border-color .15s}
.aa-card:hover,.aa-link:hover{transform:translateY(-2px);box-shadow:0 8px 22px color-mix(in srgb,var(--aa-accent) 18%,transparent);border-color:var(--aa-accent)}
.aa-card{padding:18px;display:flex;flex-direction:column;gap:10px}
.aa-title{font-size:1.1rem;font-weight:650;line-height:1.3;color:var(--aa-text)}
.aa-tag{font-size:.78rem;color:var(--aa-muted);margin-top:-6px}
.aa-desc{font-size:.9rem;line-height:1.55;flex:1;color:var(--aa-text);opacity:.92}
.aa-pills{display:flex;flex-wrap:wrap;gap:6px}
.aa-pill{font-size:.72rem;padding:2px 9px;border-radius:999px;background:color-mix(in srgb,var(--aa-accent) 14%,transparent);color:var(--aa-accent)!important;border:1px solid color-mix(in srgb,var(--aa-accent) 30%,transparent)}
.aa-actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:2px}
.aa-btn{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border-radius:8px;border:1px solid var(--aa-border);font-size:.82rem;text-decoration:none!important;color:var(--aa-text)!important;transition:border-color .15s,color .15s}
.aa-btn:hover{border-color:var(--aa-accent);color:var(--aa-accent)!important}
.aa-btn.primary{background:var(--aa-accent);border-color:var(--aa-accent);color:var(--aa-on)!important}
.aa-btn.primary:hover{opacity:.9;color:var(--aa-on)!important}
.aa-link{display:flex;align-items:center;gap:14px;padding:14px 16px;text-decoration:none!important;color:var(--aa-text)!important}
.aa-link .aa-ico{width:44px;height:44px;flex:none;border-radius:12px;display:grid;place-items:center;background:color-mix(in srgb,var(--aa-text) 8%,transparent);color:var(--aa-text)}
.aa-link .aa-name{font-weight:600;font-size:.98rem;color:var(--aa-text)}
.aa-link .aa-sub{font-size:.78rem;color:var(--aa-muted);word-break:break-all}
"""


def icon_svg(name: str, size: int = 22) -> str:
    if name in ICONS:
        color, path = ICONS[name]
        fill = "currentColor" if name == "GitHub" else f"#{color}"
    elif name in GLYPHS:
        path, fill = GLYPHS[name], GLYPH_COLORS[name]
    else:
        return ""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" style="fill:{fill}" '
        f'aria-hidden="true"><path d="{path}"/></svg>'
    )


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def pretty_url(url: str) -> str:
    if url.startswith("mailto:") or url.startswith("tel:"):
        return url.split(":", 1)[1]
    return url.split("://", 1)[-1].replace("www.", "").rstrip("/")


def external(url: str) -> str:
    return "" if url.startswith(("mailto:", "tel:")) else ' target="_blank" rel="noopener noreferrer"'


def link_cards(links) -> str:
    items = "".join(
        f'<a class="aa-link" href="{esc(u)}"{external(u)}>'
        f'<span class="aa-ico">{icon_svg(n, 24)}</span>'
        f'<span><div class="aa-name">{esc(n)}</div><div class="aa-sub">{esc(pretty_url(u))}</div></span></a>'
        for n, u in links
    )
    return f'<div class="aa-grid">{items}</div>'


def project_cards(projects) -> str:
    cards = []
    for p in projects:
        pills = "".join(f'<span class="aa-pill">{esc(t)}</span>' for t in p.get("stack", []))
        btns = []
        gh = p.get("github")
        if gh:
            btns.append(f'<a class="aa-btn primary" href="{esc(gh)}"{external(gh)}>{icon_svg("GitHub", 15)} Code</a>')
            if readme_url(gh):
                btns.append(f'<a class="aa-btn" href="{esc(readme_url(gh))}"{external(gh)}>{icon_svg("README", 15)} README</a>')
        for live in p.get("live", []):
            btns.append(
                f'<a class="aa-btn" href="{esc(live["url"])}"{external(live["url"])}>{icon_svg("Live", 15)} {esc(live["label"])}</a>'
            )
        cards.append(
            '<div class="aa-card">'
            f'<div class="aa-title">{esc(p["name"])}</div>'
            + (f'<div class="aa-tag">{esc(p["tag"])}</div>' if p.get("tag") else "")
            + f'<div class="aa-desc">{esc(p.get("description", ""))}</div>'
            f'<div class="aa-pills">{pills}</div>'
            f'<div class="aa-actions">{"".join(btns)}</div></div>'
        )
    return f'<div class="aa-grid">{"".join(cards)}</div>'


def certification_cards(certs) -> str:
    cards = []
    for name, issuer, url in certs:
        btn = f'<div class="aa-actions"><a class="aa-btn" href="{esc(url)}"{external(url)}>{icon_svg("Live", 15)} View credential</a></div>' if url else ""
        cards.append(f'<div class="aa-card"><div class="aa-title">{esc(name)}</div><div class="aa-tag">{esc(issuer)}</div>{btn}</div>')
    return f'<div class="aa-grid">{"".join(cards)}</div>'


# ---------- main ----------
if st.session_state.get("theme") not in THEMES:
    st.session_state.theme = st.query_params.get("theme", DEFAULT_THEME)
    if st.session_state.theme not in THEMES:
        st.session_state.theme = DEFAULT_THEME
st.markdown(theme_css(st.session_state.theme), unsafe_allow_html=True)
st.title("Ask Arpit")
st.caption("A first-round screen, on demand. Every answer comes straight from his resume.")

_links = [(n, u) for n, u in LINKS if u and "YOUR_" not in u]
_tabs = ["💬 Chat", "🚀 Projects", "🔗 Links"] + (["🏅 Certifications"] if CERTIFICATIONS else [])
tab_chat, tab_projects, tab_links, *tab_certs = st.tabs(_tabs)

with tab_projects:
    projects = load_projects()
    if projects:
        st.markdown(project_cards(projects), unsafe_allow_html=True)
    else:
        st.info("Projects will show up here.")

with tab_links:
    st.markdown(link_cards(_links), unsafe_allow_html=True)
    _resume_path = os.path.join(BACKEND_DIR, RESUME_FILE)
    if os.path.exists(_resume_path):
        with open(_resume_path, "rb") as f:
            st.download_button(
                "📄 Download resume (PDF)", f.read(), file_name=RESUME_FILE,
                mime="application/pdf", use_container_width=True, type="primary",
            )

if tab_certs:
    with tab_certs[0]:
        st.markdown(certification_cards(CERTIFICATIONS), unsafe_allow_html=True)

full = st.session_state.turns >= MAX_TURNS

with tab_chat:
    if "flash" in st.session_state:
        st.warning(st.session_state.pop("flash"))

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if not st.session_state.messages:
        st.write("")
        for i in range(0, len(STARTERS), 2):
            for col, s in zip(st.columns(2), STARTERS[i:i + 2]):
                col.button(s, key=f"starter-{s}", on_click=use_starter, args=(s,), use_container_width=True)

    if full:
        st.info(f"This chat reached its {MAX_TURNS}-question limit. Start a new chat to keep asking.")

# Top level (no indentation, outside any tab) so Streamlit pins it to the bottom, like ChatGPT
typed = st.chat_input("Ask about experience, skills, projects…", max_chars=MAX_CHARS, disabled=full)
question = typed or st.session_state.pop("pending", None)

if question and not full:
    with tab_chat:
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
    with st.expander("🎨 Theme", expanded=False):
        st.radio(
            "Color theme", list(THEMES), key="theme",
            format_func=lambda k: THEMES[k][0], label_visibility="collapsed",
        )
        st.query_params["theme"] = st.session_state.theme

    st.radio(
        "Answer mode",
        ["Low", "Medium"],
        key="effort",
        horizontal=True,
        help="Low is faster. Medium thinks longer for more thorough answers.",
    )

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