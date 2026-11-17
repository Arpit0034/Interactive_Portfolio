# from fastapi import FastAPI 
# from pathlib import Path 
# from pypdf import PdfReader
# from dotenv import load_dotenv
# from groq import Groq
# from pydantic import BaseModel , Field
# import os
# import json

# load_dotenv()
# my_api_key=os.getenv("GROQ_API_KEY")

# if not my_api_key:
#     raise ValueError("API key kaha hai bhai")

# client=Groq(api_key=my_api_key)
# model = "openai/gpt-oss-120b"

# class Experience(BaseModel):
#     company: str | None = None
#     role: str | None = None
#     duration: str | None = None
#     description: str | None = None
#     skills_used: list[str] = []

# class Resume(BaseModel):
#     name: str | None = None
#     email: str | None = None
#     phone: str | None = None

#     total_experience_years: float | None = None

#     skills: list[str] = []
#     experiences: list[Experience] = []
#     education: list[str] = []
#     projects: list[str] = []
#     certifications: list[str] = []

# class ChatRequest(BaseModel):
#     question: str = ""

# resume_schema = Resume.model_json_schema()

# def ask_question(question: str, resume:Resume):
#     system_prompt = f"""
# You are an AI assistant representing a jon candidate.
# Below is everything you know about the candidate.

# {resume.model_dump_json(indent=2)}
# Rules:
# 1. Answer only using this information.
# 2. Never hallucinate.
# 3. If information is unavailable,
# say
# "I don't have enough information to answer that."
# 4. Be professional
# 5. Answer as if HR is interviewing this candidate.
# 6. Ensure response in paragraph format.
# """
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {
#                 "role":"system",
#                 "content":system_prompt
#             },
#             {
#                 "role":"user",
#                 "content":question
#             }
#         ]
#     )
#     return response.choices[0].message.content

# #pdf extraction
# def read_pdf(file_path: Path):
#     reader = PdfReader(file_path)
#     text = ""
#     for page in reader.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text + "\n"
#     return text

# #parse resume
# def parse_resume(resume_text):
#     system_prompt = f"""
#     You are an expert resume parser.

#     Extract information from the resume based on its meaning,
#     not only based on exact section headings.

#     Different resumes may use different headings.

#     For example:
#     - Experience
#     - Professional Experience
#     - Work History
#     - Employment
#     - Internships

#     These may all contain relevant experience.

#     Skills may also appear in the skills section, work experience,
#     internships or projects.

#     Return ONLY valid JSON matching this schema:

#     {resume_schema}

#     Important rules:

#     1. Do not invent information.
#     2. If a value is not available, return null.
#     3. If a list has no information, return an empty list.
#     4. Include internships inside experiences.
#     5. Extract skills mentioned across the entire resume.
#     """
#     user_prompt = f"""
#     Parse the following resume:

#     {resume_text}
#     """
#     message_system={
#         "role" : "system",
#         "content" : system_prompt
#     }
#     message_user={
#         "role" : "user",
#         "content" : user_prompt
#     }
#     messages=[message_system, message_user]
#     response_format={
#         "type": "json_object"
#     }
#     response=client.chat.completions.create(model=model, messages=messages, response_format=response_format)
#     raw_output = response.choices[0].message.content
#     data = json.loads(raw_output)
#     resume = Resume(**data)
#     return resume

# app = FastAPI()
# @app.get("/")
# def home():
#     return {
#         "message":"Backend Running"
#     }

# @app.post("/chat")
# def chat(request: ChatRequest):
#     resume_text = read_pdf(Path("Arpit_Kandari_Resume_Updated.pdf"))
#     resume = parse_resume(resume_text)
#     ans = ask_question(request.question,resume)
#     return {
#         "answer":ans
#     }

from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
import json
import os
import sqlite3
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"

RESUME_PATH = Path("Arpit_Kandari_Resume_Updated.pdf")
DB_PATH = "chat_history.db"
MAX_HISTORY_MESSAGES = 20  # how many recent messages are sent to the model
MAX_TURNS_PER_CHAT = 50    # 1 turn = 1 question + 1 answer
MAX_PROMPT_CHARS = 500     # max length of a single user question
MAX_ANSWER_TOKENS = 500    # max length of a single model answer


# ---------- Schemas ----------
class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    turns_used: int
    turns_remaining: int


resume_schema = Resume.model_json_schema()
resume_cache: dict[str, Resume] = {}


# ---------- Database (persistent chat storage) ----------
@contextmanager
def get_db():
    # New connection per call: safe with FastAPI's threadpool
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)"
        )


def get_recent_history(session_id: str, limit: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    """Last `limit` messages of a chat, oldest first (what the model needs)."""
    with get_db() as db:
        rows = db.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def get_full_history(session_id: str) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def count_turns(session_id: str) -> int:
    """Number of question/answer pairs already stored for this chat."""
    with get_db() as db:
        row = db.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ? AND role = 'user'",
            (session_id,),
        ).fetchone()
    return row["n"]


def save_turn(session_id: str, question: str, answer: str):
    with get_db() as db:
        db.executemany(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            [(session_id, "user", question), (session_id, "assistant", answer)],
        )


def delete_session(session_id: str) -> int:
    with get_db() as db:
        cur = db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        return cur.rowcount


def list_sessions() -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            """
            SELECT session_id,
                   COUNT(*)        AS message_count,
                   MIN(created_at) AS started_at,
                   MAX(created_at) AS last_active,
                   (SELECT content FROM messages m2
                     WHERE m2.session_id = m.session_id AND m2.role = 'user'
                     ORDER BY m2.id LIMIT 1) AS first_question
            FROM messages m
            GROUP BY session_id
            ORDER BY last_active DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


# ---------- PDF + parsing ----------
def read_pdf(file_path: Path) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def parse_resume(resume_text: str) -> Resume:
    system_prompt = f"""
You are an expert resume parser.

Extract information from the resume based on its meaning,
not only based on exact section headings.

Different resumes may use different headings, for example:
- Experience
- Professional Experience
- Work History
- Employment
- Internships

These may all contain relevant experience.

Skills may also appear in the skills section, work experience,
internships or projects.

Return ONLY valid JSON matching this schema:

{json.dumps(resume_schema)}

Important rules:
1. Do not invent information.
2. If a value is not available, return null.
3. If a list has no information, return an empty list.
4. Include internships inside experiences.
5. Extract skills mentioned across the entire resume.
"""
    user_prompt = f"Parse the following resume:\n\n{resume_text}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return Resume(**data)


# ---------- Chat ----------
def build_system_prompt(resume: Resume) -> str:
    return f"""
You are an AI assistant representing a job candidate.
Below is everything you know about the candidate.

{resume.model_dump_json(indent=2)}

Rules:
1. Answer only using this information.
2. Never hallucinate.
3. If information is unavailable, say
   "I don't have enough information to answer that."
4. Be professional.
5. Answer as if HR is interviewing this candidate.
6. Ensure response is in paragraph format.
7. Use the earlier conversation to understand follow-up questions
   (e.g. "tell me more about that", "what about his second job?").
"""


def build_messages(question: str, resume: Resume, history: list[dict]) -> list[dict]:
    return (
        [{"role": "system", "content": build_system_prompt(resume)}]
        + history
        + [{"role": "user", "content": question}]
    )


def ask_question(question: str, resume: Resume, history: list[dict]) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(question, resume, history),
        max_completion_tokens=MAX_ANSWER_TOKENS,
    )
    return response.choices[0].message.content


def ask_question_stream(question: str, resume: Resume, history: list[dict]):
    """Yields the answer piece by piece as the model generates it."""
    stream = client.chat.completions.create(
        model=model,
        messages=build_messages(question, resume, history),
        max_completion_tokens=MAX_ANSWER_TOKENS,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content  # reasoning tokens (if any) are ignored
        if token:
            yield token


# ---------- App ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    resume_cache["resume"] = parse_resume(read_pdf(RESUME_PATH))  # parsed once
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return {"message": "Backend Running"}


def prepare_chat(request: ChatRequest) -> tuple[str, str, int]:
    """Shared validation for /chat and /chat/stream. Runs BEFORE any model call."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())

    turns_used = count_turns(session_id)
    if turns_used >= MAX_TURNS_PER_CHAT:
        raise HTTPException(
            status_code=429,
            detail=f"Chat limit reached ({MAX_TURNS_PER_CHAT} messages). Please start a new chat.",
        )
    return session_id, question, turns_used


def sse(payload: dict) -> str:
    """Format one Server-Sent Event."""
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Non-streaming: returns the full answer at once."""
    session_id, question, turns_used = prepare_chat(request)

    history = get_recent_history(session_id)
    answer = ask_question(question, resume_cache["resume"], history)
    save_turn(session_id, question, answer)

    turns_used += 1
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        turns_used=turns_used,
        turns_remaining=MAX_TURNS_PER_CHAT - turns_used,
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Streaming: sends the answer token by token as Server-Sent Events."""
    session_id, question, turns_used = prepare_chat(request)  # 400/422/429 are raised here
    history = get_recent_history(session_id)

    def event_generator():
        yield sse({"type": "meta", "session_id": session_id})

        parts: list[str] = []
        try:
            for token in ask_question_stream(question, resume_cache["resume"], history):
                parts.append(token)
                yield sse({"type": "token", "content": token})
        except Exception:
            yield sse({"type": "error", "detail": "Generation failed. Please try again."})
            return  # nothing is saved for a failed turn

        answer = "".join(parts)
        if not answer.strip():
            yield sse({"type": "error", "detail": "Empty response. Please try again."})
            return

        # Reached only if the stream finished (not on client disconnect)
        save_turn(session_id, question, answer)
        used = turns_used + 1
        yield sse({
            "type": "done",
            "turns_used": used,
            "turns_remaining": MAX_TURNS_PER_CHAT - used,
        })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/chats")
def all_chats():
    """Sidebar-style list of all saved conversations."""
    return {"chats": list_sessions()}


@app.get("/chat/{session_id}")
def get_chat(session_id: str):
    """Full saved history of one conversation."""
    messages = get_full_history(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"session_id": session_id, "messages": messages}


@app.delete("/chat/{session_id}")
def reset_chat(session_id: str):
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"message": "Conversation deleted"}