"""
Voice of Grower — FastAPI + lightweight HTML/JS frontend.

Same-origin app: this process serves both the API and the UI (Jinja2 shell
+ vanilla JS, with Chart.js/marked/DOMPurify vendored under static/vendor
so there is no CDN dependency and the CSP can stay strict). All business
logic lives in vog_core.py — this file is purely the web-framework glue
(routing, sessions, streaming, downloads).
"""

import datetime
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from collections import OrderedDict, deque

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

import shared_state
import vog_core

log = logging.getLogger("vog")
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ── Credentials: fail SAFE, never fail open ──
# Previously these fell back to literals committed to a public repo
# ("dev-only-secret-change-me" / "admin123"), so a missing or mistyped env
# var silently booted a publicly-known password. Crashing on boot would be
# the textbook fix, but it turns a config slip into an outage for a live
# service — so instead: an unset session secret gets a random per-process
# value (sessions simply don't survive a restart, which is already true of
# every other piece of state here), and an unset admin password disables
# admin login outright rather than accepting a guessable default.
SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_urlsafe(48)
if not os.getenv("SESSION_SECRET"):
    log.warning("SESSION_SECRET unset — using a random per-process secret; sessions will not survive a restart.")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ADMIN_ENABLED = bool(ADMIN_PASSWORD)
if not ADMIN_ENABLED:
    log.warning("ADMIN_PASSWORD unset — the admin panel is DISABLED for this process.")

MAX_QUERY_CHARS = 500
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

APP_BUILD = "2026-09-02-v11 (audit remediation: ingestion correctness, true counts, hardening, CI + heartbeat)"

app = FastAPI(title="Voice of Grower")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    https_only=True,
    same_site="lax",
    max_age=8 * 60 * 60,
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ── Security headers ──
# The CSP is the backstop for the markdown-render path: even if something
# slips past DOMPurify, inline/injected script has no origin it may load
# from. 'unsafe-inline' is still required for style because the templates
# use a few inline style attributes.
_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "base-uri 'none'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = _CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ── Rate limiting ──
# A single /chat turn can fan out to several paid LLM/embedding calls, and
# /admin/login is otherwise an unthrottled password oracle. Fixed-window
# per-IP counters; in-process like everything else here, so they reset on
# restart — adequate as a cost/abuse brake, not a security boundary.
_RATE_BUCKETS: dict[str, deque] = {}
_RATE_RULES = {"chat": (30, 60), "login": (5, 900)}  # (max_hits, window_seconds)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limit(request: Request, rule: str):
    max_hits, window = _RATE_RULES[rule]
    key = f"{rule}:{_client_ip(request)}"
    now = time.monotonic()
    bucket = _RATE_BUCKETS.setdefault(key, deque())
    while bucket and now - bucket[0] > window:
        bucket.popleft()
    if len(bucket) >= max_hits:
        raise HTTPException(status_code=429, detail="Too many requests — please wait a moment and try again.")
    bucket.append(now)
    if len(_RATE_BUCKETS) > 10_000:  # crude guard against unbounded key growth
        for k in [k for k, v in _RATE_BUCKETS.items() if not v][:5_000]:
            _RATE_BUCKETS.pop(k, None)

# ── Download store ──
# Each finalized chat turn may produce CSV/Excel/PPTX bytes. Rather than
# round-tripping those through the SSE stream as base64, we stash them here
# and hand the browser a plain link. Entries are bound to the owning
# session and expire on a clock, so a leaked link (browser history, proxy
# logs) is not an indefinite bearer token for someone else's export.
# Backed by shared_state, so a Redis URL makes this multi-instance-safe.
_MAX_STORED = 200
_DOWNLOAD_TTL = 30 * 60  # seconds


def _store_downloads(downloads: dict | None, owner_sid: str) -> str | None:
    if not downloads:
        return None
    download_id = uuid.uuid4().hex
    shared_state.put_download(download_id, {
        "downloads": downloads,
        "owner": owner_sid,
        "created": time.time(),
    }, max_entries=_MAX_STORED)
    return download_id


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Session store: chat history + follow-up context ──
# The signed cookie holds only a small opaque "sid"; conversation content
# lives server-side (full message text would blow past the ~4KB cookie
# limit within a few turns). True LRU with a TTL, so the user who has been
# chatting longest is not the first evicted.
_MAX_SESSIONS = 500
_MAX_HISTORY_PER_SESSION = 60
_SESSION_TTL = 12 * 60 * 60  # seconds


def _get_sid(request: Request) -> str:
    sid = request.session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        request.session["sid"] = sid
    return sid


def _get_session_data(sid: str) -> dict:
    return shared_state.get_session(sid, max_entries=_MAX_SESSIONS, ttl=_SESSION_TTL)


def _save_session_data(sid: str, data: dict):
    shared_state.put_session(sid, data, max_entries=_MAX_SESSIONS, ttl=_SESSION_TTL)


def _append_history(session_data: dict, entry: dict):
    session_data["history"].append(entry)
    if len(session_data["history"]) > _MAX_HISTORY_PER_SESSION:
        session_data["history"] = session_data["history"][-_MAX_HISTORY_PER_SESSION:]


_MAX_FEEDBACK_LOG = 200


def _log_feedback(message: str):
    shared_state.append_feedback({
        "message": message,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }, max_entries=_MAX_FEEDBACK_LOG)


# ==========================================
# PAGES
# ==========================================

@app.get("/", response_class=HTMLResponse)
def chat_page(request: Request):
    sid = _get_sid(request)
    session_data = _get_session_data(sid)
    # Escape "<" so nothing in stored text can be parsed as markup inside the
    # JSON data island — this covers the "<!--<script" script-data-escaped
    # sequence that a bare "</" replacement misses, which could otherwise
    # leave the page permanently unable to load its own JS.
    history_json = json.dumps(session_data["history"]).replace("<", "\\u003c")
    return templates.TemplateResponse(request, "chat.html", {
        "suggested_prompts": vog_core.SUGGESTED_PROMPTS_QUICK,
        "app_build": APP_BUILD,
        "history_json": history_json,
    })


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    authenticated = bool(request.session.get("authenticated"))
    return templates.TemplateResponse(request, "admin.html", {
        "authenticated": authenticated,
        "error": None,
        "app_build": APP_BUILD,
        "admin_enabled": ADMIN_ENABLED,
        "feedback_log": shared_state.list_feedback() if authenticated else [],
        "feedback_durable": shared_state.is_durable(),
    })


@app.post("/admin/login", response_class=HTMLResponse)
def admin_login(request: Request, password: str = Form(...)):
    _rate_limit(request, "login")
    if ADMIN_ENABLED and hmac.compare_digest(password, ADMIN_PASSWORD):
        # Regenerate the session id on privilege change so a pre-auth
        # session cannot be fixated into an authenticated one.
        request.session.clear()
        request.session["sid"] = uuid.uuid4().hex
        request.session["authenticated"] = True
        return RedirectResponse(url="/admin", status_code=303)
    log.warning("Failed admin login from %s", _client_ip(request))
    return templates.TemplateResponse(request, "admin.html", {
        "authenticated": False,
        "error": "Invalid credentials" if ADMIN_ENABLED else "Admin is disabled: ADMIN_PASSWORD is not configured.",
        "app_build": APP_BUILD,
        "admin_enabled": ADMIN_ENABLED,
        "feedback_log": [],
        "feedback_durable": shared_state.is_durable(),
    })


@app.post("/admin/logout")
def admin_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/admin", status_code=303)


@app.get("/admin/toolcall-probe")
async def admin_toolcall_probe(request: Request):
    """Stage 0 go/no-go for the agentic rebuild: can this model reliably
    pick the right tool with the right arguments? Runs ~10 real questions
    against the live model using the server's configured key, so nobody
    has to move a credential around. Read-only — it plans tool calls but
    never executes them, so it cannot touch the index."""
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not GROQ_API_KEY:
        raise HTTPException(status_code=400, detail="Groq API key is not configured")
    try:
        from scripts.toolcall_probe import run_probe
        return await run_in_threadpool(run_probe, GROQ_API_KEY, vog_core.GROQ_MODEL)
    except Exception as e:
        ref = uuid.uuid4().hex[:8]
        log.exception("Tool-call probe failed [ref=%s]", ref)
        raise HTTPException(status_code=500, detail=f"Probe failed (ref {ref}): {type(e).__name__}")


@app.post("/admin/feedback-log/clear")
def admin_clear_feedback_log(request: Request):
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    shared_state.clear_feedback()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/ingest")
async def admin_ingest(request: Request, file: UploadFile = File(...), purge_first: str = Form("")):
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not PINECONE_API_KEY:
        raise HTTPException(status_code=400, detail="Pinecone API key is not configured")

    # Read with a hard cap. An .xlsx is a ZIP archive, so an unbounded read
    # lets a small crafted file expand into an OOM on a 512MB instance.
    chunks, total = [], 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            return Response(
                content=json.dumps({"success": False, "error": f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)}MB limit."}),
                status_code=413, media_type="application/json")
        chunks.append(chunk)
    file_bytes = b"".join(chunks)

    try:
        # run_ingestion is fully synchronous and can run for minutes. Called
        # directly from an async handler it would block the event loop and
        # take the whole server down (including /health, which makes Render
        # restart the instance mid-upload).
        result = await run_in_threadpool(
            vog_core.run_ingestion, file_bytes, PINECONE_API_KEY,
            str(purge_first).lower() in ("1", "true", "on", "yes"),
        )
        return {
            "success": True,
            "total_records": result["total_records"],
            "skipped": result.get("skipped", []),
            "summary": result.get("summary", {}),
        }
    except ValueError as e:
        return Response(content=json.dumps({"success": False, "error": str(e)}), status_code=400, media_type="application/json")
    except Exception as e:
        ref = uuid.uuid4().hex[:8]
        log.exception("Ingestion failed [ref=%s]", ref)
        return Response(
            content=json.dumps({"success": False, "error": f"Ingestion failed unexpectedly (ref {ref}). Check server logs."}),
            status_code=500, media_type="application/json")


# ==========================================
# CHAT (Server-Sent Events)
# ==========================================

def _user_error(exc: Exception, what: str) -> str:
    """Log the real exception, hand the user a reference id. Provider SDK
    exceptions embed index hostnames, request ids and quota details, so
    str(e) must never reach the browser."""
    ref = uuid.uuid4().hex[:8]
    log.exception("%s [ref=%s]", what, ref)
    return f"{what} (ref {ref}). Please try again — if it persists, share this reference with the team."


@app.get("/chat")
def chat(request: Request, q: str = Query(..., min_length=1, max_length=MAX_QUERY_CHARS)):
    """ Streams the answer as Server-Sent Events. Event types: - "start": {badge, header} — sent once, right before token streaming begins (kind="normal" only). - "token": {token} — one LLM token at a time. - "final": {kind, badge, header, reply|full_response, chart, download_id} — always the last event; frontend renders the finished bubble + chart + download links from this. - "error": {message} Follow-up context (see vog_core.detect_followup_reference) and full chat history are both kept server-side per session — nothing round-trips through the client beyond the query text itself, so a page refresh doesn't lose either. """
    _rate_limit(request, "chat")
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    sid = _get_sid(request)
    session_data = _get_session_data(sid)
    prior_context = session_data.get("prior_context")

    _append_history(session_data, {"role": "user", "content": q})
    _save_session_data(sid, session_data)

    def event_stream():
        # Open the socket immediately. The pipeline below can take many
        # seconds before the first token, and a silent socket gets closed
        # by the proxy — which surfaces to the user as a generic failure
        # while the server keeps working and keeps spending.
        yield _sse("status", {"state": "working"})

        try:
            state = vog_core.process_chat_query(q, PINECONE_API_KEY, GROQ_API_KEY, prior_context=prior_context)
        except Exception as e:
            yield _sse("error", {"message": _user_error(e, "Could not process that question")})
            return

        kind = state["kind"]

        if kind == "meta_feedback":
            _log_feedback(q)

        if kind in ("blocked", "no_key", "no_data", "meta_feedback", "capability"):
            if "context" in state:
                session_data["prior_context"] = state["context"]
            _append_history(session_data, {
                "role": "assistant", "kind": kind, "badge": None,
                "content": state["reply"], "chart": None, "download_id": None,
            })
            _save_session_data(sid, session_data)
            yield _sse("final", {
                "kind": kind, "badge": None, "header": "",
                "reply": state["reply"], "chart": None, "download_id": None,
            })
            return

        if kind in ("ranking", "trend"):
            download_id = _store_downloads(state.get("downloads"), sid)
            _append_history(session_data, {
                "role": "assistant", "kind": kind, "badge": state.get("badge"),
                "content": state["reply"], "chart": state.get("chart"),
                "download_id": download_id,
            })
            _save_session_data(sid, session_data)
            yield _sse("final", {
                "kind": kind, "badge": state.get("badge"), "header": "",
                "reply": state["reply"], "chart": state.get("chart"),
                "download_id": download_id,
            })
            return

        # kind == "normal" — stream Groq tokens live
        badge = state["badge"]
        header = state["header"]
        yield _sse("start", {"badge": badge, "header": header})

        full_response = ""
        stream = None
        llm_failed = False
        try:
            stream = vog_core.call_groq(
                state["system_prompt"], state["user_prompt"], GROQ_API_KEY,
                max_tokens=state["response_token_budget"]
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_response += token
                    yield _sse("token", {"token": token})
        except Exception as e:
            llm_failed = True
            full_response = _user_error(e, "The answer could not be generated")
            yield _sse("token", {"token": full_response})
        finally:
            # Close the upstream stream even if the client disconnected
            # mid-response, so the connection is not left to GC.
            try:
                if stream is not None and hasattr(stream, "close"):
                    stream.close()
            except Exception:
                pass

        if llm_failed:
            # Don't build a PPTX out of an error string, and don't make a
            # second Groq call for suggestions when Groq just failed.
            _append_history(session_data, {
                "role": "assistant", "kind": "normal", "badge": badge,
                "content": full_response, "chart": None, "download_id": None,
            })
            _save_session_data(sid, session_data)
            yield _sse("final", {
                "kind": "normal", "badge": badge, "header": "",
                "reply": full_response, "chart": None, "download_id": None,
            })
            return

        result = vog_core.finalize_normal_response(state, full_response)
        download_id = _store_downloads(result["downloads"], sid)
        if "context" in state:
            # Carry the reply text forward too, so a later "what other
            # insights..." follow-up can be told what was already said and
            # avoid repeating it.
            session_data["prior_context"] = {**state["context"], "last_reply": full_response}

        # Best-effort follow-up suggestions — never let a failure here
        # affect the main answer, which is already fully delivered above.
        suggestions = vog_core.generate_followup_suggestions(
            state["query_intent"], state.get("subject_label"), state["timeframe_label"],
            full_response, GROQ_API_KEY,
        )

        _append_history(session_data, {
            "role": "assistant", "kind": "normal", "badge": badge,
            "content": header + full_response, "chart": result["chart"],
            "download_id": download_id, "suggestions": suggestions,
        })
        _save_session_data(sid, session_data)
        yield _sse("final", {
            "kind": "normal", "badge": badge, "header": header,
            "reply": full_response, "chart": result["chart"],
            "download_id": download_id, "suggestions": suggestions,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.post("/chat/clear")
def chat_clear(request: Request):
    sid = _get_sid(request)
    shared_state.clear_session(sid, max_entries=_MAX_SESSIONS, ttl=_SESSION_TTL)
    return {"success": True}


# ==========================================
# DOWNLOADS
# ==========================================

_DOWNLOAD_META = {
    "csv":   ("text/csv", "chart_data.csv"),
    "excel": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "vog_export.xlsx"),
    "pptx":  ("application/vnd.openxmlformats-officedocument.presentationml.presentation", "vog_report.pptx"),
}


@app.get("/download/{download_id}/{kind}")
def download(request: Request, download_id: str, kind: str):
    entry = shared_state.get_download(download_id)
    if not entry or kind not in _DOWNLOAD_META:
        raise HTTPException(status_code=404, detail="Download not found or expired")

    # Bind to the owning session: the id travels in an <a href>, so it ends
    # up in browser history and proxy logs. Without this it is an
    # indefinite bearer token for someone else's exported data.
    if entry.get("owner") and entry["owner"] != _get_sid(request):
        raise HTTPException(status_code=404, detail="Download not found or expired")
    if time.time() - entry.get("created", 0) > _DOWNLOAD_TTL:
        raise HTTPException(status_code=404, detail="Download not found or expired")

    blob = (entry.get("downloads") or {}).get(kind)
    if not blob:
        raise HTTPException(status_code=404, detail="Download not found or expired")

    media_type, filename = _DOWNLOAD_META[kind]
    return Response(
        content=blob, media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        }
    )


@app.get("/health")
def health():
    """Liveness only — deliberately does not echo the build string, model
    name or QA notes to unauthenticated callers."""
    return {"status": "ok"}


@app.get("/health/detail")
def health_detail(request: Request):
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "status": "ok",
        "build": APP_BUILD,
        "state_backend": shared_state.backend_name(),
        "durable_state": shared_state.is_durable(),
        "admin_enabled": ADMIN_ENABLED,
    }
