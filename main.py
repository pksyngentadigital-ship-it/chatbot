"""
Voice of Grower — FastAPI + lightweight HTML/JS frontend.

Same-origin app: this process serves both the API and the UI (Jinja2 shell
+ vanilla JS + Chart.js from a CDN), so there's no CORS to configure and
no JS build step. All business logic lives in vog_core.py — this file is
purely the web-framework glue (routing, sessions, streaming, downloads).
"""

import json
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

import vog_core

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-only-secret-change-me")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

APP_BUILD = "2026-07-15-v1 (FastAPI + HTML/JS rebuild, Plan B)"

app = FastAPI(title="Voice of Grower")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── In-memory download store ──
# Each finalized chat turn (ranking / trend / normal) may produce CSV/Excel/
# PPTX bytes. Rather than round-tripping those bytes through the SSE stream
# as base64 JSON, we stash them here under a short-lived id and hand the
# browser a plain download link. Fine for a single-process internal tool;
# a multi-instance deployment would swap this for Redis/S3.
_DOWNLOAD_STORE: dict[str, dict] = {}
_DOWNLOAD_ORDER: list[str] = []
_MAX_STORED = 200


def _store_downloads(downloads: dict | None) -> str | None:
    if not downloads:
        return None
    download_id = uuid.uuid4().hex
    _DOWNLOAD_STORE[download_id] = downloads
    _DOWNLOAD_ORDER.append(download_id)
    if len(_DOWNLOAD_ORDER) > _MAX_STORED:
        oldest = _DOWNLOAD_ORDER.pop(0)
        _DOWNLOAD_STORE.pop(oldest, None)
    return download_id


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ==========================================
# PAGES
# ==========================================

@app.get("/", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "suggested_prompts": vog_core.SUGGESTED_PROMPTS,
        "app_build": APP_BUILD,
    })


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "authenticated": bool(request.session.get("authenticated")),
        "error": None,
        "app_build": APP_BUILD,
    })


@app.post("/admin/login", response_class=HTMLResponse)
def admin_login(request: Request, password: str = Form(...)):
    if password == ADMIN_PASSWORD:
        request.session["authenticated"] = True
        return RedirectResponse(url="/admin", status_code=303)
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "authenticated": False,
        "error": "Invalid credentials",
        "app_build": APP_BUILD,
    })


@app.post("/admin/logout")
def admin_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/ingest")
async def admin_ingest(request: Request, file: UploadFile = File(...)):
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not PINECONE_API_KEY:
        raise HTTPException(status_code=400, detail="Pinecone API key is not configured")

    file_bytes = await file.read()
    try:
        result = vog_core.run_ingestion(file_bytes, PINECONE_API_KEY)
        return {"success": True, "total_records": result["total_records"]}
    except ValueError as e:
        return Response(content=json.dumps({"success": False, "error": str(e)}), status_code=400, media_type="application/json")
    except Exception as e:
        return Response(content=json.dumps({"success": False, "error": str(e)}), status_code=500, media_type="application/json")


# ==========================================
# CHAT (Server-Sent Events)
# ==========================================

@app.get("/chat")
def chat(q: str):
    """ Streams the answer as Server-Sent Events. Event types: - "start": {badge, header} — sent once, right before token streaming begins (kind="normal" only). - "token": {token} — one LLM token at a time. - "final": {kind, badge, header, reply|full_response, chart, download_id} — always the last event; frontend renders the finished bubble + chart + download links from this. - "error": {message} """

    def event_stream():
        try:
            state = vog_core.process_chat_query(q, PINECONE_API_KEY, GROQ_API_KEY)
        except Exception as e:
            yield _sse("error", {"message": str(e)})
            return

        kind = state["kind"]

        if kind in ("blocked", "no_key", "no_data"):
            yield _sse("final", {
                "kind": kind, "badge": None, "header": "",
                "reply": state["reply"], "chart": None, "download_id": None,
            })
            return

        if kind in ("ranking", "trend"):
            download_id = _store_downloads(state.get("downloads"))
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
            full_response = f"Operational Processing Error: {e}"
            yield _sse("token", {"token": full_response})

        result = vog_core.finalize_normal_response(state, full_response)
        download_id = _store_downloads(result["downloads"])
        yield _sse("final", {
            "kind": "normal", "badge": badge, "header": header,
            "reply": full_response, "chart": result["chart"],
            "download_id": download_id,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


# ==========================================
# DOWNLOADS
# ==========================================

_DOWNLOAD_META = {
    "csv":   ("text/csv", "chart_data.csv"),
    "excel": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "vog_export.xlsx"),
    "pptx":  ("application/vnd.openxmlformats-officedocument.presentationml.presentation", "vog_report.pptx"),
}


@app.get("/download/{download_id}/{kind}")
def download(download_id: str, kind: str):
    entry = _DOWNLOAD_STORE.get(download_id)
    if not entry or kind not in _DOWNLOAD_META or not entry.get(kind):
        raise HTTPException(status_code=404, detail="Download not found or expired")
    media_type, filename = _DOWNLOAD_META[kind]
    return Response(
        content=entry[kind], media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.get("/health")
def health():
    return {"status": "ok", "build": APP_BUILD}
