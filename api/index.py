"""Vercel entrypoint. Three endpoints, no server-side state.

What is deliberately NOT here, versus the Render app:

- Sessions. Every invocation may be a different machine, so an in-process
  session store is a store that silently forgets. Conversation history and
  follow-up context live in the browser and ride along with each request.
- The admin panel and ingestion. Uploading a workbook is a long, stateful
  job that does not belong in a request-scoped function, and it kept a
  password in the deployment. Ingestion runs from Streamlit or locally.
- Download ids. Nothing can hold bytes between two invocations, so exports
  are regenerated on demand from the (deterministic) plan.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vog import compose, llm, retrieval
from vog.catalog import SUGGESTED_PROMPTS_QUICK
from vog.plan import MODE_REPLY, build_plan

MAX_QUERY_CHARS = 500
MAX_CONTEXT_CHARS = 4000

app = FastAPI(title="Voice of Grower", docs_url=None, redoc_url=None)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _context(raw: str | None) -> dict:
    """Follow-up context from the browser. Untrusted: it decides only which
    product/crop/intent to inherit, and every one of those is re-validated
    downstream against the real catalog before it reaches a query."""
    if not raw or len(raw) > MAX_CONTEXT_CHARS:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {k: parsed.get(k) for k in ("product", "crop", "intent", "last_reply")}


def _prepare(q: str, ctx: str | None):
    """Plan -> evidence -> answer. Shared by /chat and /export so a download
    is built from exactly the same data as the answer above it."""
    # Planned before connecting: a capability question or an off-topic one
    # is answered without retrieval, and shouldn't fail because the index is
    # unreachable or ask the user to wait for a round trip it never needs.
    prior = _context(ctx)
    plan = build_plan(q, prior_context=prior)
    if plan.mode == MODE_REPLY:
        return plan, retrieval.Evidence(), compose.compose(plan, retrieval.Evidence())

    if not PINECONE_API_KEY:
        raise HTTPException(503, "Search is not configured (PINECONE_API_KEY is unset).")
    pc, index = retrieval.connect(PINECONE_API_KEY)
    latest = retrieval.dataset_extent(index)
    plan = build_plan(q, latest_month_year=latest, prior_context=prior)

    # Only when the regexes found nothing to go on is a classification round
    # trip worth its latency — and what comes back is validated against the
    # real catalogs before it can influence anything.
    if plan.needs_assist and GROQ_API_KEY:
        assist = llm.classify_query(q, GROQ_API_KEY)
        if assist:
            plan = build_plan(q, latest_month_year=latest, prior_context=prior, assist=assist)

    evidence = retrieval.gather(plan, index, pc)
    return plan, evidence, compose.compose(plan, evidence)


@app.get("/api/prompts")
def prompts():
    return {"prompts": SUGGESTED_PROMPTS_QUICK}


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "pinecone_configured": bool(PINECONE_API_KEY),
        "groq_configured": bool(GROQ_API_KEY),
        "model": llm.GROQ_MODEL,
    }


@app.get("/api/chat")
def chat(q: str = Query(..., min_length=1, max_length=MAX_QUERY_CHARS),
         ctx: str | None = Query(None)):
    def stream():
        try:
            plan, evidence, answer = _prepare(q, ctx)
        except HTTPException as e:
            yield _sse("error", {"message": e.detail})
            return
        except Exception:
            # The exception text can carry index names and key fragments, so
            # it goes to the logs, not to the browser.
            import traceback; traceback.print_exc()
            yield _sse("error", {"message": "Couldn't reach the feedback index. Please try again."})
            return

        yield _sse("start", {"badge": answer.badge})

        # Anything with a checkable right answer is already written; only
        # prose streams. The header goes out as the first token either way,
        # so the client just accumulates whatever arrives.
        if answer.kind != "prompt":
            text = answer.text
            if answer.top and GROQ_API_KEY:
                extra = compose.narrate_result(
                    plan.rank_dimension or "month", answer.top[0], answer.top[1],
                    _bullets(evidence), GROQ_API_KEY)
                if extra:
                    text += '\n\n' + extra
            yield _sse("token", {"token": text})
            yield from _finish(plan, answer, text)
            return

        if not GROQ_API_KEY:
            yield _sse("error", {"message": "The language model is not configured (GROQ_API_KEY is unset)."})
            return

        collected = []
        if answer.header:
            yield _sse("token", {"token": answer.header})
        try:
            for chunk in llm.stream_answer(answer.system_prompt, answer.user_prompt,
                                           GROQ_API_KEY, max_tokens=answer.token_budget):
                token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if token:
                    collected.append(token)
                    yield _sse("token", {"token": token})
        except Exception:
            import traceback; traceback.print_exc()
            yield _sse("error", {"message": "The model stopped responding partway through."})
            return

        body = "".join(collected).strip()
        if not body:
            # An empty completion is silent failure, and looks identical to a
            # working app with nothing to say. Name it.
            yield _sse("error", {"message": "The model returned an empty response. Please try again."})
            return
        yield from _finish(plan, answer, answer.header + body)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-transform",
                                      "X-Accel-Buffering": "no"})


def _bullets(evidence) -> list[str]:
    """Feedback lines to ground the narration in. Counting modes carry raw
    matches rather than sentiment buckets, so read the values directly."""
    if evidence.segments:
        return [b for s in evidence.segments for b in (s.negative + s.positive)]
    out = []
    for m in evidence.matches[:200]:
        value = str((m.get("metadata") or {}).get("value", "")).strip()
        if value:
            out.append(value)
    return out


def _finish(plan, answer, full_text: str):
    suggestions = (compose.suggest_followups(plan, full_text, GROQ_API_KEY)
                   if GROQ_API_KEY else [])
    yield _sse("final", {
        "reply": full_text,
        "badge": answer.badge,
        "chart": answer.chart,
        "suggestions": suggestions,
        # The browser stores this and sends it back on the next turn.
        "context": {**answer.context, "last_reply": full_text[:1500]},
        # Exports are regenerated from the query, so the client only needs
        # to know whether there is anything worth downloading.
        "exportable": bool(answer.export_rows),
    })


_EXPORTS = {
    "csv": ("text/csv", "csv"),
    "excel": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
    "pptx": ("application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx"),
}


@app.post("/api/export")
async def export(request: Request):
    """Regenerate a download. Stateless by necessity — there is no memory
    between invocations to have stashed the bytes in."""
    payload = await request.json()
    kind = str(payload.get("kind", ""))
    q = str(payload.get("q", ""))[:MAX_QUERY_CHARS]
    if kind not in _EXPORTS or not q:
        raise HTTPException(400, "Unknown export kind.")

    _, _, answer = _prepare(q, payload.get("ctx"))
    files = compose.build_exports(answer, str(payload.get("answer_text", ""))[:4000])
    if kind not in files:
        raise HTTPException(404, "Nothing to export for that question.")

    media_type, ext = _EXPORTS[kind]
    return Response(
        content=files[kind], media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="voice-of-grower.{ext}"'},
    )


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException):
    return JSONResponse({"message": exc.detail}, status_code=exc.status_code)
