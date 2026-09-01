"""
Shared state backend: sessions, downloads, and the flagged-feedback log.

Two interchangeable implementations behind one API:

  * Redis, when REDIS_URL is set — survives restarts and is correct with
    more than one server process.
  * In-process dicts otherwise — zero setup, but everything is lost on
    restart and is wrong across replicas.

The in-memory path is a real LRU with TTL (not the FIFO-by-creation it
replaced, which evicted the longest-active user first), so behaviour is
consistent whichever backend is in play.

Only the feedback log genuinely needs durability: the chatbot tells users
"I've logged this for the team to review", and on a free-tier instance
that spins down after ~15 minutes idle, an in-memory list makes that
sentence false. is_durable() lets the caller tell the truth either way.
"""

import json
import os
import time
from collections import OrderedDict

_REDIS_URL = os.getenv("REDIS_URL")

_redis = None
if _REDIS_URL:
    try:
        import redis  # type: ignore

        _redis = redis.Redis.from_url(_REDIS_URL, decode_responses=True, socket_timeout=3)
        _redis.ping()
    except Exception as e:  # pragma: no cover - depends on deployment
        print(f"[shared_state] REDIS_URL set but unusable ({e}); falling back to in-memory.")
        _redis = None

_SESSION_PREFIX = "vog:session:"
_DOWNLOAD_PREFIX = "vog:download:"
_FEEDBACK_KEY = "vog:feedback"

# In-memory fallbacks
_sessions: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_downloads: "OrderedDict[str, dict]" = OrderedDict()
_feedback: list[dict] = []


def backend_name() -> str:
    return "redis" if _redis else "memory"


def is_durable() -> bool:
    """True when state survives a process restart. Drives the wording of
    the user-facing 'I've logged this' acknowledgement."""
    return _redis is not None


# ──────────────────────────── sessions ────────────────────────────

def _new_session() -> dict:
    return {"history": [], "prior_context": None}


def get_session(sid: str, max_entries: int, ttl: int) -> dict:
    if _redis:
        raw = _redis.get(_SESSION_PREFIX + sid)
        if raw:
            try:
                return json.loads(raw)
            except ValueError:
                pass
        return _new_session()

    _expire_sessions(ttl)
    entry = _sessions.get(sid)
    if entry is None:
        data = _new_session()
        _sessions[sid] = (time.time(), data)
    else:
        data = entry[1]
        _sessions[sid] = (time.time(), data)  # refresh recency
    _sessions.move_to_end(sid)
    _trim(_sessions, max_entries)
    return data


def put_session(sid: str, data: dict, max_entries: int, ttl: int):
    if _redis:
        _redis.setex(_SESSION_PREFIX + sid, ttl, json.dumps(data))
        return
    _sessions[sid] = (time.time(), data)
    _sessions.move_to_end(sid)
    _trim(_sessions, max_entries)


def clear_session(sid: str, max_entries: int, ttl: int):
    """Reset a conversation in place. Deliberately routed through
    put_session rather than assigning to the dict directly — a direct
    assignment used to create an entry the LRU never registered, so it
    could never be evicted and unauthenticated calls grew memory forever."""
    put_session(sid, _new_session(), max_entries=max_entries, ttl=ttl)


def _expire_sessions(ttl: int):
    cutoff = time.time() - ttl
    stale = [k for k, (ts, _) in _sessions.items() if ts < cutoff]
    for k in stale:
        _sessions.pop(k, None)


def _trim(store: OrderedDict, max_entries: int):
    while len(store) > max_entries:
        store.popitem(last=False)


# ─────────────────────────── downloads ────────────────────────────

def put_download(download_id: str, entry: dict, max_entries: int):
    if _redis:
        # entry holds raw bytes; Redis needs a hash, so store the parts
        # separately and keep the metadata as JSON.
        key = _DOWNLOAD_PREFIX + download_id
        mapping = {
            "owner": entry["owner"],
            "created": str(entry["created"]),
        }
        for kind, blob in (entry["downloads"] or {}).items():
            if blob:
                mapping[f"blob:{kind}"] = blob.decode("latin-1")
        _redis.hset(key, mapping=mapping)
        _redis.expire(key, 30 * 60)
        return
    _downloads[download_id] = entry
    _downloads.move_to_end(download_id)
    _trim(_downloads, max_entries)


def get_download(download_id: str) -> dict | None:
    if _redis:
        raw = _redis.hgetall(_DOWNLOAD_PREFIX + download_id)
        if not raw:
            return None
        downloads = {
            k.split(":", 1)[1]: v.encode("latin-1")
            for k, v in raw.items()
            if k.startswith("blob:")
        }
        return {
            "owner": raw.get("owner", ""),
            "created": float(raw.get("created", "0")),
            "downloads": downloads,
        }
    return _downloads.get(download_id)


# ───────────────────────── feedback log ───────────────────────────

def append_feedback(entry: dict, max_entries: int):
    if _redis:
        _redis.lpush(_FEEDBACK_KEY, json.dumps(entry))
        _redis.ltrim(_FEEDBACK_KEY, 0, max_entries - 1)
        return
    _feedback.insert(0, entry)
    del _feedback[max_entries:]


def list_feedback() -> list[dict]:
    if _redis:
        out = []
        for raw in _redis.lrange(_FEEDBACK_KEY, 0, -1):
            try:
                out.append(json.loads(raw))
            except ValueError:
                continue
        return out
    return list(_feedback)


def clear_feedback():
    if _redis:
        _redis.delete(_FEEDBACK_KEY)
        return
    _feedback.clear()


def reset_all_for_tests():
    """Test helper — clears the in-memory stores between test cases."""
    _sessions.clear()
    _downloads.clear()
    _feedback.clear()
