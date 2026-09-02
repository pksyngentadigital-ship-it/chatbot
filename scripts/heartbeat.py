#!/usr/bin/env python3
"""
Synthetic monitor: asks the live app a real question and checks it answers.

Rationale: on 2026-09-01 Groq retired every Llama model without warning and
every single chat response began failing. Nothing noticed. /health kept
returning 200 the whole time, because it only proves the web process is
up — it never touches the model. The outage was found by a human happening
to chat with production.

This sends an actual query through the real SSE endpoint and fails if the
reply is empty, an error, or too slow. Run it on a schedule (GitHub
Actions cron, Render cron job, or any external uptime checker that can run
a command) and alert on a non-zero exit.

    python scripts/heartbeat.py --url https://<your-project>.vercel.app

Exit codes:
    0  healthy
    1  unhealthy (details on stderr)
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

DEFAULT_QUERY = "what do growers think about isabion?"
# Render's free tier cold-starts, which legitimately takes a while; the
# threshold is for "the model stopped answering", not for cold-start speed.
DEFAULT_TIMEOUT = 90


def check(base_url: str, query: str, timeout: int) -> tuple[bool, str]:
    url = f"{base_url.rstrip('/')}/api/chat?q={urllib.parse.quote(query)}"
    started = time.monotonic()

    try:
        req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False, f"HTTP {resp.status}"
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return False, f"request failed: {e}"

    elapsed = time.monotonic() - started

    # Parse the SSE stream: an "error" event, or a "final" event whose reply
    # is empty, both mean the model is not actually answering.
    events = []
    for block in raw.split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if name and data:
            events.append((name, data))

    if not events:
        return False, f"no SSE events received in {elapsed:.1f}s"

    for name, data in events:
        if name == "error":
            return False, f"error event: {data[:300]}"

    finals = [d for n, d in events if n == "final"]
    if not finals:
        return False, f"stream ended with no final event after {elapsed:.1f}s"

    try:
        payload = json.loads(finals[-1])
    except ValueError:
        return False, "final event was not valid JSON"

    reply = (payload.get("reply") or "").strip()
    if not reply:
        # This is the exact shape of the gpt-oss-20b failure: a clean
        # stream, no error, and zero content tokens.
        return False, f"final reply was EMPTY after {elapsed:.1f}s — the model returned no content"
    if "Operational Processing Error" in reply or "could not be generated" in reply:
        return False, f"reply carried an error: {reply[:200]}"

    return True, f"ok in {elapsed:.1f}s, {len(reply)} chars"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Base URL of the deployment")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = ap.parse_args()

    healthy, detail = check(args.url, args.query, args.timeout)
    stream = sys.stdout if healthy else sys.stderr
    print(f"[heartbeat] {'HEALTHY' if healthy else 'UNHEALTHY'}: {detail}", file=stream)
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
