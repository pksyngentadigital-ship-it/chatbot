"""Groq access — one place, so the model and its quirks are configured once."""

import json

import re

from vog.catalog import GROQ_MODEL

# gpt-oss models are reasoning models: without this they can spend the whole
# token budget on hidden chain-of-thought and return an empty completion with
# no error at all, which is indistinguishable from a working-but-silent app.
REASONING_EFFORT = "low"


def _client(api_key: str):
    # Lazy: keeps import-time cost off the cold start path.
    from groq import Groq
    return Groq(api_key=api_key)


def stream_answer(system_prompt: str, user_prompt: str, api_key: str, max_tokens: int = 600):
    """Streaming completion. Caller forwards chunks however it likes."""
    return _client(api_key).chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
        reasoning_effort=REASONING_EFFORT,
        stream=True,
    )


def complete(system_prompt: str | None, user_prompt: str, api_key: str,
             max_tokens: int = 300, temperature: float = 0.2) -> str:
    """Non-streaming completion. Returns "" on any failure — every caller
    here is an enhancement, never the main answer, so a failure must
    degrade quietly rather than break the response."""
    if not api_key:
        return ""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        resp = _client(api_key).chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=REASONING_EFFORT,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def complete_json(system_prompt: str, user_prompt: str, api_key: str, max_tokens: int = 250):
    """Completion parsed as JSON, or None. Models asked for raw JSON still
    wrap it in a markdown fence often enough to be worth stripping."""
    raw = complete(system_prompt, user_prompt, api_key, max_tokens=max_tokens, temperature=0.0)
    if not raw:
        return None
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        return None


def embed(texts: list[str], pc, input_type: str = "query") -> list[list[float]]:
    """Embed a batch in ONE call. The old code embedded one candidate word
    per request inside a loop, so an unrecognised multi-word question cost
    a round trip per word before the answer even started."""
    from vog.catalog import EMBEDDING_DIMENSION
    resp = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=texts,
        parameters={"input_type": input_type, "dimension": EMBEDDING_DIMENSION},
    )
    return [item.values for item in resp]
