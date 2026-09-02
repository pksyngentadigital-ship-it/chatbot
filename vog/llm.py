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


_CLASSIFY_PROMPT = (
    "You are a query classifier for an agricultural grower-feedback chatbot. "
    "Classify the user's question.\n\n"
    'Return ONLY a JSON object: {"product": "<name or null>", "crop": "<name or null>", '
    '"intent": "<one of: complaint, positive, suggestion, sentiment, topics>"}\n\n'
    "Intent definitions — pick by what the user actually wants, not by "
    "which words appear:\n"
    "- complaint: problems, issues, dissatisfaction, things going wrong, "
    "root causes, why something failed.\n"
    "- positive: praise, what is working well, satisfaction, successes.\n"
    "- suggestion: what people want changed, asked for, or improved; "
    "requests, expectations, ideas.\n"
    "- topics: what is being discussed most; themes, subjects, "
    "'what are people talking about' — NOT whether it is good or bad.\n"
    "- sentiment: a general read of both good and bad together; use this "
    "only when none of the above fits better.\n\n"
    "Examples:\n"
    'Q: "why are growers unhappy with delivery" -> {"product": null, "crop": null, "intent": "complaint"}\n'
    'Q: "what is working well this season" -> {"product": null, "crop": null, "intent": "positive"}\n'
    'Q: "what do growers wish we did differently" -> {"product": null, "crop": null, "intent": "suggestion"}\n'
    'Q: "what is dominating the conversation" -> {"product": null, "crop": null, "intent": "topics"}\n'
    'Q: "give me a read on how wheat growers feel" -> {"product": null, "crop": "wheat", "intent": "sentiment"}\n\n'
    "If you are not confident about product or crop, use null. Never invent a "
    "product or crop name — only name one that appears in the user's own question."
)


def classify_query(user_query: str, api_key: str) -> dict | None:
    """Propose product / crop / intent for a question the regexes couldn't read.

    The guardrail is the validation, not the prompt: every proposed value is
    checked against the real catalogs and the fixed intent enum, so the model
    can only ever hit a name that already exists — it cannot introduce one.
    That check is what caught an invented product name during testing before
    it reached anybody. Returns None when nothing survives validation.
    """
    parsed = complete_json(_CLASSIFY_PROMPT, user_query, api_key, max_tokens=200)
    if not isinstance(parsed, dict):
        return None

    from vog.catalog import CROP_LIST, PRODUCT_LIST
    from vog.plan import INTENTS

    result = {"product": None, "crop": None, "intent": None}
    product = parsed.get("product")
    if isinstance(product, str) and product.strip().lower() in PRODUCT_LIST:
        result["product"] = product.strip().lower()
    crop = parsed.get("crop")
    if isinstance(crop, str) and crop.strip().lower() in CROP_LIST:
        result["crop"] = crop.strip().lower()
    if parsed.get("intent") in INTENTS:
        result["intent"] = parsed["intent"]

    return result if any(result.values()) else None
