# actionable items-> in a video if we have to perform an action. In the context of this
# project, actionable items refer to specific tasks, follow-ups, or responsibilities
# mentioned during a meeting or video that require someone to take action.
# decisions taken, questions asked in meetings

import hashlib
import json

from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Mistral's free "Experiment" tier is capped at ~1 request/second but 500k
# tokens/minute. Module-level singleton so every getllm() call shares the
# same bucket instead of resetting it.
mistral_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.8,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

# Groq's free tier TPM (tokens/minute) cap is much tighter than Mistral's.
groq_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.4,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

# Gemini free tier (gemini-2.5-flash) is roughly 10 RPM / 250k TPM depending
# on your account — conservative pacing here, tune if you know your actual cap.
gemini_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.35,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


def getllm():
    return ChatMistralAI(
        model="mistral-medium-latest",
        temperature=0.2,
        rate_limiter=mistral_rate_limiter,
    ).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


def getllm2():
    # No retry on a TPD-exhausted provider — with_fallbacks() below handles
    # failover to Mistral/Gemini instead, which is faster than retrying Groq
    # itself against a quota that won't reset for hours.
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        rate_limiter=groq_rate_limiter,
    )


def getllm3():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.2,
        rate_limiter=gemini_rate_limiter,
    ).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)


def splitter_text():
    return RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=500)


# --- Combined extraction chain -------------------------------------------
# Previously: actionable_items(), key_decisions(), and extract_questions()
# each split the transcript and ran their own full batch of LLM calls over
# every chunk — 3x the token volume for one transcript. This single chain
# asks for all three in one JSON response per chunk instead, cutting total
# tokens sent (and TPM pressure) roughly to a third.
_COMBINED_SYSTEM_PROMPT = """You are an expert meeting analyst. From the meeting transcript chunk provided, extract three things and respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:

{{
  "action_items": [
    {{"task": "...", "owner": "...", "deadline": "..."}}
  ],
  "key_decisions": ["..."],
  "open_questions": ["..."]
}}

Rules:
- For action_items: owner is who is responsible (write "Not specified" if unclear), deadline is "Not specified" if not mentioned.
- For key_decisions: only decisions actually made in this chunk.
- For open_questions: unresolved questions or topics needing follow-up.
- If a category has nothing in this chunk, return an empty list for it.
- Do not invent content that isn't in the transcript chunk."""


def _prompt():
    return ChatPromptTemplate.from_messages([
        ("system", _COMBINED_SYSTEM_PROMPT),
        ("human", "{text}"),
    ])


# Build each provider's chain once at module scope (mirrors the rate-limiter
# singleton pattern above).
_chain_mistral = _prompt() | getllm() | StrOutputParser()
_chain_groq = _prompt() | getllm2() | StrOutputParser()
_chain_gemini = _prompt() | getllm3() | StrOutputParser()

# Each provider falls through to the other two on failure (rate limit, 429,
# outage, etc.) instead of blowing up the whole batch. Groq's free tier has a
# low tokens-per-day cap on top of its per-minute one, so once it's tapped
# out for the day, calls routed to it need somewhere else to go automatically
# — with_fallbacks() handles that per-call, no manual retry logic needed.
_resilient_mistral = _chain_mistral.with_fallbacks([_chain_gemini, _chain_groq])
_resilient_groq = _chain_groq.with_fallbacks([_chain_mistral, _chain_gemini])
_resilient_gemini = _chain_gemini.with_fallbacks([_chain_mistral, _chain_groq])

# Round-robin chunks across three resilient chains so no single provider's
# TPM/TPD/RPM cap takes the full token load of a transcript, and any one
# provider being exhausted doesn't stall the whole run.
_PROVIDER_CHAINS = [_resilient_mistral, _resilient_groq, _resilient_gemini]


def _parse_chunk_result(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # If a chunk ever comes back malformed, don't kill the whole run —
        # just treat it as empty and move on.
        return {"action_items": [], "key_decisions": [], "open_questions": []}
    return {
        "action_items": data.get("action_items", []) or [],
        "key_decisions": data.get("key_decisions", []) or [],
        "open_questions": data.get("open_questions", []) or [],
    }


# Simple in-memory cache keyed by transcript hash, so if actionable_items(),
# key_decisions(), and extract_questions() are all called on the same
# transcript (e.g. from different parts of the app), we only hit the LLM
# once total instead of once per function.
_extract_cache: dict[str, dict] = {}


def _extract_all(transcript: str) -> dict:
    key = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    if key in _extract_cache:
        return _extract_cache[key]

    chunks = splitter_text().split_text(transcript)

    # Split chunks into up to 3 groups, one per provider, and batch each
    # group against its own resilient chain. If a provider in the rotation
    # is out of quota (TPM or TPD), with_fallbacks() routes that call to a
    # live provider instead of failing the whole batch.
    groups: dict[int, list[str]] = {}
    for i, chunk in enumerate(chunks):
        groups.setdefault(i % len(_PROVIDER_CHAINS), []).append(chunk)

    raw_results = []
    for provider_idx, group_chunks in groups.items():
        chain = _PROVIDER_CHAINS[provider_idx]
        raw_results.extend(
            chain.batch(
                [{"text": chunk} for chunk in group_chunks],
                config={"max_concurrency": 2},
            )
        )

    merged = {"action_items": [], "key_decisions": [], "open_questions": []}
    for raw in raw_results:
        parsed = _parse_chunk_result(raw)
        merged["action_items"].extend(parsed["action_items"])
        merged["key_decisions"].extend(parsed["key_decisions"])
        merged["open_questions"].extend(parsed["open_questions"])

    _extract_cache[key] = merged
    return merged


# --- Public functions (same signatures/return shape as before) -----------

def actionable_items(transcript: str) -> str:
    items = _extract_all(transcript)["action_items"]
    if not items:
        return "No action items found."
    lines = []
    for i, item in enumerate(items, 1):
        task = item.get("task", "Not specified")
        owner = item.get("owner", "Not specified")
        deadline = item.get("deadline", "Not specified")
        lines.append(f"{i}. {task}\n   Owner: {owner}\n   Deadline: {deadline}")
    return "\n".join(lines)


def key_decisions(transcript: str) -> str:
    decisions = _extract_all(transcript)["key_decisions"]
    if not decisions:
        return "No key decisions found."
    return "\n".join(f"{i}. {d}" for i, d in enumerate(decisions, 1))


def extract_questions(transcript: str) -> str:
    questions = _extract_all(transcript)["open_questions"]
    if not questions:
        return "No open questions found."
    return "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
