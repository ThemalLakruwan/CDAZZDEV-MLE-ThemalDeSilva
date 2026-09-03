from __future__ import annotations

import json
import logging
import re
import time
from typing import Literal, Optional

import requests
from pydantic import BaseModel, Field, ValidationError

import config
from data_pipeline import NewsHeadline



""" AI-ASSISTED: Claude (claude-sonnet-4-6), 
Prompt: 'Design a Pydantic schema + retry strategy for validating structured JSON output from an LLM sentiment classifier 
and trade-signal generator. Include a GroqClient wrapper for the OpenAI-compatible API,
Date: 2026-09-02"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LLMResponseError(Exception):
    """Raised when the LLM's output cannot be validated after all retries."""



# Pydantic schemas
class HeadlineSentiment(BaseModel):
    headline: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str
 
 
class HeadlineSentimentBatch(BaseModel):
    """Wrapper schema for classifying every headline in a single LLM call."""
    items: list[HeadlineSentiment]
 
 
class TradeSignal(BaseModel):
    signal: Literal["Buy", "Hold", "Sell"]
    justification: str
    key_factors: list[str]
 

# LLM client wrapper
 
class GroqClient:
    def __init__(self, api_key: Optional[str] = None, model: str = config.OPENROUTER_MODEL):
        api_key = api_key or config.OPENROUTER_API_KEY
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. In Colab: os.environ['OPENROUTER_API_KEY'] = "
                "userdata.get('OPENROUTER_API_KEY')  (using Colab Secrets, never hardcode it)."
            )
        self._api_key = api_key
        self._model = model
        self._last_call_ts: float = 0.0
 
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        wait = config.LLM_MIN_SECONDS_BETWEEN_CALLS - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.monotonic()
 
    def complete_json(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None) -> dict:
        payload = self._post_with_429_backoff(
            system_prompt, user_prompt, force_json_mode=True, max_tokens=max_tokens
        )
 
        if "error" in payload:
            raise RuntimeError(f"OpenRouter error: {payload['error']}")
 
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices. Full payload: {payload}")
 
        message = choices[0]["message"]
        raw = message.get("content")
 
        if not raw or not raw.strip():
            finish_reason = choices[0].get("finish_reason")
            reasoning_preview = (message.get("reasoning") or "")[:200]
            raise RuntimeError(
                "OpenRouter returned an empty (or whitespace-only) message "
                f"content. finish_reason={finish_reason!r}, "
                f"reasoning_preview={reasoning_preview!r}. "
                "Likely causes: token budget too low, or the model requires "
                "reasoning tokens it wasn't given room for. Try raising "
                "config.LLM_MAX_TOKENS or switching config.OPENROUTER_MODEL."
            )
 
        return self._parse_json(raw)
    """ASSISTED: Claude (claude-sonnet-4-6),
 Prompt: 'Fix the groq client to handle openrouter rate limits and add batching for headline classification.
 Implement a retry strategy with exponential backoff for 429 errors, and ensure the client can parse JSON output from the LLM, even if wrapped in markdown fences or with stray text.',
 Date: 2026-09-03"""
 
    def _post_with_429_backoff(
        self,
        system_prompt: str,
        user_prompt: str,
        force_json_mode: bool,
        max_tokens: Optional[int] = None,
    ) -> dict:
        last_exc: Optional[Exception] = None
        for attempt in range(1, config.LLM_429_MAX_RETRIES + 1):
            self._throttle()
            try:
                return self._post(system_prompt, user_prompt, force_json_mode, max_tokens)
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 400 and force_json_mode:
                    logger.warning(
                        "response_format=json_object rejected (HTTP 400), retrying "
                        "without it: %s", exc.response.text[:300]
                    )
                    force_json_mode = False
                    continue
                if status == 429:
                    body_text = exc.response.text if exc.response is not None else ""
                    # OpenRouter free models enforce TWO separate 429 causes:
                    # (a) 20 requests/minute -- transient, worth waiting for.
                    # (b) 50 requests/day (or 1000/day with $10+ lifetime
                    #     credits purchased) -- a hard daily quota that does
                    #     NOT refill by waiting seconds or minutes; it only
                    #     resets at day rollover (UTC), or by adding credits.
                    # We check the error body for daily-cap language and, if
                    # found, stop retrying immediately rather than burning
                    # 5 rounds of exponential backoff (~150s+) for nothing.
                    if re.search(r"\bday\b|\bdaily\b", body_text, flags=re.IGNORECASE):
                        raise RuntimeError(
                            "Hit OpenRouter's DAILY free-model quota (50 requests/day "
                            "without purchased credits, 1000/day with $10+ lifetime "
                            "credits). This will NOT be fixed by waiting seconds or "
                            "minutes -- it resets at UTC midnight, or you can add "
                            "credits at https://openrouter.ai/credits. "
                            f"Server said: {body_text[:300]}"
                        ) from exc
 
                    last_exc = exc
                    retry_after = None
                    if exc.response is not None:
                        retry_after = exc.response.headers.get("Retry-After")
                    if retry_after:
                        wait = float(retry_after)
                    else:
                        wait = config.LLM_429_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    logger.warning(
                        "429 rate limited (attempt %d/%d, likely per-minute cap). "
                        "Waiting %.1fs before retrying...",
                        attempt, config.LLM_429_MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(
            f"Still rate limited after {config.LLM_429_MAX_RETRIES} attempts: {last_exc}"
        )
 
    def _post(
        self,
        system_prompt: str,
        user_prompt: str,
        force_json_mode: bool,
        max_tokens: Optional[int] = None,
    ) -> dict:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
            "reasoning": {"enabled": False},
        }
        if force_json_mode:
            body["response_format"] = {"type": "json_object"}
 
        response = requests.post(
            url=config.OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(body),
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
 
    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text)
 
 
def _call_with_validation(
    client: GroqClient,
    system_prompt: str,
    user_prompt: str,
    schema: type[BaseModel],
    max_tokens: Optional[int] = None,
) -> BaseModel:
    last_error: Optional[Exception] = None
    prompt = user_prompt
 
    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        try:
            raw_json = client.complete_json(system_prompt, prompt, max_tokens=max_tokens)
            return schema.model_validate(raw_json)
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning("LLM output failed validation (attempt %d): %s", attempt, exc)
            prompt = (
                f"{user_prompt}\n\n"
                f"Your previous response was invalid: {exc}. "
                f"Return ONLY valid JSON matching the required schema."
            )
        except Exception as exc:  # network / API errors
            last_error = exc
            logger.error("LLM call failed (attempt %d): %s", attempt, exc)
 
    raise LLMResponseError(
        f"LLM failed to produce a valid {schema.__name__} after "
        f"{config.LLM_MAX_RETRIES} attempts: {last_error}"
    )
 
 
# Public functions
 
def classify_headline(client: GroqClient, ticker: str, headline: str) -> Optional[HeadlineSentiment]:
    user_prompt = config.SENTIMENT_USER_PROMPT_TEMPLATE.format(ticker=ticker, headline=headline)
    try:
        result = _call_with_validation(
            client, config.SENTIMENT_SYSTEM_PROMPT, user_prompt, HeadlineSentiment
        )
        return result  # type: ignore[return-value]
    except LLMResponseError as exc:
        logger.error("Skipping headline due to persistent LLM failure: %s | %s", headline, exc)
        return None
 
 
def classify_headlines_batch(
    client: GroqClient, ticker: str, headlines: list[NewsHeadline]
) -> Optional[list[HeadlineSentiment]]:
    if not headlines:
        return []
 
    numbered = "\n".join(f"{i+1}. {h.title}" for i, h in enumerate(headlines))
    user_prompt = config.BATCH_SENTIMENT_USER_PROMPT_TEMPLATE.format(
        ticker=ticker, numbered_headlines=numbered, n_headlines=len(headlines)
    )
    # Scale the token budget to the number of headlines -- a fixed budget
    # sized for one item (config.LLM_MAX_TOKENS=600) truncates mid-JSON
    # once you're generating structured output for 8-10+ headlines at
    # once, which produces invalid JSON that looks like a formatting bug
    # but is really "the model ran out of room to finish."
    batch_max_tokens = min(
        config.LLM_BATCH_TOKENS_BASE + config.LLM_BATCH_TOKENS_PER_ITEM * len(headlines),
        config.LLM_BATCH_MAX_TOKENS_CAP,
    )
    try:
        result = _call_with_validation(
            client,
            config.BATCH_SENTIMENT_SYSTEM_PROMPT,
            user_prompt,
            HeadlineSentimentBatch,
            max_tokens=batch_max_tokens,
        )
        items = result.items  # type: ignore[attr-defined]
        if len(items) != len(headlines):
            logger.warning(
                "Batch returned %d items for %d headlines -- using what came back.",
                len(items), len(headlines),
            )
        return items
    except LLMResponseError as exc:
        logger.warning("Batch headline classification failed, will fall back to per-headline: %s", exc)
        return None
 
 
def analyze_headlines(
    client: GroqClient, ticker: str, headlines: list[NewsHeadline]
) -> tuple[list[HeadlineSentiment], float]:
    results = classify_headlines_batch(client, ticker, headlines)
 
    if results is None:
        logger.info("Falling back to per-headline classification (%d calls)...", len(headlines))
        results = []
        for h in headlines:
            classified = classify_headline(client, ticker, h.title)
            if classified is not None:
                results.append(classified)
 
    if not results:
        return results, 0.0
 
    polarity = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    weighted_sum = sum(polarity[r.sentiment] * r.confidence for r in results)
    weight_total = sum(r.confidence for r in results) or 1.0
    aggregate_score = round(weighted_sum / weight_total, 3)
 
    return results, aggregate_score
 
 
def generate_trade_signal(
    client: GroqClient,
    ticker: str,
    summary: dict,
    sentiment_score: float,
    n_headlines: int,
) -> Optional[TradeSignal]:
    ind = summary["indicators"]
    user_prompt = config.SIGNAL_USER_PROMPT_TEMPLATE.format(
        ticker=ticker,
        current_price=summary["current_price"],
        sma_50=ind["sma_50"],
        sma_200=ind["sma_200"],
        rsi_14=ind["rsi_14"],
        macd_line=ind["macd_line"],
        macd_signal=ind["macd_signal"],
        macd_hist=ind["macd_hist"],
        bb_upper=ind["bb_upper"],
        bb_middle=ind["bb_middle"],
        bb_lower=ind["bb_lower"],
        ytd_return=summary["ytd_return_pct"],
        sentiment_score=sentiment_score,
        n_headlines=n_headlines,
    )
    try:
        result = _call_with_validation(
            client, config.SIGNAL_SYSTEM_PROMPT, user_prompt, TradeSignal
        )
        return result  # type: ignore[return-value]
    except LLMResponseError as exc:
        logger.error("Could not generate trade signal: %s", exc)
        return None