from __future__ import annotations

import json
import logging
import re
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
 
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Call the model and parse its output as JSON. Raises on hard failure."""
        try:
            payload = self._post(system_prompt, user_prompt, force_json_mode=True)
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                logger.warning(
                    "response_format=json_object rejected (HTTP 400), retrying "
                    "without it: %s", exc.response.text[:300]
                )
                payload = self._post(system_prompt, user_prompt, force_json_mode=False)
            else:
                raise
 
        if "error" in payload:
            raise RuntimeError(f"OpenRouter error: {payload['error']}")
 
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices. Full payload: {payload}")
 
        message = choices[0]["message"]
        raw = message.get("content")
 
        if not raw:
            finish_reason = choices[0].get("finish_reason")
            reasoning_preview = (message.get("reasoning") or "")[:200]
            raise RuntimeError(
                "OpenRouter returned an empty message content. "
                f"finish_reason={finish_reason!r}, "
                f"reasoning_preview={reasoning_preview!r}. "
                "Likely causes: token budget too low, or the model requires "
                "reasoning tokens it wasn't given room for. Try raising "
                "config.LLM_MAX_TOKENS or switching config.OPENROUTER_MODEL."
            )
 
        return self._parse_json(raw)
 
    def _post(self, system_prompt: str, user_prompt: str, force_json_mode: bool) -> dict:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": config.LLM_MAX_TOKENS,
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
) -> BaseModel:
    last_error: Optional[Exception] = None
    prompt = user_prompt
 
    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        try:
            raw_json = client.complete_json(system_prompt, prompt)
            return schema.model_validate(raw_json)
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning("LLM output failed validation (attempt %d): %s", attempt, exc)
            prompt = (
                f"{user_prompt}\n\n"
                f"Your previous response was invalid: {exc}. "
                f"Return ONLY valid JSON matching the required schema."
            )
        except Exception as exc:
            last_error = exc
            logger.error("LLM call failed (attempt %d): %s", attempt, exc)
 
    raise LLMResponseError(
        f"LLM failed to produce a valid {schema.__name__} after "
        f"{config.LLM_MAX_RETRIES} attempts: {last_error}"
    )
 
 
# Public functions
def classify_headline(client: GroqClient, ticker: str, headline: str) -> Optional[HeadlineSentiment]:
    """Classify a single headline's sentiment. Returns None on unrecoverable failure
    (logged, not raised) so one bad headline doesn't abort the whole batch."""
    user_prompt = config.SENTIMENT_USER_PROMPT_TEMPLATE.format(ticker=ticker, headline=headline)
    try:
        result = _call_with_validation(
            client, config.SENTIMENT_SYSTEM_PROMPT, user_prompt, HeadlineSentiment
        )
        return result
    except LLMResponseError as exc:
        logger.error("Skipping headline due to persistent LLM failure: %s | %s", headline, exc)
        return None
 
 
def analyze_headlines(
    client: GroqClient, ticker: str, headlines: list[NewsHeadline]
) -> tuple[list[HeadlineSentiment], float]:
    """
    Classify every headline and compute an aggregate sentiment score in
    [-1.0, 1.0]: confidence-weighted average of (+1 positive, -1 negative,
    0 neutral).
    """
    results: list[HeadlineSentiment] = []
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
    """Produce the Buy/Hold/Sell signal reasoned over the indicator combination."""
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
