import os

"""AI-ASSISTED: Claude (claude-sonnet-4-6), 
 Prompt: 'Generate a centralized config.py, 
 including application settings, technical-indicator parameters, Groq LLM configuration, retry settings, 
 and sentiment/trade-signal prompt templates. Ensure API keys are loaded from environment variables.', 
 Date: 2026-09-02"""

# General settings
TICKER = os.environ.get("TICKER", "AAPL")  
HISTORY_PERIOD = "2y"                               
MIN_NEWS_HEADLINES = 10

# Indicator parameters
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2

# LLM settings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 2


# Prompt templates
SENTIMENT_SYSTEM_PROMPT = """You are a disciplined financial news analyst.
You read a single news headline about a public company and classify its
sentiment toward the company's stock. You must be conservative: only mark
something positive or negative if the headline gives a clear, specific
reason. Ambiguous or purely factual headlines are neutral.

Respond with ONLY a JSON object matching exactly this schema, no prose,
no markdown fences:
{
  "headline": "<the exact headline text you were given>",
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": <float between 0.0 and 1.0>,
  "brief_reason": "<one short sentence, max 25 words>"
}
"""

SENTIMENT_USER_PROMPT_TEMPLATE = """Ticker: {ticker}
Headline: "{headline}"

Classify this headline's sentiment for {ticker} stock."""

SIGNAL_SYSTEM_PROMPT = """You are a senior equity research analyst producing
a first-pass technical read on a stock. You will be given a set of computed
technical indicators and an aggregate news sentiment score. You must reason
over the COMBINATION of these signals -- for example, whether price is
above or below both moving averages, whether RSI suggests overbought or
oversold conditions in the context of the trend, whether MACD confirms or
contradicts price action, and where price sits within the Bollinger Bands
-- rather than simply restating each indicator's value in isolation.

Respond with ONLY a JSON object matching exactly this schema, no prose,
no markdown fences:
{
  "signal": "Buy" | "Hold" | "Sell",
  "justification": "<3 to 5 sentences reasoning over the indicator combination>",
  "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"]
}
"""

SIGNAL_USER_PROMPT_TEMPLATE = """Ticker: {ticker}

Technical indicators (most recent trading day):
- Current price: {current_price}
- 50-day SMA: {sma_50}
- 200-day SMA: {sma_200}
- RSI (14): {rsi_14}
- MACD line: {macd_line}
- MACD signal line: {macd_signal}
- MACD histogram: {macd_hist}
- Bollinger upper band: {bb_upper}
- Bollinger middle band (SMA20): {bb_middle}
- Bollinger lower band: {bb_lower}
- YTD return: {ytd_return}%

Aggregate news sentiment score (-1.0 = very negative, +1.0 = very positive): {sentiment_score}
Based on {n_headlines} recent headlines.

Produce a Buy, Hold, or Sell signal, reasoning over how these indicators
combine (trend vs. momentum vs. mean-reversion signals vs. news context)."""
