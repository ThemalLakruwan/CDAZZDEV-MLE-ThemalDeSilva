from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

"""ASSISTED: Claude (claude-sonnet-4-6), 
Prompt: 'Implement the indicator functions from scratch using pandas and NumPy, 
including Simple Moving Average (SMA), Wilder-smoothed RSI, MACD, and Bollinger Bands without using TA-Lib. 
Ensure each function is clear, reusable, and returns correctly structured pandas Series or DataFrames.',
Date: 2026-09-02"""

# Indicator functions
def compute_sma(close: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average: the plain arithmetic mean over a rolling window."""
    return close.rolling(window=window, min_periods=window).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing (the original,
    standard formulation -- not a naive rolling-mean approximation).

    RSI = 100 - (100 / (1 + RS))
    RS  = average gain / average loss, smoothed with Wilder's method
          (an exponential moving average with alpha = 1/period).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing == EMA with alpha = 1/period, adjust=False
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)  # avoid divide-by-zero
    rsi = 100 - (100 / (1 + rs))
    # Where avg_loss is 0 (all gains), RSI is defined as 100
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).

    macd_line      = EMA(fast) - EMA(slow)
    signal_line    = EMA(macd_line, signal period)
    histogram      = macd_line - signal_line
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd_line": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


def compute_bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands: a moving average with bands at +/- num_std standard
    deviations, capturing recent volatility around the mean price.
    """
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({"bb_middle": middle, "bb_upper": upper, "bb_lower": lower})


# Data containers
@dataclass
class NewsHeadline:
    title: str
    publisher: Optional[str] = None
    published: Optional[str] = None
    link: Optional[str] = None


@dataclass
class PipelineResult:
    ticker: str
    ohlcv: pd.DataFrame
    headlines: list[NewsHeadline] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# Fetchers
def fetch_ohlcv(ticker: str, period: str = config.HISTORY_PERIOD) -> pd.DataFrame:
    """
    Fetch daily OHLCV data via yfinance and attach all computed indicators.
    Robust to yfinance returning empty/partial data.
    """
    logger.info("Fetching %s OHLCV data for period=%s", ticker, period)
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    except Exception as exc:  # network / yfinance failure
        logger.error("OHLCV fetch failed for %s: %s", ticker, exc)
        raise RuntimeError(f"Could not fetch OHLCV data for {ticker}") from exc

    if df is None or df.empty:
        raise RuntimeError(f"No OHLCV data returned for {ticker}")

    df = df.dropna(subset=["Close"]).copy()

    # Attach indicators
    df["SMA_50"] = compute_sma(df["Close"], config.SMA_SHORT_WINDOW)
    df["SMA_200"] = compute_sma(df["Close"], config.SMA_LONG_WINDOW)
    df["RSI_14"] = compute_rsi(df["Close"], config.RSI_PERIOD)

    macd_df = compute_macd(df["Close"], config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)
    df = df.join(macd_df)

    bb_df = compute_bollinger_bands(df["Close"], config.BOLLINGER_WINDOW, config.BOLLINGER_STD)
    df = df.join(bb_df)

    return df


def fetch_news(ticker: str, min_headlines: int = config.MIN_NEWS_HEADLINES) -> list[NewsHeadline]:
    headlines: list[NewsHeadline] = []

    try:
        raw_news = yf.Ticker(ticker).news or []
        for item in raw_news:
            content = item.get("content", item)
            title = content.get("title") or item.get("title")
            if not title:
                continue
            headlines.append(
                NewsHeadline(
                    title=title,
                    publisher=(content.get("provider", {}) or {}).get("displayName")
                    or item.get("publisher"),
                    published=content.get("pubDate") or item.get("providerPublishTime"),
                    link=(content.get("canonicalUrl", {}) or {}).get("url") or item.get("link"),
                )
            )
    except Exception as exc:
        logger.warning("yfinance news fetch failed for %s: %s", ticker, exc)

    if len(headlines) < min_headlines:
        logger.info(
            "Only %d headlines from yfinance, falling back to RSS for %s", len(headlines), ticker
        )
        headlines.extend(_fetch_news_rss(ticker, min_headlines - len(headlines)))

    return headlines[: max(min_headlines, len(headlines))]


def _fetch_news_rss(ticker: str, needed: int) -> list[NewsHeadline]:
    """Fallback: Yahoo Finance RSS feed, parsed with the stdlib (no extra deps)."""
    import urllib.request
    import xml.etree.ElementTree as ET

    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    extra: list[NewsHeadline] = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            tree = ET.fromstring(resp.read())
        for item in tree.findall(".//item")[:needed]:
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")
            if title_el is not None and title_el.text:
                extra.append(
                    NewsHeadline(
                        title=title_el.text,
                        publisher="Yahoo Finance RSS",
                        published=pub_el.text if pub_el is not None else None,
                        link=link_el.text if link_el is not None else None,
                    )
                )
    except Exception as exc:
        logger.warning("RSS fallback failed for %s: %s", ticker, exc)
    return extra


def fetch_fundamentals(ticker: str) -> dict:
    """PE ratio and other snapshot fields. Handles missing fields gracefully."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning("Fundamentals fetch failed for %s: %s", ticker, exc)
        info = {}
    return {
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "market_cap": info.get("marketCap"),
        "long_name": info.get("longName", ticker),
    }


# Summary builder
def _momentum_signal(latest: pd.Series) -> str:
    score = 0
    if pd.notna(latest.get("SMA_50")) and pd.notna(latest.get("SMA_200")):
        score += 1 if latest["Close"] > latest["SMA_50"] > latest["SMA_200"] else 0
        score -= 1 if latest["Close"] < latest["SMA_50"] < latest["SMA_200"] else 0
    if pd.notna(latest.get("RSI_14")):
        if latest["RSI_14"] > 70:
            score -= 1  # overbought
        elif latest["RSI_14"] < 30:
            score += 1  # oversold, potential rebound
    if pd.notna(latest.get("macd_hist")):
        score += 1 if latest["macd_hist"] > 0 else -1

    if score >= 2:
        return "bullish"
    if score <= -2:
        return "bearish"
    return "neutral"


def build_summary(df: pd.DataFrame, ticker: str, fundamentals: dict) -> dict:
    if df.empty:
        raise ValueError("Cannot build summary from empty OHLCV DataFrame")

    latest = df.iloc[-1]
    current_price = float(latest["Close"])

    window_52w = df.tail(252)
    high_52w = float(window_52w["High"].max()) if not window_52w.empty else None
    low_52w = float(window_52w["Low"].min()) if not window_52w.empty else None

    # YTD return: from first trading day of the current calendar year
    this_year = df[df.index.year == datetime.now().year]
    if not this_year.empty:
        start_price = float(this_year.iloc[0]["Close"])
        ytd_return = round((current_price - start_price) / start_price * 100, 2)
    else:
        ytd_return = None

    def safe_round(value, ndigits=2):
        return round(float(value), ndigits) if pd.notna(value) else None

    summary = {
        "ticker": ticker,
        "company_name": fundamentals.get("long_name", ticker),
        "current_price": round(current_price, 2),
        "52_week_high": round(high_52w, 2) if high_52w is not None else None,
        "52_week_low": round(low_52w, 2) if low_52w is not None else None,
        "trailing_pe": safe_round(fundamentals.get("trailing_pe")),
        "ytd_return_pct": ytd_return,
        "momentum_signal": _momentum_signal(latest),
        "indicators": {
            "sma_50": safe_round(latest.get("SMA_50")),
            "sma_200": safe_round(latest.get("SMA_200")),
            "rsi_14": safe_round(latest.get("RSI_14")),
            "macd_line": safe_round(latest.get("macd_line")),
            "macd_signal": safe_round(latest.get("macd_signal")),
            "macd_hist": safe_round(latest.get("macd_hist")),
            "bb_upper": safe_round(latest.get("bb_upper")),
            "bb_middle": safe_round(latest.get("bb_middle")),
            "bb_lower": safe_round(latest.get("bb_lower")),
        },
        "as_of_date": str(df.index[-1].date()),
    }
    return summary



# Orchestration entry point
def run_pipeline(ticker: str = config.TICKER) -> PipelineResult:
    ohlcv = fetch_ohlcv(ticker)
    headlines = fetch_news(ticker)
    fundamentals = fetch_fundamentals(ticker)
    summary = build_summary(ohlcv, ticker, fundamentals)

    logger.info("Fetched %d rows of OHLCV, %d headlines for %s", len(ohlcv), len(headlines), ticker)
    return PipelineResult(ticker=ticker, ohlcv=ohlcv, headlines=headlines, summary=summary)


if __name__ == "__main__":
    result = run_pipeline()
    print(f"\n=== Summary for {result.ticker} ===")
    for k, v in result.summary.items():
        print(f"{k}: {v}")
    print(f"\nFetched {len(result.headlines)} headlines. First 3:")
    for h in result.headlines[:3]:
        print(f" - {h.title}")
