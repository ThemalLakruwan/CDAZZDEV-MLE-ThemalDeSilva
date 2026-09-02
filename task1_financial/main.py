"""
main.py
-------
End-to-end orchestration for Task 1 (run this cell-by-cell in Colab, or
as a script). Ties together:
  Task 1A -> data_pipeline.run_pipeline()
  Task 1B -> llm_reasoning.analyze_headlines() + generate_trade_signal()
  Bonus   -> report_generator.build_markdown_report() + render_html_report()

Set GROQ_API_KEY as an environment variable / Colab secret before running.
Never hardcode it in this file.
"""

import json
import os

import config
import data_pipeline
import llm_reasoning
import report_generator


def main(ticker: str = config.TICKER):
    # Task 1A
    print(f"[1/4] Running data pipeline for {ticker}...")
    pipeline_result = data_pipeline.run_pipeline(ticker)
    print(json.dumps(pipeline_result.summary, indent=2))

    # Task 1B
    print(f"\n[2/4] Classifying {len(pipeline_result.headlines)} headlines via Groq LLM...")
    client = llm_reasoning.GroqClient()
    sentiment_results, aggregate_sentiment = llm_reasoning.analyze_headlines(
        client, ticker, pipeline_result.headlines
    )
    print(f"Aggregate sentiment score: {aggregate_sentiment:+.3f}")
    for r in sentiment_results[:5]:
        print(f"  [{r.sentiment:>8}] ({r.confidence:.2f}) {r.headline}")

    print("\n[3/4] Generating trade signal...")
    trade_signal = llm_reasoning.generate_trade_signal(
        client, ticker, pipeline_result.summary, aggregate_sentiment, len(sentiment_results)
    )
    if trade_signal:
        print(f"Signal: {trade_signal.signal}")
        print(f"Justification: {trade_signal.justification}")

    # Bonus: report
    print("\n[4/4] Rendering report...")
    md_report = report_generator.build_markdown_report(
        pipeline_result.summary, sentiment_results, aggregate_sentiment, trade_signal
    )
    os.makedirs("output", exist_ok=True)
    with open(f"output/{ticker}_research_brief.md", "w") as f:
        f.write(md_report)

    html_path = report_generator.render_html_report(
        md_report, pipeline_result.ohlcv, ticker, f"output/{ticker}_research_brief.html"
    )
    print(f"Report written to output/{ticker}_research_brief.md and {html_path}")

    return {
        "summary": pipeline_result.summary,
        "sentiment_results": sentiment_results,
        "aggregate_sentiment": aggregate_sentiment,
        "trade_signal": trade_signal,
    }


if __name__ == "__main__":
    main()
