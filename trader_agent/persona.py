import numpy as np
import pandas as pd
import re
from typing import Dict, Any


def extract_persona(trades: pd.DataFrame) -> Dict[str, Any]:
    persona = {"style": None, "risk": None, "holding": "swing",
               "preferred_assets": [], "top_tags": [], "rules": []}
    wins = trades[trades["Outcome"] == "Profit"].copy()

    def tag_ratio(s: pd.Series, pattern: str) -> float:
        return s.fillna("").str.contains(pattern, flags=re.IGNORECASE, regex=True).mean()

    momentum_ratio = tag_ratio(wins["Tags"], r"breakout|momentum|volume-spike")
    sentiment_ratio = tag_ratio(wins["Tags"], r"news-sentiment|meme")
    meanrev_ratio  = tag_ratio(wins["Tags"], r"mean-revert|rsi-divergence|range-trade")
    avg_rsi_wins = wins["RSI"].mean() if len(wins) else np.nan

    if momentum_ratio > 0.35 and (not np.isnan(avg_rsi_wins) and avg_rsi_wins > 55):
        persona["style"] = "Momentum"
    elif sentiment_ratio > 0.30 and wins["Sentiment_Score"].mean() > 0.2:
        persona["style"] = "Sentiment"
    else:
        persona["style"] = "MeanReversion" if meanrev_ratio > 0.28 else "Technical"

    exposure = trades["Asset"].value_counts(normalize=True)
    high_beta = exposure.get("DOGE", 0) + exposure.get("PEPE", 0)
    persona["risk"] = "High" if high_beta > 0.35 else ("Medium" if high_beta > 0.15 else "Low")

    asset_win = (trades.groupby("Asset")["Outcome"]
                 .apply(lambda s: (s=="Profit").mean())
                 .sort_values(ascending=False).head(3))
    persona["preferred_assets"] = asset_win.index.tolist()

    tag_counts = {}
    for tags in trades["Tags"].fillna(""):
        for t in [x.strip() for x in str(tags).split(",") if x.strip()]:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    persona["top_tags"] = sorted(tag_counts, key=tag_counts.get, reverse=True)[:5]

    if momentum_ratio > 0.35:
        persona["rules"].append("Favor entries when RSI > 55 and volume spikes.")
    if sentiment_ratio > 0.30:
        persona["rules"].append("Lean into strong positive sentiment with tight stops.")
    if high_beta > 0.25:
        persona["rules"].append("Size down on high-volatility meme assets.")
    if not persona["rules"]:
        persona["rules"].append("Wait for confluence: RSI alignment, rising volume, supportive sentiment.")
    return persona