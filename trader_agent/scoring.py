from typing import Dict, Any

TAG_WEIGHTS = {
    "breakout": 0.6, "momentum": 0.6, "volume-spike": 0.5,
    "news-sentiment": 0.4, "meme": 0.3,
    "mean-revert": 0.4, "rsi-divergence": 0.4, "range-trade": 0.3,
    "stop-loss": -0.2
}

def setup_score(meta: Dict[str, Any]) -> float:
    try:
        rsi = float(meta.get("RSI", 50))
        vch = float(meta.get("Volume_Change_Pct", 0))
        sent = float(meta.get("Sentiment_Score", 0))
    except Exception:
        rsi, vch, sent = 50.0, 0.0, 0.0

    score = 0.0
    score += 1.0 if rsi > 55 else (0.2 if rsi > 45 else 0.0)
    score += 0.8 if vch > 20 else (0.2 if vch > 5 else 0.0)
    score += 0.5 if sent > 0.2 else (0.2 if sent > 0.0 else 0.0)
    if rsi < 30: score += 0.3

    tags = str(meta.get("Tags","")).lower().split(",")
    for t in tags:
        score += TAG_WEIGHTS.get(t.strip(), 0.0)

    if str(meta.get("Outcome","")).lower()=="profit": score += 0.2
    return round(score, 3)