# Create a synthetic "Trader Past Trades" dataset with market context and save it to CSV
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from caas_jupyter_tools import display_dataframe_to_user

random.seed(42)
np.random.seed(42)

# Asset configurations: base price ranges and typical volume ranges
assets = {
    "BTC": {"price_range": (28000, 72000), "volume_range": (0.1, 3.0)},
    "ETH": {"price_range": (1500, 4200), "volume_range": (1, 30)},
    "SOL": {"price_range": (20, 220), "volume_range": (5, 150)},
    "DOGE": {"price_range": (0.05, 0.35), "volume_range": (1000, 60000)},
    "PEPE": {"price_range": (0.000001, 0.00002), "volume_range": (1_000_000, 120_000_000)},
}

tags_pool = [
    "breakout", "rsi-divergence", "meme", "event-driven", "stop-loss",
    "swing", "intraday", "long-term", "volume-spike", "news-sentiment",
    "range-trade", "momentum", "mean-revert", "trend-follow"
]

# Helper to draw a price within a range (with slight fat-tail)
def draw_price(low, high):
    # use beta distribution to bias towards the middle with occasional extremes
    a, b = 2.0, 2.0
    u = np.random.beta(a, b)
    return round(low + u * (high - low), 8)

def draw_volume(low, high):
    val = np.random.lognormal(mean=np.log((low+high)/2), sigma=0.5)
    return round(min(max(val, low), high), 8)

def draw_rsi():
    # cluster around 35-65 but allow tails 15-85
    base = np.clip(np.random.normal(50, 15), 10, 90)
    return round(base, 2)

def draw_volume_change_pct():
    # percent change with long tail for spikes
    val = np.clip(np.random.normal(10, 40), -70, 180)
    return round(val, 2)

def draw_sentiment():
    # sentiment score between -1 and 1, slightly positive on average
    val = np.clip(np.random.normal(0.1, 0.5), -1, 1)
    return round(val, 2)

def pick_tags():
    k = random.choice([2, 2, 3])  # bias toward 2-3 tags
    return ",".join(random.sample(tags_pool, k))

def likely_outcome(side, rsi, sent, vchg):
    # Simple heuristic for outcome; inject some structure + randomness
    score = 0.0
    # RSI: oversold (<30) favors Buy; overbought (>70) favors Sell
    if side == "Buy":
        score += (30 - min(rsi, 30)) * 0.15  # positive if rsi<30
        score += sent * 0.8  # positive sentiment helps
    else:  # Sell
        score += (max(rsi, 70) - 70) * 0.15  # positive if rsi>70
        score += (-sent) * 0.5  # negative sentiment helps sell outcomes
    
    # Volume change spike can help momentum strategies
    score += max(vchg, 0) * 0.02
    
    # Add noise
    score += np.random.normal(0, 0.5)
    
    if score > 0.6:
        return "Profit"
    elif score < -0.6:
        return "Loss"
    else:
        return "Neutral"

# Generate dates from 2024-01-10 to 2025-08-01
start_date = datetime(2024, 1, 10)
end_date = datetime(2025, 8, 1)
num_days = (end_date - start_date).days

rows = []
N = 80  # number of trades

for i in range(1, N + 1):
    trade_id = f"T{i:03d}"
    asset = random.choice(list(assets.keys()))
    pr = assets[asset]["price_range"]
    vr = assets[asset]["volume_range"]
    price = draw_price(*pr)
    volume = draw_volume(*vr)
    side = random.choice(["Buy", "Sell"])
    rsi = draw_rsi()
    vchg = draw_volume_change_pct()
    sent = draw_sentiment()
    outcome = likely_outcome(side, rsi, sent, vchg)
    tags = pick_tags()
    date = (start_date + timedelta(days=random.randint(0, num_days))).date().isoformat()
    
    rows.append({
        "Trade ID": trade_id,
        "Asset": asset,
        "Buy/Sell": side,
        "Price": price,
        "Volume": volume,
        "Date": date,
        "Outcome": outcome,
        "Tags": tags,
        "RSI": rsi,
        "Volume_Change_Pct": vchg,
        "Sentiment_Score": sent
    })

df = pd.DataFrame(rows)

# Sort by date to look realistic, then reset Trade IDs sequentially by date
df = df.sort_values("Date").reset_index(drop=True)
df["Trade ID"] = [f"T{i:03d}" for i in range(1, len(df) + 1)]

# Save CSV
csv_path = "/trader_past_trades.csv"
df.to_csv(csv_path, index=False)

# Show a preview to the user
#display_dataframe_to_user("Trader Past Trades (preview)", df.head(20))

