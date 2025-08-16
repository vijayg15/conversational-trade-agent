import os
from datetime import datetime

# ====== CONFIG ======
# Flip to False to use HuggingFace community models instead of OpenAI.
USE_OPENAI: bool = True

# CSV location (change if needed)
CSV_PATH: str = os.getenv("CSV_PATH", "trader_past_trades.csv")

# Fixed "today" for reproducibility (adjust if you prefer real "now")
TODAY = datetime(2025, 8, 14)

# Model names
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL  = "gpt-4o-mini"

# HuggingFace defaults (override via env if desired)
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_CHAT_REPO   = os.getenv("HF_CHAT_REPO", "mistralai/Mixtral-8x7B-Instruct-v0.1")