from typing import Dict, Any, Optional
from datetime import date
import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

def compute_lessons(trades: pd.DataFrame, start: Optional[date]=None, end: Optional[date]=None) -> Dict[str, Any]:
    tdf = trades.copy()
    tdf["Date"] = pd.to_datetime(tdf["Date"]).dt.date
    if start: tdf = tdf[tdf["Date"] >= start]
    if end:   tdf = tdf[tdf["Date"] <= end]
    if len(tdf)==0:
        return {"window": {"start": start.isoformat() if start else None,
                           "end":   end.isoformat() if end else None},
                "assets": {}, "tags": {}, "summary": "No data in window."}

    asset_win = (tdf.groupby("Asset")["Outcome"].apply(lambda s: (s=="Profit").mean())
                 .sort_values(ascending=False)).to_dict()

    tag_stats = {}
    for _, row in tdf.iterrows():
        outs = row["Outcome"]
        for t in [x.strip() for x in str(row["Tags"]).split(",") if x.strip()]:
            st = tag_stats.setdefault(t, {"n":0, "wins":0})
            st["n"] += 1; st["wins"] += int(outs == "Profit")
    for t, st in tag_stats.items():
        st["win_rate"] = st["wins"]/max(1,st["n"])

    feat_avg = {
        "RSI_win": float(tdf[tdf["Outcome"]=="Profit"]["RSI"].mean()) if (tdf["Outcome"]=="Profit").any() else None,
        "VolChg_win": float(tdf[tdf["Outcome"]=="Profit"]["Volume_Change_Pct"].mean()) if (tdf["Outcome"]=="Profit").any() else None,
        "Sent_win": float(tdf[tdf["Outcome"]=="Profit"]["Sentiment_Score"].mean()) if (tdf["Outcome"]=="Profit").any() else None,
    }
    return {"window": {"start": start.isoformat() if start else None,
                       "end":   end.isoformat() if end else None},
            "assets": asset_win, "tags": tag_stats, "features": feat_avg}


def summarize_lessons(llm, persona, stats: Dict[str, Any]) -> str:

    lessons_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an experienced trading coach. Based on structured stats, write a concise reflection for the agent:\n"
     "- 6–10 bullet points with what worked, what failed, and risk notes.\n"
     "- Reference assets/tags and thresholds (e.g., RSI>55, ΔVol>20%).\n"
     "- Keep it historical and educational; no forward-looking advice.\n"
     "Persona (for tone/bias): {persona}"),
    ("human", "Stats JSON:\n{stats_json}")
])
    lessons_chain = lessons_prompt | llm | StrOutputParser()

    return lessons_chain.invoke({"persona": json.dumps(persona), "stats_json": json.dumps(stats, ensure_ascii=False, default=str)})
