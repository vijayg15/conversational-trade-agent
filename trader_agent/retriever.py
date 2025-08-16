import re
from typing import Optional, Dict, Tuple, List, Any
from datetime import date
from langchain_core.documents import Document
from .scoring import setup_score

def extract_simple_filters(text: str) -> Dict[str,str]:
    f = {}
    m_asset = re.search(r"\\b(BTC|ETH|SOL|DOGE|PEPE)\\b", text, re.IGNORECASE)
    if m_asset: f["Asset"] = m_asset.group(1).upper()
    if re.search(r"\\b(buy)\\b", text, re.IGNORECASE): f["Buy/Sell"] = "Buy"
    if re.search(r"\\b(sell)\\b", text, re.IGNORECASE): f["Buy/Sell"] = "Sell"
    return f

def retrieve_with_filters(retriever, query: str, filters: Optional[Dict[str,str]], date_range: Tuple[Optional[date],Optional[date]], k:int=5) -> List[Document]:
    raw_docs: List[Document] = retriever.invoke(query)
    s, e = date_range
    selected = []
    for d in raw_docs:
        ok = True
        if filters:
            for kf, vf in filters.items():
                if str(d.metadata.get(kf,"")).upper() != str(vf).upper():
                    ok = False; break
        if ok and (s or e):
            try:
                from datetime import datetime
                ddate = datetime.fromisoformat(str(d.metadata.get("Date"))).date()
                if s and ddate < s: ok = False
                if e and ddate > e: ok = False
            except Exception:
                pass
        if ok:
            d.metadata["_setup_score"] = setup_score(d.metadata)
            selected.append(d)
        if len(selected) >= max(k*4, 24):
            break
    selected_sorted = sorted(selected, key=lambda d: d.metadata.get("_setup_score",0), reverse=True)
    return selected_sorted[:k]