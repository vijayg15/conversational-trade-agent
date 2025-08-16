import re
from typing import Optional, Tuple
from datetime import datetime, timedelta, date
import pandas as pd
from .config import TODAY

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June","July","August","September","October","November","December"], start=1
)}

def clamp_to_df_dates(start: Optional[date], end: Optional[date], df: pd.DataFrame) -> Tuple[Optional[date], Optional[date]]:
    dmin = pd.to_datetime(df["Date"]).min().date()
    dmax = pd.to_datetime(df["Date"]).max().date()
    if start and start < dmin: start = dmin
    if end and end > dmax: end = dmax
    return start, end

def parse_date_range(text: str, df: pd.DataFrame) -> Tuple[Optional[date], Optional[date]]:
    t = text.lower().strip()
    m = re.search(r"(between|from)\\s*(\\d{4}-\\d{2}-\\d{2})\\s*(and|to)\\s*(\\d{4}-\\d{2}-\\d{2})", t)
    if m:
        s = datetime.fromisoformat(m.group(2)).date()
        e = datetime.fromisoformat(m.group(4)).date()
        return clamp_to_df_dates(s, e, df)
    m = re.search(r"(since|after)\\s*(\\d{4}-\\d{2}-\\d{2})", t)
    if m:
        s = datetime.fromisoformat(m.group(2)).date()
        e = TODAY.date(); return clamp_to_df_dates(s, e, df)
    m = re.search(r"(in\\s+)?([a-zA-Z]+)\\s+(\\d{4})", t)
    if m and m.group(2).lower() in MONTHS:
        year = int(m.group(3)); mon = MONTHS[m.group(2).lower()]
        s = date(year, mon, 1)
        e = (date(year, mon, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        return clamp_to_df_dates(s, e, df)
    m = re.search(r"(in\\s*)?(\\d{4})-(\\d{2})", t)
    if m:
        year = int(m.group(2)); mon = int(m.group(3))
        s = date(year, mon, 1)
        e = (date(year, mon, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        return clamp_to_df_dates(s, e, df)
    m = re.search(r"last\\s+(\\d{1,3})\\s*(day|days|week|weeks|month|months)", t)
    if m:
        n = int(m.group(1)); unit = m.group(2)
        if "day" in unit: s = (TODAY - timedelta(days=n)).date()
        elif "week" in unit: s = (TODAY - timedelta(days=7*n)).date()
        else: s = (TODAY - timedelta(days=30*n)).date()
        e = TODAY.date(); return clamp_to_df_dates(s, e, df)
    if "today" in t: return clamp_to_df_dates(TODAY.date(), TODAY.date(), df)
    if "yesterday" in t:
        y = (TODAY - timedelta(days=1)).date(); return clamp_to_df_dates(y, y, df)
    if "last month" in t:
        prev = TODAY.replace(day=1) - timedelta(days=1)
        s = prev.replace(day=1).date(); e = prev.date()
        return clamp_to_df_dates(s,e,df)
    return None, None