import json, re
from typing import Optional, Dict
from pydantic import BaseModel
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate

class IntentResult(BaseModel):
    intent: Literal["persona","why_trade","list","best","reflect","generic"]
    filters: Optional[Dict[str, str]] = None


def classify_intent(llm, query: str) -> IntentResult:

    INTENT_FEW_SHOTS = [
        {"user": "What trades do you prefer and why?", "intent": "persona"},
        {"user": "Why did you buy DOGE on 2024-10-09?", "intent": "why_trade"},
        {"user": "Show recent BTC buys", "intent": "list"},
        {"user": "Which were your most profitable ETH trades?", "intent": "best"},
        {"user": "What lessons did you learn last month? What worked and what failed?", "intent": "reflect"},
    ]
    FEW_SHOTS_TEXT = "\\n".join([f"User: {x['user']}\\nIntent: {x['intent']}" for x in INTENT_FEW_SHOTS])

    intent_prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are an intent classifier for a trader assistant. "
        "Return ONLY a compact JSON object with keys 'intent' and optional 'filters'. "
        "Valid intents: persona, why_trade, list, best, reflect, generic.\\n\\n"
        "Examples:\\n{few_shots}"),
        ("human", "{query}")
    ])

    msgs = intent_prompt.format_messages(query=query, few_shots=FEW_SHOTS_TEXT)
    raw = llm.invoke(msgs).content.strip().strip("`")
    try:
        data = json.loads(raw)
        return IntentResult(**data)
    except Exception:
        ql = query.lower()
        if any(w in ql for w in ["lesson","learn","worked","failed","pattern","insight","reflect"]):
            return IntentResult(intent="reflect")
        if any(w in ql for w in ["why did","reason","explain"]):
            return IntentResult(intent="why_trade")
        if any(w in ql for w in ["prefer","style","persona"]):
            return IntentResult(intent="persona")
        if any(w in ql for w in ["show","list","recent"]):
            return IntentResult(intent="list")
        if any(w in ql for w in ["best","profitable","wins"]):
            return IntentResult(intent="best")
        return IntentResult(intent="generic")