from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import json

def docs_to_bullets(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        m = d.metadata
        lines.append(
            f"- {m['Trade ID']} | {m['Asset']} | {m['Buy/Sell']} @ {m['Price']} x {m['Volume']} | "
            f"{m['Date']} | {m['Outcome']} | Tags: {m['Tags']} | "
            f"RSI {m['RSI']}, ΔVol {m['Volume_Change_Pct']}%, Sent {m['Sentiment_Score']} | "
            f"SetupScore {m.get('_setup_score','')}"
        )
    return "\\n".join(lines) if lines else "- (none)"


def compose_answer(llm, persona: Dict[str, Any], history: str, intent: str, date_window: str, docs: List[Document], lessons: str, query: str) -> str:
    
    response_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a trader assistant with a consistent trader personality inferred from persona.\n"
     "Answer ONLY using the provided past-trade evidence and persona; do not give financial advice.\n"
     "When explaining, cite metrics (RSI, ΔVol, Sentiment, Tags, SetupScore) from retrieved rows.\n"
     "Intent: {intent}\nPersona: {persona}\nDate window: {date_window}\nLessons (if any):\n{lessons}\n"
     "Conversation (recent):\n{history}"),
    ("human", "User query: {query}\nRelevant past trades:\n{evidence}")
])
    response_chain = response_prompt | llm | StrOutputParser()

    msgs = response_chain.invoke({
        "intent": intent,
        "persona": json.dumps(persona, ensure_ascii=False),
        "date_window": date_window,
        "lessons": lessons or "(none)",
        "history": history,
        "query": query,
        "evidence": docs_to_bullets(docs)
    })

    return msgs