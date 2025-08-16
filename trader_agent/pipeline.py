from typing import TypedDict, Optional, Tuple, List, Dict, Any
from datetime import date
from langgraph.graph import START, END, StateGraph
from langchain_core.documents import Document
from .intent import classify_intent, IntentResult
from .date_utils import parse_date_range
from .retriever import retrieve_with_filters, extract_simple_filters
from .lessons import compute_lessons, summarize_lessons

class State(TypedDict):
    query: str
    intent: IntentResult
    date_range: Tuple[Optional[date], Optional[date]]
    docs: List[Document]
    lessons_text: Optional[str]

def build_graph(df, retriever, llm):
    graph = StateGraph(State)


    def node_intent(state: State):
        return {"intent": classify_intent(llm, state["query"])}

    def node_dates(state: State):
        return {"date_range": parse_date_range(state["query"], df)}

    def node_retrieve(state: State):
        filters = (state["intent"].filters or {}).copy()
        # merge fallback regex filters
        for k,v in extract_simple_filters(state["query"]).items():
            filters.setdefault(k, v)

        docs = retrieve_with_filters(retriever, state["query"], filters, state["date_range"], k=5)
        return {"docs": docs}

    def node_lessons(state: State):
        s,e = state["date_range"]
        stats = compute_lessons(df, s, e)
        summary = summarize_lessons(llm, {}, stats)  # persona injected later in composer prompt
        return {"lessons_text": summary}
    
    graph.add_node("intent_node", node_intent)
    graph.add_node("date_node", node_dates)
    graph.add_node("retrieve_node", node_retrieve)
    graph.add_node("lessons_node", node_lessons)

    graph.add_edge(START, "intent_node")
    graph.add_edge("intent_node", "date_node")
    graph.add_edge("date_node", "retrieve_node")


    def when_reflect(state: State) -> bool:
        return state["intent"].intent == "reflect"

    graph.add_conditional_edges("retrieve_node", when_reflect, {True: "lessons_node", False: END})
    graph.add_edge("lessons_node", END)

    return graph.compile()