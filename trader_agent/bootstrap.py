import os
from typing import Callable, Dict, Any
from .config import USE_OPENAI, CSV_PATH, OPENAI_CHAT_MODEL, OPENAI_EMBED_MODEL, HF_EMBED_MODEL, HF_CHAT_REPO
from .data_store import load_csv, build_docs, build_vectorstore, load_vectorstore
from .persona import extract_persona
from .memory import Memory
from .intent import classify_intent
from .pipeline import build_graph
from .composer import compose_answer
from .retriever import extract_simple_filters

# Backends
if USE_OPENAI:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    EMBEDDINGS = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    LLM = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.2)
else:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFaceHub
    EMBEDDINGS = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    LLM = HuggingFaceHub(repo_id=HF_CHAT_REPO,
                         huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                         model_kwargs={"temperature": 0.2, "max_new_tokens": 512})

def boot_agent(csv_path: str = CSV_PATH) -> Dict[str, Any]:
    df = load_csv(csv_path)
    docs = build_docs(df)
    #For first time rum
    vectorstore = build_vectorstore(docs, EMBEDDINGS)

    #Load from local
    #vectorstore = load_vectorstore("vectordb", EMBEDDINGS)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    # persona snapshot
    persona = extract_persona(df)

    # langgraph app
    app = build_graph(df, retriever, LLM)

    # short-term memory
    memory = Memory()

    def chat(user_text: str) -> str:
        memory.add("user", user_text)
        # ✅ Only pass 'query' into invoke()
        state = app.invoke({"query": user_text})

        intent_obj = state.get("intent")
        if intent_obj is None:
            # Extremely defensive fallback (shouldn't happen)
            intent_obj = classify_intent(LLM, user_text)

        intent = intent_obj.intent if hasattr(intent_obj, "intent") else getattr(intent_obj, "intent", "generic")
        s,e = state.get("date_range", (None, None))
        docs = state.get("docs", [])
        lessons = state.get("lessons_text","") if intent=="reflect" else ""

        reply = compose_answer(
            llm=LLM,
            intent=intent,
            persona=persona,
            date_window=f"{s} → {e}",
            lessons=lessons or "(none)",         
            history=memory.last_context(),
            query=user_text,
            docs=docs,            
        )
        memory.add("assistant", reply)
        return reply

    return {
        "df": df,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "persona": persona,
        "app": app,
        "llm": LLM,
        "chat": chat,
        "memory": memory
    }