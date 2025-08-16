import pandas as pd
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def row_to_text(row) -> str:
    fields = [
        f"Trade {row['Trade ID']}",
        f"Asset {row['Asset']}",
        f"Side {row['Buy/Sell']}",
        f"Price {row['Price']}",
        f"Volume {row['Volume']}",
        f"Date {row['Date']}",
        f"Outcome {row['Outcome']}",
        f"Tags {row['Tags']}",
        f"RSI {row['RSI']}",
        f"VolumeChangePct {row['Volume_Change_Pct']}",
        f"Sentiment {row['Sentiment_Score']}",
    ]
    return " | ".join(map(str, fields))

def build_docs(df: pd.DataFrame) -> List[Document]:
    return [
        Document(page_content=row_to_text(r), metadata={k: r[k] for k in df.columns})
        for _, r in df.iterrows()
    ]

def build_vectorstore(docs: List[Document], embeddings, path_db: str="vectordb") -> Tuple[FAISS, Any]:
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(path_db)
    return vectorstore

def load_vectorstore(path_db: str, embeddings):
    db = FAISS.load_local(path_db, embeddings, allow_dangerous_deserialization=True)
    return db