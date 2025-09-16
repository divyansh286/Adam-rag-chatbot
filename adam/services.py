from .core import add_documents, rag_pipeline

def load_docs(docs: list):
    """Wrapper to load documents"""
    add_documents(docs)
    return {"status": "documents loaded", "count": len(docs)}

def ask_question(query: str):
    """Wrapper for RAG query"""
    return {"answer": rag_pipeline(query)}
