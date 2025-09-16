from fastapi import FastAPI
from pydantic import BaseModel
from adam.services import load_docs, ask_question

app = FastAPI(title="Adam RAG Chatbot", version="1.0")

class DocsRequest(BaseModel):
    docs: list

@app.post("/add-docs")
def add_docs(req: DocsRequest):
    return load_docs(req.docs)

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask(req: QueryRequest):
    return ask_question(req.query)

@app.get("/health")
def health():
    return {"status": "ok"}
