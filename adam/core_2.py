import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FAISS index (in-memory)
dimension = 384  # MiniLM embedding size
index = faiss.IndexFlatL2(dimension)
documents = []
doc_embeddings = None

# Load generation model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
summarizer = pipeline("text2text-generation", model=generator, tokenizer=tokenizer)

def add_documents(texts):
    """Embed and add documents to FAISS index"""
    global doc_embeddings, documents
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    if doc_embeddings is None:
        doc_embeddings = embeddings
    else:
        doc_embeddings = np.vstack((doc_embeddings, embeddings))
    index.add(embeddings)
    documents.extend(texts)

def retrieve(query, top_k=3):
    """Retrieve top-k documents using FAISS"""
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    return [documents[i] for i in I[0]]

def generate_answer(query, context):
    """Generate answer using FLAN-T5 given retrieved context"""
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    outputs = summarizer(prompt, max_length=128, do_sample=False)
    return outputs[0]["generated_text"]

def rag_pipeline(query):
    """Full retrieval-augmented generation pipeline"""
    retrieved_docs = retrieve(query)
    context = " ".join(retrieved_docs)
    return generate_answer(query, context)
