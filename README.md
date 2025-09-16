# 🤖 Adam – RAG-Based AI Chatbot

**Adam** is a **Retrieval-Augmented Generation (RAG) pipeline** that combines **embeddings, vector search, and Transformer-based text generation** into an interactive chatbot.  
It demonstrates strong **GenAI engineering practices**, with **FastAPI, Gradio, and Streamlit interfaces**, making it deployable as both a backend microservice and a user-facing app.  

---

## 🚀 Pipeline Overview

1. **📄 Data Ingestion & Chunking**  
   - Documents are split into manageable chunks using **LangChain’s RecursiveCharacterTextSplitter**.  

2. **🔎 Embedding & Indexing**  
   - Chunks are embedded using **MiniLM (SentenceTransformers)**.  
   - Stored in a **FAISS vector database** for efficient semantic search.  

3. **❓ Query Processing**  
   - User query is embedded and matched against the FAISS index to retrieve relevant context.  

4. **📝 Answer Generation**  
   - Context + Query passed to **FLAN-T5 (HuggingFace Transformers)** for natural language response generation.  

5. **⚡ Multi-Interface Deployment**  
   - **FastAPI** → REST endpoints for integration in services.  
   - **Gradio** → Lightweight demo UI.  
   - **Streamlit** → Sleek frontend for interactive chat.  
   - **Docker** → Containerized for deployment.  

---

## ✨ Features

- **RAG Architecture** → Retrieval + Generation pipeline.  
- **Embeddings** → `MiniLM` for semantic similarity.  
- **Generation** → `FLAN-T5` for context-aware answers.  
- **APIs & Frontends**:  
  - FastAPI backend (REST endpoints).  
  - Gradio UI for quick testing.  
  - Streamlit app for interactive chat.  
- **Deployment Ready** → Comes with `requirements.txt` + `Dockerfile`.  

---

## 🛠️ Setup

Clone repo and install requirements:
### bash
git clone https://github.com/divyansh286/ADAM-RAG-Chatbot.git
cd ADAM-RAG-Chatbot
pip install -r requirements.txt

### Run API (FastAPI):
uvicorn api:app --reload --port 8000

### Run Gradio UI:
python ui.py

### Run Streamlit UI:
streamlit run streamlit_app.py

### Run with Docker:

docker build -t adam-rag .
docker run -p 8000:8000 adam-rag

## 📁 Files

- **adam/core.py** –> Core RAG pipeline (embeddings + retrieval + generation).

- **adam/services.py** –> Service wrappers for API/UI.

- **api.py** –> FastAPI microservice exposing /add-docs and /ask.

**ui.py** –> Gradio UI (upload docs + chat).

- **streamlit_app.py** –> Streamlit UI (interactive chatbot).

- **requirements.txt** –> Python dependencies.

- **Dockerfile** –> Deployment container.

## 🧑‍💻 Tech Stack

- **LLMs & Transformers** → FLAN-T5, HuggingFace Pipelines

- **Embeddings** → MiniLM (SentenceTransformers)

- **Vector Search** → FAISS

- **Frameworks** → LangChain, FastAPI, Gradio, Streamlit

- **Deployment** → Docker, REST APIs 

## 👤 Author

### Divyansh
**AI Engineer | Data Science & GenAI Enthusiast**

### 💻 GitHub: [divyansh286](https://github.com/divyansh286)

### 🌐 Skills: LLMs, Transformers, FastAPI, Django REST, LangChain, Docker, Streamlit, Gradio
