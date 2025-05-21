import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# Sample dataset
documents = [
    "The sun is the star at the center of the Solar System.",
    "The water cycle describes how water evaporates, rises, cools, and returns as rain.",
    "Mahatma Gandhi led India to independence from British rule through nonviolent resistance.",
    "Photosynthesis is how plants use sunlight to synthesize food from carbon dioxide and water.",
    "A black hole is a spacetime region where gravity is so strong that not even light can escape."
]

# Chunk and embed
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents(documents)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding_model)

# Setup retrieval chain
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
llm = HuggingFacePipeline(pipeline=qa_pipeline)
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Streamlit UI
st.set_page_config(page_title="adam - RAG Chatbot")
st.title("🤖 adam - RAG Chatbot")
query = st.text_input("Ask me anything:")

if query:
    with st.spinner("Thinking..."):
        answer = rag_chain.run(query)
        st.write("💡 adam:", answer)
