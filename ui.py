import gradio as gr
from adam.services import load_docs, ask_question

docs = []

def upload_docs(text):
    global docs
    docs.append(text)
    load_docs([text])
    return f"Document added. Total docs: {len(docs)}"

def chat(query):
    return ask_question(query)["answer"]

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Adam - RAG Chatbot")

    with gr.Tab("📄 Upload Docs"):
        doc_in = gr.Textbox(label="Enter text/document")
        doc_btn = gr.Button("Add Document")
        doc_out = gr.Textbox(label="Status")
        doc_btn.click(upload_docs, inputs=doc_in, outputs=doc_out)

    with gr.Tab("💬 Ask Adam"):
        q_in = gr.Textbox(label="Your Question")
        q_btn = gr.Button("Ask")
        q_out = gr.Textbox(label="Answer")
        q_btn.click(chat, inputs=q_in, outputs=q_out)

if __name__ == "__main__":
    demo.launch()
