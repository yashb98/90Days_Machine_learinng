# app.py
import gradio as gr
from rag import load_pdf, create_vectorstore, rag_query

collection = None  # global Chroma collection


def upload_and_index(pdf_file):
    global collection
    chunks = load_pdf(pdf_file.name)
    collection = create_vectorstore(chunks)
    return f"✅ PDF loaded with {len(chunks)} chunks."


def ask_question(query, k):
    if not collection:
        return "Please upload a PDF first.", ""
    result = rag_query(query, k=k, collection=collection)
    sources = "\n".join([
        f"- Page {r['page']} | Chunk {r['chunk_id']}" for r in result["sources"]
    ])
    return result["answer"], sources


demo = gr.Blocks()

with demo:
    gr.Markdown("# 📘 Gemini RAG System — PDF Q&A")

    with gr.Row():
        pdf_file = gr.File(label="Upload your PDF")
        upload_btn = gr.Button("Process PDF")

    output_status = gr.Textbox(label="Status")

    with gr.Row():
        query = gr.Textbox(label="Ask a question about the PDF", lines=2)
        k = gr.Slider(1, 10, value=3, step=1, label="Top K Chunks")
        ask_btn = gr.Button("Ask")

    answer = gr.Textbox(label="Answer", lines=6)
    sources = gr.Textbox(label="Sources", lines=6)

    upload_btn.click(upload_and_index, inputs=[
                     pdf_file], outputs=[output_status])
    ask_btn.click(ask_question, inputs=[query, k], outputs=[answer, sources])

demo.launch()
