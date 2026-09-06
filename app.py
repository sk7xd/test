import os

import gradio as gr
import numpy as np
import spaces
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

client = InferenceClient(
    model=MODEL_ID,
    token=os.environ.get("HF_TOKEN"),
    provider="featherless-ai",
)
embedder = SentenceTransformer(EMBED_MODEL_ID)

SYSTEM_PROMPT = "You are Qwen, a helpful and knowledgeable AI assistant."


@spaces.GPU
def _zerogpu_warmup():
    # This Space calls the Hugging Face Inference API rather than running the
    # chat model locally, so no GPU work happens here. The decorator only
    # exists to satisfy ZeroGPU hardware, which requires a @spaces.GPU
    # function to be present at startup.
    return True


def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def process_pdfs(files):
    if not files:
        return [], None, "No documents loaded."

    chunks = []
    for f in files:
        reader = PdfReader(f.name)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        chunks.extend(chunk_text(text))

    if not chunks:
        return [], None, "Could not extract any text from the uploaded PDF(s)."

    embeddings = embedder.encode(chunks, normalize_embeddings=True)
    status = f"Loaded {len(chunks)} chunks from {len(files)} document(s)."
    return chunks, embeddings, status


def retrieve_context(query, chunks, embeddings):
    if not chunks or embeddings is None:
        return ""
    query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
    scores = np.array(embeddings) @ query_embedding
    top_indices = np.argsort(scores)[::-1][:TOP_K]
    return "\n\n---\n\n".join(chunks[i] for i in top_indices)


def respond(message, history, system_prompt, max_tokens, temperature, top_p, chunks, embeddings):
    context = retrieve_context(message, chunks, embeddings)
    if context:
        grounded_prompt = (
            f"{system_prompt}\n\n"
            "Use the following excerpts from the user's uploaded documents to answer "
            "their question. If the answer isn't contained in the excerpts, say so "
            "explicitly rather than guessing.\n\n"
            f"{context}"
        )
    else:
        grounded_prompt = system_prompt

    messages = [{"role": "system", "content": grounded_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    partial = ""
    stream = client.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content
        if token:
            partial += token
            yield partial


with gr.Blocks() as demo:
    gr.Markdown(
        f"# Qwen2.5-32B-Instruct\n"
        f"Chat with [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) via the Hugging Face "
        "Inference API, optionally grounded on PDFs you upload below."
    )

    chunks_state = gr.State([])
    embeddings_state = gr.State(None)

    with gr.Accordion("Ground on PDF documents (optional)", open=False):
        pdf_upload = gr.File(
            file_types=[".pdf"],
            file_count="multiple",
            label="Upload PDF(s)",
        )
        status_box = gr.Textbox(label="Document status", interactive=False)
        pdf_upload.change(
            process_pdfs,
            inputs=pdf_upload,
            outputs=[chunks_state, embeddings_state, status_box],
        )

    gr.ChatInterface(
        fn=respond,
        additional_inputs=[
            gr.Textbox(value=SYSTEM_PROMPT, label="System prompt"),
            gr.Slider(minimum=64, maximum=8192, value=2048, step=64, label="Max new tokens"),
            gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
            gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-p"),
            chunks_state,
            embeddings_state,
        ],
    )

if __name__ == "__main__":
    _zerogpu_warmup()
    demo.queue().launch(ssr_mode=False)
