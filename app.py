import os
import tempfile

import gradio as gr
import numpy as np
import spaces
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
TTS_MODEL_ID = "Qwen/Qwen3-TTS"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

client = InferenceClient(
    model=MODEL_ID,
    token=os.environ.get("HF_TOKEN"),
    provider="featherless-ai",
)
# TODO: set the correct provider once confirmed from the Qwen3-TTS model
# card's "Inference Providers" widget (same issue we had with the chat
# model needing provider="featherless-ai" set explicitly).
tts_client = InferenceClient(
    token=os.environ.get("HF_TOKEN"),
    provider="auto",
)
embedder = SentenceTransformer(EMBED_MODEL_ID)

SYSTEM_PROMPT = "You are Qwen, a helpful and knowledgeable AI assistant."


@spaces.GPU
def _zerogpu_warmup():
    # This Space calls the Hugging Face Inference API rather than running
    # models locally, so no GPU work happens here. The decorator only exists
    # to satisfy ZeroGPU hardware, which requires a @spaces.GPU function to
    # be present at startup.
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


def synthesize_speech(text):
    if not text.strip():
        return None
    audio_bytes = tts_client.text_to_speech(text, model=TTS_MODEL_ID)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


def user_submit(message, history):
    history = history + [{"role": "user", "content": message}]
    return "", history


def bot_respond(history, system_prompt, max_tokens, temperature, top_p, chunks, embeddings):
    user_message = history[-1]["content"]
    context = retrieve_context(user_message, chunks, embeddings)
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
    messages.extend(history[:-1])
    messages.append({"role": "user", "content": user_message})

    history = history + [{"role": "assistant", "content": ""}]

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
            history[-1]["content"] = partial
            yield history, None

    audio_path = synthesize_speech(partial)
    yield history, audio_path


with gr.Blocks() as demo:
    gr.Markdown(
        f"# Qwen2.5-32B-Instruct with voice replies\n"
        f"Ask [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) a question and hear the "
        f"answer spoken back via [{TTS_MODEL_ID}](https://huggingface.co/{TTS_MODEL_ID}), "
        "optionally grounded on PDFs you upload below."
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

    with gr.Accordion("Generation settings", open=False):
        system_prompt_box = gr.Textbox(value=SYSTEM_PROMPT, label="System prompt")
        max_tokens_slider = gr.Slider(minimum=64, maximum=8192, value=2048, step=64, label="Max new tokens")
        temperature_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature")
        top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-p")

    chatbot = gr.Chatbot(type="messages", label="Chat")
    audio_output = gr.Audio(label="Spoken response", autoplay=True)
    msg_box = gr.Textbox(label="Message", placeholder="Ask something...")

    msg_box.submit(
        user_submit,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
    ).then(
        bot_respond,
        inputs=[
            chatbot,
            system_prompt_box,
            max_tokens_slider,
            temperature_slider,
            top_p_slider,
            chunks_state,
            embeddings_state,
        ],
        outputs=[chatbot, audio_output],
    )

if __name__ == "__main__":
    _zerogpu_warmup()
    demo.queue().launch(ssr_mode=False)
