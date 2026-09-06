import os

import gradio as gr
import spaces
from huggingface_hub import InferenceClient

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"

client = InferenceClient(model=MODEL_ID, token=os.environ.get("HF_TOKEN"))

SYSTEM_PROMPT = "You are Qwen, a helpful and knowledgeable AI assistant."


@spaces.GPU
def _zerogpu_warmup():
    # This Space calls the Hugging Face Inference API rather than running the
    # model locally, so no GPU work happens here. The decorator only exists
    # to satisfy ZeroGPU hardware, which requires a @spaces.GPU function to
    # be present at startup.
    return True


def respond(message, history, system_prompt, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
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
        token = chunk.choices[0].delta.content
        if token:
            partial += token
            yield partial


demo = gr.ChatInterface(
    fn=respond,
    title="Qwen2.5-32B-Instruct",
    description=(
        f"Chat with [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) via the "
        "Hugging Face Inference API."
    ),
    additional_inputs=[
        gr.Textbox(value=SYSTEM_PROMPT, label="System prompt"),
        gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Max new tokens"),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-p"),
    ],
)

if __name__ == "__main__":
    _zerogpu_warmup()
    demo.queue().launch()
