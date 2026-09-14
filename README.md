---
title: Qwen2.5 32B Instruct Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
models:
  - Qwen/Qwen2.5-32B-Instruct
---

# Qwen2.5-32B-Instruct Demo

A chat demo for [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct), served through the
Hugging Face Inference API (the model is a 32B-parameter LLM, too large to load on typical Space hardware).

## Setup

This Space calls the Hugging Face Inference API, so it needs a token with `read` access:

1. Create a token at https://huggingface.co/settings/tokens.
2. In your Space settings, add a secret named `HF_TOKEN` with that token's value.

## Local development

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxx
python app.py
```
