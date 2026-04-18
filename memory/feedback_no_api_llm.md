---
name: Never suggest paid API LLMs
description: Always use open-source local LLMs (Qwen/Llama), never suggest OpenAI/Anthropic/Google API calls
type: feedback
---

Never suggest calling paid LLM APIs (OpenAI, Anthropic, Google). User has A100 80GB GPUs available — always use open-source models (Qwen, Llama, etc.) running locally via HuggingFace transformers.

**Why:** User considers this a bad suggestion. Research should use free, reproducible, locally-runnable models. No dependency on external services or API keys.

**How to apply:** For any LLM-related task, default to the best open-source model that fits on A100 80GB. Currently: Qwen2.5-72B-Instruct or Llama-3.1-70B-Instruct (both fit in bf16 on 80GB).