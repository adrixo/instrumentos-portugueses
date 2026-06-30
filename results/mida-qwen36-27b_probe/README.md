# MIDA Qwen3.6-27B Probe

Date: 2026-06-30

Endpoint tested from this workstation:

```text
http://100.113.15.12:8080/v1
model: qwen36-27b
```

The Hugging Face model metadata for `unsloth/Qwen3.6-27B-GGUF` reports:

```json
{
  "pipeline_tag": "image-text-to-text",
  "files": [
    "Qwen3.6-27B-Q4_K_M.gguf",
    "mmproj-F16.gguf"
  ]
}
```

Current `/v1/models` probe:

```json
{
  "id": "qwen36-27b",
  "capabilities": ["completion", "multimodal"],
  "n_ctx": 262144,
  "n_params": 26895998464
}
```

Thinking control probe:

```json
{
  "request_extra": {
    "chat_template_kwargs": {"enable_thinking": false}
  },
  "response_content": "{\"decision\":\"present\",\"confidence\":0.5}"
}
```

Image probe:

```json
{
  "status": "ok",
  "response_content": "{\"ok\":true}",
  "prompt_tokens": 1047
}
```

Conclusion: MIDA is currently usable as a llama.cpp/OpenAI-compatible multimodal backend. For runs from
`esalab-big`, use `http://100.127.120.42:8080/v1` and set `VLM_DISABLE_THINKING=true`.
