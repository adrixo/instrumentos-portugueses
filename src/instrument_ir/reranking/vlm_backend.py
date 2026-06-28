"""Backends VLM intercambiables (ADR §4 B4).

- OpenAICompatVLMBackend: cliente OpenAI-compatible (p.ej. vLLM sirviendo Qwen2.5-VL). Único backend
  real; cambiar de modelo = cambiar el servido. Requiere el extra [rerank] y un servidor con GPU.
- MockVLMBackend: determinista, sin red/GPU, para tests del pipeline de reranking.

`generate(prompt, images)` recibe imágenes PIL y devuelve el texto crudo de la respuesta.
"""

from __future__ import annotations

import base64
import io
from typing import Protocol


class VLMBackend(Protocol):
    model_id: str

    def generate(self, prompt: str, images: list, *, temperature: float, seed: int,
                 max_new_tokens: int) -> str: ...


def _data_url(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class OpenAICompatVLMBackend:
    """Habla con un endpoint OpenAI-compatible (vLLM). Envía imágenes como data URLs base64."""

    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY"):
        from openai import OpenAI

        self.model_id = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, prompt: str, images: list, *, temperature: float, seed: int,
                 max_new_tokens: int) -> str:
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": _data_url(img)}})
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            seed=seed,
            max_tokens=max_new_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


class MockVLMBackend:
    """Backend determinista para tests. Devuelve respuestas JSON de un guion (en orden) o por callable."""

    model_id = "mock-vlm"

    def __init__(self, scripted: list[str] | None = None, decision_fn=None):
        self._scripted = list(scripted or [])
        self._i = 0
        self._fn = decision_fn

    def generate(self, prompt: str, images: list, *, temperature: float, seed: int,
                 max_new_tokens: int) -> str:
        if self._fn is not None:
            return self._fn(prompt, images)
        if self._scripted:
            out = self._scripted[self._i % len(self._scripted)]
            self._i += 1
            return out
        return '{"decision":"uncertain","confidence":0.0,"visual_evidence":[],"negative_evidence":[],"score":0}'
