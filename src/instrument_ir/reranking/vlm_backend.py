"""Backends VLM intercambiables (ADR §4 B4).

- OpenAICompatVLMBackend: cliente OpenAI-compatible (p.ej. vLLM sirviendo Qwen2.5-VL). Único backend
  real; cambiar de modelo = cambiar el servido. Requiere el extra [rerank] y un servidor con GPU.
- MockVLMBackend: determinista, sin red/GPU, para tests del pipeline de reranking.

`generate(prompt, images)` recibe imágenes PIL y devuelve el texto crudo de la respuesta.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
from pathlib import Path
from typing import Protocol


class VLMBackend(Protocol):
    model_id: str

    def generate(self, prompt: str, images: list, *, temperature: float, seed: int,
                 max_new_tokens: int) -> str: ...


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_json(name: str) -> dict:
    value = os.environ.get(name)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a valid JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a valid JSON object")
    return parsed


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _prepare_image(pil_image):
    image = pil_image.convert("RGB")
    max_side = _env_int("VLM_MAX_IMAGE_SIDE", 768)
    if max_side > 0 and max(image.size) > max_side:
        from PIL import Image

        image = image.copy()
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
        image.thumbnail((max_side, max_side), resampling)
    return image


def _encode_image(pil_image) -> tuple[str, str]:
    buf = io.BytesIO()
    quality = max(1, min(95, _env_int("VLM_JPEG_QUALITY", 85)))
    _prepare_image(pil_image).save(buf, format="JPEG", quality=quality, optimize=True)
    payload = buf.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    b64 = base64.b64encode(payload).decode()
    return f"data:image/jpeg;base64,{b64}", digest


def _data_url(pil_image) -> str:
    return _encode_image(pil_image)[0]


class OpenAICompatVLMBackend:
    """Habla con un endpoint OpenAI-compatible (vLLM). Envía imágenes como data URLs base64."""

    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY"):
        from openai import OpenAI

        self.model_id = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.cache_enabled = os.environ.get("VLM_CACHE", "1").lower() not in {"0", "false", "no"}
        self.cache_dir = Path(os.environ.get("VLM_CACHE_DIR", "outputs/cache/vlm_openai"))
        self.extra_body = self._build_extra_body()
        self._cache_lock = threading.Lock()
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_extra_body(self) -> dict:
        extra = _env_json("VLM_EXTRA_BODY_JSON")
        if _env_bool("VLM_DISABLE_THINKING", False):
            extra = _deep_merge(extra, {"chat_template_kwargs": {"enable_thinking": False}})
        return extra

    def _cache_key(
        self,
        prompt: str,
        image_digests: list[str],
        temperature: float,
        seed: int,
        max_new_tokens: int,
    ) -> str:
        data = {
            "model": self.model_id,
            "prompt": prompt,
            "images": image_digests,
            "temperature": temperature,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "max_image_side": _env_int("VLM_MAX_IMAGE_SIDE", 768),
            "jpeg_quality": max(1, min(95, _env_int("VLM_JPEG_QUALITY", 85))),
            "extra_body": self.extra_body,
        }
        raw = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _read_cache(self, key: str) -> str | None:
        if not self.cache_enabled:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        except Exception:
            return None

    def _write_cache(self, key: str, text: str) -> None:
        if not self.cache_enabled:
            return
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")
        with self._cache_lock:
            tmp.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)

    def generate(self, prompt: str, images: list, *, temperature: float, seed: int,
                 max_new_tokens: int) -> str:
        encoded = [_encode_image(img) for img in images]
        key = self._cache_key(
            prompt, [digest for _, digest in encoded], temperature, seed, max_new_tokens
        )
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        content = [{"type": "text", "text": prompt}]
        for data_url, _ in encoded:
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        request = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_new_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.extra_body:
            request["extra_body"] = self.extra_body
        resp = self.client.chat.completions.create(**request)
        text = resp.choices[0].message.content or ""
        self._write_cache(key, text)
        return text


class HFVLMBackend:
    """VLM in-process con transformers (sirve en MPS/CUDA/CPU; alternativa a vLLM en Mac).

    Para B4/B5 cuando no hay servidor OpenAI-compatible (p.ej. macOS sin CUDA). Greedy (determinista).
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model
        self._torch = torch
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, torch_dtype=dtype
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model)

    def generate(self, prompt: str, images: list, *, temperature: float, seed: int,
                 max_new_tokens: int) -> str:
        torch = self._torch
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=list(images), return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]


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
