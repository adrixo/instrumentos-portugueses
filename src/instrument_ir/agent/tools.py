"""Tools del agente B5 (ADR §5). Dependencias (VLM) inyectadas → mockeables sin GPU.

- ask_vlm_vqa: VQA de presencia sobre una imagen/crop (reutiliza el prompt y el parseo de B4).
- caption_image: descripción breve de la imagen.
No usan ground truth ni nombres de archivo.
"""

from __future__ import annotations

from ..reranking.prompts import build_rerank_prompt
from ..reranking.schemas import parse_vlm_response
from ..reranking.scoring import normalize_score
from ..reranking.vlm_backend import VLMBackend

CAPTION_PROMPT = (
    "Describe in one short sentence the musical instruments visible in this image. "
    "Use only visible evidence. Do not guess from metadata."
)


def ask_vlm_vqa(
    backend: VLMBackend, image, instrument: dict, *, temperature: float, seed: int,
    max_new_tokens: int = 512,
) -> dict:
    """Pregunta si el instrumento está presente. Devuelve decision/confidence/score/evidence."""
    prompt = build_rerank_prompt(instrument)
    raw = backend.generate(
        prompt, [image], temperature=temperature, seed=seed, max_new_tokens=max_new_tokens
    )
    parsed = parse_vlm_response(raw)
    return {
        "decision": parsed["decision"],
        "confidence": float(parsed.get("confidence", 0.0)),
        "score": normalize_score(parsed["decision"], parsed.get("confidence", 0.0)),
        "evidence": parsed.get("visual_evidence", [])[:4],
    }


def caption_image(
    backend: VLMBackend, image, *, temperature: float, seed: int, max_new_tokens: int = 128
) -> str:
    raw = backend.generate(
        CAPTION_PROMPT, [image], temperature=temperature, seed=seed, max_new_tokens=max_new_tokens
    )
    return raw.strip()[:300]
