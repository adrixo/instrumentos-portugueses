"""Prompt cerrado del reranker VLM (ADR §4 B4).

Usa el nombre y la definición del instrumento (de instruments.yaml), NUNCA el ground truth de la
imagen candidata ni su nombre de archivo. Salida JSON estricta.
"""

from __future__ import annotations

VLM_RERANK_PROMPT = """You are evaluating whether an image contains a target traditional Portuguese musical instrument.

Target instrument: {instrument_name}
Target definition: {instrument_definition}
Visual cues: {visual_cues}

Rules:
- Use only visible evidence in the image.
- Do not infer from filename, path, dataset split or metadata.
- If unsure, choose "uncertain".
- Return only valid JSON.

JSON schema:
{{
  "decision": "present" | "absent" | "uncertain",
  "confidence": 0.0,
  "visual_evidence": ["short visible evidence"],
  "negative_evidence": ["short uncertainty or absence evidence"],
  "score": 0
}}"""


def build_rerank_prompt(instrument: dict) -> str:
    """instrument = entrada de instruments.yaml (canonical_name, definitions, visual_cues)."""
    name = instrument.get("canonical_name", "the instrument")
    definition = instrument.get("definitions", {}).get("en", "")
    cues = ", ".join(instrument.get("visual_cues", []))
    return VLM_RERANK_PROMPT.format(
        instrument_name=name, instrument_definition=definition, visual_cues=cues
    )
