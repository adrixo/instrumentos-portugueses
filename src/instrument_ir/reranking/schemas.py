"""Parseo y validación de la salida JSON del VLM contra el JSON Schema (ADR §4, §19.14)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

SCHEMA_PATH = Path("schemas/vlm_rerank.schema.json")


@lru_cache(maxsize=4)
def _load_schema(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_json(raw: str) -> dict:
    """Extrae el primer objeto JSON del texto (tolerante a texto alrededor)."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No se encontró JSON en la respuesta del VLM")


def parse_vlm_response(raw: str, schema_path: Path = SCHEMA_PATH) -> dict:
    """Devuelve un dict válido. Si no parsea o no valida, devuelve un 'uncertain' seguro."""
    safe = {
        "decision": "uncertain",
        "confidence": 0.0,
        "visual_evidence": [],
        "negative_evidence": ["unparseable VLM response"],
        "score": 0.0,
    }
    try:
        data = extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        return safe

    # Normalizar tipos mínimos.
    data.setdefault("visual_evidence", [])
    data.setdefault("negative_evidence", [])
    data.setdefault("confidence", 0.0)
    data.setdefault("score", 0.0)
    if data.get("decision") not in ("present", "absent", "uncertain"):
        return safe

    try:
        import jsonschema

        jsonschema.validate(data, _load_schema(str(schema_path)))
    except ImportError:
        pass
    except Exception:
        # No valida el schema: degradar a uncertain en vez de propagar basura.
        return safe
    return data
