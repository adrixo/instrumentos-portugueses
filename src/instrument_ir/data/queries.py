"""Generación y carga de queries por instrumento (ADR §5.4).

Una query es "encuéntrame imágenes con <instrumento>" expresada como texto en pt/en/es.
El texto sale de `configs/instruments.yaml` (campo `names`), editable por el estudiante.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_LANGUAGES = ("pt", "en", "es")


@dataclass(frozen=True)
class Query:
    query_id: str
    instrument_id: str
    text: str
    language: str


def _qid(instrument_id: str, language: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", instrument_id.lower()).strip("_")
    return f"q_{slug}_{language}"


def load_instruments(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def generate_queries(
    instruments: dict, languages: tuple[str, ...] = DEFAULT_LANGUAGES
) -> dict[str, list[dict]]:
    """Devuelve {instrument_id: [{id, text, language}, ...]} listo para volcar a YAML."""
    out: dict[str, list[dict]] = {}
    for instrument_id, spec in instruments.items():
        names = spec.get("names", {})
        qlist: list[dict] = []
        for lang in languages:
            text = names.get(lang)
            if not text:
                continue
            qlist.append({"id": _qid(instrument_id, lang), "text": text, "language": lang})
        if qlist:
            out[instrument_id] = qlist
    return out


def write_queries_yaml(queries_by_instrument: dict[str, list[dict]], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Queries por instrumento (ADR §5.4). Generado desde configs/instruments.yaml.\n"
        "# Editable: puedes mejorar textos, sinónimos o añadir idiomas.\n"
    )
    body = yaml.safe_dump(queries_by_instrument, allow_unicode=True, sort_keys=True)
    out_path.write_text(header + body, encoding="utf-8")
    return out_path


def load_queries(path: Path) -> list[Query]:
    """Carga queries.yaml a una lista plana de Query."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    queries: list[Query] = []
    for instrument_id, qlist in data.items():
        for q in qlist:
            queries.append(
                Query(
                    query_id=q["id"],
                    instrument_id=instrument_id,
                    text=q["text"],
                    language=q["language"],
                )
            )
    return queries
