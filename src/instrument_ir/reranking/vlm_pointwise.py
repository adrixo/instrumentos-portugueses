"""B4 — Reranker VLM pointwise (ADR §4).

Para cada candidato del top-N dense: carga la imagen, pregunta al VLM (prompt JSON cerrado), normaliza
el score y reordena. Genera trazas estructuradas (una por candidato). Determinista (temperature=0, seed).
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..data.queries import Query
from ..utils.io import ImageProvider
from .base import Candidate, Reranker, RerankedDoc
from .prompts import build_rerank_prompt
from .schemas import parse_vlm_response
from .scoring import normalize_score
from .vlm_backend import VLMBackend


class VLMPointwiseReranker(Reranker):
    def __init__(
        self,
        backend: VLMBackend,
        provider: ImageProvider,
        temperature: float = 0.0,
        seed: int = 42,
        max_new_tokens: int = 512,
    ):
        self.backend = backend
        self.provider = provider
        self.temperature = temperature
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.name = f"vlm_{backend.model_id}".replace("/", "-").lower()

    def rerank(
        self, query: Query, instrument: dict, candidates: list[Candidate], run_id: str
    ) -> tuple[list[RerankedDoc], list[dict]]:
        prompt = build_rerank_prompt(instrument)
        scored: list[RerankedDoc] = []
        traces: list[dict] = []

        for cand in candidates:
            img = self.provider.load(cand.image_id)
            raw = self.backend.generate(
                prompt, [img], temperature=self.temperature, seed=self.seed,
                max_new_tokens=self.max_new_tokens,
            )
            parsed = parse_vlm_response(raw)
            decision = parsed["decision"]
            confidence = float(parsed.get("confidence", 0.0))
            final = normalize_score(decision, confidence)

            scored.append(
                RerankedDoc(
                    image_id=cand.image_id, final_score=final, final_rank=-1,
                    dense_rank=cand.dense_rank, dense_score=cand.dense_score,
                    decision=decision, confidence=confidence,
                )
            )
            traces.append(
                {
                    "run_id": run_id,
                    "query_id": query.query_id,
                    "instrument": query.instrument_id,
                    "image_id": cand.image_id,
                    "dense_rank": cand.dense_rank,
                    "dense_score": cand.dense_score,
                    "vlm_decision": decision,
                    "vlm_confidence": confidence,
                    "vlm_score": final,
                    "final_score": final,
                    "visual_evidence": parsed.get("visual_evidence", [])[:4],
                    "negative_evidence": parsed.get("negative_evidence", [])[:4],
                    "model": self.backend.model_id,
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Orden final: final_score desc, desempate por dense_score desc (ADR §4).
        order = sorted(
            range(len(scored)),
            key=lambda i: (scored[i].final_score, scored[i].dense_score),
            reverse=True,
        )
        reranked = []
        for rank, i in enumerate(order, start=1):
            d = scored[i]
            reranked.append(
                RerankedDoc(
                    image_id=d.image_id, final_score=d.final_score, final_rank=rank,
                    dense_rank=d.dense_rank, dense_score=d.dense_score,
                    decision=d.decision, confidence=d.confidence,
                )
            )
            traces[i]["final_rank"] = rank
        return reranked, traces
