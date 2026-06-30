"""B5 — Reranker agéntico determinista (grafo propio, sin LangGraph). ADR §5.

Flujo por candidato:
    full_image_vqa
      -> si confidence >= high_conf (o full_image_only): score = score_full
      -> si no: caption + generate_crops + crop_vqa + evidence_fusion
    -> ranking final (desempate dense_score)

Determinista (temperature=0, seed). Implementa la interfaz Reranker (igual que B4). Ablaciones por flags:
use_crops, use_caption, fusion ("max"|"weighted"), full_image_only.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ..data.queries import Query
from ..reranking.base import Candidate, Reranker, RerankedDoc
from ..reranking.vlm_backend import VLMBackend
from ..utils.io import ImageProvider
from .crop_generation import generate_crops
from .scoring import fuse
from .tools import ask_vlm_vqa, caption_image


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


class AgenticReranker(Reranker):
    def __init__(
        self,
        backend: VLMBackend,
        provider: ImageProvider,
        high_confidence_threshold: float = 0.80,
        low_confidence_threshold: float = 0.40,
        max_crops: int = 5,
        crop_strategies: list[str] | None = None,
        use_crops: bool = True,
        use_caption: bool = True,
        fusion: str = "max",
        full_image_only: bool = False,
        temperature: float = 0.0,
        seed: int = 42,
        dense_retriever_name: str = "dense",
    ):
        self.backend = backend
        self.provider = provider
        self.high_conf = high_confidence_threshold
        self.low_conf = low_confidence_threshold
        self.max_crops = max_crops
        self.crop_strategies = crop_strategies or ["center", "grid2x2"]
        self.use_crops = use_crops and not full_image_only
        self.use_caption = use_caption
        self.fusion = fusion
        self.temperature = temperature
        self.seed = seed
        self.dense_retriever_name = dense_retriever_name
        self.name = f"agent_{backend.model_id}".replace("/", "-").lower()

    def _vqa(self, image, instrument):
        return ask_vlm_vqa(
            self.backend, image, instrument,
            temperature=self.temperature, seed=self.seed,
        )

    def _process_candidate(self, query: Query, instrument: dict, cand: Candidate, run_id: str):
        image = self.provider.load(cand.image_id)
        steps: list[dict] = []

        full = self._vqa(image, instrument)
        steps.append({
            "tool": "ask_vlm_full_image", "decision": full["decision"],
            "confidence": full["confidence"], "score": full["score"], "evidence": full["evidence"],
        })

        score_full = full["score"]
        final_decision, final_score = full["decision"], score_full

        need_crops = self.use_crops and full["confidence"] < self.high_conf
        if need_crops:
            score_caption = 0.0
            if self.use_caption:
                caption = caption_image(
                    self.backend, image, temperature=self.temperature, seed=self.seed
                )
                steps.append({"tool": "caption_image", "caption": caption})
                if instrument.get("canonical_name", "").lower() in caption.lower():
                    score_caption = 1.0

            crops = generate_crops(image, self.crop_strategies, self.max_crops)
            steps.append({"tool": "generate_crops", "crops": [cid for cid, _ in crops]})

            best_crop_score, best_crop_decision = 0.0, "absent"
            for crop_id, crop_img in crops:
                cr = self._vqa(crop_img, instrument)
                steps.append({
                    "tool": "ask_vlm_crop", "crop_id": crop_id, "decision": cr["decision"],
                    "confidence": cr["confidence"], "score": cr["score"], "evidence": cr["evidence"],
                })
                if cr["score"] > best_crop_score:
                    best_crop_score, best_crop_decision = cr["score"], cr["decision"]

            dense_norm = max(0.0, min(1.0, cand.dense_score))
            final_score = fuse(
                self.fusion, score_full=score_full, score_crop=best_crop_score,
                score_caption=score_caption, dense_norm=dense_norm,
            )
            final_decision = full["decision"] if score_full >= best_crop_score else best_crop_decision
            steps.append({"tool": "score_evidence", "score": final_score})

        trace = {
            "run_id": run_id, "query_id": query.query_id, "instrument": query.instrument_id,
            "image_id": cand.image_id, "dense_rank": cand.dense_rank, "dense_score": cand.dense_score,
            "steps": steps, "final_decision": final_decision, "final_score": final_score,
            "final_rank": -1, "seed": self.seed, "temperature": self.temperature,
            "model_versions": {
                "dense_retriever": self.dense_retriever_name,
                "vlm": self.backend.model_id, "agent_framework": "custom_graph",
            },
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        doc = RerankedDoc(
            image_id=cand.image_id, final_score=final_score, final_rank=-1,
            dense_rank=cand.dense_rank, dense_score=cand.dense_score,
            decision=final_decision, confidence=full["confidence"],
        )
        return doc, trace

    def rerank(
        self, query: Query, instrument: dict, candidates: list[Candidate], run_id: str
    ) -> tuple[list[RerankedDoc], list[dict]]:
        workers = max(1, _env_int("VLM_WORKERS", 1))
        if workers == 1 or len(candidates) <= 1:
            results = [
                self._process_candidate(query, instrument, cand, run_id) for cand in candidates
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self._process_candidate, query, instrument, cand, run_id)
                    for cand in candidates
                ]
                results = [future.result() for future in futures]

        docs = [doc for doc, _ in results]
        traces = [trace for _, trace in results]

        order = sorted(
            range(len(docs)), key=lambda i: (docs[i].final_score, docs[i].dense_score), reverse=True
        )
        reranked = []
        for rank, i in enumerate(order, start=1):
            d = docs[i]
            reranked.append(RerankedDoc(
                image_id=d.image_id, final_score=d.final_score, final_rank=rank,
                dense_rank=d.dense_rank, dense_score=d.dense_score,
                decision=d.decision, confidence=d.confidence,
            ))
            traces[i]["final_rank"] = rank
        return reranked, traces
