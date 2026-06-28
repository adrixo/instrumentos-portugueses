"""CLI de Instrument Retrieval Lab (ADR §10).

Fase 1 implementa: prepare-data, gen-queries, build-qrels, retrieve (dummy/openclip), evaluate, smoke.
Las etapas pesadas (embed/build-index/rerank-vlm/rerank-agent/report/serve) son stubs hasta su fase.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .utils.reproducibility import set_global_determinism

app = typer.Typer(add_completion=False, help="Instrument Retrieval Lab — visual IR (B1/B3/B4/B5).")

# Rutas por defecto del proyecto.
RAW_DEFAULT = Path("data/raw/portuguese_instruments")
PROCESSED_DEFAULT = Path("data/processed")
INSTRUMENTS_YAML = Path("configs/instruments.yaml")
QUERIES_YAML = Path("configs/queries.yaml")
RUNS_DIR = Path("outputs/runs")
METRICS_DIR = Path("outputs/metrics")


@app.command("gen-queries")
def gen_queries(
    instruments: Path = typer.Option(INSTRUMENTS_YAML, help="configs/instruments.yaml"),
    out: Path = typer.Option(QUERIES_YAML, help="Salida queries.yaml"),
    force: bool = typer.Option(False, help="Sobrescribir si ya existe"),
):
    """Genera configs/queries.yaml desde instruments.yaml (pt/en/es)."""
    from .data.queries import generate_queries, load_instruments, write_queries_yaml

    if out.exists() and not force:
        typer.echo(f"{out} ya existe (usa --force para regenerar). No se toca.")
        return
    spec = load_instruments(instruments)
    queries = generate_queries(spec)
    write_queries_yaml(queries, out)
    n = sum(len(v) for v in queries.values())
    typer.echo(f"Escritas {n} queries para {len(queries)} instrumentos en {out}")


@app.command("prepare-data")
def prepare_data(
    raw: Path = typer.Option(RAW_DEFAULT, help="Raíz del dataset COCO"),
    out: Path = typer.Option(PROCESSED_DEFAULT, help="Salida procesada"),
    splits: str = typer.Option("train,valid,test", help="Splits separados por coma"),
):
    """Parsea COCO, anonimiza y escribe mapping privado + corpus público. Genera queries.yaml."""
    from .data.prepare_dataset import prepare_dataset

    split_tuple = tuple(s.strip() for s in splits.split(",") if s.strip())
    result = prepare_dataset(raw, out, split_tuple)
    typer.echo(f"Mapping privado: {result.mapping_path}")
    for split, path in result.corpus_paths.items():
        typer.echo(f"  corpus {split}: {result.n_images[split]} imágenes -> {path}")
    typer.echo(f"Instrumentos (clases): {len(result.categories)}")

    # Generar queries.yaml si no existe.
    if not QUERIES_YAML.exists():
        gen_queries(instruments=INSTRUMENTS_YAML, out=QUERIES_YAML, force=False)


@app.command("build-qrels")
def build_qrels_cmd(
    split: str = typer.Option(..., help="train|valid|test"),
    processed: Path = typer.Option(PROCESSED_DEFAULT),
    queries: Path = typer.Option(QUERIES_YAML),
    out: Path = typer.Option(None, help="Por defecto data/processed/qrels/{split}.qrels"),
):
    """Construye qrels TREC del split desde el mapping privado + queries."""
    from .data.prepare_dataset import load_mapping
    from .data.qrels import build_qrels, write_qrels_trec
    from .data.queries import load_queries

    mapping = load_mapping(processed / "image_id_mapping.parquet", split=split)
    qs = load_queries(queries)
    rows = build_qrels(mapping, qs)
    out_path = out or (processed / "qrels" / f"{split}.qrels")
    write_qrels_trec(rows, out_path)
    typer.echo(f"qrels {split}: {len(rows)} pares relevantes -> {out_path}")


def _build_retriever(model: str, split: str, processed: Path, raw: Path, seed: int):
    from .data.prepare_dataset import load_mapping
    from .retrieval.factory import build_retriever, resolve_model
    from .utils.io import ImageProvider

    cfg = resolve_model(model)
    cfg.setdefault("seed", seed)
    if cfg["type"] == "dummy":
        return build_retriever(cfg)

    mapping = load_mapping(processed / "image_id_mapping.parquet", split=split)
    provider = ImageProvider(mapping, raw)
    return build_retriever(cfg, provider=provider, split=split)


@app.command("retrieve")
def retrieve(
    split: str = typer.Option(..., help="train|valid|test"),
    model: str = typer.Option(
        "dummy", help="dummy | openclip-vitb32 | openclip-vitl14 | jinaclip | ruta YAML"
    ),
    top_k: int = typer.Option(100, help="Top-K a recuperar"),
    processed: Path = typer.Option(PROCESSED_DEFAULT),
    raw: Path = typer.Option(RAW_DEFAULT),
    queries: Path = typer.Option(QUERIES_YAML),
    run_name: str = typer.Option(None, help="Nombre del run (por defecto B1_{model}_{split})"),
    seed: int = typer.Option(42),
):
    """Recupera top-K por query y escribe un runfile TREC."""
    import pandas as pd

    from .data.queries import load_queries
    from .utils.trec import write_run_trec

    set_global_determinism(seed)
    corpus = pd.read_parquet(processed / split / "corpus.parquet")
    image_ids = corpus["image_id"].tolist()
    qs = load_queries(queries)

    retriever = _build_retriever(model, split, processed, raw, seed)
    rankings = retriever.rank(qs, image_ids, top_k)

    name = run_name or f"B1_{retriever.name}_{split}"
    out_path = RUNS_DIR / f"{name}.trec"
    write_run_trec(rankings, name, out_path)
    typer.echo(f"runfile: {out_path}  ({len(qs)} queries, top_k={top_k})")


@app.command("evaluate")
def evaluate_cmd(
    run: Path = typer.Option(..., help="Runfile TREC"),
    qrels: Path = typer.Option(..., help="Fichero qrels TREC"),
    queries: Path = typer.Option(QUERIES_YAML),
    out: Path = typer.Option(None, help="Por defecto outputs/metrics/{run_stem}.json"),
):
    """Evalúa un runfile contra qrels (Recall/nDCG/mAP/MRR + macro por instrumento)."""
    from .data.queries import load_queries
    from .evaluation.ranx_eval import evaluate_run

    qs = load_queries(queries) if Path(queries).exists() else None
    out_path = out or (METRICS_DIR / f"{Path(run).stem}.json")
    result = evaluate_run(run, qrels, queries=qs, out_path=out_path)
    typer.echo(f"métricas -> {out_path}")
    for m, v in result["metrics"].items():
        typer.echo(f"  {m}: {v:.4f}")
    if "macro_metrics" in result:
        typer.echo("  (macro por instrumento)")
        for m, v in result["macro_metrics"].items():
            typer.echo(f"    macro_{m}: {v:.4f}")

    # Espejo en MLflow (file-store); no-op si no está instalado.
    from .utils.tracking import log_run

    macro = {f"macro_{m}": v for m, v in result.get("macro_metrics", {}).items()}
    log_run(
        experiment="portuguese_instruments_ir",
        run_name=Path(run).stem,
        params={"run": Path(run).stem, "qrels": str(qrels), "n_queries": result["n_queries"]},
        metrics={**result["metrics"], **macro},
    )


@app.command("smoke")
def smoke(
    raw: Path = typer.Option(RAW_DEFAULT),
    processed: Path = typer.Option(PROCESSED_DEFAULT),
    n_images: int = typer.Option(50),
    instruments: str = typer.Option("adufe,cavaquinho", help="2 instrumentos"),
    seed: int = typer.Option(42),
):
    """Smoke test end-to-end (ADR §21): ~50 imágenes, 2 instrumentos, dummy, Recall@10. <10 min."""
    import pandas as pd

    from .data.coco_parser import parse_split
    from .data.anonymize import build_mapping
    from .data.qrels import build_qrels, write_qrels_trec
    from .data.queries import Query
    from .evaluation.ranx_eval import evaluate_run
    from .retrieval.base import DummyRetriever
    from .utils.trec import write_run_trec

    set_global_determinism(seed)
    smoke_dir = Path("outputs/smoke")
    smoke_dir.mkdir(parents=True, exist_ok=True)
    targets = [s.strip() for s in instruments.split(",") if s.strip()]

    # Subconjunto del split valid.
    coco = parse_split(raw, "valid")
    mapping = build_mapping(coco)

    is_target = mapping["gt_instruments"].apply(lambda gt: any(t in gt for t in targets))
    positives = mapping[is_target]
    fillers = mapping[~is_target].head(max(0, n_images - len(positives)))
    subset = pd.concat([positives.head(n_images), fillers], ignore_index=True).head(n_images)
    typer.echo(f"smoke subset: {len(subset)} imágenes ({int(is_target.head(n_images).sum())} positivas aprox)")

    queries = [Query(f"q_{t}_en", t, t, "en") for t in targets]

    rows = build_qrels(subset, queries)
    qrels_path = smoke_dir / "smoke.qrels"
    write_qrels_trec(rows, qrels_path)

    retriever = DummyRetriever(seed=seed)
    rankings = retriever.rank(queries, subset["image_id"].tolist(), top_k=10)
    run_path = smoke_dir / "smoke.trec"
    write_run_trec(rankings, "smoke_dummy", run_path)

    result = evaluate_run(
        run_path, qrels_path, metrics=("recall@10", "ndcg@10", "map"),
        queries=queries, out_path=smoke_dir / "smoke_metrics.json",
    )
    typer.echo("SMOKE OK:")
    for m, v in result["metrics"].items():
        typer.echo(f"  {m}: {v:.4f}")


def _not_implemented(name: str):
    typer.echo(f"[stub] '{name}' se implementa en su fase correspondiente (ver ADR/plan).")


@app.command("embed")
def embed(
    split: str = typer.Option(..., help="train|valid|test"),
    model: str = typer.Option("openclip-vitb32", help="atajo o ruta YAML de configs/models/"),
    processed: Path = typer.Option(PROCESSED_DEFAULT),
    raw: Path = typer.Option(RAW_DEFAULT),
    seed: int = typer.Option(42),
):
    """Precomputa y cachea los embeddings del corpus de un modelo/split (acelera retrieve)."""
    import pandas as pd

    set_global_determinism(seed)
    corpus = pd.read_parquet(processed / split / "corpus.parquet")
    image_ids = corpus["image_id"].tolist()
    retriever = _build_retriever(model, split, processed, raw, seed)
    if not hasattr(retriever, "encode_corpus"):
        raise typer.BadParameter(f"El modelo '{model}' no soporta embed (¿dummy?).")
    emb = retriever.encode_corpus(image_ids)  # encode_corpus cachea internamente
    typer.echo(f"embeddings {retriever.name} {split}: {emb.shape} cacheados")


@app.command("build-index")
def build_index():
    """[stub] Construir índice (Fase 2)."""
    _not_implemented("build-index")


@app.command("rerank-vlm")
def rerank_vlm(
    dense_run: Path = typer.Option(..., help="Runfile dense (B1/B3) con los candidatos"),
    split: str = typer.Option(..., help="train|valid|test (para resolver imágenes)"),
    backend: str = typer.Option("mock", help="mock | openai"),
    vlm_model: str = typer.Option("qwen2.5-vl", help="Modelo VLM servido"),
    base_url: str = typer.Option("http://localhost:8001/v1", help="Endpoint OpenAI-compatible"),
    top_n: int = typer.Option(200, help="Candidatos del dense a rerankear"),
    final_top_k: int = typer.Option(100, help="Top-K final"),
    processed: Path = typer.Option(PROCESSED_DEFAULT),
    raw: Path = typer.Option(RAW_DEFAULT),
    queries: Path = typer.Option(QUERIES_YAML),
    instruments: Path = typer.Option(INSTRUMENTS_YAML),
    run_name: str = typer.Option(None),
    seed: int = typer.Option(42),
):
    """B4 — Reranker VLM pointwise sobre el top-N de un runfile dense."""
    from .data.prepare_dataset import load_mapping
    from .data.queries import load_instruments, load_queries
    from .reranking.base import load_candidates_from_run
    from .reranking.runner import run_pointwise_rerank
    from .reranking.vlm_backend import MockVLMBackend, OpenAICompatVLMBackend
    from .reranking.vlm_pointwise import VLMPointwiseReranker
    from .utils.io import ImageProvider

    set_global_determinism(seed)
    mapping = load_mapping(processed / "image_id_mapping.parquet", split=split)
    provider = ImageProvider(mapping, raw)
    qs = load_queries(queries)
    instr = load_instruments(instruments)
    cands = load_candidates_from_run(dense_run, top_n)

    if backend == "openai":
        vlm = OpenAICompatVLMBackend(model=vlm_model, base_url=base_url)
    else:
        vlm = MockVLMBackend()
    reranker = VLMPointwiseReranker(vlm, provider, seed=seed)

    name = run_name or f"B4_{Path(dense_run).stem}_{vlm.model_id}_{split}".replace("/", "-")
    run_path = run_pointwise_rerank(reranker, qs, instr, cands, name, final_top_k=final_top_k)
    typer.echo(f"B4 runfile: {run_path}")
    typer.echo(f"  traces: outputs/rerank_traces/{name}.jsonl")
    typer.echo(f"  candidates: outputs/candidates/{name}.parquet")


@app.command("rerank-agent")
def rerank_agent():
    """[stub] Reranker agéntico B5 (Fase 5)."""
    _not_implemented("rerank-agent")


@app.command("report")
def report():
    """[stub] Informe (Fase 6)."""
    _not_implemented("report")


@app.command("serve")
def serve():
    """[stub] Buscador web (Fase 6)."""
    _not_implemented("serve")


if __name__ == "__main__":
    app()
