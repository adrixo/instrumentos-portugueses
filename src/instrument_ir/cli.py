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


@app.command("prepare-mini")
def prepare_mini(
    instruments_sel: str = typer.Option("adufe,concertina,cavaquinho", help="Instrumentos del subset"),
    n_images: int = typer.Option(60, help="Tamaño del corpus mini"),
    source_split: str = typer.Option("valid"),
    processed: Path = typer.Option(PROCESSED_DEFAULT),
    queries: Path = typer.Option(QUERIES_YAML),
):
    """Crea un split 'mini' (subconjunto pequeño) para un end-to-end completo en GPU modesta/MPS."""
    import pandas as pd

    from .data.prepare_dataset import load_mapping
    from .data.qrels import build_qrels, write_qrels_trec
    from .data.queries import load_queries

    targets = [s.strip() for s in instruments_sel.split(",") if s.strip()]
    full = load_mapping(processed / "image_id_mapping.parquet", split=source_split)
    is_pos = full["gt_instruments"].apply(lambda g: any(t in g for t in targets))
    subset = pd.concat(
        [full[is_pos], full[~is_pos].head(max(0, n_images - int(is_pos.sum())))]
    ).head(n_images).reset_index(drop=True)

    mini_dir = processed / "mini"
    mini_dir.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(mini_dir / "image_id_mapping.parquet", index=False)  # conserva split/file_name reales
    subset[["image_id", "split", "width", "height"]].to_parquet(mini_dir / "corpus.parquet", index=False)

    qs = load_queries(queries)
    rows = build_qrels(subset, qs)
    write_qrels_trec(rows, processed / "qrels" / "mini.qrels")

    # Queries reducidas (instrumentos del subset, solo EN) para un end-to-end ágil.
    from .data.queries import generate_queries, load_instruments, write_queries_yaml

    spec = load_instruments(INSTRUMENTS_YAML)
    sub_spec = {k: spec[k] for k in targets if k in spec}
    mini_q = generate_queries(sub_spec, languages=("en",))
    write_queries_yaml(mini_q, Path("configs/queries_mini.yaml"))

    typer.echo(f"split mini: {len(subset)} imágenes ({int(is_pos.sum())} positivas de {targets})")
    typer.echo(f"  corpus: {mini_dir}/corpus.parquet  qrels: {processed}/qrels/mini.qrels")
    typer.echo("  queries reducidas: configs/queries_mini.yaml (EN)")


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
    from .data.prepare_dataset import resolve_mapping
    from .retrieval.factory import build_retriever, resolve_model
    from .utils.io import ImageProvider

    cfg = resolve_model(model)
    cfg.setdefault("seed", seed)
    if cfg["type"] == "dummy":
        return build_retriever(cfg)

    mapping = resolve_mapping(processed, split)
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


def _build_vlm_backend(backend: str, vlm_model: str, base_url: str):
    """mock | openai (vLLM/servidor) | hf (transformers in-process, MPS/CUDA/CPU)."""
    from .reranking.vlm_backend import HFVLMBackend, MockVLMBackend, OpenAICompatVLMBackend

    if backend == "openai":
        return OpenAICompatVLMBackend(model=vlm_model, base_url=base_url)
    if backend == "hf":
        return HFVLMBackend(model=vlm_model)
    return MockVLMBackend()


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
    backend: str = typer.Option("mock", help="mock | openai | hf (transformers in-process)"),
    vlm_model: str = typer.Option("qwen2.5-vl", help="Modelo VLM (servido o HF id)"),
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
    from .data.prepare_dataset import resolve_mapping
    from .data.queries import load_instruments, load_queries
    from .reranking.base import load_candidates_from_run
    from .reranking.runner import run_pointwise_rerank
    from .reranking.vlm_pointwise import VLMPointwiseReranker
    from .utils.io import ImageProvider

    set_global_determinism(seed)
    mapping = resolve_mapping(processed, split)
    provider = ImageProvider(mapping, raw)
    qs = load_queries(queries)
    instr = load_instruments(instruments)
    cands = load_candidates_from_run(dense_run, top_n)

    vlm = _build_vlm_backend(backend, vlm_model, base_url)
    reranker = VLMPointwiseReranker(vlm, provider, seed=seed)

    name = run_name or f"B4_{Path(dense_run).stem}_{vlm.model_id}_{split}".replace("/", "-")
    run_path = run_pointwise_rerank(reranker, qs, instr, cands, name, final_top_k=final_top_k)
    typer.echo(f"B4 runfile: {run_path}")
    typer.echo(f"  traces: outputs/rerank_traces/{name}.jsonl")
    typer.echo(f"  candidates: outputs/candidates/{name}.parquet")


_ABLATIONS = {
    "full": {},
    "no_crops": {"use_crops": False},
    "no_caption": {"use_caption": False},
    "full_image_only": {"full_image_only": True},
    "max_score_only": {"fusion": "max"},
    "weighted_fusion": {"fusion": "weighted"},
}


@app.command("rerank-agent")
def rerank_agent(
    dense_run: Path = typer.Option(..., help="Runfile dense (B1/B3) con los candidatos"),
    split: str = typer.Option(..., help="train|valid|test"),
    backend: str = typer.Option("mock", help="mock | openai"),
    vlm_model: str = typer.Option("qwen2.5-vl"),
    base_url: str = typer.Option("http://localhost:8001/v1"),
    ablation: str = typer.Option("full", help="full|no_crops|no_caption|full_image_only|weighted_fusion"),
    top_n: int = typer.Option(200),
    final_top_k: int = typer.Option(100),
    high_conf: float = typer.Option(0.80),
    max_crops: int = typer.Option(5),
    processed: Path = typer.Option(PROCESSED_DEFAULT),
    raw: Path = typer.Option(RAW_DEFAULT),
    queries: Path = typer.Option(QUERIES_YAML),
    instruments: Path = typer.Option(INSTRUMENTS_YAML),
    run_name: str = typer.Option(None),
    seed: int = typer.Option(42),
):
    """B5 — Reranker agéntico determinista (grafo propio) sobre el top-N dense."""
    from .agent.graph import AgenticReranker
    from .data.prepare_dataset import resolve_mapping
    from .data.queries import load_instruments, load_queries
    from .reranking.base import load_candidates_from_run
    from .reranking.runner import run_pointwise_rerank
    from .utils.io import ImageProvider

    if ablation not in _ABLATIONS:
        raise typer.BadParameter(f"ablation desconocida: {ablation} ({list(_ABLATIONS)})")

    set_global_determinism(seed)
    mapping = resolve_mapping(processed, split)
    provider = ImageProvider(mapping, raw)
    qs = load_queries(queries)
    instr = load_instruments(instruments)
    cands = load_candidates_from_run(dense_run, top_n)

    vlm = _build_vlm_backend(backend, vlm_model, base_url)
    reranker = AgenticReranker(
        vlm, provider, high_confidence_threshold=high_conf, max_crops=max_crops,
        dense_retriever_name=Path(dense_run).stem, seed=seed, **_ABLATIONS[ablation],
    )
    name = run_name or f"B5_{Path(dense_run).stem}_{ablation}_{split}".replace("/", "-")
    run_path = run_pointwise_rerank(reranker, qs, instr, cands, name, final_top_k=final_top_k)
    typer.echo(f"B5 runfile: {run_path}")
    typer.echo(f"  traces: outputs/rerank_traces/{name}.jsonl  (ablation={ablation})")


@app.command("rerank-metrics")
def rerank_metrics_cmd(
    dense_run: Path = typer.Option(..., help="Runfile dense (candidatos)"),
    reranked_run: Path = typer.Option(..., help="Runfile reordenado (B4/B5)"),
    qrels: Path = typer.Option(...),
    n: int = typer.Option(200, help="top-N de candidatos"),
    k: int = typer.Option(100),
    out: Path = typer.Option(None),
):
    """Métricas de reranking (ADR §6.2): candidate/oracle recall, gain, delta nDCG/mAP."""
    import json

    from .data.qrels import load_qrels_trec
    from .evaluation.rerank_metrics import (
        candidate_recall_at_n, delta_metric, oracle_recall_at_k, rerank_gain_at_k,
    )
    from .utils.trec import load_run_trec

    qr = load_qrels_trec(qrels)
    dense = load_run_trec(dense_run)
    rer = load_run_trec(reranked_run)
    res = {
        f"candidate_recall@{n}": candidate_recall_at_n(dense, qr, n)["macro"],
        f"oracle_recall@{k}": oracle_recall_at_k(dense, qr, n, k)["macro"],
        f"rerank_gain@{k}": rerank_gain_at_k(rer, dense, qr, k)["macro"],
        f"delta_ndcg@{k}": delta_metric(rer, dense, qr, f"ndcg@{k}"),
        "delta_map": delta_metric(rer, dense, qr, "map"),
    }
    for kk, vv in res.items():
        typer.echo(f"  {kk}: {vv:+.4f}")
    out_path = out or (METRICS_DIR / f"{Path(reranked_run).stem}__rerankmetrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    typer.echo(f"-> {out_path}")


@app.command("compare")
def compare_cmd(
    a: Path = typer.Option(..., help="metrics JSON del sistema A"),
    b: Path = typer.Option(..., help="metrics JSON del sistema B"),
    metric: str = typer.Option("recall@100"),
    seed: int = typer.Option(42),
):
    """Compara dos sistemas en una métrica: delta, IC 95% bootstrap y test pareado (ADR §6.4)."""
    import json

    from .evaluation.statistical_tests import compare_systems

    da = json.loads(Path(a).read_text())["per_query"]
    db = json.loads(Path(b).read_text())["per_query"]
    pa = {q: v[metric] for q, v in da.items() if metric in v}
    pb = {q: v[metric] for q, v in db.items() if metric in v}
    res = compare_systems(pa, pb, seed=seed)
    typer.echo(f"{metric}: A={res['mean_a']:.4f}  B={res['mean_b']:.4f}  n={res['n']}")
    ci = res["delta_ci"]
    typer.echo(f"  delta(A-B)={ci['mean']:+.4f}  IC95%=[{ci['lo']:+.3f}, {ci['hi']:+.3f}]  p={res['p_value']:.4f}")


@app.command("error-analysis")
def error_analysis_cmd(
    run: Path = typer.Option(...),
    qrels: Path = typer.Option(...),
    mapping: Path = typer.Option(PROCESSED_DEFAULT / "image_id_mapping.parquet"),
    k: int = typer.Option(10),
    out: Path = typer.Option(Path("outputs/reports/error_analysis.json")),
):
    """Falsos positivos/negativos por query (ADR §13)."""
    import json

    from .evaluation.error_analysis import analyze_run

    res = analyze_run(run, qrels, mapping, k=k)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    tot_fp = sum(v["n_fp"] for v in res.values())
    tot_fn = sum(v["n_fn"] for v in res.values())
    typer.echo(f"error analysis: {len(res)} queries, FP={tot_fp}, FN={tot_fn} -> {out}")


@app.command("report")
def report(
    metrics_dir: Path = typer.Option(METRICS_DIR),
    out_dir: Path = typer.Option(Path("outputs/reports")),
    slides: bool = typer.Option(True, help="Generar también slides Marp"),
):
    """Genera informe final (MD/HTML + tablas MD/LaTeX + figuras) y slides Marp."""
    from .reporting.report_generator import generate_report
    from .reporting.slides_generator import generate_slides

    md = generate_report(metrics_dir, out_dir)
    typer.echo(f"informe: {md}")
    typer.echo(f"  tablas: {out_dir}/tables/  figuras: {out_dir}/figures/")
    if slides:
        deck = generate_slides(metrics_dir, Path("outputs/slides/slides.md"))
        typer.echo(f"slides Marp: {deck}")


@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(7860),
):
    """Lanza el buscador web (Gradio) sobre los runfiles generados."""
    from .serving.app import launch

    launch(server_name=host, server_port=port)


if __name__ == "__main__":
    app()
