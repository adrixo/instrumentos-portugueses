"""Test nocturno RESUMIBLE: ejecuta B1, B4, B5(+ablaciones) sobre el subset mini en MPS.

B3 (ColQwen) se omite: el LoRA de colqwen2-v1.0 no es compatible con transformers 5.12
(language_model.* -> el adaptador no se aplica). Requiere entorno fijado en el box GPU.

Robusto y resumible: cada paso declara su salida; si ya existe, se salta. Así el watchdog puede
relanzar este script tras un cuelgue sin recomputar lo ya hecho. Timeout por paso. Marca DONE al final.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NIGHT = ROOT / "outputs" / "night"
NIGHT.mkdir(parents=True, exist_ok=True)

_VENV_BIN = ROOT / ".venv" / "bin"
ENV = {
    **os.environ,
    "INSTRUMENT_IR_NO_FAISS": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")),
    # Asegura que `instrument-ir` se resuelve aunque el venv no esté "activado".
    "PATH": f"{_VENV_BIN}:{os.environ.get('PATH', '')}",
}
ENV.pop("MLFLOW_TRACKING_URI", None)

INSTR = "adufe,concertina,cavaquinho,gaita-de-foles,guitarra-portuguesa,violao"
N, TOPN = "120", "30"
VLM = "Qwen/Qwen2.5-VL-3B-Instruct"
DENSE = "outputs/runs/B1_mini.trec"
QM = "configs/queries_mini.yaml"
QR = "data/processed/qrels/mini.qrels"

# (nombre, comando, timeout_s, salida_para_resume|None)
STEPS = [
    ("prepare_mini", ["instrument-ir", "prepare-mini", "--instruments-sel", INSTR, "--n-images", N], 600,
     "data/processed/mini/corpus.parquet"),
    ("b1_retrieve", ["instrument-ir", "retrieve", "--split", "mini", "--model", "openclip-vitb32",
                     "--top-k", TOPN, "--queries", QM, "--run-name", "B1_mini"], 1200,
     "outputs/runs/B1_mini.trec"),
    ("b1_eval", ["instrument-ir", "evaluate", "--run", DENSE, "--qrels", QR, "--queries", QM], 600,
     "outputs/metrics/B1_mini.json"),
    ("b4_rerank", ["instrument-ir", "rerank-vlm", "--dense-run", DENSE, "--split", "mini",
                   "--backend", "hf", "--vlm-model", VLM, "--top-n", TOPN, "--final-top-k", TOPN,
                   "--queries", QM, "--run-name", "B4_mini"], 7200, "outputs/runs/B4_mini.trec"),
    ("b4_eval", ["instrument-ir", "evaluate", "--run", "outputs/runs/B4_mini.trec", "--qrels", QR,
                 "--queries", QM], 600, "outputs/metrics/B4_mini.json"),
    ("b4_rerankmetrics", ["instrument-ir", "rerank-metrics", "--dense-run", DENSE,
                          "--reranked-run", "outputs/runs/B4_mini.trec", "--qrels", QR,
                          "--n", TOPN, "--k", TOPN], 600, "outputs/metrics/B4_mini__rerankmetrics.json"),
    ("b5_full", ["instrument-ir", "rerank-agent", "--dense-run", DENSE, "--split", "mini",
                 "--backend", "hf", "--vlm-model", VLM, "--top-n", TOPN, "--final-top-k", TOPN,
                 "--ablation", "full", "--queries", QM, "--run-name", "B5_full_mini"], 14400,
     "outputs/runs/B5_full_mini.trec"),
    ("b5_full_eval", ["instrument-ir", "evaluate", "--run", "outputs/runs/B5_full_mini.trec",
                      "--qrels", QR, "--queries", QM], 600, "outputs/metrics/B5_full_mini.json"),
    ("b5_nocrops", ["instrument-ir", "rerank-agent", "--dense-run", DENSE, "--split", "mini",
                    "--backend", "hf", "--vlm-model", VLM, "--top-n", TOPN, "--final-top-k", TOPN,
                    "--ablation", "no_crops", "--queries", QM, "--run-name", "B5_nocrops_mini"], 7200,
     "outputs/runs/B5_nocrops_mini.trec"),
    ("b5_nocrops_eval", ["instrument-ir", "evaluate", "--run", "outputs/runs/B5_nocrops_mini.trec",
                         "--qrels", QR, "--queries", QM], 600, "outputs/metrics/B5_nocrops_mini.json"),
    ("b5_fullimg", ["instrument-ir", "rerank-agent", "--dense-run", DENSE, "--split", "mini",
                    "--backend", "hf", "--vlm-model", VLM, "--top-n", TOPN, "--final-top-k", TOPN,
                    "--ablation", "full_image_only", "--queries", QM, "--run-name", "B5_fullimg_mini"], 7200,
     "outputs/runs/B5_fullimg_mini.trec"),
    ("b5_fullimg_eval", ["instrument-ir", "evaluate", "--run", "outputs/runs/B5_fullimg_mini.trec",
                         "--qrels", QR, "--queries", QM], 600, "outputs/metrics/B5_fullimg_mini.json"),
    ("b5_rerankmetrics", ["instrument-ir", "rerank-metrics", "--dense-run", DENSE,
                          "--reranked-run", "outputs/runs/B5_full_mini.trec", "--qrels", QR,
                          "--n", TOPN, "--k", TOPN, "--out",
                          "outputs/metrics/B5_full_mini__rerankmetrics.json"], 600,
     "outputs/metrics/B5_full_mini__rerankmetrics.json"),
    ("error_b4", ["instrument-ir", "error-analysis", "--run", "outputs/runs/B4_mini.trec",
                  "--qrels", QR, "--k", "10", "--out", "outputs/reports/error_B4_mini.json"], 600,
     "outputs/reports/error_B4_mini.json"),
    ("error_b5", ["instrument-ir", "error-analysis", "--run", "outputs/runs/B5_full_mini.trec",
                  "--qrels", QR, "--k", "10", "--out", "outputs/reports/error_B5_mini.json"], 600,
     "outputs/reports/error_B5_mini.json"),
    ("report", ["instrument-ir", "report"], 600, None),
]


def main():
    summary = []
    t_start = time.time()
    for name, cmd, timeout, out in STEPS:
        if out and (ROOT / out).exists():
            summary.append({"step": name, "status": "skip(exists)", "seconds": 0})
            (NIGHT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[skip] {name}", flush=True)
            continue
        t0 = time.time()
        status = "ok"
        try:
            with (NIGHT / f"{name}.log").open("w") as f:
                subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout, check=True)
        except subprocess.TimeoutExpired:
            status = "timeout"
        except subprocess.CalledProcessError as e:
            status = f"error(rc={e.returncode})"
        except Exception as e:  # noqa: BLE001
            status = f"exception({type(e).__name__})"
        dur = round(time.time() - t0, 1)
        summary.append({"step": name, "status": status, "seconds": dur})
        (NIGHT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[{status}] {name} ({dur}s)", flush=True)

    (NIGHT / "DONE").write_text(f"total_seconds={round(time.time()-t_start,1)}\n", encoding="utf-8")
    print("NIGHT RUN DONE", flush=True)


if __name__ == "__main__":
    main()
