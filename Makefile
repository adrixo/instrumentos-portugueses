# Instrument Retrieval Lab — targets (ADR §10).
# Fase 1 implementada: prepare/smoke/eval/test. b1..b5/report/serve/slides llegan en sus fases.

COMPOSE = docker compose -f docker/docker-compose.gpu.yml
SPLIT ?= valid

.PHONY: help build check-gpu prepare queries qrels retrieve eval smoke test b1 b3 b4 b5 report serve slides repro

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

build: ## Construir imagen Docker GPU
	$(COMPOSE) build

check-gpu: ## Comprobar CUDA dentro del contenedor
	bash scripts/check_gpu.sh

prepare: ## Parsear COCO + anonimizar + corpus/mapping + queries.yaml
	instrument-ir prepare-data

queries: ## (Re)generar configs/queries.yaml
	instrument-ir gen-queries --force

qrels: ## Construir qrels del split (SPLIT=valid)
	instrument-ir build-qrels --split $(SPLIT)

retrieve: ## Retrieve dummy del split (SPLIT=valid)
	instrument-ir retrieve --split $(SPLIT) --model dummy

eval: ## Evaluar el run dummy del split (SPLIT=valid)
	instrument-ir evaluate --run outputs/runs/B1_dummy_$(SPLIT).trec --qrels data/processed/qrels/$(SPLIT).qrels

smoke: ## Smoke test end-to-end (<10 min) — ADR §21
	instrument-ir smoke

test: ## Tests (incl. anti-fuga) — ADR §15.2
	pytest -q

b1: ## [Fase 2] Dense global
	@echo "Fase 2 — pendiente"
b3: ## [Fase 3] Late-interaction
	@echo "Fase 3 — pendiente"
b4: ## [Fase 4] VLM reranker
	@echo "Fase 4 — pendiente"
b5: ## [Fase 5] Agente reranker
	@echo "Fase 5 — pendiente"
report: ## [Fase 6] Informe
	@echo "Fase 6 — pendiente"
serve: ## [Fase 6] Buscador
	@echo "Fase 6 — pendiente"
slides: ## [Fase 6] Slides
	@echo "Fase 6 — pendiente"

repro: prepare qrels retrieve eval ## Reproducir el flujo Fase 1 (SPLIT=valid)
