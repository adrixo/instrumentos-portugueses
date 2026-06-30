# Despliegue en GPU (RunPod / Lambda / Vast / AWS) — runbook

Experimento pequeño en datos pero pesado en VLM. Basta **1 GPU potente** (recomendado **H100 80GB**;
sirve también A100 40GB / L40S 48GB / g5.xlarge). Estrategia: **(1) smoke** para verificar que todo
funciona (incluido B3, que falló en Mac) → **(2) run completo**.

## 0. Crear el Pod (RunPod)
- Pod **GPU H100 80GB** (o A100 80GB / L40S 48GB), plantilla **RunPod PyTorch 2.x**.
- Disco: **Container ~30 GB + Volume ~80 GB** (modelos: Qwen2.5-VL-7B ~16 GB + ColQwen ~5 GB + dataset).
- Abre el **Web Terminal** (o SSH). Trabaja en `/workspace` (persiste en el volume).

## 1. Servidor VLM (vLLM) — venv aparte, SIN docker (en RunPod no hay docker-in-docker)
```bash
cd /workspace
python -m venv vllm-env && source vllm-env/bin/activate
pip install -U pip vllm
nohup vllm serve Qwen/Qwen2.5-VL-7B-Instruct --served-model-name qwen2.5-vl --port 8001 \
  > /workspace/vllm.log 2>&1 &
deactivate
# espera a "Application startup complete":
tail -f /workspace/vllm.log     # Ctrl-C cuando aparezca; queda en http://localhost:8001/v1
```
El venv aparte evita el choque de `transformers` entre vLLM y ColQwen.

## 2. Entorno del proyecto (instrument-ir) — venv aparte del de vLLM
```bash
cd /workspace
git clone <REPO_URL> instrumentos-portugueses && cd instrumentos-portugueses
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dense,colpali,extras]" openai
```
**Importante (B3)**: el fallo en Mac fue por `transformers 5.x` con el LoRA de `colqwen2-v1.0`. Deja
que `colpali-engine` fije su `transformers` compatible (no lo fuerces a >5). El **smoke** lo verifica;
si B3 falla, prueba: `pip install 'transformers>=4.49,<4.52'` y repite el smoke.

## 2b. Auth de git en el Pod (para que los resultados se suban solos)
Los scripts hacen `commit + push` de `outputs/` tras cada experimento → así los resultados aparecen en
GitHub en tiempo real (verificación remota, sin scp). Configura el push una vez:
```bash
git config user.email "tu@email"; git config user.name "tu-nombre"
gh auth login          # o: git remote set-url origin https://<TOKEN>@github.com/adrixo/instrumentos-portugueses
```
Si no configuras auth, el run NO falla: guarda en local y al final haces `git push origin main`.

## 3. Dataset
Colócalo en `data/raw/portuguese_instruments/{train,valid,test}`. Opciones:
- **scp/rsync** desde tu Mac (1.4 GB), o
- **descarga directa en el Pod** desde Mendeley (DOI `10.17632/pk7txkgt4v.2`) y descomprime en esa ruta.

## 4. FASE 1 — Smoke (verifica TODO, ~minutos)
```bash
export VLM_BASE_URL=http://localhost:8001/v1 VLM_MODEL=qwen2.5-vl
bash scripts/gpu_smoke.sh
```
Revisa que **B3 da métricas razonables** (no aleatorias) y que B4/B5 producen runfile+traces. Mira
`outputs/reports/final_report.md`. Si todo bien → fase 2.

## 5. FASE 2 — Run completo (cifras del paper, ~horas)
```bash
export VLM_BASE_URL=http://localhost:8001/v1 VLM_MODEL=qwen2.5-vl
bash scripts/gpu_full.sh                 # B1×3 + B3 (valid+test) + B4 + B5(+ablaciones) en test
# opcional, más rápido/barato:  TOPN=100 bash scripts/gpu_full.sh
```
Resultados: `outputs/reports/final_report.md` + `outputs/reports/tables/` (macro, per-class, gain con
p-value) + traces en `outputs/rerank_traces/`. Bájalos con scp.

## Alertas y auto-apagado (opcional, recomendado)
Para que te avise al terminar y apague el Pod solo (no malgastar saldo):
```bash
# Alerta push gratis sin cuenta: instala la app "ntfy" (iOS/Android) y suscríbete a TU topic único.
export NTFY_TOPIC=instr-ir-adrixo-7f3k        # invéntate uno difícil de adivinar
# (alternativa Telegram: export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=8413253)

# Auto-terminar el Pod al acabar el GORDO (los resultados ya están en GitHub):
export RUNPOD_API_KEY=<tu_key>                # RUNPOD_POD_ID ya viene puesto en el Pod
SHUTDOWN=1 bash scripts/gpu_full.sh           # apaga 60s después de terminar (Ctrl-C para cancelar)
```
Recibirás alerta también si algo **falla**. El smoke solo avisa (no apaga).

## Notas
- Coste estimado en H100: smoke ~$1, run completo ~$5–12 (2–3 h).
- Si la GPU tiene <24 GB, usa `Qwen/Qwen2.5-VL-3B-Instruct` en el paso 1 y `VLM_MODEL` acorde.
- En Linux NO hace falta `INSTRUMENT_IR_NO_FAISS` (faiss va bien).
- Apaga la instancia al terminar para no seguir pagando.

## Servidores disponibles

La lista versionada esta en `configs/servers.yaml`. No guarda secretos.

- `esalab-big`: servidor Tailscale `esalab-big.taild1b22.ts.net` (`100.69.221.87`),
  usuario `esalab`, proyecto en `/home/esalab/Escritorio/instrumentos_portugueses_ir`.
  Tiene una RTX 3090 Ti de 24 GB; para evitar presion de VRAM con vLLM + retrievers se recomienda:

```bash
export VLM_MODEL_HF=Qwen/Qwen2.5-VL-3B-Instruct
export VLLM_IMAGE=vllm/vllm-openai:v0.10.1.1
export VLLM_MAX_MODEL_LEN=4096
export VLLM_GPU_MEMORY_UTILIZATION=0.60
export VLM_MAX_IMAGE_SIDE=768
export VLM_JPEG_QUALITY=85
export VLM_WORKERS=8
export VLM_CACHE=1
export VLM_CACHE_DIR=outputs/cache/vlm_openai
docker compose -f docker/docker-compose.gpu.yml up -d vlm-server
docker compose -f docker/docker-compose.gpu.yml run --rm irlab bash scripts/gpu_smoke.sh
```

El resize de imagen es importante para Qwen-VL via vLLM: sin `VLM_MAX_IMAGE_SIDE`, algunas imagenes
superan el contexto visual de `max_model_len=4096`. La cache evita repetir llamadas VLM entre B4 y las
ablaciones B5.

Monitor local desde esta maquina:

```powershell
pip install -e ".[monitor]"
$env:ESALAB_BIG_PASSWORD="..."
python scripts/monitor_remote.py --server esalab-big
```

El snapshot del run completo ejecutado en `esalab-big` esta en
`results/esalab-big/2026-06-30_gpu_full/`.
