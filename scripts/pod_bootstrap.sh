#!/usr/bin/env bash
# Arranque autónomo del Pod (RunPod). Clona, baja dataset del release, instala, arranca vLLM,
# corre el smoke o el gordo, sube resultados a GitHub, avisa por Telegram y apaga el Pod.
#
# Env requeridos (se pasan al crear el Pod):
#   GH_TOKEN              token GitHub (scope repo) — clone privado + push resultados + release download
#   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   alertas
#   RUNPOD_API_KEY        para auto-apagado (RUNPOD_POD_ID lo inyecta RunPod)
# Env opcionales:
#   PHASE=smoke|full     (default smoke)
#   VLM_MODEL_HF=Qwen/Qwen2.5-VL-7B-Instruct   (modelo a servir; 3B si poca VRAM)
set -uo pipefail
PHASE="${PHASE:-smoke}"
VLM_MODEL_HF="${VLM_MODEL_HF:-Qwen/Qwen2.5-VL-7B-Instruct}"
REPO="adrixo/instrumentos-portugueses"
WS=/workspace
mkdir -p "$WS"; cd "$WS"
exec > >(tee -a "$WS/bootstrap.log") 2>&1
echo "===== POD BOOTSTRAP ($(date)) PHASE=$PHASE ====="

export NTFY_TOPIC="${NTFY_TOPIC:-}"
note(){ echo ">> $1"; [ -d "$WS/instrumentos-portugueses" ] && bash "$WS/instrumentos-portugueses/scripts/notify.sh" "$1" 2>/dev/null || \
        { [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" -d chat_id="${TELEGRAM_CHAT_ID:-}" -d text="$1" >/dev/null 2>&1; }; }
fail(){ note "❌ Pod bootstrap FALLÓ: $1"; powerdown; exit 1; }
powerdown(){ [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ] && \
  curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}" >/dev/null 2>&1; }

note "🚀 Pod arrancado, preparando entorno ($PHASE)..."

# 1. sistema (sin gh: clone y release vía token+curl)
apt-get update -y >/dev/null 2>&1 || true
apt-get install -y git curl unzip python3-venv jq >/dev/null 2>&1 || true

# 2. clone (token embebido en la URL del origin -> push de resultados funciona sin más auth)
[ -d instrumentos-portugueses ] || git clone "https://x-access-token:${GH_TOKEN}@github.com/${REPO}.git" || fail "git clone"
cd instrumentos-portugueses
git config user.email "pod@runpod"; git config user.name "runpod-bot"

# Heartbeat de logs -> sube bootstrap.log + vllm.log a GitHub (rama pod-logs) cada 30s.
# Lectura en vivo desde fuera:  git fetch origin pod-logs && git show origin/pod-logs:pod_logs/live.log
( while true; do python3 scripts/pod_logpush.py >/dev/null 2>&1 || true; sleep 30; done ) &
echo "$!" > /tmp/heartbeat.pid
note "📡 heartbeat de logs activo (rama pod-logs)"

# 3. dataset desde el release (API GitHub + curl)
if [ ! -d data/raw/portuguese_instruments/train ]; then
  note "⬇️ bajando dataset del release..."
  mkdir -p /tmp/ds data/raw/portuguese_instruments
  asset_id="$(curl -fsSL -H "Authorization: token ${GH_TOKEN}" \
    "https://api.github.com/repos/${REPO}/releases/tags/dataset-v2" \
    | jq -r '.assets[] | select(.name|endswith(".zip")) | .id' | head -1)"
  [ -z "$asset_id" ] && fail "no encuentro asset del dataset"
  curl -fL -H "Authorization: token ${GH_TOKEN}" -H "Accept: application/octet-stream" \
    "https://api.github.com/repos/${REPO}/releases/assets/${asset_id}" -o /tmp/ds/data.zip || fail "descarga dataset"
  unzip -q /tmp/ds/data.zip -d /tmp/ds/extract || fail "unzip"
  inner="$(find /tmp/ds/extract -maxdepth 2 -type d -name train | head -1 | xargs dirname)"
  [ -z "$inner" ] && fail "estructura dataset inesperada"
  mv "$inner"/* data/raw/portuguese_instruments/
  rm -rf /tmp/ds
fi

# 4. vLLM (venv aparte)
if ! curl -s http://localhost:8001/v1/models 2>/dev/null | grep -q qwen2.5-vl; then
  note "🧠 instalando+arrancando vLLM ($VLM_MODEL_HF)..."
  python3 -m venv "$WS/vllm-env"
  "$WS/vllm-env/bin/pip" install -q -U pip vllm || fail "pip vllm"
  nohup "$WS/vllm-env/bin/vllm" serve "$VLM_MODEL_HF" --served-model-name qwen2.5-vl --port 8001 \
    --max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.92 \
    > "$WS/vllm.log" 2>&1 &
  for i in $(seq 1 90); do
    curl -s http://localhost:8001/v1/models 2>/dev/null | grep -q qwen2.5-vl && break
    sleep 20
  done
  curl -s http://localhost:8001/v1/models 2>/dev/null | grep -q qwen2.5-vl || fail "vLLM no arrancó"
fi

# 5. entorno del proyecto
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -U pip
  .venv/bin/pip install -q -e ".[dense,colpali,extras]" openai || fail "pip proyecto"
fi
source .venv/bin/activate
export VLM_BASE_URL=http://localhost:8001/v1 VLM_MODEL=qwen2.5-vl

# 6. ejecutar fase
note "▶️ ejecutando fase: $PHASE"
if [ "$PHASE" = "full" ]; then
  SHUTDOWN=1 bash scripts/gpu_full.sh || fail "gpu_full"
else
  bash scripts/gpu_smoke.sh || fail "gpu_smoke"
  note "✅ SMOKE terminado. Verifica en GitHub; lanza el gordo cuando confirmes."
  powerdown   # smoke siempre apaga el Pod (verificación barata); el gordo se lanza en un Pod nuevo
fi
echo "===== BOOTSTRAP FIN ====="
