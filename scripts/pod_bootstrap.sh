#!/usr/bin/env bash
# Arranque autónomo del Pod (RunPod) con: SSH siempre, logs en vivo (rama pod-logs), descarga en
# PARALELO de dataset+venv+modelo, fix de arranque vLLM, y auto-apagado.
#
# Env (los pasa runpod.py al crear el Pod):
#   GH_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, RUNPOD_API_KEY, PHASE(smoke|full), VLM_MODEL_HF
#   PUBLIC_KEY (lo inyecta RunPod con tu clave SSH de la cuenta)
set -uo pipefail
PHASE="${PHASE:-smoke}"
VLM_MODEL_HF="${VLM_MODEL_HF:-Qwen/Qwen2.5-VL-7B-Instruct}"
REPO="adrixo/instrumentos-portugueses"
WS=/workspace
mkdir -p "$WS"; cd "$WS"
exec > >(tee -a "$WS/bootstrap.log") 2>&1
echo "===== POD BOOTSTRAP ($(date)) PHASE=$PHASE model=$VLM_MODEL_HF ====="

note(){ echo ">> $1"
  if [ -f "$WS/instrumentos-portugueses/scripts/notify.sh" ]; then bash "$WS/instrumentos-portugueses/scripts/notify.sh" "$1" 2>/dev/null
  elif [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" -d chat_id="${TELEGRAM_CHAT_ID:-}" -d text="$1" >/dev/null 2>&1; fi; true; }
powerdown(){ [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ] && \
  curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}" >/dev/null 2>&1; }
fail(){ note "❌ Pod FALLÓ: $1 (revisa rama pod-logs)"; powerdown; exit 1; }

note "🚀 Pod arrancado ($PHASE)"

# --- deps + SSH SIEMPRE ---
apt-get update -y >/dev/null 2>&1 || true
apt-get install -y git curl unzip jq python3-venv openssh-server >/dev/null 2>&1 || true
mkdir -p ~/.ssh && chmod 700 ~/.ssh
[ -n "${PUBLIC_KEY:-}" ] && { echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; }
mkdir -p /run/sshd; (/usr/sbin/sshd 2>/dev/null || service ssh start 2>/dev/null || true)
note "🔑 SSH activo -> runpodctl ssh connect ${RUNPOD_POD_ID:-?}"

# --- clone + heartbeat de logs ---
[ -d instrumentos-portugueses ] || git clone "https://x-access-token:${GH_TOKEN}@github.com/${REPO}.git" || fail "git clone"
cd instrumentos-portugueses
git config user.email "pod@runpod"; git config user.name "runpod-bot"
( while true; do python3 scripts/pod_logpush.py >/dev/null 2>&1 || true; sleep 30; done ) &
note "📡 heartbeat logs activo (rama pod-logs)"

# --- SETUP EN PARALELO: modelo(vLLM) + dataset + venv ---
# smoke: --enforce-eager (arranque rápido, pocas llamadas). gordo: sin él -> CUDA graphs = más throughput.
EAGER=""; [ "$PHASE" = "smoke" ] && EAGER="--enforce-eager"
note "⚙️ descargando en paralelo: vLLM($VLM_MODEL_HF $EAGER) + dataset + venv"

# vLLM: instala y arranca el serve (la descarga del modelo ocurre aquí, en paralelo).
( python3 -m venv "$WS/vllm-env" \
  && "$WS/vllm-env/bin/pip" install -q -U pip vllm \
  && exec "$WS/vllm-env/bin/vllm" serve "$VLM_MODEL_HF" --served-model-name qwen2.5-vl --port 8001 \
       --max-model-len 8192 $EAGER --gpu-memory-utilization 0.92 \
  ) > "$WS/vllm.log" 2>&1 &

# dataset
( if [ ! -d data/raw/portuguese_instruments/train ]; then
    mkdir -p /tmp/ds data/raw/portuguese_instruments
    aid="$(curl -fsSL -H "Authorization: token ${GH_TOKEN}" "https://api.github.com/repos/${REPO}/releases/tags/dataset-v2" | jq -r '.assets[]|select(.name|endswith(".zip"))|.id'|head -1)"
    [ -z "$aid" ] && exit 1
    curl -fL -H "Authorization: token ${GH_TOKEN}" -H "Accept: application/octet-stream" "https://api.github.com/repos/${REPO}/releases/assets/${aid}" -o /tmp/ds/d.zip || exit 1
    unzip -q /tmp/ds/d.zip -d /tmp/ds/x || exit 1
    inner="$(find /tmp/ds/x -maxdepth 2 -type d -name train|head -1|xargs dirname)"; [ -z "$inner" ] && exit 1
    mv "$inner"/* data/raw/portuguese_instruments/ && rm -rf /tmp/ds
  fi ) & DPID=$!

# venv del proyecto
( [ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -U pip && .venv/bin/pip install -q -e ".[dense,colpali,extras]" openai; } ) & PPID=$!

wait $DPID || fail "descarga dataset"
note "📦 dataset listo"
wait $PPID || fail "instalación venv proyecto"
note "🐍 venv listo"

# esperar a que vLLM responda (la descarga del modelo iba en paralelo)
echo "[vllm] esperando readiness..."
for i in $(seq 1 120); do curl -s http://localhost:8001/v1/models 2>/dev/null | grep -q qwen2.5-vl && break; sleep 15; done
curl -s http://localhost:8001/v1/models 2>/dev/null | grep -q qwen2.5-vl || fail "vLLM no respondió (mira pod-logs / vllm.log)"
note "✅ vLLM listo. Ejecutando $PHASE"

# --- ejecutar fase ---
source .venv/bin/activate
export VLM_BASE_URL=http://localhost:8001/v1 VLM_MODEL=qwen2.5-vl
if [ "$PHASE" = "full" ]; then
  SHUTDOWN=1 bash scripts/gpu_full.sh || fail "gpu_full"
else
  bash scripts/gpu_smoke.sh || fail "gpu_smoke"
  note "✅ SMOKE terminado — verifica en GitHub; lanza el gordo cuando confirmes."
  powerdown
fi
echo "===== BOOTSTRAP FIN ====="
