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

# 1. sistema + gh
apt-get update -y >/dev/null 2>&1 || true
apt-get install -y git curl unzip python3-venv >/dev/null 2>&1 || true
if ! command -v gh >/dev/null 2>&1; then
  (type curl >/dev/null && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null; \
   echo "deb [signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list; \
   apt-get update -y >/dev/null 2>&1; apt-get install -y gh >/dev/null 2>&1) || pip install -q ghapi 2>/dev/null || true
fi
export GH_TOKEN
git config --global credential.helper store
printf "https://x-access-token:%s@github.com\n" "$GH_TOKEN" > ~/.git-credentials

# 2. clone
[ -d instrumentos-portugueses ] || git clone "https://x-access-token:${GH_TOKEN}@github.com/${REPO}.git" || fail "git clone"
cd instrumentos-portugueses
git config user.email "pod@runpod"; git config user.name "runpod-bot"

# 3. dataset desde el release
if [ ! -d data/raw/portuguese_instruments/train ]; then
  note "⬇️ bajando dataset del release..."
  mkdir -p /tmp/ds data/raw
  gh release download dataset-v2 --repo "$REPO" --dir /tmp/ds --clobber || fail "release download"
  unzip -q /tmp/ds/*.zip -d /tmp/ds/extract || fail "unzip"
  # la carpeta interna contiene train/valid/test
  inner="$(find /tmp/ds/extract -maxdepth 2 -type d -name train | head -1 | xargs dirname)"
  [ -z "$inner" ] && fail "estructura dataset inesperada"
  mkdir -p data/raw/portuguese_instruments
  mv "$inner"/* data/raw/portuguese_instruments/
  rm -rf /tmp/ds
fi

# 4. vLLM (venv aparte)
if ! curl -s http://localhost:8001/v1/models 2>/dev/null | grep -q qwen2.5-vl; then
  note "🧠 instalando+arrancando vLLM ($VLM_MODEL_HF)..."
  python3 -m venv "$WS/vllm-env"
  "$WS/vllm-env/bin/pip" install -q -U pip vllm || fail "pip vllm"
  nohup "$WS/vllm-env/bin/vllm" serve "$VLM_MODEL_HF" --served-model-name qwen2.5-vl --port 8001 \
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
