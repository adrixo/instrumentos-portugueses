#!/usr/bin/env bash
# Termina (borra) el Pod actual de RunPod para dejar de pagar. Los resultados ya están en GitHub.
# Requiere RUNPOD_API_KEY (y RUNPOD_POD_ID, que RunPod inyecta como env en el Pod).
# Espera GRACE segundos antes de apagar para poder cancelar con Ctrl-C.
set -u
GRACE="${SHUTDOWN_GRACE:-60}"

if [ -z "${RUNPOD_API_KEY:-}" ]; then
  echo "[shutdown] sin RUNPOD_API_KEY -> NO apago. Apaga el Pod a mano."
  exit 0
fi
PID="${RUNPOD_POD_ID:-}"
if [ -z "$PID" ]; then
  echo "[shutdown] sin RUNPOD_POD_ID -> NO apago (¿estás dentro de un Pod RunPod?)."
  exit 0
fi

echo "[shutdown] el Pod $PID se TERMINARÁ en ${GRACE}s. Ctrl-C para cancelar..."
sleep "$GRACE"
curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${PID}\\\"}) }\"}" >/dev/null 2>&1 \
  && echo "[shutdown] solicitud de terminación enviada para $PID" \
  || echo "[shutdown] fallo al terminar; hazlo a mano en la consola de RunPod"
