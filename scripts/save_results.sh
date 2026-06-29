#!/usr/bin/env bash
# Guarda los resultados en git tras cada experimento: commit + push a main (best-effort).
# No aborta el run si git falla (p.ej. sin auth de push): solo avisa.
# Uso: bash scripts/save_results.sh "mensaje del paso"
msg="${1:-results}"

git add outputs configs/queries_mini.yaml 2>/dev/null || true
if git diff --cached --quiet 2>/dev/null; then
  echo "[save] nada nuevo que guardar ($msg)"
  exit 0
fi
git commit -q -m "results: $msg" 2>/dev/null || { echo "[save] commit falló ($msg)"; exit 0; }
if git push -q origin main 2>/dev/null; then
  echo "[save] commit+push OK -> $msg"
else
  echo "[save] commit OK, push falló (¿auth?) -> $msg  (se subirá al final con: git push origin main)"
fi
# aviso por Telegram de que ESE experimento acabó
bash "$(dirname "$0")/notify.sh" "✅ $msg — resultados guardados en GitHub" 2>/dev/null || true
