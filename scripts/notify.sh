#!/usr/bin/env bash
# Envía una alerta (best-effort, nunca falla el run). Configurable por env:
#   NTFY_TOPIC          -> push gratis sin cuenta: instala la app ntfy y suscríbete a ese topic.
#   TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID -> alerta por Telegram.
msg="${1:-done}"
sent=0
if [ -n "${NTFY_TOPIC:-}" ]; then
  curl -s -H "Title: Instrument Retrieval Lab" -d "$msg" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 && { echo "[notify] ntfy -> $msg"; sent=1; }
fi
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
  curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" -d text="$msg" >/dev/null 2>&1 && { echo "[notify] telegram -> $msg"; sent=1; }
fi
[ "$sent" = 0 ] && echo "[notify] (sin NTFY_TOPIC ni TELEGRAM_*; mensaje: $msg)"
true
