#!/usr/bin/env bash
# B4 zero-shot comparison against Qwen3.6-27B served by llama.cpp/OpenAI API.
# Run this from esalab-big; from that host MIDA is reachable at 100.127.120.42.
set -euo pipefail

VLM_BASE_URL="${VLM_BASE_URL:-http://100.127.120.42:8080/v1}"
VLM_MODEL="${VLM_MODEL:-qwen36-27b}"
VLM_DISABLE_THINKING="${VLM_DISABLE_THINKING:-true}"
TOPN="${TOPN:-50}"
FINAL_K="${FINAL_K:-50}"
DENSE_MODEL="${DENSE_MODEL:-openclip-vitl14}"
RUN_NAME="${RUN_NAME:-B4_qwen36_zero_shot_test}"

export VLM_DISABLE_THINKING

probe_image_support() {
  python3 - "$VLM_BASE_URL" "$VLM_MODEL" <<'PY'
import base64
import json
import sys
import urllib.error
import urllib.request

base_url, model = sys.argv[1], sys.argv[2]
jpg_1x1 = base64.b64encode(
    bytes.fromhex(
        "ffd8ffe000104a46494600010101006000600000ffdb0043000302020302020303030304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e0b0b1016101113141515150c0f171816141812141514ffdb00430103040405040509050509140d0b0d1414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414ffc00011080001000103012200021101031101ffc4001400010000000000000000000000000000000000000008ffc4001410010000000000000000000000000000000000000000ffc4001401010000000000000000000000000000000000000008ffc4001411010000000000000000000000000000000000000000ffda000c03010002110311003f00b2c001ffd9"
    )
).decode()
payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Return only JSON: {\"ok\":true}."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpg_1x1}"}},
            ],
        }
    ],
    "max_tokens": 32,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    base_url.rstrip("/") + "/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
)
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        print(resp.read().decode("utf-8", "replace")[:500])
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", "replace")
    raise SystemExit(
        "Qwen3.6 endpoint is not accepting image input yet. "
        "Start llama-server with ENABLE_MMPROJ=1 / --mmproj first.\n"
        f"HTTP {exc.code}: {body}"
    )
PY
}

echo "### preflight: Qwen3.6 image support at $VLM_BASE_URL"
curl -fsS "$VLM_BASE_URL/models" >/dev/null
probe_image_support

echo "### prepare-data + test qrels"
instrument-ir prepare-data
instrument-ir build-qrels --split test

echo "### dense candidates top-$TOPN (dense=$DENSE_MODEL, test)"
instrument-ir retrieve --split test --model "$DENSE_MODEL" --top-k "$TOPN" --run-name "DENSE_qwen36_test"
DENSE_RUN="outputs/runs/DENSE_qwen36_test.trec"

echo "### $RUN_NAME (zero-shot, top_n=$TOPN, final_k=$FINAL_K)"
instrument-ir rerank-vlm --dense-run "$DENSE_RUN" --split test --backend openai \
  --base-url "$VLM_BASE_URL" --vlm-model "$VLM_MODEL" --top-n "$TOPN" --final-top-k "$FINAL_K" \
  --run-name "$RUN_NAME"
instrument-ir evaluate --run "outputs/runs/${RUN_NAME}.trec" --qrels data/processed/qrels/test.qrels
instrument-ir rerank-metrics --dense-run "$DENSE_RUN" --reranked-run "outputs/runs/${RUN_NAME}.trec" \
  --qrels data/processed/qrels/test.qrels --n "$TOPN" --k "$FINAL_K" \
  --out "outputs/metrics/${RUN_NAME}__rerankmetrics.json"
instrument-ir report

echo "Qwen3.6 zero-shot comparison complete: outputs/runs/${RUN_NAME}.trec"
