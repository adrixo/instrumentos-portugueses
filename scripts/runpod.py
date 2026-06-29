#!/usr/bin/env python3
"""Gestor de Pods RunPod para el experimento (ejecútalo en tu Mac; usa .env + `gh auth token`).

Uso:
  python3 scripts/runpod.py create smoke     # crea Pod H100 que corre el smoke y se apaga solo
  python3 scripts/runpod.py create full       # crea Pod H100 que corre el GORDO y se apaga solo
  python3 scripts/runpod.py list              # lista tus pods y su estado
  python3 scripts/runpod.py terminate <id>    # termina (borra) un pod

Secretos: lee .env (RUNPOD_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) y `gh auth token` (GH_TOKEN).
El Pod arranca scripts/pod_bootstrap.sh: clona, baja dataset del release, vLLM, smoke/gordo,
sube resultados a GitHub, avisa por Telegram y se auto-apaga.
"""
import json, os, subprocess, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "adrixo/instrumentos-portugueses"
GPU = "NVIDIA H100 80GB HBM3"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def load_env():
    env = {}
    p = os.path.join(ROOT, ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def gql(api_key, query, variables=None):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(f"https://api.runpod.io/graphql?api_key={api_key}", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        return {"httpError": e.code, "body": e.read().decode()}


def create(phase):
    env = load_env()
    rp = env["RUNPOD_API_KEY"]
    gh = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    assert gh, "no `gh auth token` (haz `gh auth login`)"
    docker_args = ('bash -c "apt-get update -y >/dev/null 2>&1; apt-get install -y git curl >/dev/null 2>&1; '
                   'rm -rf /workspace/instrumentos-portugueses; '
                   'git clone https://x-access-token:$GH_TOKEN@github.com/%s.git '
                   '/workspace/instrumentos-portugueses && '
                   'bash /workspace/instrumentos-portugueses/scripts/pod_bootstrap.sh"' % REPO)
    inp = {
        "gpuCount": 1, "gpuTypeId": GPU, "name": f"instr-ir-{phase}", "imageName": IMAGE,
        "containerDiskInGb": 80, "volumeInGb": 0, "dockerArgs": docker_args,
        "env": [
            {"key": "GH_TOKEN", "value": gh},
            {"key": "TELEGRAM_BOT_TOKEN", "value": env.get("TELEGRAM_BOT_TOKEN", "")},
            {"key": "TELEGRAM_CHAT_ID", "value": env.get("TELEGRAM_CHAT_ID", "")},
            {"key": "RUNPOD_API_KEY", "value": rp},
            {"key": "PHASE", "value": phase},
        ],
    }
    q = "mutation($input: PodFindAndDeployOnDemandInput!){ podFindAndDeployOnDemand(input:$input){ id machineId costPerHr } }"
    for cloud in ("COMMUNITY", "SECURE"):
        inp["cloudType"] = cloud
        r = gql(rp, q, {"input": inp})
        pod = (r.get("data") or {}).get("podFindAndDeployOnDemand")
        if pod:
            print(f"✅ Pod creado ({cloud}): id={pod['id']}  ${pod.get('costPerHr')}/h  fase={phase}")
            print(f"   monitor:   python3 scripts/runpod.py list")
            print(f"   terminar:  python3 scripts/runpod.py terminate {pod['id']}")
            print(f"   resultados: aparecerán en GitHub (commits) y aviso por Telegram.")
            return
        print(f"   {cloud}: sin disponibilidad/erro -> {json.dumps(r)[:200]}")
    print("❌ No se pudo crear el Pod (sin H100 disponible ahora). Reintenta o prueba otra GPU/region.")


def list_pods():
    env = load_env()
    r = gql(env["RUNPOD_API_KEY"], "query{ myself{ clientBalance pods{ id name desiredStatus costPerHr runtime{ uptimeInSeconds } } } }")
    me = (r.get("data") or {}).get("myself") or {}
    print(f"saldo: ${me.get('clientBalance')}")
    for p in me.get("pods", []):
        up = (p.get("runtime") or {}).get("uptimeInSeconds")
        print(f"  {p['id']}  {p['name']}  {p['desiredStatus']}  ${p.get('costPerHr')}/h  up={up}s")
    if not me.get("pods"):
        print("  (sin pods activos)")


def terminate(pid):
    env = load_env()
    r = gql(env["RUNPOD_API_KEY"], "mutation($id:String!){ podTerminate(input:{podId:$id}) }", {"id": pid})
    print("terminado" if "errors" not in r else r)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "create":
        create(sys.argv[2] if len(sys.argv) > 2 else "smoke")
    elif cmd == "terminate":
        terminate(sys.argv[2])
    else:
        list_pods()
