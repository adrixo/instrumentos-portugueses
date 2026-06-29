#!/usr/bin/env python3
"""Sube los logs del Pod a GitHub (rama 'pod-logs', fichero pod_logs/live.log) vía API.
Pensado para correr en bucle en el Pod -> lectura en vivo desde fuera con:
    git fetch origin pod-logs && git show origin/pod-logs:pod_logs/live.log
Requiere GH_TOKEN en el entorno. No toca el working tree (todo por API)."""
import base64, json, os, urllib.request
REPO = "adrixo/instrumentos-portugueses"
BRANCH = "pod-logs"
FILE = "pod_logs/live.log"
TOK = os.environ["GH_TOKEN"]


def api(method, path, data=None):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/{path}",
        data=(json.dumps(data).encode() if data else None),
        headers={"Authorization": f"token {TOK}", "Accept": "application/vnd.github+json",
                 "User-Agent": "pod-logpush"}, method=method)
    try:
        return json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.HTTPError as e:
        return {"_err": e.code, "_body": e.read().decode()[:200]}


def tail(p, n=14000):
    try:
        return open(p, encoding="utf-8", errors="replace").read()[-n:]
    except Exception as e:
        return f"(sin {p}: {e})"


def main():
    import datetime
    content = (f"hora: {datetime.datetime.utcnow().isoformat()}Z  PHASE={os.environ.get('PHASE','?')}  "
               f"pod={os.environ.get('RUNPOD_POD_ID','?')}\n\n"
               f"===== bootstrap.log =====\n{tail('/workspace/bootstrap.log')}\n\n"
               f"===== vllm.log =====\n{tail('/workspace/vllm.log')}\n")
    b64 = base64.b64encode(content.encode()).decode()

    # crear rama pod-logs si no existe (desde main)
    cur = api("GET", f"contents/{FILE}?ref={BRANCH}")
    if cur.get("_err") == 404:
        main_sha = api("GET", "git/ref/heads/main").get("object", {}).get("sha")
        if main_sha:
            api("POST", "git/refs", {"ref": f"refs/heads/{BRANCH}", "sha": main_sha})
        cur = api("GET", f"contents/{FILE}?ref={BRANCH}")

    body = {"message": "pod log heartbeat", "content": b64, "branch": BRANCH}
    if isinstance(cur, dict) and cur.get("sha"):
        body["sha"] = cur["sha"]
    r = api("PUT", f"contents/{FILE}", body)
    print("ok" if "content" in r else r)


if __name__ == "__main__":
    main()
