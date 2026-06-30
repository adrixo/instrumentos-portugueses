#!/usr/bin/env python3
"""Remote GPU experiment monitor with a Rich TUI.

Run locally:
  python scripts/monitor_remote.py --server esalab-big
  python scripts/monitor_remote.py --server esalab-big --phase full --once

Secrets are not read from repo files. Use SSH keys or set ESALAB_BIG_PASSWORD.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def configure_output_encoding() -> None:
    """Avoid Windows code-page crashes when remote logs contain Unicode."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

DEFAULT_SERVERS: dict[str, dict[str, Any]] = {
    "esalab-big": {
        "host": "esalab-big.taild1b22.ts.net",
        "tailscale_ip": "100.69.221.87",
        "user": "esalab",
        "project_dir": "/home/esalab/Escritorio/instrumentos_portugueses_ir",
    }
}

EXPECTED_ARTIFACTS = {
    "smoke": [
        "outputs/runs/B1_smoke.trec",
        "outputs/runs/B3_smoke.trec",
        "outputs/runs/B4_smoke.trec",
        "outputs/runs/B5_smoke.trec",
        "outputs/metrics/B1_smoke.json",
        "outputs/metrics/B3_smoke.json",
        "outputs/metrics/B4_smoke.json",
        "outputs/metrics/B5_smoke.json",
        "outputs/reports/final_report.md",
    ],
    "full": [
        "outputs/runs/B1_openclip-vitb32_valid.trec",
        "outputs/runs/B1_openclip-vitl14_valid.trec",
        "outputs/runs/B1_jinaclip_valid.trec",
        "outputs/runs/B1_openclip-vitb32_test.trec",
        "outputs/runs/B1_openclip-vitl14_test.trec",
        "outputs/runs/B1_jinaclip_test.trec",
        "outputs/runs/B3_colqwen_valid.trec",
        "outputs/runs/B3_colqwen_test.trec",
        "outputs/runs/DENSE_test.trec",
        "outputs/runs/B4_test.trec",
        "outputs/runs/B5_full_test.trec",
        "outputs/runs/B5_no_crops_test.trec",
        "outputs/runs/B5_no_caption_test.trec",
        "outputs/runs/B5_full_image_only_test.trec",
        "outputs/runs/B5_max_score_only_test.trec",
        "outputs/runs/B5_weighted_fusion_test.trec",
        "outputs/metrics/B1_openclip-vitb32_valid.json",
        "outputs/metrics/B1_openclip-vitl14_valid.json",
        "outputs/metrics/B1_jinaclip_valid.json",
        "outputs/metrics/B1_openclip-vitb32_test.json",
        "outputs/metrics/B1_openclip-vitl14_test.json",
        "outputs/metrics/B1_jinaclip_test.json",
        "outputs/metrics/B3_colqwen_valid.json",
        "outputs/metrics/B3_colqwen_test.json",
        "outputs/metrics/B4_test.json",
        "outputs/metrics/B5_full_test.json",
        "outputs/metrics/B5_no_crops_test.json",
        "outputs/metrics/B5_no_caption_test.json",
        "outputs/metrics/B5_full_image_only_test.json",
        "outputs/metrics/B5_max_score_only_test.json",
        "outputs/metrics/B5_weighted_fusion_test.json",
        "outputs/metrics/B4_test__rerankmetrics.json",
        "outputs/metrics/B5_full_test__rerankmetrics.json",
        "outputs/reports/error_B4_test.json",
        "outputs/reports/error_B5_test.json",
        "outputs/reports/final_report.md",
    ],
}

SMOKE_STEPS = [
    "prepare-data + subset mini",
    "B1 dense",
    "B3 late-interaction",
    "B4 VLM reranker",
    "B5 agente",
    "report",
]

FULL_STEPS = [
    "prepare-data + qrels",
    "B1 dense",
    "B3 ColQwen",
    "candidatos",
    "B4 VLM reranker",
    "B5 agente + ablaciones",
    "error-analysis + report",
]


REMOTE_COLLECTOR = r"""
import glob
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

root = Path(os.environ["PROJECT_DIR"])
tail_lines = int(os.environ.get("TAIL_LINES", "40"))


def run(args, timeout=8):
    try:
        p = subprocess.run(
            args,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {"ok": p.returncode == 0, "rc": p.returncode, "out": p.stdout}
    except Exception as exc:
        return {"ok": False, "rc": None, "out": f"{type(exc).__name__}: {exc}"}


def read_text(path, limit=None):
    try:
        data = Path(path).read_text(encoding="utf-8", errors="replace")
        return data[-limit:] if limit else data
    except Exception:
        return ""


def tail(path, n):
    text = read_text(path, limit=250_000)
    if not text:
        return []
    return text.replace("\r", "").splitlines()[-n:]


def file_age(path):
    try:
        return round(time.time() - Path(path).stat().st_mtime, 1)
    except Exception:
        return None


def pid_running(pid):
    try:
        return bool(pid) and Path(f"/proc/{int(pid)}").exists()
    except Exception:
        return False


def job_status(name):
    base = root / "outputs" / "remote" / name
    pid_text = read_text(str(base) + ".pid").strip()
    exit_text = read_text(str(base) + ".exit").strip()
    log_path = str(base) + ".log"
    log_text = read_text(log_path, limit=2_000_000)
    log_tail = log_text.replace("\r", "").splitlines()[-tail_lines:] if log_text else []
    markers = [line.strip() for line in log_text.splitlines() if line.strip().startswith("###")]
    build_steps = re.findall(r"\[(\d+)/(\d+)\]", log_text)
    progress = None
    if build_steps:
        done, total = max((int(a), int(b)) for a, b in build_steps if int(b) > 0)
        progress = {"done": done, "total": total}
    if exit_text == "0":
        state = "done"
    elif exit_text:
        state = "failed"
    elif pid_running(pid_text):
        state = "running"
    elif pid_text:
        state = "stopped"
    else:
        state = "missing"
    return {
        "name": name,
        "pid": pid_text or None,
        "running": pid_running(pid_text),
        "exit": exit_text or None,
        "state": state,
        "log_age_seconds": file_age(log_path),
        "log_size_bytes": Path(log_path).stat().st_size if Path(log_path).exists() else 0,
        "tail": log_tail,
        "markers": markers[-12:],
        "progress": progress,
    }


def parse_table(out, columns):
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= columns:
            rows.append(parts)
    return rows


def rel_paths(pattern):
    return sorted(str(Path(p).relative_to(root)).replace("\\", "/") for p in glob.glob(str(root / pattern)))


gpu_out = run([
    "nvidia-smi",
    "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits",
], timeout=5)["out"]
gpu_rows = parse_table(gpu_out, 6)

proc_out = run([
    "nvidia-smi",
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
], timeout=5)["out"]
gpu_processes = parse_table(proc_out, 3)

docker_out = run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"], timeout=5)["out"]
containers = []
for line in docker_out.splitlines():
    parts = line.split("\t")
    if len(parts) >= 3:
        containers.append({"name": parts[0], "image": parts[1], "status": parts[2]})

vlm_ready = run(["curl", "-fsS", "--max-time", "2", "http://localhost:8001/v1/models"], timeout=4)
vlm_container = next((c["name"] for c in containers if "vlm-server" in c["name"]), None)
vlm_tail = []
if vlm_container:
    vlm_tail = run(["docker", "logs", "--tail", str(min(tail_lines, 40)), vlm_container], timeout=6)["out"].splitlines()

df_out = run(["df", "-h", str(root)], timeout=5)["out"].splitlines()
disk = df_out[-1].split() if df_out else []

paths = []
for pattern in [
    "outputs/runs/*",
    "outputs/metrics/*",
    "outputs/rerank_traces/*",
    "outputs/reports/*",
    "outputs/reports/tables/*",
    "outputs/remote/*",
]:
    paths.extend(rel_paths(pattern))
paths = sorted(set(paths))

latest = []
for p in paths:
    full = root / p
    try:
        st = full.stat()
        latest.append({"path": p, "size": st.st_size, "mtime": st.st_mtime})
    except Exception:
        pass
latest = sorted(latest, key=lambda x: x["mtime"], reverse=True)[:20]

data_counts = {}
for split in ["train", "valid", "test"]:
    d = root / "data" / "raw" / "portuguese_instruments" / split
    data_counts[split] = len([p for p in d.rglob("*") if p.is_file()]) if d.exists() else 0


def line_count(path):
    try:
        with Path(path).open("rb") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


dense_target = line_count(root / "outputs" / "runs" / "DENSE_test.trec")
trace_counts = {}
for name in [
    "B4_test",
    "B5_full_test",
    "B5_no_crops_test",
    "B5_no_caption_test",
    "B5_full_image_only_test",
    "B5_max_score_only_test",
    "B5_weighted_fusion_test",
]:
    path = root / "outputs" / "rerank_traces" / f"{name}.jsonl"
    if path.exists():
        trace_counts[name] = {
            "lines": line_count(path),
            "target": dense_target,
            "size": path.stat().st_size,
        }

cache_files = len(glob.glob(str(root / "outputs" / "cache" / "vlm_openai" / "**" / "*.json"), recursive=True))

print(json.dumps({
    "ok": True,
    "host": socket.gethostname(),
    "time": time.time(),
    "root": str(root),
    "exists": root.exists(),
    "gpu": gpu_rows,
    "gpu_processes": gpu_processes,
    "containers": containers,
    "vlm_ready": vlm_ready["ok"],
    "vlm_response": vlm_ready["out"][:500],
    "vlm_tail": vlm_tail[-tail_lines:],
    "disk": disk,
    "data_counts": data_counts,
    "trace_counts": trace_counts,
    "cache_files": cache_files,
    "paths": paths,
    "latest": latest,
    "jobs": {
        "build_irlab": job_status("build_irlab"),
        "vlm_server": job_status("vlm_server"),
        "gpu_smoke": job_status("gpu_smoke"),
        "gpu_full": job_status("gpu_full"),
    },
}, ensure_ascii=True))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor remote GPU experiments over SSH.")
    parser.add_argument("--server", default="esalab-big", help="Server key from configs/servers.yaml")
    parser.add_argument("--host", help="Override SSH host")
    parser.add_argument("--user", help="Override SSH user")
    parser.add_argument("--project-dir", help="Override remote project directory")
    parser.add_argument("--identity-file", help="SSH private key path")
    parser.add_argument("--password-env", default="ESALAB_BIG_PASSWORD")
    parser.add_argument("--phase", choices=["auto", "smoke", "full"], default="auto")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--tail-lines", type=int, default=36)
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--plain", action="store_true", help="Disable Rich UI")
    return parser.parse_args()


def load_servers() -> dict[str, dict[str, Any]]:
    servers = dict(DEFAULT_SERVERS)
    cfg = ROOT / "configs" / "servers.yaml"
    if not cfg.exists():
        return servers
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        for name, value in (data.get("servers") or {}).items():
            servers[name] = {**servers.get(name, {}), **(value or {})}
    except Exception:
        pass
    return servers


def import_paramiko():
    try:
        import paramiko  # type: ignore

        return paramiko
    except ImportError:
        print(
            "Missing dependency: paramiko. Install with `pip install -e .[monitor]` "
            "or `pip install paramiko rich`.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def connect(args: argparse.Namespace, server: dict[str, Any]):
    paramiko = import_paramiko()
    host = args.host or server.get("host") or server.get("tailscale_ip")
    user = args.user or server.get("user")
    if not host or not user:
        raise SystemExit("Missing host/user. Use --host and --user.")

    password = os.environ.get(args.password_env)
    key_filename = args.identity_file
    if not password and not key_filename:
        password = getpass.getpass(f"SSH password for {user}@{host} ({args.password_env}): ")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        username=user,
        password=password or None,
        key_filename=key_filename,
        look_for_keys=True,
        allow_agent=True,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def collect(client, project_dir: str, tail_lines: int) -> dict[str, Any]:
    command = (
        f"PROJECT_DIR={shlex.quote(project_dir)} "
        f"TAIL_LINES={int(tail_lines)} "
        f"python3 - <<'PY'\n{REMOTE_COLLECTOR}\nPY"
    )
    _, stdout, stderr = client.exec_command(command, timeout=45)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        raise RuntimeError(f"remote collector failed rc={rc}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return json.loads(out)


def detect_phase(snapshot: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    jobs = snapshot.get("jobs", {})
    if jobs.get("gpu_full", {}).get("state") in {"running", "done", "failed", "stopped"}:
        return "full"
    if jobs.get("gpu_smoke", {}).get("state") in {"running", "done", "failed", "stopped"}:
        return "smoke"
    return "smoke"


def artifact_progress(snapshot: dict[str, Any], phase: str) -> tuple[int, int, list[str]]:
    expected = EXPECTED_ARTIFACTS[phase]
    paths = set(snapshot.get("paths", []))
    done = [p for p in expected if p in paths]
    missing = [p for p in expected if p not in paths]
    return len(done), len(expected), missing


def job_percent(job: dict[str, Any]) -> int:
    if job.get("state") == "done":
        return 100
    if job.get("state") == "failed":
        return 100
    progress = job.get("progress") or {}
    done, total = progress.get("done"), progress.get("total")
    if done and total:
        return max(1, min(99, int(done * 100 / total)))
    if job.get("state") == "running":
        return 10
    return 0


def phase_marker_progress(snapshot: dict[str, Any], phase: str) -> tuple[int, int]:
    job = snapshot.get("jobs", {}).get(f"gpu_{phase}", {})
    markers = job.get("markers") or []
    total = len(SMOKE_STEPS if phase == "smoke" else FULL_STEPS)
    return min(len(markers), total), total


def choose_log(snapshot: dict[str, Any], phase: str) -> tuple[str, list[str]]:
    jobs = snapshot.get("jobs", {})
    for name in [f"gpu_{phase}", "build_irlab", "vlm_server"]:
        job = jobs.get(name, {})
        if job.get("state") == "running" and job.get("tail"):
            return name, job.get("tail") or []
    for name in [f"gpu_{phase}", "build_irlab", "vlm_server"]:
        job = jobs.get(name, {})
        if job.get("tail"):
            return name, job.get("tail") or []
    if snapshot.get("vlm_tail"):
        return "docker vlm-server", snapshot.get("vlm_tail") or []
    return "logs", ["No logs yet."]


def fmt_age(seconds: Any) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def render_plain(snapshot: dict[str, Any], phase: str) -> None:
    done, total, missing = artifact_progress(snapshot, phase)
    marker_done, marker_total = phase_marker_progress(snapshot, phase)
    print("\033[2J\033[H", end="")
    print(f"{snapshot.get('host')} | {time.strftime('%Y-%m-%d %H:%M:%S')} | {snapshot.get('root')}")
    print(f"phase={phase} artifacts={done}/{total} steps={marker_done}/{marker_total} vlm={snapshot.get('vlm_ready')}")
    for gpu in snapshot.get("gpu", []):
        idx, name, used, total_mem, util, temp = gpu[:6]
        print(f"GPU {idx}: {name} mem={used}/{total_mem} MiB util={util}% temp={temp}C")
    print("Jobs:")
    for name, job in (snapshot.get("jobs") or {}).items():
        print(
            f"  {name:12} {job.get('state'):8} pid={job.get('pid') or '-'} "
            f"exit={job.get('exit') or '-'} log_age={fmt_age(job.get('log_age_seconds'))}"
        )
    traces = snapshot.get("trace_counts") or {}
    if traces:
        print("Trace progress:")
        for name, info in traces.items():
            lines = int(info.get("lines") or 0)
            target = int(info.get("target") or 0)
            suffix = f"/{target}" if target else ""
            print(f"  {name:24} {lines}{suffix} candidates")
        print(f"  vlm cache files          {snapshot.get('cache_files', 0)}")
    print("Latest artifacts:")
    for item in snapshot.get("latest", [])[:10]:
        print(f"  {item['path']} ({item['size']} bytes)")
    print("Missing expected:")
    for item in missing[:8]:
        print(f"  {item}")
    name, lines = choose_log(snapshot, phase)
    print(f"\n--- {name} tail ---")
    print("\n".join(lines[-30:]))


def rich_available() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


def make_rich_dashboard(snapshot: dict[str, Any], phase: str):
    from rich import box
    from rich.columns import Columns
    from rich.console import Group
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
    from rich.table import Table
    from rich.text import Text

    jobs = snapshot.get("jobs") or {}
    artifact_done, artifact_total, missing = artifact_progress(snapshot, phase)
    marker_done, marker_total = phase_marker_progress(snapshot, phase)

    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="right")
    disk = snapshot.get("disk") or []
    disk_text = " ".join(disk[-5:]) if disk else "disk: unknown"
    header.add_row(
        f"[bold]esalab-big[/bold] -> {snapshot.get('host')}  [dim]{snapshot.get('root')}[/dim]",
        time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    header.add_row(
        f"phase=[cyan]{phase}[/cyan]  vLLM={'[green]ready[/green]' if snapshot.get('vlm_ready') else '[yellow]starting/offline[/yellow]'}",
        disk_text,
    )

    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        expand=True,
    )
    progress.add_task("image build", total=100, completed=job_percent(jobs.get("build_irlab", {})))
    progress.add_task(f"{phase} steps", total=marker_total, completed=marker_done)
    progress.add_task(f"{phase} artifacts", total=artifact_total, completed=artifact_done)
    for name, info in (snapshot.get("trace_counts") or {}).items():
        target = int(info.get("target") or 0)
        lines = int(info.get("lines") or 0)
        if target > 0:
            progress.add_task(f"{name} trace", total=target, completed=min(lines, target))
    progress.add_task("vLLM readiness", total=100, completed=100 if snapshot.get("vlm_ready") else 30)

    gpu_table = Table(title="GPU", box=box.SIMPLE_HEAVY, expand=True)
    for col in ["idx", "name", "mem", "util", "temp"]:
        gpu_table.add_column(col)
    for row in snapshot.get("gpu", []):
        idx, name, used, total_mem, util, temp = row[:6]
        gpu_table.add_row(idx, name, f"{used}/{total_mem} MiB", f"{util}%", f"{temp} C")
    if not snapshot.get("gpu"):
        gpu_table.add_row("-", "nvidia-smi unavailable", "-", "-", "-")

    proc_table = Table(title="GPU Processes", box=box.SIMPLE_HEAVY, expand=True)
    for col in ["pid", "process", "mem"]:
        proc_table.add_column(col)
    for row in snapshot.get("gpu_processes", [])[:6]:
        proc_table.add_row(row[0], Path(row[1]).name, f"{row[2]} MiB")
    if not snapshot.get("gpu_processes"):
        proc_table.add_row("-", "none", "-")

    jobs_table = Table(title="Jobs", box=box.SIMPLE_HEAVY, expand=True)
    for col in ["job", "state", "pid", "exit", "log age"]:
        jobs_table.add_column(col)
    for name, job in jobs.items():
        state = job.get("state") or "-"
        style = "green" if state == "done" else "red" if state == "failed" else "cyan" if state == "running" else "dim"
        jobs_table.add_row(
            name,
            f"[{style}]{state}[/{style}]",
            job.get("pid") or "-",
            job.get("exit") or "-",
            fmt_age(job.get("log_age_seconds")),
        )

    artifact_table = Table(title="Latest Artifacts", box=box.SIMPLE_HEAVY, expand=True)
    artifact_table.add_column("path", overflow="fold")
    artifact_table.add_column("size", justify="right")
    for item in snapshot.get("latest", [])[:10]:
        artifact_table.add_row(item["path"], str(item["size"]))
    if not snapshot.get("latest"):
        artifact_table.add_row("none yet", "-")

    missing_text = "\n".join(missing[:10]) if missing else "All expected artifacts are present."
    log_name, log_lines = choose_log(snapshot, phase)
    log_text = Text("\n".join(log_lines[-34:]) or "No logs yet.")

    return Group(
        Panel(header, border_style="blue"),
        Panel(progress, title="Progress", border_style="cyan"),
        Columns([gpu_table, proc_table], equal=True, expand=True),
        Columns([jobs_table, artifact_table], equal=True, expand=True),
        Panel(missing_text, title="Missing Expected Artifacts", border_style="yellow"),
        Panel(log_text, title=f"{log_name} tail", border_style="magenta"),
    )


def main() -> None:
    configure_output_encoding()
    args = parse_args()
    server = load_servers().get(args.server)
    if not server:
        raise SystemExit(f"Unknown server {args.server!r}.")
    project_dir = args.project_dir or server.get("project_dir")
    if not project_dir:
        raise SystemExit("Missing project directory. Use --project-dir.")

    client = connect(args, server)
    use_rich = not args.plain and rich_available()
    if args.once:
        snapshot = collect(client, project_dir, args.tail_lines)
        phase = detect_phase(snapshot, args.phase)
        if use_rich:
            from rich.console import Console

            Console().print(make_rich_dashboard(snapshot, phase))
        else:
            render_plain(snapshot, phase)
        return

    if use_rich:
        from rich.live import Live

        with Live(refresh_per_second=4, screen=True) as live:
            while True:
                snapshot = collect(client, project_dir, args.tail_lines)
                phase = detect_phase(snapshot, args.phase)
                live.update(make_rich_dashboard(snapshot, phase))
                time.sleep(args.interval)
    else:
        print("Rich is not installed; using plain monitor. Install with `pip install -e .[monitor]`.")
        while True:
            snapshot = collect(client, project_dir, args.tail_lines)
            phase = detect_phase(snapshot, args.phase)
            render_plain(snapshot, phase)
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nmonitor stopped")
