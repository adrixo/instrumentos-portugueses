"""Watchdog del test nocturno: mantiene night_run.py vivo y desatascado.

- Si terminó (outputs/night/DONE): imprime DONE y sale.
- Si no corre: lo relanza (es resumible -> barato).
- Si corre pero no progresa (≈0s de CPU desde la última comprobación): lo mata y relanza.
- Garantiza que `caffeinate` sigue activo para que el Mac no se duerma.

Pensado para ejecutarse cada ~10 min desde un cron.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NIGHT = ROOT / "outputs" / "night"
NIGHT.mkdir(parents=True, exist_ok=True)
CPUFILE = NIGHT / "_cputime"


def _parse_time(t: str) -> float:
    t = t.strip()
    if not t:
        return 0.0
    parts = t.replace("-", ":").split(":")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def _orchestrator_pid() -> int | None:
    out = subprocess.run(["pgrep", "-f", "night_run.py"], capture_output=True, text=True).stdout
    for p in out.split():
        comm = subprocess.run(["ps", "-o", "comm=", "-p", p], capture_output=True, text=True).stdout
        if "python" in comm.lower():
            return int(p)
    return None


def _pids() -> list[int]:
    pid = _orchestrator_pid()
    return [pid] if pid else []


def _group_cpu(pgid: int) -> float:
    """Suma la CPU de TODO el grupo (orquestador + hijo instrument-ir que hace el trabajo real)."""
    out = subprocess.run(["ps", "-axo", "pgid=,time="], capture_output=True, text=True).stdout
    total = 0.0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        gid, _, tstr = line.partition(" ")
        try:
            if int(gid) == pgid:
                total += _parse_time(tstr)
        except ValueError:
            continue
    return total


def _relaunch() -> None:
    """Lanza el orquestador con el python del venv (así el python es líder de grupo -> pgid fiable)."""
    log = (NIGHT / "orchestrator.log").open("a")
    py = str(ROOT / ".venv" / "bin" / "python")
    subprocess.Popen(
        [py, "-u", "scripts/night_run.py"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    CPUFILE.write_text("0")


def _ensure_caffeinate(pids: list[int]) -> None:
    running = subprocess.run(["pgrep", "-f", "caffeinate -dimsu"], capture_output=True, text=True).stdout.strip()
    if not running and pids:
        subprocess.Popen(["caffeinate", "-dimsu", "-w", str(pids[0])], start_new_session=True)


def main() -> None:
    if (NIGHT / "DONE").exists():
        print("DONE")
        return

    orch = _orchestrator_pid()
    if orch is None:
        _relaunch()
        time.sleep(3)
        print("RELAUNCHED (estaba caído)")
        _ensure_caffeinate(_pids())
        return

    cur = _group_cpu(orch)  # CPU de orquestador + hijo trabajador
    last = float(CPUFILE.read_text()) if CPUFILE.exists() else None
    # 'stuck' solo si el grupo entero no avanza ni 2s de CPU desde la última comprobación (~10 min).
    if last is not None and cur >= last and (cur - last) < 2.0:
        subprocess.run(["pkill", "-9", "-g", str(orch)])
        time.sleep(2)
        _relaunch()
        print(f"STUCK_RESTARTED (cpu grupo {cur:.1f}s sin avanzar)")
    else:
        CPUFILE.write_text(str(cur))
        print(f"RUNNING ok (cpu grupo {cur:.1f}s)")
    _ensure_caffeinate(_pids())


if __name__ == "__main__":
    main()
