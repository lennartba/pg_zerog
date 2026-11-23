"""Running external tools (COLMAP, OpenMVS) with logging."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("pgz")

EXE = ".exe" if sys.platform == "win32" else ""


def find_tool(name: str, search_dir: str | None) -> str:
    """Resolve a tool from an explicit directory, else from PATH."""
    if search_dir:
        cand = Path(search_dir) / (name + EXE)
        if cand.exists():
            return str(cand)
        cand = Path(search_dir) / name
        if cand.exists():
            return str(cand)
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(
            f"'{name}' not found (searched {search_dir or '-'} and PATH). "
            "Set [tools] in the config."
        )
    return found


def run(cmd: list[str], log_file: Path, cwd: Path | None = None) -> None:
    """Run a command, tee output to log_file, raise on non-zero exit."""
    cmd = [str(c) for c in cmd]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log.info("$ %s", " ".join(cmd))
    t0 = time.time()
    with open(log_file, "a", encoding="utf-8", errors="replace") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(
            cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=os.environ.copy()
        )
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"{Path(cmd[0]).name} failed with exit code {proc.returncode} "
            f"after {dt:.0f}s; see {log_file}"
        )
    log.info("  done in %.0fs", dt)
