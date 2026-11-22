"""Shared helpers for the RunPod tooling tests: no network, no API key, no pod."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

RUNPOD = Path(__file__).resolve().parent.parent
REPO = RUNPOD.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run_script(
    script: str | Path,
    *args: str,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    """Run a bash script of runpod/ and capture its output."""
    path = Path(script) if Path(script).is_absolute() else RUNPOD / script
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(
        ["bash", str(path), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=full_env,
        input=stdin,
        check=False,
    )


@pytest.fixture
def stub_bin(tmp_path: Path) -> Path:
    """A PATH directory whose network and secret tools record a call and fail.

    A dry run must not read a secret, call the network or touch ssh: any call leaves a line in
    `calls.log` next to the stubs and exits 99.
    """
    bindir = tmp_path / "stubs"
    bindir.mkdir()
    log = bindir / "calls.log"
    for tool in ("security", "curl", "ssh", "scp", "ssh-keygen", "nvidia-smi"):
        stub = bindir / tool
        stub.write_text(f'#!/bin/sh\necho "{tool} $*" >> "{log}"\nexit 99\n')
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return bindir


def stub_env(stub_bin: Path, home: Path) -> dict[str, str]:
    """Environment putting the failing stubs first on PATH and a private HOME and state dir."""
    return {
        "PATH": f"{stub_bin}:{os.environ['PATH']}",
        "HOME": str(home),
        "RUNPOD_STATE_DIR": str(home),
        "RUNPOD_WATCHDOGS": str(home / ".runpod-watchdogs"),
    }


def stub_calls(stub_bin: Path) -> str:
    """The calls the stubs recorded (empty when none)."""
    log = stub_bin / "calls.log"
    return log.read_text() if log.exists() else ""
