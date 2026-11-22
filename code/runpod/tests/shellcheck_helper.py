"""The shellcheck launcher the pod-side shell tests use (shellcheck when installed, else uvx, else None)."""

import shutil


def shellcheck_command() -> list[str] | None:
    """shellcheck when installed, else through uvx (the shellcheck-py wheel), else None."""
    if shutil.which("shellcheck"):
        return ["shellcheck"]
    if shutil.which("uvx"):
        return ["uvx", "--from", "shellcheck-py", "shellcheck"]
    return None
