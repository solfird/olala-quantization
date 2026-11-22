"""Fetch popularity and maintenance data of the tools used here with ``gh api``.

DECISIONS.md embeds the table printed by this script, so the numbers there are generated, not
typed. Regenerate with ``uv run python repo_stats.py --json repo_stats.json``.
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_REPOS = (
    "UKGovernmentBEIS/inspect_ai",
    "UKGovernmentBEIS/inspect_evals",
    "evalplus/evalplus",
    "josejg/instruction_following_eval",
    "EleutherAI/lm-evaluation-harness",
    "openai/simple-evals",
    "allenai/IFBench",
    "vllm-project/vllm",
)

Runner = Callable[[list[str]], str | None]


def gh(args: list[str]) -> str | None:
    """Run ``gh <args>`` and return stdout, or ``None`` if the command fails.

    Args:
        args: Arguments after ``gh``.

    Returns:
        Standard output text, or ``None`` on a non-zero exit.
    """
    proc = subprocess.run(  # noqa: S603
        ["gh", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def fetch_repo(repo: str, runner: Runner | None = None) -> dict[str, Any]:
    """Collect stars, last push, license and latest release of one GitHub repository.

    Args:
        repo: ``owner/name``.
        runner: Function running ``gh`` (default: :func:`gh`); injectable for tests.

    Returns:
        A flat dict; ``stars`` is ``None`` and ``error`` is set if the repository is unreadable.
    """
    run = runner or gh
    raw = run(["api", f"repos/{repo}"])
    if raw is None:
        return {"repo": repo, "stars": None, "error": "gh api failed"}
    info = json.loads(raw)
    rel_raw = run(["api", f"repos/{repo}/releases/latest"])
    rel = json.loads(rel_raw) if rel_raw else {}
    return {
        "repo": repo,
        "stars": info.get("stargazers_count"),
        "pushed_at": info.get("pushed_at"),
        "license": (info.get("license") or {}).get("spdx_id"),
        "archived": info.get("archived"),
        "latest_release": rel.get("tag_name"),
        "latest_release_published": rel.get("published_at"),
    }


def render_markdown(rows: Sequence[dict[str, Any]], queried: str) -> str:
    """Render the rows as a markdown table.

    Args:
        rows: Output of :func:`fetch_repo` for each repository.
        queried: Date of the query, shown in the caption.

    Returns:
        Markdown text.
    """

    def day(value: str | None) -> str:
        return value[:10] if value else "none"

    lines = [
        f"Queried with `gh api` on {queried} (GitHub REST API).",
        "",
        "| Repository | Stars | Last push | Latest release | Release date | License | Archived |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("stars") is None:
            lines.append(f"| {r['repo']} | unavailable | | | | | |")
            continue
        lines.append(
            f"| {r['repo']} | {r['stars']} | {day(r['pushed_at'])} | "
            f"{r['latest_release'] or 'none'} | {day(r['latest_release_published'])} | "
            f"{r['license'] or 'none'} | {r['archived']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(prog="repo_stats.py", description=__doc__.split("\n")[0])
    ap.add_argument("--repos", nargs="+", default=list(DEFAULT_REPOS))
    ap.add_argument("--json", help="write the raw rows here")
    args = ap.parse_args(argv)
    now = datetime.now(UTC)
    rows = [fetch_repo(r, gh) for r in args.repos]
    if args.json:
        payload = {"queried_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "rows": rows}
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sys.stdout.write(render_markdown(rows, now.strftime("%Y-%m-%d")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
