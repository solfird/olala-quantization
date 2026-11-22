"""Tests for repo_stats.py (popularity / maintenance data pulled with `gh api`)."""

import json

import repo_stats

REPO = {
    "stargazers_count": 1234,
    "pushed_at": "2026-11-01T10:00:00Z",
    "license": {"spdx_id": "MIT"},
    "archived": False,
}
RELEASE = {"tag_name": "v1.2.3", "published_at": "2026-10-20T08:00:00Z"}


def fake_runner(calls):
    def run(args):
        calls.append(args)
        if args[-1].endswith("/releases/latest"):
            return json.dumps(RELEASE) if "has/release" in args[-1] else None
        return json.dumps(REPO)

    return run


def test_fetch_repo_collects_stars_push_date_license_and_latest_release():
    calls = []
    row = repo_stats.fetch_repo("has/release", fake_runner(calls))
    assert row == {
        "repo": "has/release",
        "stars": 1234,
        "pushed_at": "2026-11-01T10:00:00Z",
        "license": "MIT",
        "archived": False,
        "latest_release": "v1.2.3",
        "latest_release_published": "2026-10-20T08:00:00Z",
    }
    assert calls[0] == ["api", "repos/has/release"]
    assert calls[1] == ["api", "repos/has/release/releases/latest"]


def test_a_repository_without_releases_gets_none_for_the_release_fields():
    row = repo_stats.fetch_repo("no/release", fake_runner([]))
    assert row["latest_release"] is None
    assert row["latest_release_published"] is None


def test_a_repository_that_cannot_be_read_is_reported_as_unavailable():
    row = repo_stats.fetch_repo("x/y", lambda args: None)
    assert row["stars"] is None
    assert row["error"]


def test_markdown_table_lists_every_repository_with_the_query_date():
    rows = [
        repo_stats.fetch_repo("has/release", fake_runner([])),
        repo_stats.fetch_repo("no/release", fake_runner([])),
    ]
    md = repo_stats.render_markdown(rows, "2026-11-20")
    assert "2026-11-20" in md
    assert "has/release" in md and "no/release" in md
    assert "1234" in md
    assert "v1.2.3" in md
    assert "2026-11-01" in md


def test_main_writes_json_and_prints_markdown(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(repo_stats, "gh", fake_runner([]))
    code = repo_stats.main(["--json", str(tmp_path / "s.json"), "--repos", "has/release"])
    assert code == 0
    saved = json.loads((tmp_path / "s.json").read_text())
    assert saved["rows"][0]["stars"] == 1234
    assert saved["queried_utc"]
    assert "has/release" in capsys.readouterr().out
