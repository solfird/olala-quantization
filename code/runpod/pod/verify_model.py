r"""Check a downloaded model folder against the Hugging Face Hub's own file metadata.

Every file is compared by size; LFS files by sha256 (the hash the Hub stores for them), the small
files by their git blob id (sha1 of "blob <size>\0" + content, the `blobId` of the Hub listing).
The metadata comes from the public API without any token
(`/api/models/REPO/revision/REV?blobs=true`) or from a JSON file (`--meta`, used by the tests).
Exit 0 when every expected file is present and equal, 1 otherwise. Standard library only: it runs
on the pod before anything else is installed.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

CHUNK = 8 * 1024 * 1024
HUB = "https://huggingface.co/api/models"


def fetch_meta(repo: str, revision: str) -> dict:
    """The Hub's file listing (sizes, blob ids, LFS hashes) of one revision; public, no token."""
    url = f"{HUB}/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
        return json.load(response)


def sha256_of(path: Path) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_id(path: Path) -> str:
    """The git object id of a file: sha1 of 'blob <size>' NUL then the content."""
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def expected_files(meta: dict, ignore: list[str], only: list[str]) -> list[dict]:
    """The siblings of the listing that must be on disk."""
    files = []
    for sibling in meta["siblings"]:
        name = sibling["rfilename"]
        if any(fnmatch.fnmatch(name, pattern) for pattern in ignore):
            continue
        if only and name not in only:
            continue
        files.append(sibling)
    return files


def verify(directory: Path, files: list[dict]) -> tuple[list[str], int]:
    """Compare each file; return the report lines and the number of problems."""
    lines: list[str] = []
    problems = 0
    for sibling in files:
        name = sibling["rfilename"]
        path = directory / name
        if not path.is_file():
            lines.append(f"MISSING {name}")
            problems += 1
            continue
        size = path.stat().st_size
        if size != sibling["size"]:
            lines.append(f"FAIL {name} size {size} != {sibling['size']}")
            problems += 1
            continue
        lfs = sibling.get("lfs")
        if lfs:
            got = sha256_of(path)
            if got != lfs["sha256"]:
                lines.append(f"FAIL {name} sha256 mismatch: got {got}, hub {lfs['sha256']}")
                problems += 1
                continue
            lines.append(f"OK {name} {size} bytes sha256 {got[:16]}")
        else:
            got = git_blob_id(path)
            if got != sibling["blobId"]:
                lines.append(f"FAIL {name} git-blob mismatch: got {got}, hub {sibling['blobId']}")
                problems += 1
                continue
            lines.append(f"OK {name} {size} bytes git-blob {got[:16]}")
    return lines, problems


def main(argv: list[str]) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--meta", help="a JSON file in the Hub's listing format")
    source.add_argument("--repo", help="fetch the listing of this public repository")
    parser.add_argument("--revision", default="main", help="with --repo")
    parser.add_argument("--expect-revision", help="the commit the listing must belong to")
    parser.add_argument("--ignore", action="append", default=[], help="fnmatch pattern, repeatable")
    parser.add_argument(
        "--only", action="append", default=[], help="check only this file, repeatable"
    )
    args = parser.parse_args(argv)

    meta = (
        json.loads(Path(args.meta).read_text())
        if args.meta
        else fetch_meta(args.repo, args.revision)
    )
    problems = 0
    if args.expect_revision and meta.get("sha") != args.expect_revision:
        print(
            f"FAIL revision: the listing is for {meta.get('sha')}, expected {args.expect_revision}"
        )
        problems += 1
    files = expected_files(meta, args.ignore, args.only)
    lines, bad = verify(Path(args.dir), files)
    problems += bad
    for line in lines:
        print(line, flush=True)
    total = sum(f["size"] for f in files)
    if problems:
        sha = meta.get("sha")
        print(f"NOT VERIFIED: {problems} problem(s) in {len(files)} files (revision {sha})")
        return 1
    print(f"VERIFIED {len(files)} files, {total} bytes (revision {meta.get('sha')})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
