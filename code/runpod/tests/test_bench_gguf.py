"""pod/bench_gguf.sh: the short speed sweep of arm Q4 (amendment 24). No GPU, no network, no pod.

The script runs from a COPY of runpod/pod in a fake /workspace. gguf_setup.sh, serve_gguf.sh, smoke_gguf.sh,
stop_server.sh and gguf_provenance.sh are stubs that record their arguments; `vllm`, `pgrep` and `timeout` are stubs on
PATH; python3 is a wrapper that gives the real q4_tool.py a fake huggingface_hub (a snapshot_download that honours
allow_patterns, so that a download of weights would show).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import RUNPOD, stub_calls, stub_env

POD_SRC = RUNPOD / "pod"
SCRIPT = "bench_gguf.sh"
Q4 = "q4_k_m-ovhrecipe"
Q5 = "q5_k_m"
BOTH = f"{Q4},{Q5}"
MODEL_REVISION = "b40f8165ee2254c18601220e19842600c6d91794"

REPO_FILES = {
    "config.json": json.dumps({"auto_map": {"AutoConfig": "configuration_olala.OlalaConfig"}}),
    "configuration_olala.py": "class OlalaConfig: pass\n",
    "tokenizer.json": "{}",
    "tokenizer_config.json": json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}),
    "chat_template.jinja": "{{ messages }}",
    "special_tokens_map.json": "{}",
    "model-00001-of-00004.safetensors": "weights",
    "README.md": "readme",
}

STUBS = {
    "gguf_setup.sh": r"""#!/bin/bash
echo "gguf_setup.sh $*" >> "$STUB_LOG"
""",
    "stop_server.sh": r"""#!/bin/bash
echo "stop_server.sh" >> "$STUB_LOG"
rm -f "$OLALA_WORK/run/server.arm"
""",
    "serve_gguf.sh": r"""#!/bin/bash
echo "serve_gguf.sh $*" >> "$STUB_LOG"
v=""; while [ $# -gt 0 ]; do case $1 in --variant) v=$2; shift 2 ;; *) shift ;; esac; done
case ",${STUB_SERVE_FAIL:-}," in *",$v,"*) echo "error: serve failed" >&2; exit 1 ;; esac
mkdir -p "$OLALA_WORK/run"; echo "gguf-$v" > "$OLALA_WORK/run/server.arm"
""",
    "smoke_gguf.sh": r"""#!/bin/bash
echo "smoke_gguf.sh $*" >> "$STUB_LOG"
v=$(sed 's/^gguf-//' "$OLALA_WORK/run/server.arm")
case ",${STUB_SMOKE_FAIL:-}," in *",$v,"*) echo "smoke fail"; exit 1 ;; esac
case ",${STUB_SMOKE_WARN:-}," in *",$v,"*) echo "smoke warn"; exit 2 ;; esac
echo "smoke ok"
""",
    "gguf_provenance.sh": r"""#!/bin/bash
echo "gguf_provenance.sh" >> "$STUB_LOG"
echo "gguf_variant=$(sed 's/^gguf-//' "$OLALA_WORK/run/server.arm")"
""",
}

VLLM_STUB = r"""#!/bin/bash
echo "vllm $*" >> "$STUB_LOG"
echo "state: served=$(cat "$OLALA_WORK/run/server.arm" 2>/dev/null) done=$([ -e "$STUB_DONE_GLOB" ] && echo present || echo absent)" >> "$STUB_LOG"
conc=""; n=""; dir=""; name=""
while [ $# -gt 0 ]; do
  case $1 in
    --max-concurrency) conc=$2; shift 2 ;;
    --num-prompts) n=$2; shift 2 ;;
    --result-dir) dir=$2; shift 2 ;;
    --result-filename) name=$2; shift 2 ;;
    *) shift ;;
  esac
done
failed=0
case ",${STUB_FAIL_POINTS:-}," in *",$name,"*) failed=${STUB_FAILED:-2} ;; esac
[ "${STUB_VLLM_RC:-0}" = 0 ] || { echo "vllm crashed" >&2; exit "$STUB_VLLM_RC"; }
printf '{"completed": %s, "failed": %s, "num_prompts": %s, "output_throughput": %s, "median_ttft_ms": 40.0, "median_tpot_ms": 9.0}\n' \
  $((n - failed)) "$failed" "$n" "$((conc * 50))" > "$dir/$name"
echo "============ Serving Benchmark Result ============"
"""

PGREP_STUB = r"""#!/bin/bash
[ -z "${STUB_PGREP:-}" ] || echo "$STUB_PGREP"
"""

PYTHON_STUB = r"""#!/bin/bash
case "$*" in
  *q4_tool.py*) exec "$REAL_PY" -c "$FAKE_BOOT" "$@" ;;
  *) exec "$REAL_PY" "$@" ;;
esac
"""

FAKE_BOOT = r"""
import fnmatch, json, os, runpy, sys, types
i = [k for k, a in enumerate(sys.argv) if a.endswith("q4_tool.py")][0]
script = sys.argv[i]
sys.argv = sys.argv[i:]
def snapshot_download(**kw):
    with open(os.environ["FAKE_HF_LOG"], "a") as f:
        f.write(json.dumps({**kw, "HF_TOKEN_SEEN": os.environ.get("HF_TOKEN")}) + "\n")
    files = json.loads(os.environ["FAKE_REPO_FILES"])
    dest = kw["local_dir"]
    os.makedirs(dest, exist_ok=True)
    for name, content in files.items():
        if any(fnmatch.fnmatch(name, p) for p in kw["allow_patterns"]):
            with open(os.path.join(dest, name), "w") as f:
                f.write(content)
    return dest
hub = types.ModuleType("huggingface_hub")
hub.snapshot_download = snapshot_download
sys.modules["huggingface_hub"] = hub
runpy.run_path(script, run_name="__main__")
"""


def write_exec(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Bench:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.work = tmp / "ws"
        self.pod = tmp / "pod"
        self.log = tmp / "calls.log"
        self.log.write_text("")
        shutil.copytree(POD_SRC, self.pod, ignore=shutil.ignore_patterns("__pycache__"))
        for name, body in STUBS.items():
            write_exec(self.pod / name, body)
        bindir = tmp / "bin"
        bindir.mkdir()
        write_exec(bindir / "vllm", VLLM_STUB)
        write_exec(bindir / "pgrep", PGREP_STUB)
        write_exec(bindir / "python3", PYTHON_STUB)
        write_exec(bindir / "timeout", '#!/bin/bash\nshift\nexec "$@"\n')
        self.hf_log = tmp / "hf.log"
        (tmp / "cg").mkdir()
        self.env = {
            "OLALA_WORK": str(self.work),
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "REAL_PY": sys.executable,
            "FAKE_BOOT": FAKE_BOOT,
            "FAKE_HF_LOG": str(self.hf_log),
            "FAKE_REPO_FILES": json.dumps(REPO_FILES),
            "STUB_LOG": str(self.log),
            "STUB_DONE_GLOB": str(self.work / "results" / "bench-L1.done"),
            "HF_TOKEN": "must-not-leak",
            "BENCH_STOP_WAIT": "0",
        }

    @property
    def results(self) -> Path:
        return self.work / "results"

    @property
    def tokdir(self) -> Path:
        return self.work / "models" / "olala-tokenizer"

    @property
    def done(self) -> Path:
        return self.results / "bench-L1.done"

    @property
    def manifest(self) -> Path:
        return self.pod / "gguf_files.json"

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.pod / SCRIPT), *args],
            capture_output=True,
            text=True,
            env={**os.environ, **self.env, **env},
            check=False,
            timeout=120,
        )

    def calls(self) -> list[str]:
        return [ln for ln in self.log.read_text().splitlines() if ln]

    def add_q4_to_manifest(self) -> None:
        data = json.loads(self.manifest.read_text())
        data["gguf"]["files"][Q4] = {
            "name": "Olala-v1.0-mini-7A1B-q4_k_m-ovhrecipe.gguf",
            "size": 1,
            "sha256": "a" * 64,
            "derived_from": "bf16",
            "note": "produced on this pod",
        }
        self.manifest.write_text(json.dumps(data, indent=2))


@pytest.fixture
def bench(tmp_path):
    b = Bench(tmp_path)
    b.add_q4_to_manifest()
    return b


def vllm_calls(b: Bench) -> list[str]:
    return [c for c in b.calls() if c.startswith("vllm ")]


def sequence(b: Bench) -> list[str]:
    """The scripted calls, one short word each, in order."""
    out = []
    for c in b.calls():
        if c.startswith("state:"):
            continue
        head = c.split()[0]
        if head == "vllm":
            conc = c.split("--max-concurrency ")[1].split()[0]
            rep = c.split("--result-filename ")[1].split()[0]
            out.append(f"vllm c{conc} {rep}")
        elif head in ("serve_gguf.sh", "gguf_setup.sh"):
            out.append(" ".join(c.split()[:3]))
        else:
            out.append(head)
    return out


# --- arguments and dry run ------------------------------------------------------------------------------------


def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(POD_SRC / SCRIPT)], check=False).returncode == 0


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("L1",),
        ("L1", BOTH, "extra"),
        ("a/b", BOTH),
        ("-x", BOTH),
        ("L1", "q9_k_z"),
        ("L1", f"{Q4},"),
        ("L1", f"{Q4},{Q4}"),
        ("L1", ""),
        ("--bogus", "L1", BOTH),
    ],
)
def test_bad_arguments_are_refused_before_anything_is_run(bench, args):
    out = bench.run(*args)
    assert out.returncode == 2
    assert "BENCH_GGUF_FAILED" in out.stdout and bench.calls() == []
    assert not bench.work.exists()


def test_the_dry_run_prints_the_plan_runs_nothing_and_writes_nothing(bench, tmp_path, stub_bin):
    out = bench.run("--dry-run", "L1", BOTH, **stub_env(stub_bin, tmp_path / "home"))
    assert out.returncode == 0, out.stderr
    assert bench.calls() == [] and stub_calls(stub_bin) == "" and not bench.work.exists()
    text = out.stdout
    assert text.index(Q4) < text.index(Q5)
    assert "OVHaiLLM/Olala-v1.0-mini-7A1B" in text and MODEL_REVISION in text
    assert f"--tokenizer {bench.tokdir}" in text
    assert (
        "--random-input-len 512 --random-output-len 512 --num-prompts 8 --max-concurrency 1" in text
    )
    assert (
        "--num-prompts 12 --max-concurrency 4" in text
        and "--num-prompts 36 --max-concurrency 12" in text
    )
    assert text.strip().splitlines()[-1] == "BENCH_GGUF_DRY_RUN (nothing was run)"


# --- the tokenizer --------------------------------------------------------------------------------------------


def test_only_tokenizer_and_config_files_are_downloaded_once_at_the_pinned_revision(bench):
    assert bench.run("L1", BOTH).returncode == 0
    lines = bench.hf_log.read_text().splitlines()
    assert len(lines) == 1
    hf = json.loads(lines[0])
    assert hf["repo_id"] == "OVHaiLLM/Olala-v1.0-mini-7A1B" and hf["revision"] == MODEL_REVISION
    assert hf["token"] is False and hf["HF_TOKEN_SEEN"] is None
    assert not any("safetensors" in p or p in ("*", "*.bin") for p in hf["allow_patterns"])
    assert (bench.tokdir / "tokenizer.json").is_file() and not list(
        bench.tokdir.glob("*.safetensors")
    )


@pytest.mark.parametrize(
    "missing", ["tokenizer.json", "config.json", "chat_template.jinja", "configuration_olala.py"]
)
def test_a_missing_tokenizer_file_refuses_the_sweep_before_any_server_starts(bench, missing):
    files = {k: v for k, v in REPO_FILES.items() if k != missing}
    out = bench.run("L1", BOTH, FAKE_REPO_FILES=json.dumps(files))
    assert out.returncode == 1 and "BENCH_GGUF_FAILED:TOKENIZER" in out.stdout
    assert bench.calls() == []
    assert bench.done.read_text().splitlines()[0].startswith("BENCH_GGUF_FAILED:TOKENIZER")


def test_the_vllm_commands_name_the_tokenizer_folder(bench):
    assert bench.run("L1", BOTH).returncode == 0
    assert all(f"--tokenizer {bench.tokdir} " in c for c in vllm_calls(bench))


# --- order of a real run --------------------------------------------------------------------------------------


def test_each_variant_is_fetched_served_smoke_gated_swept_and_stopped_in_order(bench):
    out = bench.run("L1", BOTH)
    assert out.returncode == 0, out.stdout + out.stderr
    points = [f"vllm c{c} c{c}_r{r}.json" for r in (1, 2) for c in (1, 4, 12)]

    def one(v: str) -> list[str]:
        return [
            f"gguf_setup.sh --variant {v}",
            "stop_server.sh",
            f"serve_gguf.sh --variant {v}",
            "smoke_gguf.sh",
            *points,
            "gguf_provenance.sh",
            "stop_server.sh",
        ]

    assert sequence(bench) == [*one(Q4), *one(Q5)]


def test_the_file_is_made_present_with_setup_but_the_fork_is_not_rebuilt(bench):
    assert bench.run("L1", BOTH).returncode == 0
    setups = [c for c in bench.calls() if c.startswith("gguf_setup.sh")]
    assert setups == [
        f"gguf_setup.sh --variant {Q4} --skip-build",
        f"gguf_setup.sh --variant {Q5} --skip-build",
    ]


def test_the_server_is_started_with_the_default_twelve_slots(bench):
    assert bench.run("L1", BOTH).returncode == 0
    serves = [c for c in bench.calls() if c.startswith("serve_gguf.sh")]
    assert serves == [f"serve_gguf.sh --variant {Q4}", f"serve_gguf.sh --variant {Q5}"]
    assert not any("--np" in c for c in serves)


def test_the_smoke_gate_is_run_with_a_timeout(bench):
    assert bench.run("L1", Q5).returncode == 0
    assert [c for c in bench.calls() if c.startswith("smoke_gguf.sh")] == [
        "smoke_gguf.sh --timeout 900"
    ]


# --- the points -----------------------------------------------------------------------------------------------


def test_the_vllm_command_line_of_a_point_is_the_one_of_the_amendment(bench):
    assert bench.run("L1", Q5).returncode == 0
    first = vllm_calls(bench)[0]
    d = bench.results / f"bench-L1-{Q5}"
    assert first == (
        "vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions "
        f"--model olala --tokenizer {bench.tokdir} --trust-remote-code --dataset-name random --random-input-len 512 "
        "--random-output-len 512 --num-prompts 8 --max-concurrency 1 --ignore-eos --save-result "
        f"--result-dir {d} --result-filename c1_r1.json"
    )


def test_the_prompts_are_three_times_the_concurrency_and_at_least_eight_with_two_repeats(bench):
    assert bench.run("L1", Q5).returncode == 0
    pairs = []
    for c in vllm_calls(bench):
        pairs.append(
            (
                int(c.split("--max-concurrency ")[1].split()[0]),
                int(c.split("--num-prompts ")[1].split()[0]),
            )
        )
    assert sorted(pairs) == sorted([(1, 8), (4, 12), (12, 36)] * 2)


def test_the_result_json_lands_under_the_label_and_variant_directory(bench):
    assert bench.run("L1", BOTH).returncode == 0
    for v in (Q4, Q5):
        d = bench.results / f"bench-L1-{v}"
        assert sorted(p.name for p in d.glob("c*_r*.json")) == sorted(
            f"c{c}_r{r}.json" for c in (1, 4, 12) for r in (1, 2)
        )
        assert (d / "summary.json").is_file() and (d / "provenance.txt").is_file()
        assert (d / "smoke.txt").is_file() and (d / "machine.txt").is_file()


def test_the_repeats_can_be_set_by_the_environment(bench):
    assert bench.run("L1", Q5, BENCH_REPEATS="1").returncode == 0
    assert len(vllm_calls(bench)) == 3
    assert bench.run("L2", Q5, BENCH_REPEATS="0").returncode == 2


def test_every_point_is_run_while_its_variant_is_the_one_served(bench):
    assert bench.run("L1", BOTH).returncode == 0
    states = [c for c in bench.calls() if c.startswith("state:")]
    assert states[:6] == [f"state: served=gguf-{Q4} done=absent"] * 6
    assert states[6:] == [f"state: served=gguf-{Q5} done=absent"] * 6


# --- failures are reported, not hidden ------------------------------------------------------------------------


def test_a_point_with_failed_requests_is_reported_kept_and_flagged_in_the_done_file(bench):
    out = bench.run("L1", Q5, STUB_FAIL_POINTS="c4_r2.json", STUB_FAILED="2")
    assert out.returncode == 3
    assert f"BENCH_GGUF_POINT_FAILED:{Q5}:c=4:r=2" in out.stdout and "failed=2" in out.stdout
    assert f"BENCH_GGUF_DONE:{Q5}" in out.stdout  # the data is kept and the variant finished
    summary = json.loads((bench.results / f"bench-L1-{Q5}" / "summary.json").read_text())
    assert summary["points_with_failed_requests"] == 1
    lines = bench.done.read_text().splitlines()
    assert lines[0] == "BENCH_GGUF_ALL_DONE" and "points_with_failed_requests=1" in lines


def test_a_point_whose_command_crashed_is_a_failed_point_too(bench):
    out = bench.run("L1", Q5, STUB_VLLM_RC="1")
    assert out.returncode == 3 and f"BENCH_GGUF_POINT_FAILED:{Q5}:c=1:r=1" in out.stdout
    assert "no result" in out.stdout


def test_a_failed_smoke_gate_skips_that_variant_and_the_sweep_goes_on_to_the_next(bench):
    out = bench.run("L1", BOTH, STUB_SMOKE_FAIL=Q4)
    assert out.returncode == 1
    assert f"BENCH_GGUF_FAILED:{Q4}:SMOKE" in out.stdout and f"BENCH_GGUF_DONE:{Q5}" in out.stdout
    assert "BENCH_GGUF_ALL_DONE" not in out.stdout and f"BENCH_GGUF_DONE:{Q4}" not in out.stdout
    assert len(vllm_calls(bench)) == 6  # none for Q4
    assert bench.done.read_text().splitlines()[0].startswith(f"BENCH_GGUF_FAILED:{Q4}:SMOKE")
    assert [
        c for c in bench.calls() if c == "stop_server.sh"
    ]  # the server of the failed variant is stopped


def test_a_smoke_gate_with_warnings_is_accepted(bench):
    out = bench.run("L1", Q5, STUB_SMOKE_WARN=Q5)
    assert out.returncode == 0 and "BENCH_GGUF_ALL_DONE" in out.stdout


def test_a_server_that_does_not_start_fails_that_variant(bench):
    out = bench.run("L1", BOTH, STUB_SERVE_FAIL=Q4)
    assert out.returncode == 1 and f"BENCH_GGUF_FAILED:{Q4}:SERVE" in out.stdout
    assert (
        len([c for c in bench.calls() if c.startswith("smoke_gguf.sh")]) == 1
    )  # only Q5 got that far
    assert len(vllm_calls(bench)) == 6


def test_a_gpu_that_is_not_exclusive_fails_the_variant_before_any_point(bench):
    out = bench.run("L1", Q5, STUB_PGREP="1234 python run_tasks.py --model olala")
    assert out.returncode == 1 and f"BENCH_GGUF_FAILED:{Q5}:GPU_NOT_EXCLUSIVE" in out.stdout
    assert vllm_calls(bench) == []


# --- markers and the done file --------------------------------------------------------------------------------


def test_the_markers_are_printed_per_variant_then_for_the_whole_sweep(bench):
    out = bench.run("L1", BOTH)
    assert out.returncode == 0, out.stdout + out.stderr
    marks = [ln for ln in out.stdout.splitlines() if ln.startswith("BENCH_GGUF_")]
    assert marks == [f"BENCH_GGUF_DONE:{Q4}", f"BENCH_GGUF_DONE:{Q5}", "BENCH_GGUF_ALL_DONE"]


def test_the_done_file_is_written_last_and_is_absent_while_the_points_run(bench):
    assert bench.run("L1", BOTH).returncode == 0
    assert bench.done.read_text().splitlines()[0] == "BENCH_GGUF_ALL_DONE"
    others = [p for p in bench.results.rglob("*") if p.is_file() and p != bench.done]
    assert others and all(p.stat().st_mtime_ns <= bench.done.stat().st_mtime_ns for p in others)
    assert not any("done=present" in c for c in bench.calls())


def test_a_stale_done_file_is_removed_at_the_start(bench):
    bench.results.mkdir(parents=True)
    bench.done.write_text("BENCH_GGUF_ALL_DONE\nstale\n")
    out = bench.run("L1", Q5, STUB_SMOKE_FAIL=Q5)
    assert out.returncode == 1 and "stale" not in bench.done.read_text()


def test_an_existing_result_directory_is_never_overwritten(bench):
    (bench.results / f"bench-L1-{Q5}").mkdir(parents=True)
    out = bench.run("L1", Q5)
    assert out.returncode == 2 and "exists" in (out.stdout + out.stderr) and bench.calls() == []


# --- the q4 variant in the pod's manifest ---------------------------------------------------------------------


def test_a_manifest_that_lost_the_q4_variant_is_restored_from_the_record_of_q4_prepare(bench):
    data = json.loads(bench.manifest.read_text())
    entry = data["gguf"]["files"].pop(Q4)
    bench.manifest.write_text(json.dumps(data))
    (bench.work / "q4").mkdir(parents=True)
    (bench.work / "q4" / "variant.json").write_text(json.dumps({"variant": Q4, "entry": entry}))
    out = bench.run("L1", Q4)
    assert out.returncode == 0, out.stdout + out.stderr
    assert Q4 in json.loads(bench.manifest.read_text())["gguf"]["files"]


def test_a_missing_q4_variant_without_a_record_fails_the_variant(bench):
    data = json.loads(bench.manifest.read_text())
    data["gguf"]["files"].pop(Q4)
    bench.manifest.write_text(json.dumps(data))
    out = bench.run("L1", Q4)
    assert out.returncode == 1 and f"BENCH_GGUF_FAILED:{Q4}:MANIFEST" in out.stdout
    assert not any(c.startswith("serve_gguf.sh") for c in bench.calls())


def test_no_token_name_is_used_outside_an_unset():
    code = [
        ln for ln in (POD_SRC / SCRIPT).read_text().splitlines() if not ln.lstrip().startswith("#")
    ]
    for token in ("HF_TOKEN", "GITHUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        assert all("unset" in ln for ln in code if token in ln), token
