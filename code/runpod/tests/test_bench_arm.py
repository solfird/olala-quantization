"""pod/bench_arm.sh: one arm's throughput sweep on a fresh pod (serve, smoke gate, provenance, sweep, BENCH_DONE).

No GPU, no network, no pod. The script is run from a copy of pod/ whose /workspace is a temp folder and whose steps
(stop_server.sh, serve_olala.sh, smoke.sh, bench_sweep.sh, the loader patch, olala_quant.convert and .verify) are stubs
that record their arguments in one calls log, so the ORDER of the steps is what the tests read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from ablation_stub import NAMES
from ablation_stub import source as ablation_source
from conftest import REPO, RUNPOD, run_script, stub_calls

POD = RUNPOD / "pod"
BANNERS = (
    "INFO Quantizing 3 layers of types: Fp8LinearMethod\n"
    "INFO Selected FlashInferFP8ScaledMMLinearKernel for Fp8LinearMethod\n"
    "INFO Using FLASHINFER_CUTLASS MoE backend\n"
    "INFO unrelated line that must not be copied\n"
)
CKPT_FILES = {
    "model.safetensors.index.json": '{"weight_map": {}}\n',
    "config.json": '{"model_type": "olala"}\n',
    "quantization_info.json": '{"recipe": "fp8"}\n',
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stub_script(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Bench:
    """A pod's /workspace in a temp folder, with every step of bench_arm.sh replaced by a recording stub."""

    def __init__(self, tmp: Path) -> None:
        self.work = tmp / "ws"
        self.pod = self.work / "bundle" / "pod"
        self.calls = tmp / "calls.log"
        self.calls.write_text("")
        shutil.copytree(POD, self.pod)
        for name in ("bench_arm.sh", "job_common.sh"):  # the scripts hard-code /workspace
            path = self.pod / name
            if path.exists():
                path.write_text(path.read_text().replace("/workspace", str(self.work)))
        with (self.pod / "job_common.sh").open(
            "a"
        ) as f:  # the loader patch needs the real vLLM site-packages
            f.write(
                f'\napply_patch() {{ echo apply_patch >> "{self.calls}"; return "${{PATCH_RC:-0}}"; }}\n'
            )
        c = str(self.calls)
        stub_script(
            self.pod / "stop_server.sh", f'echo "stop_server.sh" >> "{c}"\nexit "${{STOP_RC:-0}}"\n'
        )
        stub_script(
            self.pod / "serve_olala.sh",
            f'echo "serve_olala.sh $*" >> "{c}"\necho "started $1"\nexit "${{SERVE_RC:-0}}"\n',
        )
        stub_script(
            self.pod / "smoke.sh",
            f'echo "smoke.sh $*" >> "{c}"\necho "smoke verdict"\nexit "${{SMOKE_RC:-0}}"\n',
        )
        stub_script(
            self.pod / "bench_sweep.sh",
            f"""echo "bench_sweep.sh $*" >> "{c}"
[ -z "${{SWEEP_EARLY_DONE:-}}" ] || echo BENCH_DONE
out="{self.work}/results/bench-$1"
if [ -z "${{SWEEP_NOJSON:-}}" ]; then mkdir -p "$out"; echo '{{}}' > "$out/short_c1_r1.json"; fi
[ -z "${{SWEEP_POINTFAIL:-}}" ] || echo "POINT FAILED: short c=128 rep=1 (see x.log)"
echo "bench $1: output tok/s summary"
[ -n "${{SWEEP_NODONE:-}}" ] || echo BENCH_DONE
exit "${{SWEEP_RC:-0}}"
""",
        )
        # olala_quant.convert / verify: recorded, and convert writes a checkpoint
        pkg = self.work / "bundle" / "quant" / "olala_quant"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        files = repr(CKPT_FILES)
        (pkg / "convert.py").write_text(
            f"""import os, pathlib, sys
a = sys.argv[1:]
open({c!r}, "a").write("convert " + " ".join(a) + "\\n")
print("convert says hi")
if os.environ.get("CONVERT_RC", "0") != "0":
    print("convert boom"); sys.exit(int(os.environ["CONVERT_RC"]))
dst = pathlib.Path(a[a.index("--dst") + 1]); dst.mkdir(parents=True, exist_ok=True)
for n, t in {files}.items():
    (dst / n).write_text(t)
if "--select" in a:
    (dst / "quantization_info.json").write_text('{{"recipe": "nvfp4", "selection": {{"spec": "' + a[a.index("--select") + 1] + '"}}}}\\n')
"""
        )
        (pkg / "ablation.py").write_text(ablation_source(self.calls))
        (pkg / "verify.py").write_text(
            f"""import os, pathlib, sys
a = sys.argv[1:]
open({c!r}, "a").write("verify " + " ".join(a) + "\\n")
if os.environ.get("VERIFY_RC", "0") != "0":
    print("verify violations", file=sys.stderr); sys.exit(int(os.environ["VERIFY_RC"]))
if "--report" in a:
    pathlib.Path(a[a.index("--report") + 1]).write_text("{{}}")
print("OK: 3 tensors checked, no violation. {{}}")
"""
        )
        self.bin = tmp / "bin"
        self.bin.mkdir()
        stub_script(self.bin / "nvidia-smi", "echo 'NVIDIA RTX PRO 4500 Blackwell, 580.95'\n")
        stub_script(self.bin / "pgrep", 'printf "%s" "${PGREP_OUT:-}"\n[ -n "${PGREP_OUT:-}" ]\n')
        if shutil.which("sha256sum") is None:
            stub_script(self.bin / "sha256sum", 'exec shasum -a 256 "$@"\n')
        for sub in ("logs/fp8pc", "logs/ckpt-fp8-I", "results"):
            (self.work / sub).mkdir(parents=True, exist_ok=True)
        for sub in ("fp8pc", "ckpt-fp8-I"):
            (self.work / "logs" / sub / "server-1.log").write_text(BANNERS)
        self.ckpt = self.work / "ckpt" / "fp8-I"

    def make_ckpt(self, missing: tuple[str, ...] = (), selection: str | None = None) -> None:
        self.ckpt.mkdir(parents=True, exist_ok=True)
        for n, t in CKPT_FILES.items():
            if n not in missing:
                (self.ckpt / n).write_text(t)
        if selection is not None and "quantization_info.json" not in missing:
            info = {"recipe": "nvfp4", "selection": {"spec": selection}}
            (self.ckpt / "quantization_info.json").write_text(json.dumps(info) + "\n")

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.pod / "bench_arm.sh"), *args],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "BENCH_STOP_WAIT": "0",
                **env,
                "PATH": f"{self.bin}:{os.environ['PATH']}",
            },
            check=False,
            timeout=60,
        )

    def steps(self) -> list[str]:
        return [ln for ln in self.calls.read_text().splitlines() if ln]

    def names(self) -> list[str]:
        return [s.split()[0] for s in self.steps()]

    @property
    def benchlog(self) -> Path:
        return self.work / "logs" / "bench-F1.log"

    @property
    def prov(self) -> Path:
        return self.work / "results" / "provenance-F1.txt"


@pytest.fixture
def bench(tmp_path):
    return Bench(tmp_path)


def last_line(out: subprocess.CompletedProcess[str]) -> str:
    return out.stdout.strip().splitlines()[-1]


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


# --- dry run ---------------------------------------------------------------------------------------------------------


def test_dry_run_online_prints_the_plan_and_touches_nothing(bench):
    before = tree(bench.work)
    out = bench.run("--dry-run", "F1", "online:fp8pc")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "serve_olala.sh fp8pc" in out.stdout
    assert "smoke.sh fp8pc --timeout 900" in out.stdout
    assert "bench_sweep.sh F1 --repeats 2" in out.stdout
    assert "BENCH_PIPELINE_FAILED" not in out.stdout
    assert bench.steps() == [] and tree(bench.work) == before


def test_dry_run_ckpt_lists_patch_convert_verify_hashes_and_the_log_name(bench):
    out = bench.run("F2", "--dry-run", "ckpt:/workspace/ckpt/fp8-I:fp8", BENCH_REPEATS="5")
    assert out.returncode == 0, out.stdout + out.stderr
    text = out.stdout
    assert "olala_quant.convert" in text and "--recipe fp8" in text
    assert "olala_quant.verify" in text
    assert "model.safetensors.index.json config.json quantization_info.json" in text
    assert "serve_olala.sh ckpt /workspace/ckpt/fp8-I" in text
    assert "smoke.sh ckpt-fp8-I --timeout 900" in text
    assert "bench_sweep.sh F2 --repeats 5" in text
    assert bench.steps() == []


def test_dry_run_uses_no_network_secret_or_gpu_tool(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for tool in ("security", "curl", "ssh", "scp", "nvidia-smi", "pgrep", "vllm"):
        stub_script(stubs / tool, f'echo "{tool} $*" >> "{stubs}/calls.log"\nexit 99\n')
    out = run_script(
        "pod/bench_arm.sh",
        "--dry-run",
        "F1",
        "online:fp8pc",
        env={"PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(tmp_path)},
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert stub_calls(stubs) == ""


# --- argument refusal ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("F1",),
        ("F1", "magic:1"),
        ("F1", "online:"),
        ("F1", "online:fp8pc:extra"),
        ("F1", "gguf:q5_k_m"),
        ("F1", "ckpt:/workspace/ckpt/x"),
        ("F1", "ckpt:/workspace/ckpt/x:"),
        ("F1", "ckpt:/workspace/ckpt/x:fp4"),
        ("F1", "ckpt:relative/dir:fp8"),
        ("F1", "ckpt::fp8"),
        ("a/b", "online:fp8pc"),
        ("-x", "online:fp8pc"),
        ("a b", "online:fp8pc"),
        ("../x", "online:fp8pc"),
        ("F1", "online:fp8pc", "extra"),
        ("F1", "online:fp8pc", "--bogus"),
    ],
)
def test_bad_arguments_are_refused_with_exit_2_before_anything_runs(bench, args):
    out = bench.run(*args)
    assert out.returncode == 2, out.stdout + out.stderr
    assert "BENCH_PIPELINE_FAILED" in out.stdout and "usage" in (out.stdout + out.stderr).lower()
    assert bench.steps() == [] and not bench.benchlog.exists()
    assert (
        out.returncode == bench.run("--dry-run", *args).returncode
    )  # a dry run refuses the same way


@pytest.mark.parametrize("value", ["0", "x", "-1", "2.5", ""])
def test_bad_repeats_are_refused(bench, value):
    out = bench.run("F1", "online:fp8pc", BENCH_REPEATS=value)
    assert out.returncode == 2 and "BENCH_REPEATS" in out.stdout + out.stderr
    assert "BENCH_PIPELINE_FAILED" in out.stdout and bench.steps() == []


def test_an_existing_bench_directory_is_refused_before_a_server_is_started(bench):
    (bench.work / "results" / "bench-F1").mkdir()
    out = bench.run("F1", "online:fp8pc")
    assert out.returncode == 2 and "bench-F1" in out.stdout + out.stderr
    assert "BENCH_PIPELINE_FAILED" in out.stdout and bench.steps() == []


# --- order of the steps and the files they leave -----------------------------------------------------------------------


def test_online_mode_serves_smoke_gates_sweeps_then_stops_the_server_and_says_done(bench):
    out = bench.run("F1", "online:fp8pc")
    assert out.returncode == 0, out.stdout + out.stderr
    assert last_line(out) == "BENCH_PIPELINE_DONE"
    steps = bench.steps()
    assert [s.split()[0] for s in steps] == [
        "stop_server.sh",
        "serve_olala.sh",
        "smoke.sh",
        "bench_sweep.sh",
        "stop_server.sh",
    ]
    assert steps[1] == "serve_olala.sh fp8pc"
    assert steps[2].startswith("smoke.sh fp8pc --timeout 900")
    assert steps[3] == "bench_sweep.sh F1 --repeats 2"
    assert "apply_patch" not in steps  # online arms need no loader patch (as in arm_pipeline.sh)


def test_the_provenance_has_the_header_the_gpu_and_the_kernel_banners_and_nothing_else_of_the_log(
    bench,
):
    assert bench.run("F1", "online:fp8pc").returncode == 0
    text = bench.prov.read_text()
    assert text.startswith("run=F1 mode=online:fp8pc started=")
    assert "NVIDIA RTX PRO 4500 Blackwell, 580.95" in text
    assert (
        "Quantizing 3 layers of types" in text
        and "Selected FlashInfer" in text
        and "MoE backend" in text
    )
    assert "unrelated line" not in text


def test_repeats_come_from_the_environment(bench):
    assert bench.run("F1", "online:fp8pc", BENCH_REPEATS="3").returncode == 0
    assert "bench_sweep.sh F1 --repeats 3" in bench.steps()


def test_the_results_directory_the_collector_pulls_gets_provenance_smoke_and_the_sweep_output(
    bench,
):
    assert bench.run("F1", "online:fp8pc").returncode == 0
    d = bench.work / "results" / "bench-F1"
    assert (d / "provenance-F1.txt").read_text() == bench.prov.read_text()
    assert "smoke verdict" in (d / "smoke.txt").read_text()
    assert "bench F1: output tok/s summary" in (d / "sweep.log").read_text()
    # bench_sweep.sh's summary globs *.json and splits NAME_cN_rN: only point files may be JSON in that folder
    assert sorted(p.name for p in d.glob("*.json")) == ["short_c1_r1.json"]


def test_the_bench_log_holds_the_sweep_output_and_ends_with_exactly_one_bench_done(bench):
    assert bench.run("F1", "online:fp8pc").returncode == 0
    lines = bench.benchlog.read_text().splitlines()
    assert "bench F1: output tok/s summary" in lines
    assert lines[-1] == "BENCH_DONE" and lines.count("BENCH_DONE") == 1


def test_ckpt_mode_converts_a_missing_checkpoint_verifies_hashes_serves_and_sweeps_in_order(bench):
    out = bench.run("F1", f"ckpt:{bench.ckpt}:fp8")
    assert out.returncode == 0, out.stdout + out.stderr
    assert last_line(out) == "BENCH_PIPELINE_DONE"
    steps = bench.steps()
    assert [s.split()[0] for s in steps] == [
        "stop_server.sh",
        "apply_patch",
        "convert",
        "verify",
        "serve_olala.sh",
        "smoke.sh",
        "bench_sweep.sh",
        "stop_server.sh",
    ]
    src = f"{bench.work}/models/olala-bf16"
    assert steps[2] == f"convert --src {src} --dst {bench.ckpt} --recipe fp8 --device cuda"
    assert steps[3].startswith(f"verify --src {src} --dst {bench.ckpt} --report ")
    assert steps[4] == f"serve_olala.sh ckpt {bench.ckpt}"
    assert steps[5].startswith("smoke.sh ckpt-fp8-I --timeout 900")
    text = bench.prov.read_text()
    for name, content in CKPT_FILES.items():
        assert f"{sha(content)}  {name}" in text
    assert "OK: 3 tensors checked" in text  # the verify verdict is recorded
    assert "Quantizing 3 layers" in text  # banners come after the hashes
    assert text.index("quantization_info.json") < text.index("Quantizing 3 layers")


def test_ckpt_mode_does_not_convert_an_existing_checkpoint_but_still_verifies_it(bench):
    bench.make_ckpt()
    assert bench.run("F1", f"ckpt:{bench.ckpt}:fp8").returncode == 0
    assert bench.names() == [
        "stop_server.sh",
        "apply_patch",
        "verify",
        "serve_olala.sh",
        "smoke.sh",
        "bench_sweep.sh",
        "stop_server.sh",
    ]


def test_verify_options_come_from_the_environment_for_calibrated_checkpoints(bench):
    bench.make_ckpt()
    assert (
        bench.run("F1", f"ckpt:{bench.ckpt}:nvfp4", BENCH_VERIFY_ARGS="--calibrated").returncode
        == 0
    )
    verify = next(s for s in bench.steps() if s.startswith("verify"))
    assert verify.endswith("--calibrated") or " --calibrated " in verify


def test_a_smoke_exit_2_is_accepted(bench):
    out = bench.run("F1", "online:fp8pc", SMOKE_RC="2")
    assert out.returncode == 0 and last_line(out) == "BENCH_PIPELINE_DONE"
    assert "bench_sweep.sh" in bench.names()


# --- BENCH_DONE only after a complete sweep ----------------------------------------------------------------------------


def test_a_stale_bench_done_from_an_earlier_attempt_is_removed_when_the_pipeline_fails_early(bench):
    bench.benchlog.write_text("old sweep\nBENCH_DONE\n")
    out = bench.run("F1", "online:fp8pc", SMOKE_RC="1")
    assert out.returncode == 1 and "BENCH_DONE" not in (
        bench.benchlog.read_text() if bench.benchlog.exists() else ""
    )


def test_a_failed_sweep_leaves_no_bench_done_even_if_the_sweep_printed_one(bench):
    out = bench.run("F1", "online:fp8pc", SWEEP_RC="2", SWEEP_EARLY_DONE="1", SWEEP_NODONE="1")
    assert out.returncode == 1 and last_line(out).startswith("BENCH_PIPELINE_FAILED")
    assert "sweep" in last_line(out)
    assert "BENCH_DONE" not in bench.benchlog.read_text()
    assert bench.names()[-1] == "stop_server.sh"


def test_a_sweep_that_printed_bench_done_but_wrote_no_result_is_a_failure(bench):
    out = bench.run("F1", "online:fp8pc", SWEEP_NOJSON="1")
    assert out.returncode == 1 and last_line(out).startswith("BENCH_PIPELINE_FAILED")
    assert "BENCH_DONE" not in bench.benchlog.read_text()


def test_a_sweep_that_never_reached_its_end_marker_is_a_failure(bench):
    out = bench.run("F1", "online:fp8pc", SWEEP_NODONE="1")
    assert out.returncode == 1 and last_line(out).startswith("BENCH_PIPELINE_FAILED")
    assert "BENCH_DONE" not in bench.benchlog.read_text()


def test_failed_points_keep_the_data_collectable_but_the_pipeline_still_says_failed(bench):
    out = bench.run("F1", "online:fp8pc", SWEEP_POINTFAIL="1")
    assert out.returncode == 1 and last_line(out).startswith("BENCH_PIPELINE_FAILED")
    assert "point" in last_line(out).lower()
    assert (
        bench.benchlog.read_text().splitlines()[-1] == "BENCH_DONE"
    )  # the collector pulls the partial data
    assert (bench.work / "results" / "bench-F1" / "provenance-F1.txt").exists()
    assert bench.names()[-1] == "stop_server.sh"


# --- failure paths: reason printed, no sweep, no BENCH_DONE, server stopped ---------------------------------------------


@pytest.mark.parametrize(
    ("env", "reason"),
    [
        ({"SERVE_RC": "1"}, "serve"),
        ({"SMOKE_RC": "1"}, "smoke gate"),
        ({"SMOKE_RC": "3"}, "smoke gate"),
        ({"PGREP_OUT": "4242 python run_tasks.py\n"}, "GPU"),
    ],
)
def test_online_failures_stop_before_the_sweep_and_name_the_reason(bench, env, reason):
    out = bench.run("F1", "online:fp8pc", **env)
    assert out.returncode == 1, out.stdout + out.stderr
    assert last_line(out).startswith("BENCH_PIPELINE_FAILED")
    assert reason in last_line(out)
    assert "bench_sweep.sh" not in bench.names() and bench.names()[-1] == "stop_server.sh"
    assert not bench.benchlog.exists() or "BENCH_DONE" not in bench.benchlog.read_text()


@pytest.mark.parametrize(
    ("env", "reason", "stopped_after"),
    [
        ({"PATCH_RC": "1"}, "loader patch", ["stop_server.sh", "apply_patch"]),
        ({"CONVERT_RC": "1"}, "convert", ["stop_server.sh", "apply_patch", "convert"]),
        ({"VERIFY_RC": "1"}, "verify", ["stop_server.sh", "apply_patch", "convert", "verify"]),
    ],
)
def test_ckpt_failures_before_serving_name_the_reason_and_serve_nothing(
    bench, env, reason, stopped_after
):
    out = bench.run("F1", f"ckpt:{bench.ckpt}:fp8", **env)
    assert out.returncode == 1, out.stdout + out.stderr
    assert last_line(out).startswith("BENCH_PIPELINE_FAILED") and reason in last_line(out)
    assert bench.names()[: len(stopped_after)] == stopped_after
    assert "serve_olala.sh" not in bench.names() and "bench_sweep.sh" not in bench.names()


def test_a_checkpoint_missing_one_of_the_hashed_files_is_a_failure(bench):
    bench.make_ckpt(missing=("quantization_info.json",))
    out = bench.run("F1", f"ckpt:{bench.ckpt}:fp8")
    assert out.returncode == 1 and "provenance" in last_line(out)
    assert "serve_olala.sh" not in bench.names()


def test_the_verify_exit_status_is_not_hidden_by_a_pipe(bench):
    bench.make_ckpt()
    out = bench.run("F1", f"ckpt:{bench.ckpt}:fp8", VERIFY_RC="2")
    assert (
        out.returncode == 1 and "verify" in last_line(out) and "serve_olala.sh" not in bench.names()
    )


def test_a_failing_final_stop_does_not_turn_a_complete_sweep_into_a_failure(bench):
    out = bench.run("F1", "online:fp8pc", STOP_RC="1")
    assert out.returncode == 0 and last_line(out) == "BENCH_PIPELINE_DONE"


def test_the_gpu_exclusivity_check_runs_after_the_smoke_gate_and_before_the_sweep(bench):
    # the smoke test itself uses the GPU; the busy check must not trip on it, and must come right before the sweep
    out = bench.run("F1", "online:fp8pc", PGREP_OUT="9 vllm bench serve\n")
    assert out.returncode == 1
    assert bench.names() == ["stop_server.sh", "serve_olala.sh", "smoke.sh", "stop_server.sh"]


def test_a_server_log_without_kernel_banners_is_noted_in_the_provenance_not_hidden(bench):
    (bench.work / "logs" / "fp8pc" / "server-1.log").write_text("INFO nothing of interest\n")
    out = bench.run("F1", "online:fp8pc")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "warning: no kernel banner found" in bench.prov.read_text()


@pytest.mark.parametrize("value", ["--calibrated; touch x", "$(id)", "--a`b`", "--a|b"])
def test_verify_options_with_shell_characters_are_refused(bench, value):
    out = bench.run("F1", f"ckpt:{bench.ckpt}:fp8", BENCH_VERIFY_ARGS=value)
    assert out.returncode == 2 and "BENCH_VERIFY_ARGS" in out.stdout + out.stderr
    assert bench.steps() == []


# --- ckptsel: a component-ablation checkpoint ---------------------------------------------------------------------------------

SPECS = ["experts", "experts:layers=0-11", "non-experts:layers=0-3,5-7,12", "mamba:layers=4"]


def sel_mode(bench, spec, recipe="nvfp4"):
    return f"ckptsel:{bench.ckpt}:{recipe}:{spec}"


@pytest.mark.parametrize("spec", SPECS)
def test_dry_run_ckptsel_prints_the_plan_with_the_selection_on_every_step_and_touches_nothing(
    bench, spec
):
    before = tree(bench.work)
    out = bench.run("--dry-run", "F2", sel_mode(bench, spec), BENCH_REPEATS="5")
    assert out.returncode == 0, out.stdout + out.stderr
    text = out.stdout
    assert f"--recipe nvfp4 --device cuda --select {spec}" in text
    src = f"{bench.work}/models/olala-bf16"  # the fixture's copy of the script has /workspace rewritten
    assert f"olala_quant.verify --src {src} --dst {bench.ckpt} --report" in text
    assert f"--select {spec}" in next(ln for ln in text.splitlines() if "olala_quant.verify" in ln)
    check = next(ln for ln in text.splitlines() if "olala_quant.ablation check" in ln)
    assert f"--src {src} --dst {bench.ckpt} --select {spec}" in check
    assert "olala_quant.ablation expected" in text and "stale" in text.lower()
    assert f"serve_olala.sh ckpt {bench.ckpt}" in text and "bench_sweep.sh F2 --repeats 5" in text
    assert bench.steps() == [] and tree(bench.work) == before


@pytest.mark.parametrize(
    "mode",
    [
        "ckptsel:/w/d:nvfp4",
        "ckptsel:/w/d:nvfp4:",
        "ckptsel:/w/d:nvfp4:bogus",
        "ckptsel:/w/d:nvfp4:Experts",
        "ckptsel:/w/d:nvfp4:experts:",
        "ckptsel:/w/d:nvfp4:experts:layers=",
        "ckptsel:/w/d:nvfp4:experts:layers=5-",
        "ckptsel:/w/d:nvfp4:experts:layers=a",
        "ckptsel:/w/d:nvfp4:experts:other=1",
        "ckptsel:/w/d:nvfp4:experts:layers=1:layers=2",
        "ckptsel:/w/d:nvfp4:experts;touch x",
        "ckptsel:/w/d:fp4:experts",
        "ckptsel:/w/d::experts",
        "ckptsel:rel/d:nvfp4:experts",
        "ckptsel::nvfp4:experts",
    ],
)
def test_a_malformed_ckptsel_mode_is_refused_with_exit_2_before_anything_runs(bench, mode):
    out = bench.run("F2", mode)
    assert out.returncode == 2, out.stdout + out.stderr
    assert "BENCH_PIPELINE_FAILED" in out.stdout and "usage" in (out.stdout + out.stderr).lower()
    assert bench.steps() == [] and not (bench.work / "logs" / "bench-F2.log").exists()
    assert bench.run("--dry-run", "F2", mode).returncode == 2


def test_the_selection_names_the_scripts_accept_are_the_ones_of_olala_quant_ablation():
    text = (REPO / "quant" / "olala_quant" / "ablation.py").read_text()
    block = text.split("SELECT_NAMES: dict", 1)[1].split("\n}", 1)[0]
    real = set(re.findall(r'"([a-z-]+)":\s*frozenset', block))
    assert real == set(NAMES)
    for script in ("bench_arm.sh", "arm_pipeline.sh"):
        m = re.search(r"sel_re='\(([a-z|-]+)\)", (POD / script).read_text())
        assert m is not None, f"{script} has no selection-name grammar"
        assert set(m.group(1).split("|")) == real, f"{script} and ablation.SELECT_NAMES disagree"


@pytest.mark.parametrize("spec", SPECS)
def test_ckptsel_checks_the_selection_converts_verifies_checks_then_serves_and_sweeps(bench, spec):
    out = bench.run("F2", sel_mode(bench, spec))
    assert out.returncode == 0, out.stdout + out.stderr
    assert last_line(out) == "BENCH_PIPELINE_DONE"
    assert bench.names() == [
        "ablation",
        "stop_server.sh",
        "apply_patch",
        "convert",
        "verify",
        "ablation",
        "serve_olala.sh",
        "smoke.sh",
        "bench_sweep.sh",
        "stop_server.sh",
    ]
    steps = bench.steps()
    src = f"{bench.work}/models/olala-bf16"
    assert steps[0] == f"ablation expected --src {src} --select {spec}"
    assert (
        steps[3]
        == f"convert --src {src} --dst {bench.ckpt} --recipe nvfp4 --device cuda --select {spec}"
    )
    assert steps[4].startswith(f"verify --src {src} --dst {bench.ckpt} --report ")
    assert steps[4].endswith(f" --select {spec}")
    assert steps[5] == f"ablation check --src {src} --dst {bench.ckpt} --select {spec}"
    assert steps[6] == f"serve_olala.sh ckpt {bench.ckpt}"
    assert steps[7].startswith("smoke.sh ckpt-fp8-I --timeout 900")


def test_the_verify_options_of_the_environment_are_kept_next_to_the_selection(bench):
    out = bench.run("F2", sel_mode(bench, "experts"), BENCH_VERIFY_ARGS="--calibrated")
    assert out.returncode == 0, out.stdout + out.stderr
    verify = next(s for s in bench.steps() if s.startswith("verify"))
    assert " --calibrated" in verify and verify.count("--select experts") == 1


def test_the_provenance_records_the_selection_the_expected_counts_the_hashes_and_the_verdicts(
    bench,
):
    assert bench.run("F2", sel_mode(bench, "experts:layers=5-7,0-3")).returncode == 0
    text = (bench.work / "results" / "provenance-F2.txt").read_text()
    assert "selection=experts:layers=0-3,5-7" in text  # the canonical spelling
    assert "selection_requested=experts:layers=5-7,0-3" in text
    counts = next(ln for ln in text.splitlines() if ln.startswith("expected_counts="))
    assert json.loads(counts.split("=", 1)[1]) == {
        "ignored_tensors": 219,
        "quantized_tensors": 40,
        "spec": "experts:layers=0-3,5-7",
    }
    assert "verify: OK: 3 tensors checked" in text
    assert "ablation_check: ABLATION_CHECK_OK experts:layers=5-7,0-3" in text
    info = (bench.ckpt / "quantization_info.json").read_text()
    for name, content in (
        ("model.safetensors.index.json", CKPT_FILES["model.safetensors.index.json"]),
        ("config.json", CKPT_FILES["config.json"]),
        ("quantization_info.json", info),
    ):
        assert f"{sha(content)}  {name}" in text
    d = bench.work / "results" / "bench-F2"
    assert (d / "provenance-F2.txt").read_text() == text  # the collector pulls it with the sweep


@pytest.mark.parametrize(
    ("env", "reason", "ran"),
    [
        ({"CONVERT_RC": "1"}, "convert", ["apply_patch", "convert"]),
        ({"VERIFY_RC": "1"}, "verify", ["apply_patch", "convert", "verify"]),
        ({"CHECK_RC": "1"}, "ablation check", ["apply_patch", "convert", "verify", "ablation"]),
    ],
)
def test_a_failed_convert_verify_or_ablation_check_fails_the_pipeline_before_serving(
    bench, env, reason, ran
):
    out = bench.run("F2", sel_mode(bench, "experts"), **env)
    assert out.returncode == 1, out.stdout + out.stderr
    assert last_line(out).startswith("BENCH_PIPELINE_FAILED") and reason in last_line(out)
    assert bench.names()[2:] == ran
    assert "serve_olala.sh" not in bench.names() and "bench_sweep.sh" not in bench.names()


def test_an_ablation_check_mismatch_is_shown(bench):
    out = bench.run("F2", sel_mode(bench, "experts"), CHECK_RC="1")
    assert "MISMATCH" in out.stdout


def test_an_existing_checkpoint_of_the_same_selection_is_not_converted_but_still_verified_and_checked(
    bench,
):
    bench.make_ckpt(selection="experts:layers=0-3,5-7")
    out = bench.run("F2", sel_mode(bench, "experts:layers=5-7,0-3"))
    assert out.returncode == 0, out.stdout + out.stderr
    assert bench.names() == [
        "ablation",
        "stop_server.sh",
        "apply_patch",
        "verify",
        "ablation",
        "serve_olala.sh",
        "smoke.sh",
        "bench_sweep.sh",
        "stop_server.sh",
    ]


@pytest.mark.parametrize(
    ("selection", "missing"),
    [
        ("experts", ()),
        ("experts:layers=0-11", ()),
        (None, ()),  # a plain checkpoint
        ("experts:layers=0-3", ("quantization_info.json",)),
    ],
)
def test_an_existing_checkpoint_of_another_selection_is_refused_and_nothing_is_run(
    bench, selection, missing
):
    bench.make_ckpt(missing=missing, selection=selection)
    out = bench.run("F2", sel_mode(bench, "experts:layers=0-3,5-7"))
    assert out.returncode == 2, out.stdout + out.stderr
    assert last_line(out).startswith("BENCH_PIPELINE_FAILED")
    assert "stale" in out.stdout.lower() and str(bench.ckpt) in out.stdout
    assert bench.names() == [
        "ablation"
    ]  # only the CPU pre-check: no server stopped, no convert, no GPU
    assert not (bench.work / "logs" / "bench-F2.log").exists()
    assert not (bench.work / "results" / "provenance-F2.txt").exists()


def test_the_refusal_of_a_stale_checkpoint_names_both_selections(bench):
    bench.make_ckpt(selection="experts")
    out = bench.run("F2", sel_mode(bench, "experts:layers=5-7,0-3"))
    assert "experts:layers=0-3,5-7" in out.stdout and "'experts'" in out.stdout


def test_a_spec_that_olala_quant_ablation_refuses_is_refused_before_anything_else_runs(bench):
    out = bench.run("F2", sel_mode(bench, "experts:layers=0-999"))
    assert out.returncode == 2, out.stdout + out.stderr
    assert last_line(out).startswith("BENCH_PIPELINE_FAILED") and "layers 0 to 55" in out.stdout
    assert bench.names() == ["ablation"]


def test_the_plain_ckpt_mode_passes_no_selection_and_never_runs_the_ablation_check(bench):
    assert bench.run("F1", f"ckpt:{bench.ckpt}:fp8").returncode == 0
    assert all("--select" not in s for s in bench.steps())
    assert "ablation" not in bench.names()
    assert "selection" not in bench.prov.read_text()
