"""pod/arm_pipeline.sh: the online: and ckpt: modes as they are, and the ckptsel: mode (a component-ablation checkpoint).

No GPU, no network, no pod. The script runs from a copy of pod/ whose /workspace is a temp folder and whose steps
(stop_server.sh, serve_olala.sh, smoke.sh, run_accuracy.sh, the loader patch, olala_quant.convert, .verify and .ablation) are
stubs that record their arguments in one calls log, so the ORDER of the steps is what the tests read.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from ablation_stub import source as ablation_source
from conftest import RUNPOD

POD = RUNPOD / "pod"
BANNERS = (
    "INFO Quantizing 3 layers of types: Fp8LinearMethod\n"
    "INFO Selected FlashInferFP8ScaledMMLinearKernel for Fp8LinearMethod\n"
    "INFO unrelated line that must not be copied\n"
)
INDEX = '{"weight_map": {}}\n'
CONFIG = '{"model_type": "olala"}\n'


def qinfo(selection: str | None = None) -> str:
    info: dict[str, object] = {"recipe": "nvfp4"}
    if selection is not None:
        info["selection"] = {"spec": selection}
    return json.dumps(info) + "\n"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stub_script(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Pipe:
    """A pod's /workspace in a temp folder, every step of arm_pipeline.sh replaced by a recording stub."""

    def __init__(self, tmp: Path) -> None:
        self.work = tmp / "ws"
        self.pod = self.work / "bundle" / "pod"
        self.calls = tmp / "calls.log"
        self.calls.write_text("")
        shutil.copytree(POD, self.pod)
        for name in ("arm_pipeline.sh", "job_common.sh"):  # the scripts hard-code /workspace
            path = self.pod / name
            path.write_text(path.read_text().replace("/workspace", str(self.work)))
        c = str(self.calls)
        with (self.pod / "job_common.sh").open("a") as f:  # the loader patch needs the real vLLM
            f.write(
                f'\napply_patch() {{ echo apply_patch >> "{c}"; return "${{PATCH_RC:-0}}"; }}\n'
            )
        stub_script(self.pod / "stop_server.sh", f'echo "stop_server.sh" >> "{c}"\n')
        stub_script(
            self.pod / "serve_olala.sh",
            f'echo "serve_olala.sh $*" >> "{c}"\necho "started $1"\nexit "${{SERVE_RC:-0}}"\n',
        )
        stub_script(
            self.pod / "smoke.sh", f'echo "smoke.sh $*" >> "{c}"\nexit "${{SMOKE_RC:-0}}"\n'
        )
        stub_script(
            self.pod / "run_accuracy.sh",
            f'echo "run_accuracy.sh $*" >> "{c}"\necho RUN_ACCURACY_DONE\nexit "${{ACC_RC:-0}}"\n',
        )
        pkg = self.work / "bundle" / "quant" / "olala_quant"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "ablation.py").write_text(ablation_source(self.calls))
        (pkg / "convert.py").write_text(
            f"""import json, os, pathlib, sys
a = sys.argv[1:]
open({c!r}, "a").write("convert " + " ".join(a) + "\\n")
print("convert says hi")
if os.environ.get("CONVERT_RC", "0") != "0":
    print("convert boom"); sys.exit(int(os.environ["CONVERT_RC"]))
dst = pathlib.Path(a[a.index("--dst") + 1]); dst.mkdir(parents=True, exist_ok=True)
info = {{"recipe": a[a.index("--recipe") + 1]}}
if "--select" in a:
    info["selection"] = {{"spec": a[a.index("--select") + 1]}}
(dst / "model.safetensors.index.json").write_text({INDEX!r})
(dst / "config.json").write_text({CONFIG!r})
(dst / "quantization_info.json").write_text(json.dumps(info) + "\\n")
"""
        )
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
        if shutil.which("sha256sum") is None:
            stub_script(self.bin / "sha256sum", 'exec shasum -a 256 "$@"\n')
        (self.work / "logs" / "setup").mkdir(parents=True)
        (self.work / "logs" / "setup" / "evals-env-1.log").write_text("EVALS_ENV_DONE\n")
        for sub in ("fp8pc", "ckpt-fp8-I", "ckpt-abl"):
            (self.work / "logs" / sub).mkdir()
            (self.work / "logs" / sub / "server-1.log").write_text(BANNERS)
        (self.work / "results").mkdir()
        self.ckpt = self.work / "ckpt" / "abl"
        self.src = self.work / "models" / "olala-bf16"

    def make_ckpt(self, selection: str | None, files: tuple[str, ...] = ("i", "c", "q")) -> None:
        self.ckpt.mkdir(parents=True, exist_ok=True)
        if "i" in files:
            (self.ckpt / "model.safetensors.index.json").write_text(INDEX)
        if "c" in files:
            (self.ckpt / "config.json").write_text(CONFIG)
        if "q" in files:
            (self.ckpt / "quantization_info.json").write_text(qinfo(selection))

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.pod / "arm_pipeline.sh"), *args],
            capture_output=True,
            text=True,
            env={**os.environ, **env, "PATH": f"{self.bin}:{os.environ['PATH']}"},
            check=False,
            timeout=60,
        )

    def steps(self) -> list[str]:
        return [ln for ln in self.calls.read_text().splitlines() if ln]

    def names(self) -> list[str]:
        out = []
        for s in self.steps():
            w = s.split()
            out.append(f"ablation {w[1]}" if w[0] == "ablation" else w[0])
        return out

    def step(self, prefix: str) -> str:
        return next(s for s in self.steps() if s.startswith(prefix))

    @property
    def prov(self) -> str:
        return (self.work / "results" / "provenance-F2.txt").read_text()


@pytest.fixture
def pipe(tmp_path):
    return Pipe(tmp_path)


def last_line(out: subprocess.CompletedProcess[str]) -> str:
    return out.stdout.strip().splitlines()[-1]


SPECS = ["experts", "experts:layers=0-11", "non-experts:layers=0-3,5-7,12", "mamba:layers=4"]


def ckptsel(pipe: Pipe, spec: str, recipe: str = "nvfp4") -> str:
    return f"ckptsel:{pipe.ckpt}:{recipe}:{spec}"


# --- online: and ckpt: are what they were ---------------------------------------------------------------------------------


def test_online_mode_serves_smoke_gates_records_provenance_then_runs_the_accuracy_harness(pipe):
    out = pipe.run("F1", "online:fp8pc", "--tasks", "mgsm")
    assert out.returncode == 0, out.stdout + out.stderr
    assert last_line(out) == "ARM_PIPELINE_DONE"
    assert pipe.steps() == [
        "stop_server.sh",
        "serve_olala.sh fp8pc",
        "smoke.sh fp8pc --timeout 900",
        "run_accuracy.sh F1 --effort medium --tasks mgsm",
    ]
    lines = (pipe.work / "results" / "provenance-F1.txt").read_text().splitlines()
    assert lines[0].startswith("run=F1 mode=online:fp8pc started=")
    assert lines[1] == "NVIDIA RTX PRO 4500 Blackwell, 580.95"
    assert lines[2:] == [
        "INFO Quantizing 3 layers of types: Fp8LinearMethod",
        "INFO Selected FlashInferFP8ScaledMMLinearKernel for Fp8LinearMethod",
    ]


def test_ckpt_mode_converts_a_missing_checkpoint_without_select_and_runs_no_verify_or_ablation(
    pipe,
):
    out = pipe.run("F2", f"ckpt:{pipe.work}/ckpt/fp8-I:fp8")
    assert out.returncode == 0, out.stdout + out.stderr
    assert last_line(out) == "ARM_PIPELINE_DONE"
    d = f"{pipe.work}/ckpt/fp8-I"
    assert pipe.steps() == [
        "stop_server.sh",
        "apply_patch",
        f"convert --src {pipe.src} --dst {d} --recipe fp8 --device cuda",
        f"serve_olala.sh ckpt {d}",
        "smoke.sh ckpt-fp8-I --timeout 900",
        "run_accuracy.sh F2 --effort medium",
    ]
    text = pipe.prov
    assert f"{sha(INDEX)}  model.safetensors.index.json" in text
    assert "Quantizing 3 layers" in text and "selection" not in text


def test_ckpt_mode_does_not_convert_an_existing_checkpoint(pipe):
    pipe.ckpt = pipe.work / "ckpt" / "fp8-I"
    pipe.make_ckpt(None)
    assert pipe.run("F2", f"ckpt:{pipe.ckpt}:fp8").returncode == 0
    assert pipe.names() == [
        "stop_server.sh",
        "apply_patch",
        "serve_olala.sh",
        "smoke.sh",
        "run_accuracy.sh",
    ]


def test_an_unknown_mode_is_refused(pipe):
    out = pipe.run("F2", "magic:1")
    assert out.returncode == 1 and last_line(out) == "ARM_PIPELINE_FAILED"


# --- ckptsel: the happy path ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS)
def test_ckptsel_checks_the_selection_converts_verifies_checks_then_serves_and_runs_the_harness(
    pipe, spec
):
    out = pipe.run("F2", ckptsel(pipe, spec), "--tasks", "mgsm")
    assert out.returncode == 0, out.stdout + out.stderr
    assert last_line(out) == "ARM_PIPELINE_DONE"
    d = pipe.ckpt
    assert pipe.names() == [
        "ablation expected",
        "stop_server.sh",
        "apply_patch",
        "convert",
        "verify",
        "ablation check",
        "serve_olala.sh",
        "smoke.sh",
        "run_accuracy.sh",
    ]
    assert pipe.step("ablation expected") == f"ablation expected --src {pipe.src} --select {spec}"
    assert pipe.step("convert") == (
        f"convert --src {pipe.src} --dst {d} --recipe nvfp4 --device cuda --select {spec}"
    )
    verify = pipe.step("verify")
    assert verify.startswith(f"verify --src {pipe.src} --dst {d} --report ")
    assert verify.endswith(f" --select {spec}")
    assert pipe.step("ablation check") == (
        f"ablation check --src {pipe.src} --dst {d} --select {spec}"
    )
    assert pipe.step("serve_olala.sh") == f"serve_olala.sh ckpt {d}"
    assert pipe.step("smoke.sh").startswith("smoke.sh ckpt-abl --timeout 900")
    assert pipe.step("run_accuracy.sh") == "run_accuracy.sh F2 --effort medium --tasks mgsm"


def test_the_spec_is_split_from_the_front_so_that_its_colons_stay_in_it(pipe):
    spec = "experts:layers=12-23,30"
    assert pipe.run("F2", ckptsel(pipe, spec)).returncode == 0
    assert pipe.step("convert").endswith(f"--recipe nvfp4 --device cuda --select {spec}")
    assert f"--dst {pipe.ckpt} " in pipe.step("convert")


def test_the_provenance_records_the_selection_the_expected_counts_the_hashes_and_the_verdicts(pipe):
    assert pipe.run("F2", ckptsel(pipe, "experts:layers=5-7,0-3")).returncode == 0
    text = pipe.prov
    assert text.startswith("run=F2 mode=ckptsel:")
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
    info = json.loads((pipe.ckpt / "quantization_info.json").read_text())
    assert info["selection"]["spec"] == "experts:layers=5-7,0-3"
    for name, content in (
        ("model.safetensors.index.json", INDEX),
        ("config.json", CONFIG),
        ("quantization_info.json", (pipe.ckpt / "quantization_info.json").read_text()),
    ):
        assert f"{sha(content)}  {name}" in text
    assert "Quantizing 3 layers" in text
    assert text.index("quantization_info.json") < text.index("Quantizing 3 layers")


# --- ckptsel: gates that fail the pipeline --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "reason", "ran"),
    [
        ({"CONVERT_RC": "1"}, "convert", ["apply_patch", "convert"]),
        ({"VERIFY_RC": "1"}, "verify", ["apply_patch", "convert", "verify"]),
        (
            {"CHECK_RC": "1"},
            "ablation check",
            ["apply_patch", "convert", "verify", "ablation check"],
        ),
    ],
)
def test_a_failed_convert_verify_or_ablation_check_fails_the_pipeline_before_serving(
    pipe, env, reason, ran
):
    out = pipe.run("F2", ckptsel(pipe, "experts"), **env)
    assert out.returncode == 1, out.stdout + out.stderr
    assert last_line(out) == "ARM_PIPELINE_FAILED"
    assert reason in out.stdout
    assert pipe.names()[2:] == ran
    for never in ("serve_olala.sh", "smoke.sh", "run_accuracy.sh"):
        assert never not in pipe.names()


def test_a_failed_serve_fails_the_pipeline_before_the_smoke_gate(pipe):
    out = pipe.run("F2", ckptsel(pipe, "experts"), SERVE_RC="1")
    assert out.returncode == 1 and "serve" in out.stdout and last_line(out) == "ARM_PIPELINE_FAILED"
    assert "smoke.sh" not in pipe.names() and "run_accuracy.sh" not in pipe.names()


def test_the_ablation_check_mismatch_is_shown_in_the_log(pipe):
    out = pipe.run("F2", ckptsel(pipe, "experts"), CHECK_RC="1")
    assert "MISMATCH" in out.stdout and "selection" in out.stdout


def test_ckptsel_runs_the_checks_on_an_existing_checkpoint_of_the_same_selection_without_converting(
    pipe,
):
    pipe.make_ckpt("experts:layers=0-3,5-7")
    out = pipe.run(
        "F2", ckptsel(pipe, "experts:layers=5-7,0-3")
    )  # another spelling of the same selection
    assert out.returncode == 0, out.stdout + out.stderr
    assert pipe.names() == [
        "ablation expected",
        "stop_server.sh",
        "apply_patch",
        "verify",
        "ablation check",
        "serve_olala.sh",
        "smoke.sh",
        "run_accuracy.sh",
    ]


@pytest.mark.parametrize(
    ("selection", "files"),
    [
        ("experts", ("i", "c", "q")),  # another selection
        ("experts:layers=0-11", ("i", "c", "q")),  # same component, other layers
        (None, ("i", "c", "q")),  # a plain checkpoint, no selection recorded
        ("experts:layers=0-3", ("i", "c")),  # no quantization_info.json at all
    ],
)
def test_an_existing_checkpoint_of_another_selection_is_refused_and_nothing_is_run(
    pipe, selection, files
):
    pipe.make_ckpt(selection, files)
    out = pipe.run("F2", ckptsel(pipe, "experts:layers=0-3,5-7"))
    assert out.returncode == 1, out.stdout + out.stderr
    assert last_line(out) == "ARM_PIPELINE_FAILED"
    assert "stale" in out.stdout.lower() and str(pipe.ckpt) in out.stdout
    assert pipe.names() == ["ablation expected"]  # no server stopped, no convert, nothing served


def test_the_refusal_of_a_stale_checkpoint_names_both_selections(pipe):
    pipe.make_ckpt("experts")
    out = pipe.run("F2", ckptsel(pipe, "experts:layers=5-7,0-3"))
    assert "experts:layers=0-3,5-7" in out.stdout and "'experts'" in out.stdout


# --- ckptsel: arguments are validated before any GPU work ----------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "ckptsel:/w/d:nvfp4",  # no spec
        "ckptsel:/w/d:nvfp4:",
        "ckptsel:/w/d:nvfp4:bogus",
        "ckptsel:/w/d:nvfp4:Experts",
        "ckptsel:/w/d:nvfp4:experts:layers=",
        "ckptsel:/w/d:nvfp4:experts:layers=5-",
        "ckptsel:/w/d:nvfp4:experts:layers=a",
        "ckptsel:/w/d:nvfp4:experts:other=1",
        "ckptsel:/w/d:nvfp4:experts:layers=1:layers=2",
        "ckptsel:/w/d:fp4:experts",  # bad recipe
        "ckptsel:/w/d::experts",
        "ckptsel:rel/d:nvfp4:experts",  # relative dir
        "ckptsel::nvfp4:experts",
        "ckptsel:/w/d x:nvfp4:experts",
    ],
)
def test_a_malformed_ckptsel_mode_fails_before_any_step_runs(pipe, bad):
    out = pipe.run("F2", bad)
    assert out.returncode == 1 and last_line(out) == "ARM_PIPELINE_FAILED", out.stdout
    assert "ckptsel" in out.stdout
    assert pipe.steps() == []


def test_a_malformed_spec_is_refused_even_when_the_evals_environment_never_comes(pipe):
    (pipe.work / "logs" / "setup" / "evals-env-1.log").write_text("EVALS_ENV_FAILED\n")
    out = pipe.run("F2", ckptsel(pipe, "bogus"))
    assert "evals environment" not in out.stdout and "ckptsel" in out.stdout


def test_a_spec_that_the_ablation_module_refuses_fails_before_the_server_is_touched(pipe):
    out = pipe.run("F2", ckptsel(pipe, "experts:layers=0-999"))
    assert out.returncode == 1 and last_line(out) == "ARM_PIPELINE_FAILED"
    assert "layers 0 to 55" in out.stdout
    assert pipe.names() == ["ablation expected"]
