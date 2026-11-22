"""pod/ablation_arm.sh and pod/ablation_batch.sh: component-ablation fidelity runs, without a GPU, a network or a pod.

The scripts run from a copy of pod/ whose /workspace is a temp folder and whose steps (stop_server.sh, serve_olala.sh,
smoke.sh, the loader patch, olala_quant.convert, .verify and .ablation, evals/fidelity.py) are stubs that record their
arguments in one calls log, so the ORDER of the steps and the arguments each gets are what the tests read.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from conftest import RUNPOD, run_script, stub_calls
from shellcheck_helper import shellcheck_command

POD = RUNPOD / "pod"
SPEC = "experts:layers=12-23"
LABEL = "abl-experts-L12-23"
EXPECTED = {
    "spec": SPEC,
    "quantized_tensors": 22,
    "ignored_tensors": 237,
    "quantized_layers": 11,
    "quantized_layers_by_type": {"mlp.experts": 11},
    "ignored_layers": 212,
    "total_quantizable_layers": 223,
    "ignore_entries": 213,
}
COMPARE = {
    "ppl_ref": {"value": 10.0, "lo": 9.9, "hi": 10.1},
    "ppl_var": {"value": 10.2, "lo": 10.1, "hi": 10.3},
    "ppl_ratio": {"value": 1.02, "lo": 1.01, "hi": 1.03},
    "kld_mean": {"value": 0.0123, "lo": 0.011, "hi": 0.0136},
    "kld_median": {"value": 0.004, "lo": 0.003, "hi": 0.005},
    "kld_p99": {"value": 0.21, "lo": 0.19, "hi": 0.23},
    "top1_agreement": {"value": 0.951, "lo": 0.949, "hi": 0.953},
    "n_chunks": 64,
    "n_positions": 32768,
    "k": 20,
}


def stub_script(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Ablation:
    """A pod's /workspace in a temp folder, with every step of ablation_arm.sh replaced by a recording stub."""

    def __init__(self, tmp: Path) -> None:
        self.work = tmp / "ws"
        self.pod = self.work / "bundle" / "pod"
        self.calls = tmp / "calls.log"
        self.calls.write_text("")
        shutil.copytree(POD, self.pod)
        for name in (
            "ablation_arm.sh",
            "ablation_batch.sh",
            "job_common.sh",
        ):  # they hard-code /workspace
            path = self.pod / name
            if path.exists():
                path.write_text(path.read_text().replace("/workspace", str(self.work)))
        c = str(self.calls)
        with (self.pod / "job_common.sh").open(
            "a"
        ) as f:  # the loader patch needs the real vLLM site-packages
            f.write(
                f'\napply_patch() {{ echo apply_patch >> "{c}"; return "${{PATCH_RC:-0}}"; }}\n'
            )
        stub_script(
            self.pod / "stop_server.sh", f'echo "stop_server.sh" >> "{c}"\nexit "${{STOP_RC:-0}}"\n'
        )
        stub_script(
            self.pod / "serve_olala.sh",
            f"""echo "serve_olala.sh $*" >> "{c}"
if [ "$1" = ckpt ]; then name=ckpt-$(basename "$2"); else name=$1; fi
mkdir -p "{self.work}/logs/$name"
printf '%s' "${{SERVER_LOG_TEXT:-INFO Using MARLIN NvFp4 MoE backend}}" > "{self.work}/logs/$name/server-1.log"
echo "started $1"
exit "${{SERVE_RC:-0}}"
""",
        )
        stub_script(
            self.pod / "smoke.sh",
            f"""echo "smoke.sh $*" >> "{c}"
echo "smoke verdict $1"
case "$1" in bf16) exit "${{REF_SMOKE_RC:-0}}" ;; *) exit "${{SMOKE_RC:-0}}" ;; esac
""",
        )
        # python stubs: olala_quant.{convert,verify,ablation} and evals/fidelity.py
        pkg = self.work / "bundle" / "quant" / "olala_quant"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "convert.py").write_text(
            f"""import os, pathlib, sys
a = sys.argv[1:]
open({c!r}, "a").write("convert " + " ".join(a) + "\\n")
if os.environ.get("DONE_PROBE") and os.path.exists(os.environ["DONE_PROBE"]):
    open({c!r}, "a").write("probe: a .done exists while the run is in progress\\n")
print("convert says hi")
if os.environ.get("CONVERT_RC", "0") != "0":
    print("convert boom"); sys.exit(int(os.environ["CONVERT_RC"]))
dst = pathlib.Path(a[a.index("--dst") + 1]); dst.mkdir(parents=True, exist_ok=True)
for n in ("model.safetensors.index.json", "config.json", "quantization_info.json"):
    (dst / n).write_text('{{"n": "' + n + '"}}\\n')
(dst / "model-00001-of-00001.safetensors").write_text("weights")
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
print("OK: 945 tensors checked, no violation. {{}}")
"""
        )
        expected = json.dumps(EXPECTED)
        (pkg / "ablation.py").write_text(
            f"""import json, os, sys
a = sys.argv[1:]
open({c!r}, "a").write("ablation " + " ".join(a) + "\\n")
spec = a[a.index("--select") + 1]
if a[0] == "expected":
    if os.environ.get("EXPECTED_RC", "0") != "0":
        print("error: unknown selection", file=sys.stderr); sys.exit(2)
    print(json.dumps(dict({expected}, spec=spec)))
else:
    if os.environ.get("CHECK_RC", "0") != "0":
        print("MISMATCH: 3 quantized tensors on disk, expected 22", file=sys.stderr); sys.exit(1)
    print("ABLATION_CHECK_OK " + spec)
"""
        )
        evals = self.work / "bundle" / "evals"
        evals.mkdir(parents=True)
        compare = json.dumps(COMPARE)
        (evals / "fidelity.py").write_text(
            f"""import json, os, pathlib, sys
a = sys.argv[1:]
open({c!r}, "a").write("fidelity " + " ".join(a) + "\\n")
if a[0] == "collect":
    if os.environ.get("COLLECT_RC", "0") != "0" and "b0" not in a[a.index("--out") + 1]:
        print("collect boom"); sys.exit(1)
    if os.environ.get("REF_COLLECT_RC", "0") != "0" and "b0_" in a[a.index("--out") + 1]:
        print("reference collect boom"); sys.exit(1)
    out = pathlib.Path(a[a.index("--out") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("COLLECT_NO_FILE"):
        out.write_bytes(b"npz:" + out.name.encode())
    print("wrote " + str(out) + ": (64, 511) positions, k=20")
else:
    if os.environ.get("COMPARE_RC", "0") != "0":
        print("compare boom"); sys.exit(1)
    pathlib.Path(a[a.index("--json") + 1]).write_text(json.dumps({compare}))
    print("### Logit fidelity")
    print("| Perplexity ratio (variant / reference) | 1.0200 | [1.0100, 1.0300] |")
    print("| Mean KL(ref || variant), nats | 0.01230 | [0.01100, 0.01360] |")
    print("| 99th percentile KL, nats | 0.21000 | [0.19000, 0.23000] |")
    print("| Same top-1 token (fraction) | 0.9510 | [0.9490, 0.9530] |")
"""
        )
        self.bin = tmp / "bin"
        self.bin.mkdir()
        stub_script(self.bin / "nvidia-smi", "echo 'NVIDIA RTX PRO 4500 Blackwell, 580.95'\n")
        if shutil.which("sha256sum") is None:
            stub_script(self.bin / "sha256sum", 'exec shasum -a 256 "$@"\n')
        (self.work / "models" / "olala-bf16").mkdir(parents=True)
        self.ckpt = self.work / "ckpt" / f"abl-{LABEL}"
        self.out = self.work / "results" / f"ablation-{LABEL}"
        self.done = self.work / "results" / f"ablation-{LABEL}.done"
        self.ref = self.work / "results" / "raw" / "b0_wikitext2.npz"

    def make_reference(self) -> None:
        self.ref.parent.mkdir(parents=True, exist_ok=True)
        self.ref.write_bytes(b"a reference collected earlier on this pod")

    def run(
        self, *args: str, script: str = "ablation_arm.sh", **env: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.pod / script), *args],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "ABLATION_STOP_WAIT": "0",
                **env,
                "PATH": f"{self.bin}:{os.environ['PATH']}",
            },
            check=False,
            timeout=60,
        )

    def steps(self) -> list[str]:
        return [ln for ln in self.calls.read_text().splitlines() if ln]

    def names(self) -> list[str]:
        out = []
        for s in self.steps():
            w = s.split()
            out.append(" ".join(w[:2]) if w[0] in ("fidelity", "ablation") else w[0])
        return out

    def step(self, prefix: str) -> str:
        found = [s for s in self.steps() if s.startswith(prefix)]
        assert len(found) == 1, (prefix, self.steps())
        return found[0]

    def summary(self) -> dict:
        return json.loads((self.out / "summary.json").read_text())


@pytest.fixture
def abl(tmp_path):
    return Ablation(tmp_path)


def last_line(out: subprocess.CompletedProcess[str]) -> str:
    return out.stdout.strip().splitlines()[-1]


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


FULL_WITH_REFERENCE = [
    "stop_server.sh",
    "ablation expected",
    "apply_patch",
    "convert",
    "verify",
    "serve_olala.sh",
    "smoke.sh",
    "ablation check",
    "fidelity collect",
    "stop_server.sh",
    "fidelity compare",
]


# --- dry run and arguments -------------------------------------------------------------------------------------------


def test_dry_run_prints_the_whole_plan_and_touches_nothing(abl):
    before = tree(abl.work)
    out = abl.run("--dry-run", LABEL, SPEC)
    assert out.returncode == 0, out.stdout + out.stderr
    text = out.stdout
    for needle in (
        f"--select {SPEC}",
        "olala_quant.ablation expected",
        "olala_quant.convert",
        "--recipe nvfp4",
        "olala_quant.verify",
        "olala_quant.ablation check",
        f"serve_olala.sh ckpt {abl.work}/ckpt/abl-{LABEL}",
        f"smoke.sh ckpt-abl-{LABEL} --timeout 900",
        "fidelity.py collect",
        "fidelity.py compare",
        "b0_wikitext2.npz",
        f"{abl.work}/results/ablation-{LABEL}.done",
        "ABLATION_DONE",
    ):
        assert needle in text, needle
    assert abl.steps() == [] and tree(abl.work) == before


def test_dry_run_names_the_reference_given_by_env_and_the_recipe(abl):
    out = abl.run("--dry-run", LABEL, SPEC, ABLATION_REF="/data/ref.npz", ABLATION_RECIPE="fp8")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "/data/ref.npz" in out.stdout and "--recipe fp8" in out.stdout
    assert "serve_olala.sh bf16" not in out.stdout  # a given reference is not produced


def test_dry_run_says_it_will_produce_the_reference_when_there_is_none(abl):
    out = abl.run("--dry-run", LABEL, SPEC)
    assert "serve_olala.sh bf16" in out.stdout
    abl.make_reference()
    out = abl.run("--dry-run", LABEL, SPEC)
    assert "serve_olala.sh bf16" not in out.stdout and "reference present" in out.stdout


def test_dry_run_uses_no_network_secret_or_gpu_tool(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for tool in ("security", "curl", "ssh", "scp", "nvidia-smi", "vllm"):
        stub_script(stubs / tool, f'echo "{tool} $*" >> "{stubs}/calls.log"\nexit 99\n')
    out = run_script(
        "pod/ablation_arm.sh",
        "--dry-run",
        LABEL,
        SPEC,
        env={"PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(tmp_path)},
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert stub_calls(stubs) == ""


@pytest.mark.parametrize(
    "args",
    [
        (),
        (LABEL,),
        (LABEL, SPEC, "extra"),
        ("../evil", SPEC),
        ("a b", SPEC),
        ("-x", SPEC),
        (LABEL, "experts layers"),
        (LABEL, "experts;rm"),
        (LABEL, "$(id)"),
        (LABEL, "Experts"),
        (LABEL, ""),
        ("--bogus", LABEL, SPEC),
    ],
)
def test_bad_arguments_are_refused_with_exit_2_before_anything_runs(abl, args):
    out = abl.run(*args)
    assert out.returncode == 2, out.stdout + out.stderr
    assert "ABLATION_FAILED" in out.stdout and "usage:" in out.stderr
    assert abl.steps() == []
    assert list((abl.work / "results").glob("ablation-*")) == []


@pytest.mark.parametrize(
    ("env", "reason"),
    [
        ({"ABLATION_RECIPE": "int4"}, "ABLATION_RECIPE"),
        ({"ABLATION_RECIPE": "nvfp4_w4a4"}, "ABLATION_RECIPE"),
        ({"ABLATION_NOISE": "2"}, "ABLATION_NOISE"),
        ({"ABLATION_KEEP_CKPT": "yes"}, "ABLATION_KEEP_CKPT"),
    ],
)
def test_bad_environment_is_refused(abl, env, reason):
    out = abl.run(LABEL, SPEC, **env)
    assert out.returncode == 2 and reason in out.stdout + out.stderr
    assert abl.steps() == []


def test_an_existing_result_directory_is_never_overwritten(abl):
    abl.out.mkdir(parents=True)
    (abl.out / "summary.json").write_text("{}")
    out = abl.run(LABEL, SPEC)
    assert out.returncode == 2 and "exists" in out.stdout + out.stderr
    assert abl.steps() == [] and (abl.out / "summary.json").read_text() == "{}"
    assert not abl.done.exists()


# --- the run ---------------------------------------------------------------------------------------------------------


def test_a_run_converts_verifies_serves_smokes_checks_collects_and_compares_in_that_order(abl):
    abl.make_reference()
    out = abl.run(LABEL, SPEC)
    assert out.returncode == 0, out.stdout + out.stderr
    assert last_line(out) == "ABLATION_DONE"
    assert abl.names() == FULL_WITH_REFERENCE
    w = abl.work
    assert (
        abl.step("ablation expected")
        == f"ablation expected --src {w}/models/olala-bf16 --select {SPEC}"
    )
    assert abl.step("convert") == (
        f"convert --src {w}/models/olala-bf16 --dst {abl.ckpt} --recipe nvfp4 --device cuda --select {SPEC}"
    )
    assert abl.step("verify") == (
        f"verify --src {w}/models/olala-bf16 --dst {abl.ckpt} --report {abl.out}/verify.json --select {SPEC}"
    )
    assert abl.step("serve_olala.sh") == f"serve_olala.sh ckpt {abl.ckpt}"
    assert abl.step("smoke.sh") == f"smoke.sh ckpt-abl-{LABEL} --timeout 900"
    check = abl.step("ablation check")
    assert check.startswith(
        f"ablation check --src {w}/models/olala-bf16 --dst {abl.ckpt} --select {SPEC} --server-log "
    )
    assert check.endswith(f"/logs/ckpt-abl-{LABEL}/server-1.log")
    collect = abl.step("fidelity collect")
    assert "--chunks data/wikitext2_test_64x512.json" in collect
    assert "--base-url http://127.0.0.1:8000/v1 --model olala" in collect
    assert collect.endswith(f"--out {w}/results/raw/{LABEL}_wikitext2.npz")
    compare = abl.step("fidelity compare")
    assert compare.startswith(
        f"fidelity compare {abl.ref} {w}/results/raw/{LABEL}_wikitext2.npz --json "
    )
    assert compare.endswith(f"{abl.out}/fidelity_vs_b0.json")


def test_the_recipe_is_chosen_by_env_and_passed_to_convert(abl):
    abl.make_reference()
    out = abl.run(LABEL, SPEC, ABLATION_RECIPE="fp8")
    assert out.returncode == 0 and "--recipe fp8" in abl.step("convert")


def test_the_results_are_the_kl_the_perplexity_ratio_and_the_top1_agreement_with_their_intervals(
    abl,
):
    abl.make_reference()
    abl.run(LABEL, SPEC)
    s = abl.summary()
    assert s["label"] == LABEL and s["spec"] == SPEC and s["recipe"] == "nvfp4"
    assert s["fidelity"]["kld_mean"] == COMPARE["kld_mean"]
    assert s["fidelity"]["kld_p99"] == COMPARE["kld_p99"]
    assert s["fidelity"]["ppl_ratio"] == COMPARE["ppl_ratio"]
    assert s["fidelity"]["top1_agreement"] == COMPARE["top1_agreement"]
    assert s["fidelity"]["n_chunks"] == 64 and s["fidelity"]["n_positions"] == 32768
    assert s["expected"] == EXPECTED
    assert s["status"] == "done" and s["verify"] == "OK" and s["check"] == "ABLATION_CHECK_OK"
    assert s["reference"]["path"] == str(abl.ref) and len(s["reference"]["sha256"]) == 64
    assert s["gpu"].startswith("NVIDIA RTX PRO 4500")
    assert (abl.out / "fidelity_vs_b0.json").is_file() and (abl.out / "fidelity_vs_b0.md").is_file()
    assert "Perplexity ratio" in (abl.out / "fidelity_vs_b0.md").read_text()
    assert (abl.out / "expected.json").is_file() and (abl.out / "verify.json").is_file()
    assert (abl.out / "smoke.txt").is_file()


def test_the_headline_numbers_are_printed_in_the_log(abl):
    abl.make_reference()
    out = abl.run(LABEL, SPEC)
    for needle in ("Perplexity ratio", "Mean KL", "99th percentile KL", "top-1"):
        assert needle in out.stdout, needle


def test_the_done_marker_the_collector_waits_for_is_written_after_the_results(abl):
    abl.make_reference()
    assert not abl.done.exists()
    abl.run(LABEL, SPEC)
    assert abl.done.read_text().strip() == "ABLATION_DONE"
    assert (abl.out / "summary.json").is_file()


def test_the_provenance_records_the_checkpoint_hashes_the_verdicts_and_the_kernel_banners(abl):
    abl.make_reference()
    abl.run(LABEL, SPEC, SERVER_LOG_TEXT="INFO Using MARLIN NvFp4 MoE backend\nINFO noise line\n")
    prov = (abl.out / "provenance.txt").read_text()
    assert f"run={LABEL} spec={SPEC} recipe=nvfp4" in prov
    assert "NVIDIA RTX PRO 4500" in prov
    for name in ("model.safetensors.index.json", "config.json", "quantization_info.json"):
        assert name in prov
    assert "verify: OK: 945 tensors checked" in prov
    assert "Using MARLIN NvFp4 MoE backend" in prov and "noise line" not in prov


def test_a_run_that_has_a_reference_does_not_serve_bf16(abl):
    abl.make_reference()
    abl.run(LABEL, SPEC)
    assert "serve_olala.sh bf16" not in abl.steps()
    assert not any("b0_wikitext2" in s and s.startswith("fidelity collect") for s in abl.steps())


def test_the_reference_is_produced_once_per_pod_by_serving_bf16_first(abl):
    out = abl.run(LABEL, SPEC)
    assert out.returncode == 0, out.stdout + out.stderr
    names = abl.names()
    assert names[:8] == [
        "stop_server.sh",
        "ablation expected",
        "serve_olala.sh",
        "smoke.sh",
        "fidelity collect",
        "fidelity collect",
        "stop_server.sh",
        "fidelity compare",
    ]
    assert abl.steps()[2] == "serve_olala.sh bf16"
    assert abl.steps()[3] == "smoke.sh bf16 --timeout 900"
    assert abl.steps()[4].endswith("/results/raw/b0_wikitext2.npz")
    assert abl.steps()[5].endswith(
        "/results/raw/b0repeat_wikitext2.npz"
    )  # the noise floor of the same server
    assert abl.steps()[7].startswith(
        f"fidelity compare {abl.ref} {abl.work}/results/raw/b0repeat_wikitext2.npz"
    )
    assert abl.names()[8:] == FULL_WITH_REFERENCE[2:]
    assert abl.ref.is_file()
    s = abl.summary()
    assert s["reference"]["produced_here"] is True
    assert s["noise_floor"]["kld_mean"] == COMPARE["kld_mean"]["value"]
    assert s["noise_floor"]["top1_agreement"] == COMPARE["top1_agreement"]["value"]
    assert (abl.out / "reference" / "b0_wikitext2.npz").is_file()
    assert (abl.out / "reference" / "noise_floor.json").is_file()
    # a second run on the same pod reuses it: no bf16 serve, and it does not claim the noise floor
    abl.calls.write_text("")
    out2 = abl.run("abl-second", "mamba")
    assert out2.returncode == 0, out2.stdout + out2.stderr
    assert "serve_olala.sh bf16" not in abl.steps()
    s2 = json.loads((abl.work / "results" / "ablation-abl-second" / "summary.json").read_text())
    assert s2["reference"]["produced_here"] is False and "noise_floor" not in s2


def test_the_noise_floor_can_be_switched_off(abl):
    out = abl.run(LABEL, SPEC, ABLATION_NOISE="0")
    assert out.returncode == 0, out.stdout + out.stderr
    assert not any("b0repeat" in s for s in abl.steps())
    assert "noise_floor" not in abl.summary()


def test_a_reference_given_by_path_is_used_and_not_produced(abl, tmp_path):
    given = tmp_path / "elsewhere" / "ref.npz"
    given.parent.mkdir()
    given.write_bytes(b"given")
    out = abl.run(LABEL, SPEC, ABLATION_REF=str(given))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "serve_olala.sh bf16" not in abl.steps()
    assert abl.step("fidelity compare").startswith(f"fidelity compare {given} ")
    assert abl.summary()["reference"]["path"] == str(given)
    assert abl.summary()["reference"]["produced_here"] is False


def test_a_reference_path_that_does_not_exist_is_a_failure_before_any_gpu_work(abl, tmp_path):
    out = abl.run(LABEL, SPEC, ABLATION_REF=str(tmp_path / "missing.npz"))
    assert out.returncode == 3, out.stdout + out.stderr
    assert last_line(out) == "ABLATION_FAILED"
    assert "serve_olala.sh" not in " ".join(abl.steps()) and "convert" not in abl.names()
    assert abl.done.read_text().startswith("ABLATION_FAILED")


def test_a_checkpoint_is_removed_after_a_good_run_unless_asked_to_keep_it(abl):
    abl.make_reference()
    abl.run(LABEL, SPEC)
    assert not abl.ckpt.exists()
    assert (abl.out / "quantization_info.json").is_file() and (abl.out / "config.json").is_file()
    assert (abl.out / "model.safetensors.index.json").is_file()
    abl.out.rename(abl.out.with_name("moved"))
    abl.done.unlink()
    abl.run(LABEL, SPEC, ABLATION_KEEP_CKPT="1")
    assert abl.ckpt.is_dir() and (abl.ckpt / "model-00001-of-00001.safetensors").is_file()


def test_a_leftover_checkpoint_directory_is_a_failure_not_an_overwrite(abl):
    abl.make_reference()
    abl.ckpt.mkdir(parents=True)
    (abl.ckpt / "precious.txt").write_text("x")
    out = abl.run(LABEL, SPEC)
    assert out.returncode == 1 and last_line(out) == "ABLATION_FAILED"
    assert (abl.ckpt / "precious.txt").exists()
    assert "convert" not in abl.names()


# --- failures: every one is loud, stops the run, stops the server, and leaves a marker for the collector ---------------


FAILURES = [
    ({"EXPECTED_RC": "2"}, "selection", ["stop_server.sh", "ablation expected"]),
    ({"PATCH_RC": "1"}, "loader patch", ["stop_server.sh", "ablation expected", "apply_patch"]),
    (
        {"CONVERT_RC": "1"},
        "convert",
        ["stop_server.sh", "ablation expected", "apply_patch", "convert"],
    ),
    (
        {"VERIFY_RC": "1"},
        "verify",
        ["stop_server.sh", "ablation expected", "apply_patch", "convert", "verify"],
    ),
    (
        {"SERVE_RC": "1"},
        "serve",
        [
            "stop_server.sh",
            "ablation expected",
            "apply_patch",
            "convert",
            "verify",
            "serve_olala.sh",
            "stop_server.sh",
        ],
    ),
    (
        {"SMOKE_RC": "1"},
        "smoke",
        [
            "stop_server.sh",
            "ablation expected",
            "apply_patch",
            "convert",
            "verify",
            "serve_olala.sh",
            "smoke.sh",
            "stop_server.sh",
        ],
    ),
    (
        {"CHECK_RC": "1"},
        "check",
        [
            "stop_server.sh",
            "ablation expected",
            "apply_patch",
            "convert",
            "verify",
            "serve_olala.sh",
            "smoke.sh",
            "ablation check",
            "stop_server.sh",
        ],
    ),
    (
        {"COLLECT_RC": "1"},
        "fidelity collect",
        [
            "stop_server.sh",
            "ablation expected",
            "apply_patch",
            "convert",
            "verify",
            "serve_olala.sh",
            "smoke.sh",
            "ablation check",
            "fidelity collect",
            "stop_server.sh",
        ],
    ),
    (
        {"COLLECT_NO_FILE": "1"},
        "no logprob file",
        [
            "stop_server.sh",
            "ablation expected",
            "apply_patch",
            "convert",
            "verify",
            "serve_olala.sh",
            "smoke.sh",
            "ablation check",
            "fidelity collect",
            "stop_server.sh",
        ],
    ),
    (
        {"COMPARE_RC": "1"},
        "fidelity compare",
        FULL_WITH_REFERENCE,
    ),
]


@pytest.mark.parametrize(
    ("env", "reason", "expected_steps"), FAILURES, ids=[f[1] for f in FAILURES]
)
def test_a_failed_step_stops_the_run_leaves_a_failed_marker_and_the_server_stopped(
    abl, env, reason, expected_steps
):
    abl.make_reference()
    out = abl.run(LABEL, SPEC, **env)
    assert out.returncode == 1, out.stdout + out.stderr
    assert last_line(out) == "ABLATION_FAILED"
    assert reason in out.stdout
    assert abl.names() == expected_steps
    assert abl.done.read_text().startswith("ABLATION_FAILED") and reason in abl.done.read_text()
    s = abl.summary()
    assert s["status"] == "failed" and reason in s["reason"]
    assert "ABLATION_DONE" not in out.stdout
    assert (abl.out / "provenance.txt").is_file()


def test_a_failed_reference_stops_the_run_with_exit_3_and_never_converts(abl):
    out = abl.run(LABEL, SPEC, REF_SMOKE_RC="1")
    assert out.returncode == 3, out.stdout + out.stderr
    assert last_line(out) == "ABLATION_FAILED" and "reference" in out.stdout
    assert "convert" not in abl.names() and abl.names()[-1] == "stop_server.sh"


def test_a_reference_collection_that_wrote_nothing_is_a_failure(abl):
    out = abl.run(LABEL, SPEC, COLLECT_NO_FILE="1")
    assert out.returncode == 3 and "reference" in out.stdout
    assert not abl.ref.exists() and "convert" not in abl.names()


def test_a_smoke_with_warnings_is_accepted(abl):
    abl.make_reference()
    stub_script(
        abl.pod / "smoke.sh",
        f'echo "smoke.sh $*" >> "{abl.calls}"\necho "smoke verdict"\nexit 2\n',
    )
    out = abl.run(LABEL, SPEC)
    assert out.returncode == 0 and last_line(out) == "ABLATION_DONE"


def test_a_stale_done_marker_of_an_earlier_attempt_cannot_let_the_collector_pull_an_unfinished_run(
    abl,
):
    abl.make_reference()
    abl.done.write_text("ABLATION_DONE\n")
    out = abl.run(LABEL, SPEC, CONVERT_RC="1")
    assert out.returncode == 1
    assert abl.done.read_text().startswith("ABLATION_FAILED")


def test_the_check_receives_the_newest_server_log_of_this_checkpoint(abl):
    abl.make_reference()
    abl.run(LABEL, SPEC)
    assert (
        abl.step("ablation check").split("--server-log ")[1]
        == f"{abl.work}/logs/ckpt-abl-{LABEL}/server-1.log"
    )


# --- the batch -------------------------------------------------------------------------------------------------------


def batch(abl, *pairs: str, **env: str) -> subprocess.CompletedProcess[str]:
    return abl.run(*pairs, script="ablation_batch.sh", **env)


def test_the_batch_runs_each_label_and_spec_pair_in_order_and_reports_the_tally(abl):
    abl.make_reference()
    out = batch(abl, "abl-a", "experts", "abl-b", "mamba", "abl-c", "attention")
    assert out.returncode == 0, out.stdout + out.stderr
    assert last_line(out) == "ABLATION_BATCH_DONE ok=3 failed=0 skipped=0"
    selects = [s.split("--select ")[1] for s in abl.steps() if s.startswith("convert")]
    assert selects == ["experts", "mamba", "attention"]
    for label in ("abl-a", "abl-b", "abl-c"):
        assert (
            abl.work / "results" / f"ablation-{label}.done"
        ).read_text().strip() == "ABLATION_DONE"


def test_the_batch_goes_on_after_a_failed_spec_and_says_so(abl):
    abl.make_reference()
    out = batch(abl, "abl-a", "experts", "abl-b", "mamba", CONVERT_RC="1")  # every convert fails
    assert out.returncode == 1
    assert last_line(out) == "ABLATION_BATCH_DONE ok=0 failed=2 skipped=0"
    assert len([s for s in abl.steps() if s.startswith("convert")]) == 2


def test_the_batch_stops_when_the_reference_cannot_be_made(abl):
    out = batch(abl, "abl-a", "experts", "abl-b", "mamba", REF_SMOKE_RC="1")
    assert out.returncode == 3
    assert (
        len([s for s in abl.steps() if s.startswith("ablation expected")]) == 1
    )  # the second never started
    assert last_line(out) == "ABLATION_BATCH_DONE ok=0 failed=1 skipped=1"
    assert (abl.work / "results" / "ablation-abl-b.done").read_text().startswith("ABLATION_FAILED")


def test_the_batch_makes_the_reference_once(abl):
    out = batch(abl, "abl-a", "experts", "abl-b", "mamba")
    assert out.returncode == 0, out.stdout + out.stderr
    assert abl.steps().count("serve_olala.sh bf16") == 1


def test_the_batch_does_not_start_a_spec_after_its_deadline(abl):
    abl.make_reference()
    out = batch(abl, "abl-a", "experts", "abl-b", "mamba", ABLATION_DEADLINE_MIN="0")
    assert out.returncode == 1
    assert last_line(out) == "ABLATION_BATCH_DONE ok=0 failed=0 skipped=2"
    assert abl.steps() == []
    for label in ("abl-a", "abl-b"):
        assert "deadline" in (abl.work / "results" / f"ablation-{label}.done").read_text()
        assert (
            (abl.work / "results" / f"ablation-{label}.done")
            .read_text()
            .startswith("ABLATION_FAILED")
        )


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("only-a-label",),
        ("a", "experts", "b"),
        ("a", "experts", "a", "mamba"),
        ("a b", "experts"),
    ],
)
def test_the_batch_refuses_odd_argument_lists_and_repeated_labels(abl, args):
    out = batch(abl, *args)
    assert out.returncode == 2 and "usage:" in out.stderr
    assert abl.steps() == []


def test_the_batch_refuses_a_deadline_that_is_not_a_number(abl):
    out = batch(abl, "a", "experts", ABLATION_DEADLINE_MIN="soon")
    assert out.returncode == 2 and "ABLATION_DEADLINE_MIN" in out.stdout + out.stderr


# --- the scripts themselves --------------------------------------------------------------------------------------------

SCRIPTS = ["pod/ablation_arm.sh", "pod/ablation_batch.sh"]   # the Mac-side launcher is not part of the release


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_scripts_have_valid_bash_syntax(script):
    check = subprocess.run(
        ["bash", "-n", str(RUNPOD / script)], capture_output=True, text=True, check=False
    )
    assert check.returncode == 0, check.stderr


@pytest.mark.skipif(shellcheck_command() is None, reason="neither shellcheck nor uvx is installed")
@pytest.mark.parametrize("script", SCRIPTS)
def test_the_scripts_are_shellcheck_clean(script):
    out = subprocess.run(
        [*shellcheck_command(), "-x", "-P", "SCRIPTDIR", str(RUNPOD / script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout


def test_the_pod_scripts_kill_nothing_by_text_and_read_no_secret():
    for script in SCRIPTS:
        text = (RUNPOD / script).read_text()
        assert "pkill" not in text and "killall" not in text, script
        assert "security find-generic-password" not in text and "HF_TOKEN" not in text, script


def test_no_done_marker_exists_while_a_run_is_in_progress(abl):
    abl.make_reference()
    abl.done.write_text("ABLATION_DONE\n")  # left by an earlier attempt with the same label
    out = abl.run(LABEL, SPEC, DONE_PROBE=str(abl.done))
    assert out.returncode == 0, out.stdout + out.stderr
    assert not any(step.startswith("probe:") for step in abl.steps())
