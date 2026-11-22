"""pod/fidelity_batch.sh: the stage 1 batch of PROTOCOL.md amendment 25, without a GPU, a network or a pod.

The script runs from a copy of pod/ whose /workspace is a temp folder and whose steps (stop_server.sh, serve_olala.sh,
smoke.sh, the loader patch, olala_quant.convert, .verify and .ablation, evals/fidelity.py) are stubs that record their
arguments in one calls log; fidelity_gate.py is the real one. The converter stub writes a checkpoint directory whose shard
has a size set per arm, and the comparison stub reports a KL and a perplexity ratio set per arm, so that the gate and the
choice of the candidate are exercised end to end.
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
LABEL = "fx"
ARMS = ["N5", "C1", "C2", "C3"]
# amendment 25's default batch: (name, spec, --scale-search)
DEFAULT = [
    ("N5", "experts", False),
    ("C1", "experts-all", False),
    ("C2", "experts-all", True),
    ("C3", "experts", True),
]
# what the stubs report unless a test says otherwise: C1 misses the gate, C2 and C3 pass, C2 is smaller than C3
KL = {"N5": "0.0243", "C1": "0.0310", "C2": "0.0285", "C3": "0.0230"}
PPL = {"N5": "1.0197", "C1": "1.0250", "C2": "1.0240", "C3": "1.0190"}
SIZE = {"N5": 6000, "C1": 5000, "C2": 5000, "C3": 6000}
EXPECTED = {"quantized_tensors": 72, "ignored_tensors": 187, "quantized_layers": 36}


def stub_script(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Fidelity:
    """A pod's /workspace in a temp folder, every step of fidelity_batch.sh replaced by a recording stub."""

    def __init__(self, tmp: Path) -> None:
        self.work = tmp / "ws"
        self.pod = self.work / "bundle" / "pod"
        self.calls = tmp / "calls.log"
        self.calls.write_text("")
        shutil.copytree(POD, self.pod)
        for name in ("fidelity_batch.sh", "job_common.sh"):  # they hard-code /workspace
            path = self.pod / name
            path.write_text(path.read_text().replace("/workspace", str(self.work)))
        c = str(self.calls)
        with (self.pod / "job_common.sh").open("a") as f:
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
[ -z "${{SERVE_FAIL:-}}" ] || case "$2" in *"-${{SERVE_FAIL}}") exit 1 ;; esac
exit 0
""",
        )
        stub_script(
            self.pod / "smoke.sh",
            f"""echo "smoke.sh $*" >> "{c}"
echo "smoke verdict $1"
case "$1" in bf16) exit "${{REF_SMOKE_RC:-0}}" ;; *) [ -z "${{SMOKE_FAIL:-}}" ] || case "$1" in *"-${{SMOKE_FAIL}}") exit 1 ;; esac ;; esac
exit 0
""",
        )
        pkg = self.work / "bundle" / "quant" / "olala_quant"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "convert.py").write_text(
            f"""import json, os, pathlib, sys
a = sys.argv[1:]
open({c!r}, "a").write("convert " + " ".join(a) + "\\n")
dst = pathlib.Path(a[a.index("--dst") + 1])
name = dst.name.split("-", 2)[2]
open({c!r}, "a").write("ckpts-at-convert " + " ".join(sorted(os.listdir(dst.parent))) + "\\n")
if os.environ.get("DONE_PROBE") and os.path.exists(os.environ["DONE_PROBE"]):
    open({c!r}, "a").write("probe: a .done exists while the run is in progress\\n")
print("convert says hi")
if os.environ.get("CONVERT_FAIL") == name:
    print("convert boom"); sys.exit(1)
dst.mkdir(parents=True, exist_ok=True)
info = {{"selection": {{"spec": a[a.index("--select") + 1]}}}}
if "--scale-search" in a:
    info["scale_search"] = {{"factors": [1.0, 0.95, 0.9, 0.85, 0.8, 0.75], "version": 1}}
(dst / "quantization_info.json").write_text(json.dumps(info))
(dst / "model.safetensors.index.json").write_text("{{}}")
(dst / "config.json").write_text("{{}}")
(dst / "tokenizer.json").write_bytes(b"t" * 50)
(dst / "model-00001-of-00001.safetensors").write_bytes(b"w" * int(os.environ.get("SIZE_" + name, "1000")))
"""
        )
        (pkg / "verify.py").write_text(
            f"""import json, os, pathlib, sys
a = sys.argv[1:]
open({c!r}, "a").write("verify " + " ".join(a) + "\\n")
dst = pathlib.Path(a[a.index("--dst") + 1])
info = json.loads((dst / "quantization_info.json").read_text())
if "--scale-search" in a and "scale_search" not in info:
    print("[info_scale_search] required and absent", file=sys.stderr); sys.exit(1)
if "--no-scale-search" in a and "scale_search" in info:
    print("[info_scale_search] forbidden and present", file=sys.stderr); sys.exit(1)
if os.environ.get("VERIFY_FAIL") == dst.name.split("-", 2)[2]:
    print("verify violations", file=sys.stderr); sys.exit(1)
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
    if os.environ.get("EXPECTED_FAIL") == spec:
        print("error: unknown selection", file=sys.stderr); sys.exit(2)
    print(json.dumps(dict({expected}, spec=spec)))
else:
    dst = a[a.index("--dst") + 1]
    if os.environ.get("CHECK_FAIL") == dst.rsplit("-", 1)[-1]:
        print("MISMATCH: 3 quantized tensors on disk, expected 72", file=sys.stderr); sys.exit(1)
    print("ABLATION_CHECK_OK " + spec)
"""
        )
        evals = self.work / "bundle" / "evals"
        evals.mkdir(parents=True)
        (evals / "fidelity.py").write_text(
            f"""import json, os, pathlib, sys
a = sys.argv[1:]
open({c!r}, "a").write("fidelity " + " ".join(a) + "\\n")
KL = {KL!r}; PPL = {PPL!r}
if a[0] == "collect":
    out = pathlib.Path(a[a.index("--out") + 1])
    if os.environ.get("REF_COLLECT_FAIL") and "b0_" in out.name:
        print("reference collect boom"); sys.exit(1)
    if os.environ.get("COLLECT_FAIL") and out.name.startswith("fid-fx-" + os.environ["COLLECT_FAIL"] + "_"):
        print("collect boom"); sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not (os.environ.get("COLLECT_NO_FILE") and "b0" not in out.name):
        out.write_bytes(b"npz:" + out.name.encode())
    print("wrote " + str(out) + ": (64, 511) positions, k=20")
else:
    var = pathlib.Path(a[2]).name
    name = var.split("_")[0].split("-", 2)[2] if var.startswith("fid-") else None
    if os.environ.get("COMPARE_FAIL") and os.environ["COMPARE_FAIL"] == name:
        print("compare boom"); sys.exit(1)
    if name:
        open({c!r}, "a").write("ckpts-at-compare " + name + ": " + " ".join(sorted(os.listdir({str(self.work / "ckpt")!r}))) + "\\n")
    kl = float(os.environ.get("KL_" + str(name), KL.get(name, "0.0120")))
    ppl = float(os.environ.get("PPL_" + str(name), PPL.get(name, "1.0100")))
    def m(v): return {{"value": v, "lo": v * 0.95, "hi": v * 1.05}}
    doc = {{"ppl_ref": m(10.0), "ppl_var": m(10.2), "ppl_ratio": m(ppl), "kld_mean": m(kl), "kld_median": m(0.004),
           "kld_p99": m(0.21), "top1_agreement": m(0.951), "n_chunks": 64, "n_positions": 32704, "k": 20}}
    pathlib.Path(a[a.index("--json") + 1]).write_text(json.dumps(doc))
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
        self.results = self.work / "results"
        self.out = self.results / f"fidelity-{LABEL}"
        self.done = self.results / f"fidelity-{LABEL}.done"
        self.ref = self.results / "raw" / "b0_wikitext2.npz"
        self.ckpt = self.work / "ckpt"
        self.selected = self.ckpt / "selected"

    def make_reference(self) -> None:
        self.ref.parent.mkdir(parents=True, exist_ok=True)
        self.ref.write_bytes(b"a reference collected earlier on this pod")

    def run(
        self, *args: str, script: str = "fidelity_batch.sh", **env: str
    ) -> subprocess.CompletedProcess[str]:
        sizes = {f"SIZE_{k}": str(v) for k, v in SIZE.items()}
        return subprocess.run(
            ["bash", str(self.pod / script), *args],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                **sizes,
                "FIDELITY_STOP_WAIT": "0",
                **env,
                "PATH": f"{self.bin}:{os.environ['PATH']}",
            },
            check=False,
            timeout=120,
        )

    def steps(self) -> list[str]:
        return [ln for ln in self.calls.read_text().splitlines() if ln]

    def selects(self, prefix: str) -> list[str]:
        return [s for s in self.steps() if s.startswith(prefix)]

    def summary(self) -> dict:
        return json.loads((self.out / "summary.json").read_text())

    def selection(self) -> dict:
        return json.loads((self.out / "selection.json").read_text())

    def arm(self, name: str) -> dict:
        return {a["name"]: a for a in self.summary()["arms"]}[name]


@pytest.fixture
def fid(tmp_path):
    return Fidelity(tmp_path)


def last_line(out: subprocess.CompletedProcess[str]) -> str:
    return out.stdout.strip().splitlines()[-1]


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def good_run(fid: Fidelity, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    fid.make_reference()
    out = fid.run(LABEL, *args, **env)
    assert out.returncode in (0, 1), out.stdout + out.stderr
    return out


# --- dry run and arguments ---------------------------------------------------------------------------------------------


def test_dry_run_prints_the_whole_plan_and_touches_nothing(fid):
    before = tree(fid.work)
    out = fid.run("--dry-run", LABEL)
    assert out.returncode == 0, out.stdout + out.stderr
    for needle in (
        "arms N5,C1,C2,C3",
        "arm N5 (control): --select experts",
        "arm C1 (candidate): --select experts-all",
        "arm C2 (candidate): --select experts-all --scale-search",
        "arm C3 (candidate): --select experts --scale-search",
        "olala_quant.ablation expected",
        "olala_quant.convert",
        "olala_quant.verify",
        "olala_quant.ablation check",
        "fidelity.py collect",
        "fidelity.py compare",
        "fidelity_gate.py pick",
        f"{fid.work}/ckpt/selected",
        f"{fid.work}/results/fidelity-{LABEL}.done",
        "FIDELITY_DONE",
    ):
        assert needle in out.stdout, needle
    assert fid.steps() == [] and tree(fid.work) == before


def test_dry_run_uses_no_network_secret_or_gpu_tool(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for tool in ("security", "curl", "ssh", "scp", "nvidia-smi", "vllm"):
        stub_script(stubs / tool, f'echo "{tool} $*" >> "{stubs}/calls.log"\nexit 99\n')
    out = run_script(
        "pod/fidelity_batch.sh",
        "--dry-run",
        LABEL,
        env={"PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(tmp_path)},
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert stub_calls(stubs) == ""


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("-x",),
        ("a b",),
        ("../evil",),
        (LABEL, "Experts"),
        (LABEL, "experts layers"),
        (LABEL, "experts;rm"),
        (LABEL, "$(id)"),
        (LABEL, "experts:scale-search:scale-search"),
        (LABEL, "experts:layers=1-"),
        (LABEL, "N5=experts"),  # the control's name
        (LABEL, "A=experts", "A=experts-all"),
        (LABEL, "bad name=experts"),
        (LABEL, "--bogus"),
        (LABEL, ""),
    ],
)
def test_bad_arguments_are_refused_with_exit_2_before_anything_runs(fid, args):
    out = fid.run(*args)
    assert out.returncode == 2, out.stdout + out.stderr
    assert "FIDELITY_FAILED" in out.stdout and "usage:" in out.stderr
    assert fid.steps() == []
    assert not (fid.results / f"fidelity-{LABEL}").exists()


@pytest.mark.parametrize(
    ("env", "reason"),
    [
        ({"FIDELITY_NOISE": "2"}, "FIDELITY_NOISE"),
        ({"FIDELITY_DEADLINE_MIN": "soon"}, "FIDELITY_DEADLINE_MIN"),
        ({"FIDELITY_STOP_WAIT": "x"}, "FIDELITY_STOP_WAIT"),
    ],
)
def test_bad_environment_is_refused(fid, env, reason):
    out = fid.run(LABEL, **env)
    assert out.returncode == 2 and reason in out.stdout + out.stderr
    assert fid.steps() == []


def test_an_existing_result_directory_or_selected_checkpoint_is_never_overwritten(fid):
    fid.out.mkdir(parents=True)
    (fid.out / "summary.json").write_text("{}")
    out = fid.run(LABEL)
    assert out.returncode == 2 and "exists" in out.stdout + out.stderr
    assert (
        fid.steps() == []
        and (fid.out / "summary.json").read_text() == "{}"
        and not fid.done.exists()
    )
    shutil.rmtree(fid.out)
    fid.selected.mkdir(parents=True)
    (fid.selected / "precious").write_text("x")
    out = fid.run(LABEL)
    assert out.returncode == 2 and "selected" in out.stdout + out.stderr
    assert (fid.selected / "precious").exists() and fid.steps() == []


# --- the default batch --------------------------------------------------------------------------------------------------


def test_the_default_batch_converts_the_four_arms_of_amendment_25_with_the_right_selection_and_flag(
    fid,
):
    out = good_run(fid)
    assert out.returncode == 0, out.stdout + out.stderr
    assert last_line(out) == "FIDELITY_DONE"
    converts = fid.selects("convert ")
    assert len(converts) == 4
    for line, (name, spec, search) in zip(converts, DEFAULT, strict=True):
        assert line == (
            f"convert --src {fid.work}/models/olala-bf16 --dst {fid.ckpt}/fid-{LABEL}-{name} --recipe nvfp4 "
            f"--device cuda --select {spec}{' --scale-search' if search else ''}"
        )
    verifies = fid.selects("verify ")
    for line, (name, spec, search) in zip(verifies, DEFAULT, strict=True):
        assert line == (
            f"verify --src {fid.work}/models/olala-bf16 --dst {fid.ckpt}/fid-{LABEL}-{name} "
            f"--report {fid.out}/arms/{name}/verify.json --select {spec} "
            f"{'--scale-search' if search else '--no-scale-search'}"
        )
    for line, (name, spec, _) in zip(fid.selects("ablation check"), DEFAULT, strict=True):
        assert f"--dst {fid.ckpt}/fid-{LABEL}-{name} --select {spec} --server-log " in line
        assert line.endswith(f"/logs/ckpt-fid-{LABEL}-{name}/server-1.log")
    assert [s for s in fid.steps() if s.startswith("serve_olala.sh")] == [
        f"serve_olala.sh ckpt {fid.ckpt}/fid-{LABEL}-{n}" for n in ARMS
    ]
    assert [s for s in fid.steps() if s.startswith("smoke.sh")] == [
        f"smoke.sh ckpt-fid-{LABEL}-{n} --timeout 900" for n in ARMS
    ]


def test_each_arm_runs_its_steps_in_the_order_of_the_ablation_script(fid):
    good_run(fid)
    names = [
        s.split()[0] if not s.startswith(("ablation", "fidelity")) else " ".join(s.split()[:2])
        for s in fid.steps()
    ]
    names = [n for n in names if not n.startswith(("ckpts-", "probe"))]
    expected_all = ["stop_server.sh"] + ["ablation expected"] * 4
    assert names[:5] == expected_all  # every selection is checked first, before any GPU work
    one_arm = [
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
    assert names[5:] == one_arm * 4


def test_the_results_have_the_summary_the_selection_the_arms_and_the_marker(fid):
    good_run(fid)
    s = fid.summary()
    assert s["label"] == LABEL and s["status"] == "done" and s["selected"] == "C2"
    assert [a["name"] for a in s["arms"]] == ARMS
    assert (
        s["thresholds"]["mean_kl_max"] == "0.0304" and s["thresholds"]["ppl_ratio_max"] == "1.030"
    )
    assert "amendment 25" in s["thresholds"]["source"]
    for a in s["arms"]:
        assert a["status"] == "done" and a["verify"] == "OK" and a["check"] == "ABLATION_CHECK_OK"
        assert (
            a["file_bytes"] == SIZE[a["name"]]
        )  # the size of the shards of the checkpoint, measured before it is removed
        assert (
            a["dir_bytes"] > a["file_bytes"]
        )  # the whole directory: index, config, tokenizer and the rest
        for key in ("kld_mean", "kld_median", "kld_p99", "ppl_ratio", "top1_agreement"):
            assert set(a["fidelity"][key]) == {"value", "lo", "hi"}
        assert a["fidelity"]["kld_mean"]["value"] == float(KL[a["name"]])
        assert a["verdict"]["passes"] is (a["name"] != "C1")
        assert (fid.out / "arms" / a["name"] / "arm.json").is_file()
        for f in (
            "expected.json",
            "verify.json",
            "verify.txt",
            "check.txt",
            "smoke.txt",
            "provenance.txt",
            "fidelity_vs_b0.json",
            "fidelity_vs_b0.md",
            "server.log",
            "quantization_info.json",
            "config.json",
        ):
            assert (fid.out / "arms" / a["name"] / f).is_file(), (a["name"], f)
    assert fid.arm("N5")["role"] == "control" and fid.arm("N5")["verdict"]["passes"] is True
    assert (
        fid.arm("C1")["verdict"]["mean_kl"]["ok"] is False
        and "mean KL" in fid.arm("C1")["verdict"]["reason"]
    )
    sel = fid.selection()
    assert (
        sel["selected"] == "C2"
        and sel["checkpoint"] == f"{fid.work}/ckpt/selected"
        and sel["passing"] == ["C2", "C3"]
    )
    assert (
        sel["spec"] == "experts-all" and sel["scale_search"] is True and sel["file_bytes"] == 5000
    )
    assert fid.done.read_text().strip() == "FIDELITY_DONE selected=C2"


def test_the_gate_is_applied_at_the_exact_limits(fid):
    out = good_run(
        fid,
        KL_C1="0.0304",
        PPL_C1="1.030",
        KL_C2="0.03040000000000001",
        PPL_C3="1.0300000000000002",
    )
    s = fid.summary()
    by = {a["name"]: a for a in s["arms"]}
    assert by["C1"]["verdict"]["passes"] is True  # both exactly at their limits
    assert by["C2"]["verdict"]["passes"] is False  # one unit in the last place above on the KL
    assert by["C3"]["verdict"]["passes"] is False  # and on the perplexity ratio
    assert s["selected"] == "C1" and fid.selection()["selected"] == "C1"
    assert out.returncode == 0


def test_the_smallest_passing_candidate_is_selected_not_the_best_nor_the_largest(fid):
    env = {
        "SIZE_C1": "9000",
        "SIZE_C2": "8000",
        "SIZE_C3": "7000",
        "KL_C1": "0.0100",
        "KL_C2": "0.0200",
        "KL_C3": "0.0300",
    }
    good_run(fid, **env)
    assert fid.selection()["selected"] == "C3"  # the largest file has the lowest KL: not chosen
    assert fid.arm("C3")["file_bytes"] == 7000


def test_a_tie_on_size_goes_to_the_lowest_kl(fid):
    good_run(
        fid,
        SIZE_C1="5000",
        SIZE_C2="5000",
        SIZE_C3="5000",
        KL_C1="0.0290",
        KL_C2="0.0280",
        KL_C3="0.0270",
    )
    assert fid.selection()["selected"] == "C3"


def test_the_control_is_never_selected_even_when_it_is_the_smallest(fid):
    good_run(fid, SIZE_N5="10", KL_C1="0.0100")
    assert fid.selection()["selected"] != "N5"
    assert not (fid.ckpt / f"fid-{LABEL}-N5").exists()


# --- the checkpoints ------------------------------------------------------------------------------------------------------


def test_each_checkpoint_is_kept_until_its_fidelity_is_measured(fid):
    good_run(fid)
    for name in ARMS:
        listing = next(s for s in fid.steps() if s.startswith(f"ckpts-at-compare {name}:"))
        assert f"fid-{LABEL}-{name}" in listing.split(":", 1)[1].split(), (name, listing)
    # the candidates before it are still there when the next one is converted (kept until the end), the control's is not
    at_convert = [s for s in fid.steps() if s.startswith("ckpts-at-convert")]
    assert f"fid-{LABEL}-N5" not in at_convert[1] and f"fid-{LABEL}-C1" in at_convert[2].split()
    assert f"fid-{LABEL}-C1" in at_convert[3].split() and f"fid-{LABEL}-C2" in at_convert[3].split()


def test_the_selected_checkpoint_stays_on_the_pod_under_ckpt_selected_and_the_others_are_removed(
    fid,
):
    out = good_run(fid)
    assert out.returncode == 0
    assert sorted(p.name for p in fid.ckpt.iterdir()) == ["selected"]
    shard = fid.selected / "model-00001-of-00001.safetensors"
    assert (
        shard.stat().st_size == SIZE["C2"]
    )  # C2's shard, moved: its quantization_info still says experts-all + search
    info = json.loads((fid.selected / "quantization_info.json").read_text())
    assert info["selection"]["spec"] == "experts-all" and "scale_search" in info
    assert not list(fid.out.rglob("*.safetensors")), (
        "no checkpoint bytes under the results folder that is pulled home"
    )
    assert not list(fid.out.rglob("model-*")), "not even by name"


def test_with_no_passing_candidate_nothing_is_kept_and_the_selection_is_none(fid):
    out = good_run(fid, KL_C1="0.05", KL_C2="0.04", KL_C3="0.0305")
    assert out.returncode == 0, out.stdout + out.stderr
    assert fid.selection()["selected"] == "none" and fid.selection()["checkpoint"] is None
    assert fid.summary()["selected"] == "none"
    assert not fid.selected.exists() and list(fid.ckpt.iterdir()) == []
    assert "selected=none" in fid.done.read_text() and "selected=none" in out.stdout


def test_the_batch_moves_the_winner_and_never_copies_a_checkpoint_anywhere(fid):
    text = (POD / "fidelity_batch.sh").read_text()
    for forbidden in ("scp", "rsync", "cp -r", "cp -R", "pod_ssh", "tar "):
        assert forbidden not in text, forbidden
    assert "mv -- " in text


# --- the marker ---------------------------------------------------------------------------------------------------------------


def test_the_done_marker_is_written_last_and_never_while_the_run_is_in_progress(fid):
    fid.make_reference()
    fid.done.write_text(
        "FIDELITY_DONE selected=old\n"
    )  # left by an earlier attempt with the same label
    out = fid.run(LABEL, DONE_PROBE=str(fid.done))
    assert out.returncode == 0, out.stdout + out.stderr
    assert not any(s.startswith("probe:") for s in fid.steps())
    done_ns = fid.done.stat().st_mtime_ns
    for f in (
        fid.out / "summary.json",
        fid.out / "selection.json",
        fid.selected / "quantization_info.json",
    ):
        assert f.stat().st_mtime_ns <= done_ns, f
    assert fid.done.read_text().strip() == "FIDELITY_DONE selected=C2"


def test_the_markers_of_the_run_are_printed_in_order(fid):
    out = good_run(fid)
    lines = out.stdout.splitlines()
    arm_done = [ln for ln in lines if ln.startswith("FIDELITY_ARM_DONE")]
    assert arm_done == [f"FIDELITY_ARM_DONE {n}" for n in ARMS]
    batch = [ln for ln in lines if ln.startswith("FIDELITY_BATCH_DONE")]
    assert batch == ["FIDELITY_BATCH_DONE ok=4 failed=0 skipped=0 selected=C2"]
    assert lines.index(batch[0]) > lines.index(arm_done[-1]) and last_line(out) == "FIDELITY_DONE"
    for needle in ("Perplexity ratio", "Mean KL", "99th percentile KL", "top-1"):
        assert needle in out.stdout, needle


def test_the_provenance_records_the_checkpoint_hashes_and_the_kernel_banners(fid):
    good_run(fid)
    prov = (fid.out / "arms" / "C2" / "provenance.txt").read_text()
    assert "arm=C2 role=candidate spec=experts-all scale_search=1" in prov
    for name in ("model.safetensors.index.json", "config.json", "quantization_info.json"):
        assert name in prov
    assert "verify: OK: 945 tensors checked" in prov and "Using MARLIN NvFp4 MoE backend" in prov
    assert "NVIDIA RTX PRO 4500" in (fid.out / "provenance.txt").read_text()


# --- the reference ------------------------------------------------------------------------------------------------------------


def test_the_reference_is_produced_once_by_serving_bf16_first_when_there_is_none(fid):
    out = fid.run(LABEL)
    assert out.returncode == 0, out.stdout + out.stderr
    steps = fid.steps()
    assert steps.count("serve_olala.sh bf16") == 1
    i = steps.index("serve_olala.sh bf16")
    assert steps[i + 1] == "smoke.sh bf16 --timeout 900"
    assert steps[i + 2].endswith("/results/raw/b0_wikitext2.npz") and steps[i + 3].endswith(
        "/results/raw/b0repeat_wikitext2.npz"
    )
    assert not any(s.startswith("convert") for s in steps[:i])
    assert fid.ref.is_file() and (fid.out / "reference" / "b0_wikitext2.npz").is_file()
    assert (fid.out / "reference" / "noise_floor.json").is_file()


def test_a_reference_that_exists_is_used_and_bf16_is_not_served(fid):
    good_run(fid)
    assert "serve_olala.sh bf16" not in fid.steps()
    assert all(
        s.startswith(f"fidelity compare {fid.ref} ") for s in fid.selects("fidelity compare")
    )


def test_a_reference_given_by_path_is_used_and_must_exist(fid, tmp_path):
    given = tmp_path / "elsewhere" / "ref.npz"
    given.parent.mkdir()
    given.write_bytes(b"given")
    out = fid.run(LABEL, FIDELITY_REF=str(given))
    assert out.returncode == 0, out.stdout + out.stderr
    assert fid.selects("fidelity compare")[0].startswith(f"fidelity compare {given} ")
    shutil.rmtree(fid.out)
    fid.done.unlink()
    shutil.rmtree(fid.ckpt, ignore_errors=True)
    fid.calls.write_text("")
    out = fid.run(LABEL, FIDELITY_REF=str(tmp_path / "missing.npz"))
    assert out.returncode == 3 and "FIDELITY_FAILED" in out.stdout
    assert fid.done.read_text().startswith("FIDELITY_FAILED") and not fid.selects("convert")


def test_a_failed_reference_stops_the_batch_with_exit_3_and_never_converts(fid):
    out = fid.run(LABEL, REF_SMOKE_RC="1")
    assert out.returncode == 3, out.stdout + out.stderr
    assert last_line(out) == "FIDELITY_FAILED" and "reference" in out.stdout
    assert not fid.selects("convert") and fid.steps()[-1] == "stop_server.sh"
    assert (
        fid.done.read_text().startswith("FIDELITY_FAILED") and "reference" in fid.done.read_text()
    )
    assert fid.summary()["selected"] == "none" and fid.selection()["selected"] == "none"
    assert {a["status"] for a in fid.summary()["arms"]} == {"skipped"}


# --- failures ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "reason"),
    [
        ({"CONVERT_FAIL": "C2"}, "convert"),
        ({"VERIFY_FAIL": "C2"}, "verify"),
        ({"SERVE_FAIL": "C2"}, "serve"),
        ({"SMOKE_FAIL": "C2"}, "smoke"),
        ({"CHECK_FAIL": "C2"}, "check"),
        ({"COLLECT_FAIL": "C2"}, "fidelity collect"),
        ({"COLLECT_NO_FILE": "1", "COLLECT_FAIL": ""}, "no logprob file"),
        ({"COMPARE_FAIL": "C2"}, "fidelity compare"),
    ],
)
def test_a_failed_step_fails_that_arm_only_and_the_batch_goes_on(fid, env, reason):
    out = good_run(fid, **env)
    if "COLLECT_NO_FILE" in env:  # every arm fails the same way
        assert out.returncode == 1 and fid.summary()["selected"] == "none"
        return
    assert out.returncode == 1, out.stdout + out.stderr
    assert f"FIDELITY_ARM_FAILED C2: {reason}" in out.stdout or reason in "\n".join(
        ln for ln in out.stdout.splitlines() if ln.startswith("FIDELITY_ARM_FAILED C2")
    )
    c2 = fid.arm("C2")
    assert c2["status"] == "failed" and reason in c2["reason"] and c2["verdict"]["passes"] is False
    assert {fid.arm(n)["status"] for n in ("N5", "C1", "C3")} == {"done"}  # the other arms ran
    assert (
        fid.selection()["selected"] == "C3"
    )  # C2 cannot be chosen; C3 is the next smallest that passes
    assert "FIDELITY_BATCH_DONE ok=3 failed=1 skipped=0 selected=C3" in out.stdout
    assert last_line(out) == "FIDELITY_DONE" and fid.done.read_text().startswith("FIDELITY_DONE")
    assert (fid.out / "arms" / "C2" / "arm.json").is_file()


def test_a_failed_arm_does_not_leave_a_server_running(fid):
    good_run(fid, CHECK_FAIL="C2")
    steps = fid.steps()
    i = max(
        i
        for i, s in enumerate(steps)
        if s.startswith("ablation check") and s.endswith("server-1.log") and "-C2" in s
    )
    assert (
        steps[i + 1] == "stop_server.sh"
    )  # stopped right after the failed check, before the next arm's patch


def test_a_selection_the_code_refuses_stops_everything_before_any_gpu_work(fid):
    fid.make_reference()
    out = fid.run(LABEL, EXPECTED_FAIL="experts-all")
    assert out.returncode == 1, out.stdout + out.stderr
    assert last_line(out) == "FIDELITY_FAILED" and "selection refused" in out.stdout
    assert fid.done.read_text().startswith("FIDELITY_FAILED")
    assert not any(s.startswith(("convert", "serve_olala", "apply_patch")) for s in fid.steps())


def test_a_leftover_checkpoint_directory_is_a_failure_of_that_arm_not_an_overwrite(fid):
    fid.make_reference()
    leftover = fid.ckpt / f"fid-{LABEL}-C1"
    leftover.mkdir(parents=True)
    (leftover / "precious.txt").write_text("x")
    out = fid.run(LABEL)
    assert out.returncode == 1
    assert (leftover / "precious.txt").exists()
    assert fid.arm("C1")["status"] == "failed" and "exists" in fid.arm("C1")["reason"]
    assert [s for s in fid.selects("convert ") if f"fid-{LABEL}-C1" in s] == []


def test_the_loader_patch_is_applied_before_each_conversion_and_its_failure_fails_the_arm(fid):
    good_run(fid)
    steps = fid.steps()
    converts = [i for i, s in enumerate(steps) if s.startswith("convert ")]
    assert len(converts) == 4 and all(
        steps[i - 2] == "apply_patch" or steps[i - 1] == "apply_patch" for i in converts
    )
    assert steps.count("apply_patch") == 4


def test_a_failed_loader_patch_fails_every_arm_without_converting(fid):
    out = good_run(fid, PATCH_RC="1")
    assert out.returncode == 1 and "loader patch" in out.stdout
    assert not fid.selects("convert ") and fid.summary()["selected"] == "none"


def test_the_deadline_skips_every_arm_and_says_so(fid):
    out = good_run(fid, FIDELITY_DEADLINE_MIN="0")
    assert out.returncode == 1
    assert "FIDELITY_BATCH_DONE ok=0 failed=0 skipped=4 selected=none" in out.stdout
    assert not fid.selects("convert")
    for a in fid.summary()["arms"]:
        assert a["status"] == "skipped" and "deadline" in a["reason"]
    assert fid.done.read_text().startswith("FIDELITY_DONE selected=none")


def test_a_smoke_with_warnings_is_accepted(fid):
    fid.make_reference()
    stub_script(
        fid.pod / "smoke.sh", f'echo "smoke.sh $*" >> "{fid.calls}"\necho "smoke verdict"\nexit 2\n'
    )
    out = fid.run(LABEL)
    assert out.returncode == 0 and last_line(out) == "FIDELITY_DONE"


# --- custom candidates ---------------------------------------------------------------------------------------------------------


def test_custom_candidates_name_their_spec_and_option_and_get_c1_c2_by_default(fid):
    fid.make_reference()
    out = fid.run(
        LABEL, "experts-all", "X=experts:layers=12-23:scale-search", "experts:layers=0-11"
    )
    assert out.returncode == 0, out.stdout + out.stderr
    converts = fid.selects("convert ")
    assert [c.split("--select ")[1] for c in converts] == [
        "experts",
        "experts-all",
        "experts:layers=12-23 --scale-search",
        "experts:layers=0-11",
    ]
    assert [c.split("--dst ")[1].split()[0].rsplit("-", 1)[1] for c in converts] == [
        "N5",
        "C1",
        "X",
        "C3",
    ]
    assert [a["name"] for a in fid.summary()["arms"]] == ["N5", "C1", "X", "C3"]
    assert fid.arm("X")["scale_search"] is True and fid.arm("X")["spec"] == "experts:layers=12-23"


# --- the scripts themselves ------------------------------------------------------------------------------------------------------

SCRIPTS = ["pod/fidelity_batch.sh"]   # the Mac-side launcher is not part of the release


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


def test_the_pod_scripts_kill_nothing_by_text_and_read_no_secret_and_hold_no_threshold():
    for script in SCRIPTS:
        text = (RUNPOD / script).read_text()
        assert "pkill" not in text and "killall" not in text, script
        assert "security find-generic-password" not in text and "HF_TOKEN" not in text, script
        assert "0.0304" not in text and "1.030" not in text, (
            "the thresholds live in fidelity_gate.py only"
        )


def test_the_batch_does_not_edit_or_source_the_older_ablation_scripts():
    text = (POD / "fidelity_batch.sh").read_text()
    assert "ablation_arm.sh" in text  # named in the header: why its steps are copied
    assert 'bash "$here/ablation_arm.sh"' not in text and "sed -i" not in text
