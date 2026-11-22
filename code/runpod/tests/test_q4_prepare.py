"""pod/q4_prepare.sh: arm Q4 of amendment 24, the quantization step. No network, no GPU, no pod.

The script runs from a COPY of runpod/pod (so that the edit of the pod's manifest lands in the copy: a test proves the
repository's own gguf_files.json is untouched) in a fake /workspace. gguf_setup.sh is a stub; cmake, llama-quantize and
llama-imatrix are stubs that record their arguments; python3 is a wrapper that gives q4_tool.py a fake huggingface_hub
and a fake pyarrow (the tool itself is the real one).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import RUNPOD, stub_calls, stub_env
from test_q4_tool import GOOD_ROWS, quantize_log, tiny_gguf

POD_SRC = RUNPOD / "pod"
MANIFEST_SRC = POD_SRC / "gguf_files.json"
SCRIPT = "q4_prepare.sh"
VARIANT = "q4_k_m-ovhrecipe"
Q4_NAME = "Olala-v1.0-mini-7A1B-q4_k_m-ovhrecipe.gguf"
BF16_NAME = "Olala-v1.0-mini-7A1B-bf16.gguf"
Q4_BYTES = b"pretend this is a 4-bit gguf"
IMATRIX_BYTES = b"pretend this is an imatrix"
ROWS = [" \n", " = Valkyria Chronicles III = \n", "\n", "Senjo no Valkyria 3 : café\n"]
COMMIT = json.loads(MANIFEST_SRC.read_text())["llama_cpp"]["commit"]

HELP_QUANT = """usage: ./llama-quantize [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights]
       [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--tensor-type] [--tensor-type-file]
       model-f32.gguf [model-quant.gguf] type [nthreads]

  --imatrix file_name
                                      use data in file_name as importance matrix for quant optimizations
  --tensor-type tensor_name=ggml_type
                                      quantize this tensor to this ggml_type
  --tensor-type-file tensor_types.txt
                                      list of tensors to quantize to a specific ggml_type
"""
HELP_IMATRIX = """----- common params -----
-h,    --help, --usage                  print usage and exit
-m,    --model FNAME                    model path
-f,    --file FNAME                     a file containing the prompt
--chunks N                              max number of chunks to process (default: -1, -1 = all)
-ngl,  --gpu-layers N                   max. number of layers to store in VRAM

example usage:

    llama-imatrix -m model.gguf -f some-text.txt [-o imatrix.gguf] [--chunk 123] [--process-output]
"""

QUANT_STUB = r"""#!/bin/bash
echo "llama-quantize $*" >> "$STUB_LOG"
if [ "${1:-}" = "--help" ]; then cat "$STUB_QUANT_HELP"; exit 1; fi
echo "state: q4.done=$([ -e "$OLALA_WORK/results/q4.done" ] && echo present || echo absent)" >> "$STUB_LOG"
pos=()
while [ $# -gt 0 ]; do
  case $1 in
    --imatrix|--tensor-type) shift 2 ;;
    --tensor-type=*) shift ;;
    *) pos+=("$1"); shift ;;
  esac
done
[ "${STUB_QUANT_RC:-0}" = 0 ] || { echo "llama_model_quantize: failed" >&2; exit "$STUB_QUANT_RC"; }
cp "$STUB_Q4_BYTES" "${pos[1]}"
cat "$STUB_QUANT_LOG" >&2
exit "${STUB_QUANT_LATE_RC:-0}"
"""

IMATRIX_STUB = r"""#!/bin/bash
echo "llama-imatrix $*" >> "$STUB_LOG"
if [ "${1:-}" = "--help" ]; then cat "$STUB_IMATRIX_HELP"; exit 0; fi
echo "state: q4.done=$([ -e "$OLALA_WORK/results/q4.done" ] && echo present || echo absent)" >> "$STUB_LOG"
[ "${STUB_IMATRIX_RC:-0}" = 0 ] || exit "$STUB_IMATRIX_RC"
out=""
while [ $# -gt 0 ]; do case $1 in -o) out=$2; shift 2 ;; *) shift ;; esac; done
cp "$STUB_IMATRIX_BYTES" "$out"
echo "main: computing over ${STUB_CHUNKS:-200} chunks, n_ctx=512, batch_size=2048, n_seq=4" >&2
"""

SETUP_STUB = r"""#!/bin/bash
echo "gguf_setup.sh $*" >> "$STUB_LOG"
[ -z "${STUB_SETUP_FAIL:-}" ] || { echo "GGUF_SETUP_FAILED: $STUB_SETUP_FAIL"; exit 1; }
mkdir -p "$OLALA_WORK/models/olala-gguf" "$OLALA_WORK/llama.cpp-olala/build" "$OLALA_WORK/logs/gguf"
cp "$STUB_BF16" "$OLALA_WORK/models/olala-gguf/Olala-v1.0-mini-7A1B-bf16.gguf"
cat > "$OLALA_WORK/llama.cpp-olala/build/olala_build_record.txt" << EOF
commit=$STUB_COMMIT
tree=ff04ce13ba15d503c6f50b625550a3b31924adcf
repo=https://github.com/OVHai-LLM/llama.cpp.git
cmake_flags=-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a-real
cuda_arch=120a-real
nvcc=13.0 (/usr/local/cuda/bin/nvcc)
EOF
echo GGUF_SETUP_DONE
"""

CMAKE_STUB = r"""#!/bin/bash
echo "cmake $*" >> "$STUB_LOG"
if [ "$1" = "--build" ]; then
  [ -z "${STUB_CMAKE_FAIL:-}" ] || exit 1
  mkdir -p "$2/bin"
  cp "$STUB_TOOLS/llama-quantize" "$STUB_TOOLS/llama-imatrix" "$2/bin/"
fi
"""

PYTHON_STUB = r"""#!/bin/bash
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pip" ]; then echo "pip $*" >> "$STUB_LOG"; touch "$FAKE_PIP_DONE"; exit 0; fi
case "$*" in
  *q4_tool.py*) exec "$REAL_PY" -c "$FAKE_BOOT" "$@" ;;
  *) exec "$REAL_PY" "$@" ;;
esac
"""

FAKE_BOOT = r"""
import json, os, runpy, sys, types
i = [k for k, a in enumerate(sys.argv) if a.endswith("q4_tool.py")][0]
script = sys.argv[i]
sys.argv = sys.argv[i:]
if sys.argv[1] == "report":
    done = os.path.join(os.environ["OLALA_WORK"], "results", "q4.done")
    with open(os.environ["STUB_LOG"], "a") as f:
        f.write("state: report q4.done=" + ("present" if os.path.exists(done) else "absent") + "\n")
def hf_hub_download(**kw):
    kw["HF_TOKEN_SEEN"] = os.environ.get("HF_TOKEN")
    with open(os.environ["FAKE_HF_LOG"], "a") as f:
        f.write(json.dumps(kw) + "\n")
    return os.environ["FAKE_PARQUET"]
hub = types.ModuleType("huggingface_hub")
hub.hf_hub_download = hf_hub_download
sys.modules["huggingface_hub"] = hub
if not (os.environ.get("FAKE_NO_PYARROW") and not os.path.exists(os.environ["FAKE_PIP_DONE"])):
    class Table:
        def __init__(self, rows): self.rows = rows
        def column(self, name): return self
        def to_pylist(self): return self.rows
    pq = types.ModuleType("pyarrow.parquet")
    pq.read_table = lambda path, columns=None: Table(json.load(open(path)))
    pa = types.ModuleType("pyarrow")
    pa.parquet = pq
    sys.modules["pyarrow"] = pa
    sys.modules["pyarrow.parquet"] = pq
runpy.run_path(script, run_name="__main__")
"""


def write_exec(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Q4Pod:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.work = tmp / "ws"
        self.pod = tmp / "pod"
        self.log = tmp / "calls.log"
        self.log.write_text("")
        shutil.copytree(POD_SRC, self.pod, ignore=shutil.ignore_patterns("__pycache__"))
        write_exec(self.pod / "gguf_setup.sh", SETUP_STUB)
        bindir, tools = tmp / "bin", tmp / "tools"
        bindir.mkdir()
        tools.mkdir()
        write_exec(bindir / "cmake", CMAKE_STUB)
        write_exec(bindir / "python3", PYTHON_STUB)
        write_exec(tools / "llama-quantize", QUANT_STUB)
        write_exec(tools / "llama-imatrix", IMATRIX_STUB)
        self.quant_help, self.imatrix_help = tmp / "quant.help", tmp / "imatrix.help"
        self.quant_help.write_text(HELP_QUANT)
        self.imatrix_help.write_text(HELP_IMATRIX)
        self.quant_log = tmp / "quantize.log.txt"
        self.quant_log.write_text(quantize_log(GOOD_ROWS))
        (tmp / "q4.bytes").write_bytes(Q4_BYTES)
        (tmp / "imatrix.bytes").write_bytes(IMATRIX_BYTES)
        self.parquet = tmp / "train.parquet"
        self.parquet.write_text(json.dumps(ROWS))
        self.hf_log = tmp / "hf.log"
        self.bf16 = tiny_gguf(tmp / "bf16.gguf", [[1000, 1000], [500]])
        (tmp / "cg").mkdir()
        (tmp / "cuda").mkdir()
        self.env = {
            "OLALA_WORK": str(self.work),
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "REAL_PY": sys.executable,
            "FAKE_BOOT": FAKE_BOOT,
            "FAKE_PARQUET": str(self.parquet),
            "FAKE_HF_LOG": str(self.hf_log),
            "FAKE_PIP_DONE": str(tmp / "pip.done"),
            "STUB_LOG": str(self.log),
            "STUB_TOOLS": str(tools),
            "STUB_QUANT_HELP": str(self.quant_help),
            "STUB_IMATRIX_HELP": str(self.imatrix_help),
            "STUB_QUANT_LOG": str(self.quant_log),
            "STUB_Q4_BYTES": str(tmp / "q4.bytes"),
            "STUB_IMATRIX_BYTES": str(tmp / "imatrix.bytes"),
            "STUB_BF16": str(self.bf16),
            "STUB_COMMIT": COMMIT,
            "OLALA_REPO_RUNPOD": str(
                RUNPOD
            ),  # where the Mac's own files are: the script must never touch them
            "OLALA_NPROC": "3",
            "CGROUP_DIR": str(tmp / "cg"),
            "CUDA_HOME": str(tmp / "cuda"),
            "HF_TOKEN": "must-not-leak",
        }

    # paths of the fake workspace
    @property
    def q4(self) -> Path:
        return self.work / "q4"

    @property
    def results(self) -> Path:
        return self.work / "results"

    @property
    def done(self) -> Path:
        return self.results / "q4.done"

    @property
    def out_file(self) -> Path:
        return self.work / "models" / "olala-gguf" / Q4_NAME

    @property
    def build(self) -> Path:
        return self.work / "llama.cpp-olala" / "build"

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

    def report(self) -> dict:
        return json.loads((self.results / "q4" / "prepare.json").read_text())

    def done_first_line(self) -> str:
        return self.done.read_text().splitlines()[0]


@pytest.fixture
def q4(tmp_path):
    return Q4Pod(tmp_path)


def real_calls(calls: list[str]) -> list[str]:
    """The calls that are not state probes of the stub tools."""
    return [c for c in calls if not c.startswith("state:")]


def kinds(calls: list[str]) -> list[str]:
    out = []
    for c in real_calls(calls):
        if c.startswith("cmake --build"):
            out.append("build")
        elif c.startswith("gguf_setup.sh"):
            out.append("setup")
        elif c.startswith(("llama-quantize --help", "llama-imatrix --help")):
            out.append("help:" + c.split()[0].removeprefix("llama-"))
        elif c.startswith("llama-imatrix"):
            out.append("imatrix")
        elif c.startswith("llama-quantize"):
            out.append("quantize")
    return out


MAKE = ("--imatrix", "make")


def file_args(path: Path) -> tuple[str, ...]:
    return ("--imatrix", str(path), "--imatrix-sha", hashlib.sha256(path.read_bytes()).hexdigest())


# --- arguments ------------------------------------------------------------------------------------------------


def test_the_script_parses_and_its_header_states_how_the_text_is_made():
    path = POD_SRC / SCRIPT
    assert subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0
    header = "\n".join(ln for ln in path.read_text().splitlines()[:40] if ln.startswith("#"))
    assert "no separator" in header and "wiki.train.raw" in header and "concatenat" in header


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("--imatrix",),
        ("--imatrix", "make", "--imatrix-sha", "a" * 64),
        ("--imatrix", "/x/olala.imatrix"),
        ("--imatrix", "/x/olala.imatrix", "--imatrix-sha", "zz"),
        ("--imatrix", "/x/olala.imatrix", "--imatrix-sha", "a" * 63),
        ("--imatrix", "make", "--bogus"),
        ("make",),
    ],
)
def test_bad_arguments_are_refused_with_a_marker_and_nothing_is_run_or_written(q4, args):
    out = q4.run(*args)
    assert out.returncode == 2
    assert out.stdout.strip().splitlines()[-1].startswith("Q4_PREPARE_FAILED:BAD_ARGUMENTS")
    assert q4.calls() == [] and not q4.work.exists()


# --- dry run --------------------------------------------------------------------------------------------------


def test_the_dry_run_prints_the_plan_in_order_runs_nothing_and_writes_nothing(
    q4, tmp_path, stub_bin
):
    out = q4.run("--dry-run", *MAKE, **stub_env(stub_bin, tmp_path / "home"))
    assert out.returncode == 0, out.stderr
    assert q4.calls() == [] and stub_calls(stub_bin) == "" and not q4.work.exists()
    steps = [ln for ln in out.stdout.splitlines() if ln.startswith("== ")]
    words = [
        "bf16",
        "extra targets",
        "help",
        "wikitext",
        "imatrix",
        "quantize",
        "manifest",
        "prepare.json",
    ]
    text = " | ".join(steps).lower()
    positions = [text.index(w) for w in words]
    assert positions == sorted(positions), text
    assert out.stdout.strip().splitlines()[-1] == "Q4_PREPARE_DRY_RUN (nothing was run)"


def test_the_dry_run_shows_the_exact_commands_of_the_recipe(q4):
    out = q4.run("--dry-run", *MAKE)
    text = out.stdout
    build, work = q4.build, q4.work
    assert "gguf_setup.sh --variant bf16" in text
    assert f"cmake --build {build} --target llama-quantize llama-imatrix -j 3" in text
    assert (
        f"{build}/bin/llama-imatrix -m {work}/models/olala-gguf/{BF16_NAME} -f {work}/q4/wiki.train.raw "
        f"-o {work}/q4/olala.imatrix --chunks 200 -ngl 99"
    ) in text
    assert (
        f"{build}/bin/llama-quantize --imatrix {work}/q4/olala.imatrix --tensor-type ffn_latent_up=bf16 "
        f"--tensor-type ffn_up_exps=q8_0 {work}/models/olala-gguf/{BF16_NAME} "
        f"{work}/models/olala-gguf/{Q4_NAME} Q4_K_M 3"
    ) in text
    assert "Salesforce/wikitext" in text and "b08601e04326c79dfdd32d625aee71d232d685c3" in text
    assert "wikitext-2-raw-v1/train-00000-of-00001.parquet" in text


def test_the_dry_run_with_a_file_verifies_instead_of_making(q4, tmp_path):
    path = tmp_path / "olala.imatrix"
    path.write_bytes(IMATRIX_BYTES)
    text = q4.run("--dry-run", *file_args(path)).stdout
    assert "llama-imatrix -m" not in text and "sha256" in text and str(path) in text


# --- the order of a real run ----------------------------------------------------------------------------------


def test_the_steps_run_in_the_order_setup_build_help_imatrix_quantize(q4):
    out = q4.run(*MAKE)
    assert out.returncode == 0, out.stdout + out.stderr
    assert kinds(q4.calls()) == [
        "setup",
        "build",
        "help:quantize",
        "help:imatrix",
        "imatrix",
        "quantize",
    ]
    assert out.stdout.strip().splitlines()[-1] == "Q4_PREPARE_DONE"


def test_setup_is_the_idempotent_script_for_the_bf16_variant_and_the_extra_targets_use_its_build_directory(
    q4,
):
    assert q4.run(*MAKE).returncode == 0
    calls = real_calls(q4.calls())
    assert calls[0] == "gguf_setup.sh --variant bf16"
    build = next(c for c in calls if c.startswith("cmake --build"))
    assert build == f"cmake --build {q4.build} --target llama-quantize llama-imatrix -j 3"
    assert (q4.build / "bin" / "llama-quantize").exists()


def test_the_commit_and_the_flags_come_from_the_record_of_the_build(q4):
    assert q4.run(*MAKE).returncode == 0
    lc = q4.report()["llama_cpp"]
    assert lc["commit"] == COMMIT and "-DGGML_CUDA=ON" in lc["cmake_flags"]
    assert lc["cuda_arch"] == "120a-real" and lc["jobs"] == "3"


def test_a_failed_setup_stops_everything_with_the_reason(q4):
    out = q4.run(*MAKE, STUB_SETUP_FAIL="no GPU")
    assert out.returncode == 1
    assert out.stdout.strip().splitlines()[-1].startswith("Q4_PREPARE_FAILED:SETUP_BF16")
    assert kinds(q4.calls()) == ["setup"]
    assert q4.done_first_line().startswith("Q4_PREPARE_FAILED:SETUP_BF16")


def test_a_failed_build_of_the_extra_targets_stops_before_any_run(q4):
    out = q4.run(*MAKE, STUB_CMAKE_FAIL="1")
    assert out.returncode == 1 and "Q4_PREPARE_FAILED:BUILD_EXTRA" in out.stdout
    assert kinds(q4.calls()) == ["setup", "build"]


# --- the help check -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("which", "flag"),
    [
        ("quant", "--tensor-type"),
        ("quant", "--imatrix"),
        ("imatrix", "--chunks"),
    ],
)
def test_a_flag_missing_from_the_help_is_refused_before_anything_is_run(q4, which, flag):
    path = q4.quant_help if which == "quant" else q4.imatrix_help
    text = path.read_text()
    # --tensor-type-file and --chunk stay: only the exact flag is removed
    path.write_text(
        "\n".join(
            ln.replace(f"{flag} ", "--other ").replace(f"[{flag}]", "[--other]")
            if flag in ln
            else ln
            for ln in text.splitlines()
        )
        + "\n"
    )
    out = q4.run(*MAKE)
    assert out.returncode == 1
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("Q4_PREPARE_FAILED:FLAG_NOT_ACCEPTED") and flag in last
    assert "imatrix" not in kinds(q4.calls()) and "quantize" not in kinds(q4.calls())
    assert q4.done_first_line() == last
    assert (q4.work / "logs" / "q4" / "llama-quantize.help.txt").is_file()


def test_the_help_output_is_kept_in_the_logs_and_in_the_results(q4):
    assert q4.run(*MAKE).returncode == 0
    for name, text in (("llama-quantize", HELP_QUANT), ("llama-imatrix", HELP_IMATRIX)):
        assert (q4.work / "logs" / "q4" / f"{name}.help.txt").read_text() == text
        assert (q4.results / "q4" / "logs" / f"{name}.help.txt").read_text() == text


def test_the_tensor_type_syntax_follows_the_help(q4):
    q4.quant_help.write_text(
        HELP_QUANT.replace(
            "--tensor-type tensor_name=ggml_type", "--tensor-type=tensor_name=ggml_type"
        )
    )
    assert q4.run(*MAKE).returncode == 0
    quant = next(c for c in q4.calls() if c.startswith("llama-quantize --imatrix"))
    assert "--tensor-type=ffn_latent_up=bf16 --tensor-type=ffn_up_exps=q8_0" in quant


# --- the importance matrix ------------------------------------------------------------------------------------


def test_make_writes_the_text_from_the_pinned_parquet_and_records_its_sha(q4):
    assert q4.run(*MAKE).returncode == 0
    text = q4.q4 / "wiki.train.raw"
    assert text.read_bytes() == "".join(ROWS).encode()
    hf = json.loads(q4.hf_log.read_text().splitlines()[0])
    assert hf["repo_id"] == "Salesforce/wikitext" and hf["repo_type"] == "dataset"
    assert hf["revision"] == "b08601e04326c79dfdd32d625aee71d232d685c3"
    assert hf["filename"] == "wikitext-2-raw-v1/train-00000-of-00001.parquet"
    assert (
        hf["token"] is False and hf["HF_TOKEN_SEEN"] is None
    )  # no credential reaches the download
    assert q4.report()["wikitext"]["text_sha256"] == hashlib.sha256(text.read_bytes()).hexdigest()


def test_make_runs_llama_imatrix_with_200_chunks_on_the_bf16_file_and_nothing_else(q4):
    assert q4.run(*MAKE).returncode == 0
    cmd = next(c for c in q4.calls() if c.startswith("llama-imatrix -m"))
    work = q4.work
    assert cmd == (
        f"llama-imatrix -m {work}/models/olala-gguf/{BF16_NAME} -f {work}/q4/wiki.train.raw "
        f"-o {work}/q4/olala.imatrix --chunks 200 -ngl 99"
    )
    rep = q4.report()["imatrix"]
    assert rep["mode"] == "make" and rep["sha256"] == hashlib.sha256(IMATRIX_BYTES).hexdigest()
    assert (
        rep["command"].endswith("--chunks 200 -ngl 99")
        and "computing over 200 chunks" in rep["chunks_line"]
    )
    assert (work / "logs" / "q4" / "imatrix.log").is_file()
    assert (q4.results / "q4" / "olala.imatrix").read_bytes() == IMATRIX_BYTES


def test_an_imatrix_run_that_did_not_use_200_chunks_fails(q4):
    out = q4.run(*MAKE, STUB_CHUNKS="100")
    assert out.returncode == 1 and "Q4_PREPARE_FAILED:IMATRIX_CHUNKS" in out.stdout
    assert "quantize" not in kinds(q4.calls())


def test_a_failed_imatrix_run_stops_before_the_quantization(q4):
    out = q4.run(*MAKE, STUB_IMATRIX_RC="1")
    assert out.returncode == 1 and "Q4_PREPARE_FAILED:IMATRIX_FAILED" in out.stdout
    assert "quantize" not in kinds(q4.calls())


def test_a_pod_without_pyarrow_installs_it_once_and_goes_on(q4):
    out = q4.run(*MAKE, FAKE_NO_PYARROW="1")
    assert out.returncode == 0, out.stdout + out.stderr
    assert sum(1 for c in q4.calls() if c.startswith("pip ")) == 1


def test_a_given_imatrix_with_the_right_sha_is_used_and_no_imatrix_is_computed(q4, tmp_path):
    given = tmp_path / "given.imatrix"
    given.write_bytes(b"shipped from the first pod")
    out = q4.run(*file_args(given))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "imatrix" not in kinds(q4.calls()) and not q4.hf_log.exists()
    assert (q4.q4 / "olala.imatrix").read_bytes() == b"shipped from the first pod"
    quant = next(c for c in q4.calls() if c.startswith("llama-quantize --imatrix"))
    assert f"--imatrix {q4.q4}/olala.imatrix" in quant
    rep = q4.report()["imatrix"]
    assert rep["mode"] == "file" and rep["sha256"] == hashlib.sha256(given.read_bytes()).hexdigest()


def test_an_imatrix_sha_in_capitals_is_accepted(q4, tmp_path):
    given = tmp_path / "given.imatrix"
    given.write_bytes(b"x")
    assert (
        q4.run(
            "--imatrix", str(given), "--imatrix-sha", hashlib.sha256(b"x").hexdigest().upper()
        ).returncode
        == 0
    )


def test_a_given_imatrix_with_another_sha_stops_before_the_quantization(q4, tmp_path):
    given = tmp_path / "given.imatrix"
    given.write_bytes(b"not the one that was promised")
    out = q4.run("--imatrix", str(given), "--imatrix-sha", "0" * 64)
    assert out.returncode == 1 and "Q4_PREPARE_FAILED:IMATRIX_SHA_MISMATCH" in out.stdout
    assert "quantize" not in kinds(q4.calls()) and not q4.out_file.exists()
    assert q4.done_first_line().startswith("Q4_PREPARE_FAILED:IMATRIX_SHA_MISMATCH")


def test_a_given_imatrix_that_is_missing_is_refused(q4, tmp_path):
    out = q4.run("--imatrix", str(tmp_path / "nope"), "--imatrix-sha", "a" * 64)
    assert out.returncode == 1 and "Q4_PREPARE_FAILED:IMATRIX_MISSING" in out.stdout


def test_a_given_imatrix_already_at_the_working_path_is_verified_in_place(q4):
    q4.q4.mkdir(parents=True)
    target = q4.q4 / "olala.imatrix"
    target.write_bytes(b"shipped by the launcher")
    assert q4.run(*file_args(target)).returncode == 0
    assert target.read_bytes() == b"shipped by the launcher"


# --- the quantization -----------------------------------------------------------------------------------------


def test_quantize_gets_the_recipe_the_bf16_file_the_output_the_type_and_the_threads(q4):
    assert q4.run(*MAKE).returncode == 0
    cmd = next(c for c in q4.calls() if c.startswith("llama-quantize --imatrix"))
    work = q4.work
    assert cmd == (
        f"llama-quantize --imatrix {work}/q4/olala.imatrix --tensor-type ffn_latent_up=bf16 "
        f"--tensor-type ffn_up_exps=q8_0 {work}/models/olala-gguf/{BF16_NAME} "
        f"{work}/models/olala-gguf/{Q4_NAME} Q4_K_M 3"
    )


def test_the_log_sha_size_and_tensor_type_table_are_recorded(q4):
    assert q4.run(*MAKE).returncode == 0
    rep = q4.report()
    quant = rep["quantized"]
    assert quant["sha256"] == hashlib.sha256(Q4_BYTES).hexdigest() and quant["size_bytes"] == len(
        Q4_BYTES
    )
    assert quant["name"] == Q4_NAME and quant["path"] == str(q4.out_file)
    assert rep["quantize"]["command"].endswith("Q4_K_M 3") and rep["quantize"]["threads"] == "3"
    assert (q4.work / "logs" / "q4" / "quantize.log").read_text().count("converting to") >= 5
    assert rep["tensor_types"]["by_type"] == {"bf16": 2, "f32": 1, "q4_K": 2, "q6_K": 2, "q8_0": 2}
    assert rep["tensor_types"]["by_group"]["ffn_up_exps"] == {"q8_0": 2}
    assert (q4.results / "q4" / "tensor_types.json").is_file()


def test_bits_per_weight_use_the_parameter_count_of_the_bf16_header(q4):
    assert q4.run(*MAKE).returncode == 0
    quant = q4.report()["quantized"]
    assert quant["parameters"] == 1_000_500 and "header" in quant["parameters_source"]
    assert quant["bits_per_weight"] == pytest.approx(len(Q4_BYTES) * 8 / 1_000_500)


def test_an_unreadable_header_gives_the_stated_approximation(q4):
    q4.bf16.write_bytes(b"not a gguf")
    assert q4.run(*MAKE).returncode == 0
    quant = q4.report()["quantized"]
    assert quant["parameters"] == 6_766_000_000 and "approximate" in quant["parameters_source"]


def test_a_quantization_that_did_not_apply_the_recipe_fails_and_edits_nothing(q4):
    rows = [(n, s, "q4_K" if "ffn_up_exps" in n else f) for n, s, f in GOOD_ROWS]
    q4.quant_log.write_text(quantize_log(rows))
    before = q4.manifest.read_bytes()
    out = q4.run(*MAKE)
    assert out.returncode == 1 and "Q4_PREPARE_FAILED:RECIPE_NOT_APPLIED" in out.stdout
    assert q4.manifest.read_bytes() == before
    assert q4.done_first_line().startswith("Q4_PREPARE_FAILED:RECIPE_NOT_APPLIED")


def test_a_quantizer_that_fails_after_writing_the_file_and_the_whole_log_is_still_a_failure(q4):
    before = q4.manifest.read_bytes()
    out = q4.run(*MAKE, STUB_QUANT_LATE_RC="1")
    assert out.returncode == 1 and "Q4_PREPARE_FAILED:QUANTIZE_FAILED" in out.stdout
    assert not q4.out_file.exists() and q4.manifest.read_bytes() == before
    assert q4.done_first_line().startswith("Q4_PREPARE_FAILED:QUANTIZE_FAILED")


def test_a_failed_quantization_leaves_no_output_file_and_no_manifest_edit(q4):
    before = q4.manifest.read_bytes()
    out = q4.run(*MAKE, STUB_QUANT_RC="1")
    assert out.returncode == 1 and "Q4_PREPARE_FAILED:QUANTIZE_FAILED" in out.stdout
    assert not q4.out_file.exists() and q4.manifest.read_bytes() == before


# --- the manifest of the pod ----------------------------------------------------------------------------------


def test_the_pods_copy_of_the_manifest_gets_the_variant_with_the_measured_values(q4):
    assert q4.run(*MAKE).returncode == 0
    entry = json.loads(q4.manifest.read_text())["gguf"]["files"][VARIANT]
    assert entry["name"] == Q4_NAME and entry["size"] == len(Q4_BYTES)
    assert entry["sha256"] == hashlib.sha256(Q4_BYTES).hexdigest()
    assert "produced on this pod" in entry["note"] and "bf16" in entry["note"]
    assert (
        "Q4_K_M" in entry["note"]
        and "ffn_up_exps=q8_0" in entry["note"]
        and "200 chunks" in entry["note"]
    )


def test_the_repositorys_own_manifest_is_never_touched(q4):
    before = MANIFEST_SRC.read_bytes()
    assert q4.run(*MAKE).returncode == 0
    assert MANIFEST_SRC.read_bytes() == before
    assert VARIANT not in json.loads(before)["gguf"]["files"]
    assert VARIANT in json.loads(q4.manifest.read_text())["gguf"]["files"]  # the copy has it


def test_the_edit_is_recorded_in_the_provenance_and_the_existing_scripts_accept_the_variant(q4):
    assert q4.run(*MAKE).returncode == 0
    edit = q4.report()["manifest_edit"]
    assert edit["changed"] is True and edit["manifest"] == str(q4.manifest)
    assert edit["sha256_after"] == hashlib.sha256(q4.manifest.read_bytes()).hexdigest()
    assert (q4.results / "q4" / "manifest_edit.json").is_file()
    record = json.loads(
        (q4.q4 / "variant.json").read_text()
    )  # what bench_gguf.sh restores a lost edit from
    assert (
        record["variant"] == VARIANT
        and record["entry"]["sha256"] == hashlib.sha256(Q4_BYTES).hexdigest()
    )
    tool = subprocess.run(
        [sys.executable, "-I", str(q4.pod / "gguf_tool.py"), "resolve", VARIANT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert tool.stdout.strip() == VARIANT
    serve = subprocess.run(
        [
            "bash",
            str(q4.pod / "serve_gguf.sh"),
            "--dry-run",
            "--variant",
            VARIANT,
            "--skip-memory-check",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "OLALA_WORK": str(q4.work)},
    )
    assert serve.returncode == 0, serve.stderr
    assert f"model={q4.out_file}" in serve.stdout


# --- prepare.json and q4.done ---------------------------------------------------------------------------------


def test_prepare_json_has_the_commands_shas_sizes_and_recipe(q4):
    assert q4.run(*MAKE).returncode == 0
    rep = q4.report()
    assert rep["status"] == "done" and rep["amendment"] == 24
    assert rep["recipe"]["quant_type"] == "Q4_K_M" and rep["recipe"]["imatrix_chunks"] == 200
    assert rep["recipe"]["tensor_types"] == {"ffn_latent_up": "bf16", "ffn_up_exps": "q8_0"}
    assert (
        rep["bf16"]["sha256"]
        == json.loads(MANIFEST_SRC.read_text())["gguf"]["files"]["bf16"]["sha256"]
    )
    assert rep["quantized"]["size_bytes"] == len(Q4_BYTES)
    for step in ("setup_bf16", "build_extra", "help", "wikitext", "imatrix", "quantize"):
        assert step in rep["steps"], step


def test_q4_done_has_the_marker_first_and_the_hashes_after(q4):
    assert q4.run(*MAKE).returncode == 0
    lines = q4.done.read_text().splitlines()
    assert lines[0] == "Q4_PREPARE_DONE"
    assert f"quantized_sha256={hashlib.sha256(Q4_BYTES).hexdigest()}" in lines
    assert f"imatrix_sha256={hashlib.sha256(IMATRIX_BYTES).hexdigest()}" in lines


def test_q4_done_is_written_last_and_is_absent_while_the_tools_run(q4):
    assert q4.run(*MAKE).returncode == 0
    states = [c for c in q4.calls() if c.startswith("state:")]
    assert sorted(states) == ["state: q4.done=absent"] * 2 + ["state: report q4.done=absent"]
    done_time = q4.done.stat().st_mtime_ns
    others = [p for p in (q4.results / "q4").rglob("*") if p.is_file()]
    assert others and all(p.stat().st_mtime_ns <= done_time for p in others)
    assert (q4.results / "q4" / "prepare.json").stat().st_mtime_ns <= done_time


def test_a_stale_done_of_an_earlier_run_does_not_survive_a_new_run(q4):
    q4.results.mkdir(parents=True)
    q4.done.write_text("Q4_PREPARE_DONE\nstale\n")
    out = q4.run(*MAKE, STUB_SETUP_FAIL="boom")
    assert out.returncode == 1 and "stale" not in q4.done.read_text()
    assert q4.done_first_line().startswith("Q4_PREPARE_FAILED")


def test_a_stale_done_is_gone_while_the_new_run_works(q4):
    q4.results.mkdir(parents=True)
    q4.done.write_text("Q4_PREPARE_DONE\nstale\n")
    assert q4.run(*MAKE).returncode == 0
    states = [c for c in q4.calls() if c.startswith("state:")]
    assert states and all(st.endswith("q4.done=absent") for st in states)
    assert "stale" not in q4.done.read_text()


def test_a_failure_writes_a_failed_report_and_still_writes_done_last(q4):
    out = q4.run(*MAKE, STUB_QUANT_RC="1")
    assert out.returncode == 1
    rep = q4.report()
    assert (
        rep["status"] == "failed"
        and rep["reason"].startswith("QUANTIZE_FAILED")
        and "quantized" not in rep
    )
    assert q4.done.stat().st_mtime_ns >= (q4.results / "q4" / "prepare.json").stat().st_mtime_ns


# --- resume ---------------------------------------------------------------------------------------------------


def test_a_second_run_does_not_redo_a_finished_step(q4):
    assert q4.run(*MAKE).returncode == 0
    q4.log.write_text("")
    q4.hf_log.unlink()
    again = q4.run(*MAKE)
    assert again.returncode == 0, again.stdout + again.stderr
    assert real_calls(q4.calls()) == [] and not q4.hf_log.exists()
    assert again.stdout.count("skipped (already done)") >= 6
    assert q4.done_first_line() == "Q4_PREPARE_DONE" and q4.report()["status"] == "done"


def test_a_failed_quantization_is_resumed_from_that_step_only(q4):
    assert q4.run(*MAKE, STUB_QUANT_RC="1").returncode == 1
    q4.log.write_text("")
    assert q4.run(*MAKE).returncode == 0
    assert kinds(q4.calls()) == ["quantize"]


def test_a_rerun_step_invalidates_the_steps_after_it(q4):
    assert q4.run(*MAKE).returncode == 0
    (q4.q4 / "steps" / "imatrix.done").unlink()
    q4.log.write_text("")
    assert q4.run(*MAKE).returncode == 0
    assert kinds(q4.calls()) == ["imatrix", "quantize"]


def test_another_given_imatrix_reruns_the_imatrix_and_the_quantization(q4, tmp_path):
    first = tmp_path / "a.imatrix"
    first.write_bytes(b"first")
    assert q4.run(*file_args(first)).returncode == 0
    second = tmp_path / "b.imatrix"
    second.write_bytes(b"second")
    q4.log.write_text("")
    assert q4.run(*file_args(second)).returncode == 0
    assert kinds(q4.calls()) == ["quantize"]
    assert (q4.q4 / "olala.imatrix").read_bytes() == b"second"


def test_switching_from_make_to_a_given_imatrix_does_not_reuse_the_made_one(q4, tmp_path):
    assert q4.run(*MAKE).returncode == 0
    given = tmp_path / "g.imatrix"
    given.write_bytes(b"given")
    q4.log.write_text("")
    assert q4.run(*file_args(given)).returncode == 0
    assert kinds(q4.calls()) == ["quantize"]
    assert q4.report()["imatrix"]["mode"] == "file"


def test_the_manifest_edit_is_repeated_when_a_fresh_bundle_lost_it(q4):
    assert q4.run(*MAKE).returncode == 0
    q4.manifest.write_bytes(
        MANIFEST_SRC.read_bytes()
    )  # pod_setup --bundle-only replaced the bundle
    q4.log.write_text("")
    assert q4.run(*MAKE).returncode == 0
    assert VARIANT in json.loads(q4.manifest.read_text())["gguf"]["files"]
    assert real_calls(q4.calls()) == []


def test_no_token_name_is_used_outside_an_unset():
    code = [
        ln for ln in (POD_SRC / SCRIPT).read_text().splitlines() if not ln.lstrip().startswith("#")
    ]
    for token in ("HF_TOKEN", "GITHUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        assert all("unset" in ln for ln in code if token in ln), token
