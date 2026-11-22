"""Tests on the generated fork patch: it applies to the pristine fork file, it compiles, and the
silent 'skipping' path is gone. The fork source is read with `git show`, never executed or imported.
"""

import ast
import os
import py_compile
import re
import shutil
import subprocess
from pathlib import Path

import pytest

QUANT = Path(__file__).parents[1]
PATCH = QUANT / "fork-patch" / "olala-quant-loader.patch"
LOADER = QUANT / "fork-patch" / "olala_quant_loader.py"
FORK = Path(os.environ.get("OLALA_FORK_DIR", QUANT.parent / "recon" / "src" / "vllm"))
OLALA = "vllm/model_executor/models/olala.py"
NEW_FILE = "vllm/model_executor/models/olala_quant_loader.py"
FORK_COMMIT = "ebbadef53f515d476332a105f81fd67096ed9a57"


def _patch_sections() -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for line in PATCH.read_text().splitlines():
        if line.startswith("diff --git"):
            current = line.split(" b/")[-1]
            sections[current] = []
        elif current:
            sections[current].append(line)
    return sections


def test_patch_touches_exactly_two_files_and_adds_the_tested_module() -> None:
    sections = _patch_sections()
    assert set(sections) == {OLALA, NEW_FILE}
    added = [ln[1:] for ln in sections[NEW_FILE] if ln.startswith("+") and not ln.startswith("+++")]
    assert "\n".join(added) + "\n" == LOADER.read_text()


def test_new_module_does_not_import_vllm() -> None:
    text = LOADER.read_text()
    assert not re.search(r"^\s*(from|import)\s+vllm", text, re.M)


def test_patch_removes_the_silent_skip_and_adds_the_loud_error() -> None:
    sections = _patch_sections()
    removed = "\n".join(ln for ln in sections[OLALA] if ln.startswith("-"))
    added = "\n".join(ln for ln in sections[OLALA] if ln.startswith("+"))
    assert "not found, skipping" in removed
    assert "warning_once" in removed
    assert "not found, skipping" not in added
    assert "raise_if_unconsumed" in added
    assert "check_dtype_contract" in added
    assert "scatter_fused_expert_tensor" in added
    assert "OLALA_ALLOW_UNCONSUMED" in added


@pytest.fixture
def pristine_olala(tmp_path: Path) -> Path:
    if not (FORK / ".git").exists():
        pytest.skip("recon/src/vllm is not present")
    out = subprocess.run(
        ["git", "-C", str(FORK), "show", f"{FORK_COMMIT}:{OLALA}"],
        capture_output=True,
        check=True,
    )
    target = tmp_path / OLALA
    target.parent.mkdir(parents=True)
    target.write_bytes(out.stdout)
    return tmp_path


def test_patch_applies_cleanly_and_the_result_compiles(pristine_olala: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=pristine_olala, check=True)
    subprocess.run(["git", "apply", str(PATCH)], cwd=pristine_olala, check=True)
    patched = pristine_olala / OLALA
    py_compile.compile(str(patched), cfile=str(pristine_olala / "olala.pyc"), doraise=True)
    py_compile.compile(str(pristine_olala / NEW_FILE), cfile=str(pristine_olala / "q.pyc"), doraise=True)
    text = patched.read_text()
    assert "not found, skipping" not in text
    assert "from vllm.model_executor.models import olala_quant_loader as olq" in text
    # the old exact-name rename table is gone: names are routed by olala_quant_loader
    assert '"moe_gate.weight": "gate.weight"' not in text


def _load_weights_calls(patched_source: str) -> tuple[set[str], set[str]]:
    """Names called inside OlalaForCausalLM.load_weights (attribute or plain) and its helpers."""
    tree = ast.parse(patched_source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OlalaForCausalLM")
    funcs = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}

    def calls(fn: ast.FunctionDef) -> set[str]:
        names = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                f = node.func
                names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
        return names

    return calls(funcs["load_weights"]), set().union(*(calls(f) for f in funcs.values()))


def test_patched_load_weights_is_wired_to_the_loud_logic(pristine_olala: Path) -> None:
    subprocess.run(["git", "apply", str(PATCH)], cwd=pristine_olala, check=True)
    in_load, in_class = _load_weights_calls((pristine_olala / OLALA).read_text())
    assert {"route_checkpoint_name", "raise_if_unconsumed", "_load_one_weight"} <= in_load
    assert "_finalize_quantized_params" in in_load
    assert {"scatter_fused_expert_tensor", "check_dtype_contract", "unloaded_params"} <= in_class
    assert "warning_once" not in in_load  # the silent path is not reachable from load_weights
    # nothing in the quantized-load path swallows the error
    source = (pristine_olala / OLALA).read_text()
    assert "except olq.OlalaQuantLoadError" not in source
