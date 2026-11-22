"""Tests for olala_quant.ablation: named selections of what NVFP4 (or FP8) quantizes, on the REAL header.

The header (945 tensors, shapes and dtypes only) is the one of tests/data/olala_header.json, so every
expected number is computed here by hand from recon/quantizability.md (36 layers, attention at 4, 9, ..., 34,
list I = mixer_proj of layers 1, 7, 23 and the MoE of layers 0, 1, 5, 19) and not from the code under test.
"""

import json
from pathlib import Path

import pytest

from olala_quant import ablation, recipes
from olala_quant.checkpoint_io import TensorMeta, header_from_json
from olala_quant.plan import build_plan

DATA = Path(__file__).parent / "data"
ATTENTION_LAYERS = {4, 9, 14, 19, 24, 29, 34}
LIST_I_MIXER = {1, 7, 23}
LIST_I_MOE = {0, 1, 5, 19}

# The line the OVH fork logged for the online FP8 arm with list I (F1) and without it (F1n),
# results/pod-2026-11-20/smoke-4500/logs/fp8pc*/server-*.log, verbatim from "Quantizing" on.
SERVER_LINE_LIST_I = (
    "Quantizing 200 layers of types: gate_proj: 7 (from linear: fp8_per_channel_static); "
    "mixer_proj: 33 (from linear: fp8_per_channel_static); mlp.experts: 32 (from moe: "
    "fp8_per_channel_static); mlp.fc1_latent_proj: 32 (from linear: fp8_per_channel_static); "
    "mlp.fc2_latent_proj: 32 (from linear: fp8_per_channel_static); mlp.shared_experts.down_proj: 32 "
    "(from linear: fp8_per_channel_static); mlp.shared_experts.up_proj: 32 (from linear: "
    "fp8_per_channel_static)"
)
SERVER_LINE_NO_IGNORE = (
    "Quantizing 223 layers of types: gate_proj: 7 (from linear: fp8_per_channel_static); "
    "mixer_proj: 36 (from linear: fp8_per_channel_static); mlp.experts: 36 (from moe: "
    "fp8_per_channel_static); mlp.fc1_latent_proj: 36 (from linear: fp8_per_channel_static); "
    "mlp.fc2_latent_proj: 36 (from linear: fp8_per_channel_static); mlp.shared_experts.down_proj: 36 "
    "(from linear: fp8_per_channel_static); mlp.shared_experts.up_proj: 36 (from linear: "
    "fp8_per_channel_static)"
)

#            spec                      quantized layers, quantized tensors
EXPECTED = [
    ("all", 200, 232),
    ("experts", 32, 64),  # 36 MoE layers minus list I's 0, 1, 5, 19
    ("experts:layers=0-11", 9, 18),  # 0..11 minus 0, 1, 5
    ("experts:layers=12-23", 11, 22),  # 12..23 minus 19
    ("experts:layers=24-35", 12, 24),  # no list I layer there
    ("non-experts", 168, 168),  # 200 - 32
    ("shared-experts", 64, 64),  # fc_1 and fc_2 of 32 layers: two modules each
    ("latent", 64, 64),
    ("attention", 14, 14),  # 7 mixer_proj + 7 gate_proj
    ("mamba", 26, 26),  # 29 Mamba mixer_proj minus list I's 1, 7, 23
    ("mamba:layers=0-11", 8, 8),  # M layers 0-3, 5-8, 10, 11 (10 of them) minus 1 and 7
    ("experts:layers=24-35,0-5", 15, 30),  # union, order does not matter: 24..35 and 2, 3, 4
]
TOTAL_QUANTIZABLE_TENSORS = 259


@pytest.fixture(scope="module")
def header() -> dict[str, TensorMeta]:
    return header_from_json(json.loads((DATA / "olala_header.json").read_text()))


def _plan(header: dict[str, TensorMeta], spec: str, recipe: str = "nvfp4"):
    resolved = ablation.resolve_selection(header, spec, recipes.DEFAULT_IGNORE_I)
    return build_plan(header, recipes.RECIPES[recipe], resolved.patterns)


def _quantized_layers(plan, role: str) -> set[int]:
    return {t.layer for t in plan.tensors if t.category == "quantized" and t.role == role}


# ---------------------------------------------------------------------------------------- the spec grammar


def test_layer_kinds_come_from_the_header_and_match_the_model(header: dict[str, TensorMeta]) -> None:
    kinds = ablation.layer_kinds(header)
    assert sorted(kinds) == list(range(36))
    assert {i for i, k in kinds.items() if k == "V"} == ATTENTION_LAYERS
    assert sum(k == "M" for k in kinds.values()) == 29


def test_a_layer_that_is_neither_mamba_nor_attention_is_an_error(header: dict[str, TensorMeta]) -> None:
    bad = {n: m for n, m in header.items() if not n.startswith("model.layers.2.mixer.")}
    with pytest.raises(ablation.SelectionError, match="layer 2"):
        ablation.layer_kinds(bad)


@pytest.mark.parametrize(
    ("spec", "canonical"),
    [
        ("experts", "experts"),
        ("non-experts", "non-experts"),
        ("experts:layers=0-11", "experts:layers=0-11"),
        ("experts:layers=7", "experts:layers=7"),
        ("experts:layers=10,2-3", "experts:layers=2-3,10"),
        ("experts:layers=0-5,3-8", "experts:layers=0-8"),
        ("experts:layers=0-3,4-6", "experts:layers=0-6"),  # adjacent ranges merge
        ("all:layers=0-35", "all:layers=0-35"),
    ],
)
def test_specs_are_parsed_and_written_canonically(spec: str, canonical: str) -> None:
    assert ablation.parse_select(spec).spec == canonical


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "expert",  # typo
        "Experts",  # strict: no case folding
        "experts:",
        "experts:layers=",
        "experts:layers=5-2",  # reversed
        "experts:layers=-1",
        "experts:layers=a-b",
        "experts:layers=1-2-3",
        "experts:layers=1;2",
        "experts:depth=1-2",  # unknown key
        "experts:layers=1:layers=2",  # key twice
        "experts:layers=0-11:x=1",
        "experts, mamba",  # one class per spec
        " experts",
    ],
)
def test_malformed_specs_are_refused(spec: str) -> None:
    with pytest.raises(ablation.SelectionError):
        ablation.parse_select(spec)


def test_the_named_selections_are_the_documented_set() -> None:
    assert set(ablation.SELECT_NAMES) == {
        "all",
        "experts",
        "non-experts",
        "shared-experts",
        "latent",
        "attention",
        "mamba",
    }
    atomic = ["experts", "shared-experts", "latent", "attention", "mamba"]
    assert ablation.SELECT_NAMES["all"] == frozenset(atomic)
    assert ablation.SELECT_NAMES["non-experts"] == frozenset(atomic) - {"experts"}
    for name in atomic:
        assert ablation.SELECT_NAMES[name] == frozenset({name})


# ---------------------------------------------------------------------------------------- what is selected


@pytest.mark.parametrize(("spec", "layers", "tensors"), EXPECTED)
def test_expected_counts_on_the_real_header(
    header: dict[str, TensorMeta], spec: str, layers: int, tensors: int
) -> None:
    counts = ablation.expected_counts(header, spec)
    assert counts["quantized_layers"] == layers
    assert counts["quantized_tensors"] == tensors
    assert counts["ignored_tensors"] == TOTAL_QUANTIZABLE_TENSORS - tensors
    assert sum(counts["quantized_layers_by_type"].values()) == layers
    assert counts["spec"] == ablation.parse_select(spec).spec
    assert counts["total_quantizable_layers"] == 223


@pytest.mark.parametrize(("spec", "layers", "tensors"), EXPECTED)
def test_the_counts_are_what_the_plan_of_the_converter_quantizes(
    header: dict[str, TensorMeta], spec: str, layers: int, tensors: int
) -> None:
    plan = _plan(header, spec)
    quantized = [t for t in plan.tensors if t.category == "quantized"]
    assert len(quantized) == tensors
    assert len({t.vllm_module for t in quantized}) == layers  # one vLLM module per counted layer


def test_the_server_line_of_the_online_arms_is_what_selection_all_predicts(
    header: dict[str, TensorMeta],
) -> None:
    n, by_type = ablation.parse_quantizing_line(SERVER_LINE_LIST_I)
    counts = ablation.expected_counts(header, "all")
    assert (n, by_type) == (200, counts["quantized_layers_by_type"])
    assert counts["quantized_layers"] == n
    # and without list I (F1n): 223 modules, every one of the 7 types at its full count
    n2, by_type2 = ablation.parse_quantizing_line(SERVER_LINE_NO_IGNORE)
    nothing = ablation.expected_counts(header, "all", base_patterns=())
    assert (n2, by_type2) == (223, nothing["quantized_layers_by_type"])


def test_the_types_of_the_server_line_are_the_vllm_module_tails(header: dict[str, TensorMeta]) -> None:
    by_type = ablation.expected_counts(header, "experts:layers=24-35")["quantized_layers_by_type"]
    assert by_type == {"mlp.experts": 12}
    by_type = ablation.expected_counts(header, "attention")["quantized_layers_by_type"]
    assert by_type == {"gate_proj": 7, "mixer_proj": 7}
    by_type = ablation.expected_counts(header, "shared-experts")["quantized_layers_by_type"]
    assert by_type == {"mlp.shared_experts.down_proj": 32, "mlp.shared_experts.up_proj": 32}


def test_parse_quantizing_line_reads_the_total_and_every_type_and_refuses_nonsense() -> None:
    assert ablation.parse_quantizing_line("INFO Quantizing 3 layers of types: a: 1 (x); b.c: 2 (y)") == (
        3,
        {"a": 1, "b.c": 2},
    )
    with pytest.raises(ablation.SelectionError, match="no 'Quantizing"):
        ablation.parse_quantizing_line("nothing here")
    with pytest.raises(ablation.SelectionError, match="sum"):
        ablation.parse_quantizing_line("Quantizing 5 layers of types: a: 1 (x)")


def test_every_quantizable_role_belongs_to_exactly_one_component(header: dict[str, TensorMeta]) -> None:
    kinds = ablation.layer_kinds(header)
    for spec in recipes.QUANT_ROLES:
        assert ablation.component_of(spec.role, 4, kinds) in ablation.COMPONENTS, spec.role
        assert ablation.component_of(spec.role, 0, kinds) in ablation.COMPONENTS, spec.role
    assert ablation.component_of("mixer_proj", 4, kinds) == "attention"
    assert ablation.component_of("mixer_proj", 0, kinds) == "mamba"
    with pytest.raises(ablation.SelectionError, match="no component"):
        ablation.component_of("a_role_added_to_the_recipes_later", 0, kinds)


def test_the_five_components_partition_the_quantized_set(header: dict[str, TensorMeta]) -> None:
    atomic = ["experts", "shared-experts", "latent", "attention", "mamba"]
    every = {t.name for t in _plan(header, "all").tensors if t.category == "quantized"}
    parts = [{t.name for t in _plan(header, c).tensors if t.category == "quantized"} for c in atomic]
    assert set().union(*parts) == every
    assert sum(len(p) for p in parts) == len(every)  # pairwise disjoint
    non_experts = {t.name for t in _plan(header, "non-experts").tensors if t.category == "quantized"}
    assert non_experts == every - parts[0]


@pytest.mark.parametrize(("spec", "_layers", "_tensors"), EXPECTED)
def test_list_i_is_never_quantized_whatever_the_selection(
    header: dict[str, TensorMeta], spec: str, _layers: int, _tensors: int
) -> None:
    plan = _plan(header, spec)
    by_name = {t.name: t for t in plan.tensors}
    for layer in LIST_I_MIXER:
        assert by_name[f"model.layers.{layer}.mixer_proj.weight"].category == "ignored"
    for layer in LIST_I_MOE:
        for t in plan.tensors:
            if t.name.startswith(f"model.layers.{layer}.mlp.") and t.vllm_module is not None:
                assert t.category == "ignored", t.name
    ignore = set(plan.vllm_ignore)
    assert {f"model.layers.{i}.mixer_proj" for i in LIST_I_MIXER} <= ignore
    assert {f"model.layers.{i}.mlp.fc1_latent_proj" for i in LIST_I_MOE} <= ignore
    assert {f"re:model\\.layers\\.{i}\\.mlp\\.experts\\..*" for i in LIST_I_MOE} <= ignore


def test_the_layer_range_includes_both_ends_and_nothing_else(header: dict[str, TensorMeta]) -> None:
    assert _quantized_layers(_plan(header, "experts:layers=0-11"), "routed_up") == {
        2,
        3,
        4,
        6,
        7,
        8,
        9,
        10,
        11,
    }
    assert _quantized_layers(_plan(header, "experts:layers=12-23"), "routed_up") == (
        set(range(12, 24)) - {19}
    )
    assert _quantized_layers(_plan(header, "experts:layers=24-35"), "routed_up") == set(range(24, 36))
    assert _quantized_layers(_plan(header, "experts:layers=35"), "routed_down") == {35}
    assert _quantized_layers(_plan(header, "experts:layers=2"), "routed_down") == {2}
    # the up and the down projection always travel together (vLLM cannot split them)
    for spec in ("experts:layers=12-23", "experts:layers=2-4,30"):
        plan = _plan(header, spec)
        assert _quantized_layers(plan, "routed_up") == _quantized_layers(plan, "routed_down")
        assert plan.forced == {}


def test_a_layer_range_selects_within_a_class_only(header: dict[str, TensorMeta]) -> None:
    plan = _plan(header, "mamba:layers=0-11")
    quantized = {t.layer for t in plan.tensors if t.category == "quantized"}
    assert quantized == {0, 2, 3, 5, 6, 8, 10, 11}  # M layers of 0..11 minus list I's 1 and 7
    assert {t.role for t in plan.tensors if t.category == "quantized"} == {"mixer_proj"}
    plan = _plan(header, "attention:layers=12-24")
    assert {t.layer for t in plan.tensors if t.category == "quantized"} == {14, 19, 24}


def test_attention_is_the_out_projection_and_the_gate_of_the_attention_layers(
    header: dict[str, TensorMeta],
) -> None:
    plan = _plan(header, "attention")
    quantized = [t for t in plan.tensors if t.category == "quantized"]
    assert {t.role for t in quantized} == {"mixer_proj", "attn_gate_proj"}
    assert {t.layer for t in quantized} == ATTENTION_LAYERS
    mamba = _plan(header, "mamba")
    assert {t.layer for t in mamba.tensors if t.category == "quantized"}.isdisjoint(ATTENTION_LAYERS)


# ---------------------------------------------------------------------------------------- the ignore list


def test_every_deselected_module_is_in_the_vllm_ignore_list(header: dict[str, TensorMeta]) -> None:
    for spec in ("experts", "non-experts", "experts:layers=12-23", "mamba", "attention", "latent"):
        plan = _plan(header, spec)
        ignore = set(plan.vllm_ignore)
        for t in plan.tensors:
            if t.vllm_module is None:
                continue
            if t.category == "quantized":
                assert t.vllm_module not in ignore, (spec, t.name)
            else:
                assert t.vllm_module in ignore, (spec, t.name)
                if t.role.startswith("routed"):
                    assert f"re:{t.vllm_module.replace('.', chr(92) + '.')}\\..*" in ignore, (spec, t.name)
        assert "lm_head" in ignore


def test_experts_only_ignores_every_dense_module_and_keeps_the_experts_out_of_the_list(
    header: dict[str, TensorMeta],
) -> None:
    ignore = set(ablation_plan_ignore(header, "experts"))
    assert "model.layers.2.mixer_proj" in ignore  # 36 mixer_proj, 7 gate_proj, 4 modules x 36 layers
    assert "model.layers.4.gate_proj" in ignore
    assert "model.layers.30.mlp.shared_experts.up_proj" in ignore
    assert "model.layers.30.mlp.fc2_latent_proj" in ignore
    assert "model.layers.30.mlp.experts" not in ignore
    assert "model.layers.0.mlp.experts" in ignore  # list I
    n_plain = len([e for e in ignore if not e.startswith("re:") and e != "lm_head"])
    assert n_plain == 223 - 32  # every counted module that is not quantized


def test_non_experts_ignores_every_routed_expert_layer_with_its_regex(header: dict[str, TensorMeta]) -> None:
    ignore = set(ablation_plan_ignore(header, "non-experts"))
    for layer in range(36):
        assert f"model.layers.{layer}.mlp.experts" in ignore
        assert f"re:model\\.layers\\.{layer}\\.mlp\\.experts\\..*" in ignore
    assert "model.layers.30.mlp.shared_experts.up_proj" not in ignore


def ablation_plan_ignore(header: dict[str, TensorMeta], spec: str) -> tuple[str, ...]:
    return _plan(header, spec).vllm_ignore


# ---------------------------------------------------------------------------------------- refusals


def test_a_selection_that_quantizes_nothing_is_refused(header: dict[str, TensorMeta]) -> None:
    with pytest.raises(ablation.SelectionError, match="quantizes no tensor"):
        ablation.resolve_selection(header, "experts:layers=0-1", recipes.DEFAULT_IGNORE_I)  # both list I
    with pytest.raises(ablation.SelectionError, match="quantizes no tensor"):
        ablation.resolve_selection(header, "attention:layers=0-3", recipes.DEFAULT_IGNORE_I)  # no attention
    with pytest.raises(ablation.SelectionError, match="quantizes no tensor"):
        ablation.resolve_selection(header, "mamba:layers=4", recipes.DEFAULT_IGNORE_I)  # layer 4 is attention


@pytest.mark.parametrize("spec", ["experts:layers=36", "experts:layers=30-36", "experts:layers=0-99"])
def test_layers_beyond_the_model_are_refused_not_ignored(header: dict[str, TensorMeta], spec: str) -> None:
    with pytest.raises(ablation.SelectionError, match="layers 0 to 35"):
        ablation.resolve_selection(header, spec, recipes.DEFAULT_IGNORE_I)


def test_the_resolved_selection_records_what_it_did(header: dict[str, TensorMeta]) -> None:
    resolved = ablation.resolve_selection(header, "experts:layers=12-23", recipes.DEFAULT_IGNORE_I)
    assert resolved.selection.spec == "experts:layers=12-23"
    assert resolved.patterns[: len(recipes.DEFAULT_IGNORE_I)] == recipes.DEFAULT_IGNORE_I
    assert resolved.base_patterns == recipes.DEFAULT_IGNORE_I
    assert set(resolved.deselected) == set(resolved.patterns[len(recipes.DEFAULT_IGNORE_I) :])
    # deselected: every quantizable tensor outside layers 12-23 experts that list I does not already hold
    assert "model.layers.2.mlp.experts.experts.weight" in resolved.deselected
    assert "model.layers.12.mlp.experts.experts.weight" not in resolved.deselected
    assert "model.layers.0.mlp.experts.experts.weight" not in resolved.deselected  # list I already ignores it
    assert len(resolved.deselected) == len(set(resolved.deselected))


def test_the_default_list_is_untouched_by_the_all_selection(header: dict[str, TensorMeta]) -> None:
    resolved = ablation.resolve_selection(header, "all", recipes.DEFAULT_IGNORE_I)
    assert resolved.patterns == recipes.DEFAULT_IGNORE_I
    assert resolved.deselected == ()
    plain = build_plan(header, recipes.RECIPES["nvfp4"], recipes.DEFAULT_IGNORE_I)
    selected = build_plan(header, recipes.RECIPES["nvfp4"], resolved.patterns)
    assert selected.vllm_ignore == plain.vllm_ignore
    assert selected.output_bytes() == plain.output_bytes()
    assert selected.ignored_quantizable == plain.ignored_quantizable


def test_n2_as_planned_today_is_unchanged(header: dict[str, TensorMeta]) -> None:
    """Characterization of arm N2 (nvfp4, list I) before the ablation code existed: byte count and config."""
    import hashlib

    plan = build_plan(header, recipes.RECIPES["nvfp4"], recipes.DEFAULT_IGNORE_I)
    assert plan.output_bytes() == 6_043_754_176  # PROTOCOL: NVFP4 6.04 GB with list I
    blob = json.dumps(
        {
            "qc": recipes.RECIPES["nvfp4"].quantization_config(plan.vllm_ignore),
            "ign": list(plan.ignored_quantizable),
            "bytes": plan.output_bytes(),
            "acct": plan.category_accounting(),
        },
        sort_keys=True,
    )
    assert hashlib.sha256(blob.encode()).hexdigest() == (
        "3316e8e3ca9a2e12d36c73374659d5d6b2de6beeb6100310f44f07687d45a8f7"
    )
    plan8 = build_plan(header, recipes.RECIPES["fp8"], recipes.DEFAULT_IGNORE_I)
    assert plan8.output_bytes() == 8_375_938_080


# ---------------------------------------------------------------------------------------- experts-all
# (amendment 25)  C1 and C2 of amendment 25: the routed experts (up and down tensors) of ALL 36 layers are
# quantized, including the MoE layers 0, 1, 5, 19 that list I normally keeps in bf16; everything else stays
# bf16 and the mixer projections of layers 1, 7, 23 (the other half of list I) stay bf16 too. Every expected
# number below is worked out from the real header.

ALL_MOE = set(range(36))
LIST_I_MIXER_PATTERNS = [
    "model.layers.1.mixer_proj.weight",
    "model.layers.7.mixer_proj.weight",
    "model.layers.23.mixer_proj.weight",
]
LIST_I_MOE_PATTERNS = [f"model.layers.{i}.mlp.*" for i in (0, 1, 5, 19)]
# one routed tensor of the model: [256, 768, 384] bf16 = 150,994,944 bytes; NVFP4: n/2 packed + n/16 e4m3
# scales + 256 fp32
ROUTED_N = 256 * 768 * 384
ROUTED_BF16 = ROUTED_N * 2
ROUTED_NVFP4 = ROUTED_N // 2 + ROUTED_N // 16 + 256 * 4
HEADER_BYTES = 13_579_574_304  # every tensor of the header as stored (bf16 and fp32): the source checkpoint
N5_BYTES = 6_633_872_416  # amendment 22's correction, `experts` on the same header
C1_BYTES = 5_765_659_680  # computed below by hand, 72 routed tensors


def test_the_hand_computed_sizes_agree_with_the_figures_already_in_the_protocol(
    header: dict[str, TensorMeta],
) -> None:
    assert sum(m.nbytes for m in header.values()) == HEADER_BYTES
    saving = ROUTED_BF16 - ROUTED_NVFP4
    assert HEADER_BYTES - 64 * saving == N5_BYTES  # N5: 32 layers x 2 tensors
    assert HEADER_BYTES - 72 * saving == C1_BYTES  # C1 and C2: 36 layers x 2 tensors
    assert C1_BYTES < 5_990_551_040 < N5_BYTES  # smaller than Q5_K_M, N5 is larger


def test_experts_all_quantizes_the_routed_experts_of_all_36_layers_and_nothing_else(
    header: dict[str, TensorMeta],
) -> None:
    counts = ablation.expected_counts(header, "experts-all")
    assert counts["spec"] == "experts-all"
    assert counts["quantized_layers"] == 36  # modules: one per MoE layer
    assert counts["quantized_tensors"] == 72  # up and down of each
    assert counts["quantized_layers_by_type"] == {"mlp.experts": 36}
    assert counts["ignored_tensors"] == TOTAL_QUANTIZABLE_TENSORS - 72 == 187
    assert counts["ignored_layers"] == 223 - 36 == 187
    assert counts["total_quantizable_layers"] == 223
    assert counts["ignore_entries"] == 187 + 1  # every other module, plus lm_head; no regex of a routed layer
    plan = _plan(header, "experts-all")
    quantized = {t.name for t in plan.tensors if t.category == "quantized"}
    assert quantized == {
        f"model.layers.{i}.mlp.experts.{t}.weight" for i in ALL_MOE for t in ("experts", "output_experts")
    }


def test_experts_all_lifts_the_moe_half_of_list_i_and_keeps_its_mixer_half(
    header: dict[str, TensorMeta],
) -> None:
    plan = _plan(header, "experts-all")
    by_name = {t.name: t for t in plan.tensors}
    for layer in LIST_I_MIXER:  # the mixer projections of 1, 7, 23 stay bf16
        assert by_name[f"model.layers.{layer}.mixer_proj.weight"].category == "ignored"
    for layer in LIST_I_MOE:  # the routed experts of 0, 1, 5, 19 are quantized ...
        for t in ("experts", "output_experts"):
            assert by_name[f"model.layers.{layer}.mlp.experts.{t}.weight"].category == "quantized"
        # ... but the rest of their MLP (shared experts, latent projections) is bf16, as everywhere in this
        # selection
        for leaf in ("shared_experts.fc_1", "shared_experts.fc_2", "down_proj", "up_proj"):
            assert by_name[f"model.layers.{layer}.mlp.{leaf}.weight"].category == "ignored", (layer, leaf)
    ignore = set(plan.vllm_ignore)
    assert {f"model.layers.{i}.mixer_proj" for i in LIST_I_MIXER} <= ignore
    for layer in LIST_I_MOE:
        assert f"model.layers.{layer}.mlp.experts" not in ignore
        assert f"re:model\\.layers\\.{layer}\\.mlp\\.experts\\..*" not in ignore
        assert f"model.layers.{layer}.mlp.fc1_latent_proj" in ignore
        assert f"model.layers.{layer}.mlp.shared_experts.up_proj" in ignore


def test_experts_all_records_that_list_i_was_partly_lifted_and_which_patterns(
    header: dict[str, TensorMeta],
) -> None:
    resolved = ablation.resolve_selection(header, "experts-all", recipes.DEFAULT_IGNORE_I)
    record = resolved.record()
    assert record["spec"] == "experts-all" and record["name"] == "experts-all"
    assert record["list_i_lifted"] is True
    assert record["lifted_ignore_patterns"] == LIST_I_MOE_PATTERNS
    assert record["kept_ignore_patterns"] == LIST_I_MIXER_PATTERNS
    assert record["base_ignore_patterns"] == list(recipes.DEFAULT_IGNORE_I)  # what the caller gave, whole
    assert record["components"] == ["experts"]
    assert record["expected"]["quantized_tensors"] == 72
    # the patterns given to the plan: the kept half of list I, then one exact name per deselected tensor
    assert resolved.patterns[:3] == tuple(LIST_I_MIXER_PATTERNS)
    assert not any(p in resolved.patterns for p in LIST_I_MOE_PATTERNS)
    assert resolved.base_patterns == recipes.DEFAULT_IGNORE_I
    # nothing that list I held and the selection still wants bf16 is lost: it is in `deselected` by exact name
    assert "model.layers.19.mlp.shared_experts.fc_1.weight" in resolved.deselected
    assert "model.layers.0.mlp.down_proj.weight" in resolved.deselected
    assert "model.layers.0.mlp.experts.experts.weight" not in resolved.deselected


def test_the_selections_that_existed_before_record_nothing_about_lifting(
    header: dict[str, TensorMeta],
) -> None:
    """Their `selection` block stays what it was (a checkpoint built earlier must still verify)."""
    for spec in ("all", "experts", "non-experts", "experts:layers=12-23", "mamba", "latent"):
        record = ablation.resolve_selection(header, spec, recipes.DEFAULT_IGNORE_I).record()
        assert set(record) == {
            "spec",
            "name",
            "components",
            "layers",
            "base_ignore_patterns",
            "deselected_tensors",
            "expected",
        }, spec


def test_experts_all_with_a_layer_range_quantizes_only_those_layers_experts(
    header: dict[str, TensorMeta],
) -> None:
    counts = ablation.expected_counts(header, "experts-all:layers=0-11")
    assert (counts["quantized_layers"], counts["quantized_tensors"]) == (12, 24)  # 0 and 1 and 5 included
    plan = _plan(header, "experts-all:layers=0-11")
    assert _quantized_layers(plan, "routed_up") == set(range(12))
    assert _quantized_layers(plan, "routed_down") == set(range(12))
    assert plan.forced == {}
    # layer 19 is outside the range: its routed tensors stay bf16, by exact name now that its list I pattern
    # is lifted
    by_name = {t.name: t for t in plan.tensors}
    assert by_name["model.layers.19.mlp.experts.experts.weight"].category == "ignored"
    assert "model.layers.19.mlp.experts" in plan.vllm_ignore
    assert ablation.parse_select("experts-all:layers=7,3-5,4").spec == "experts-all:layers=3-5,7"
    assert ablation.expected_counts(header, "experts-all:layers=19")["quantized_layers"] == 1


def test_experts_all_without_a_list_i_equals_experts(header: dict[str, TensorMeta]) -> None:
    a = ablation.resolve_selection(header, "experts-all", ())
    b = ablation.resolve_selection(header, "experts", ())
    assert a.patterns == b.patterns and a.deselected == b.deselected
    assert a.record()["lifted_ignore_patterns"] == []
    assert ablation.expected_counts(header, "experts-all", ())["quantized_layers"] == 36
    assert ablation.expected_counts(header, "experts", ())["quantized_layers"] == 36


def test_only_the_patterns_that_cover_a_routed_expert_tensor_are_lifted(
    header: dict[str, TensorMeta],
) -> None:
    base = (
        "model.layers.7.mixer_proj.weight",  # a mixer: kept
        "model.layers.0.mlp.*",  # covers the routed tensors: lifted
        "model.layers.1.mlp.experts.*",  # covers only routed tensors: lifted
        "model.layers.5.mlp.shared_experts.*",  # covers no routed tensor: kept
    )
    resolved = ablation.resolve_selection(header, "experts-all", base)
    record = resolved.record()
    assert record["lifted_ignore_patterns"] == ["model.layers.0.mlp.*", "model.layers.1.mlp.experts.*"]
    assert record["kept_ignore_patterns"] == [
        "model.layers.7.mixer_proj.weight",
        "model.layers.5.mlp.shared_experts.*",
    ]
    plan = build_plan(header, recipes.RECIPES["nvfp4"], resolved.patterns)
    by_name = {t.name: t for t in plan.tensors}
    assert by_name["model.layers.5.mlp.shared_experts.fc_1.weight"].category == "ignored"
    assert by_name["model.layers.5.mlp.experts.experts.weight"].category == "quantized"


def test_experts_all_is_not_one_of_the_names_the_pod_scripts_accept_yet() -> None:
    """The names of the older ckptsel grammar (pod/arm_pipeline.sh, bench_arm.sh) are unchanged on purpose."""
    assert "experts-all" not in ablation.SELECT_NAMES
    assert set(ablation.EXTENDED_SELECTIONS) == {"experts-all"}
    assert ablation.parse_select("experts-all").components == frozenset({"experts"})


@pytest.mark.parametrize("recipe", ["nvfp4", "fp8"])
def test_the_plan_of_experts_all_has_the_size_worked_out_by_hand(
    header: dict[str, TensorMeta], recipe: str
) -> None:
    plan = _plan(header, "experts-all", recipe)
    if recipe == "nvfp4":
        assert plan.output_bytes() == C1_BYTES == 5_765_659_680
        # C2 differs from C1 only by its block-scale VALUES: same tensors, same shapes, same bytes
        assert plan.output_bytes() < _plan(header, "experts").output_bytes() == N5_BYTES
    else:
        assert plan.output_bytes() < HEADER_BYTES


def test_experts_all_is_refused_when_it_quantizes_nothing(header: dict[str, TensorMeta]) -> None:
    only_moe = {n: m for n, m in header.items() if "mlp.experts." not in n}
    with pytest.raises(ablation.SelectionError, match="quantizes no tensor"):
        ablation.resolve_selection(only_moe, "experts-all", recipes.DEFAULT_IGNORE_I)
    with pytest.raises(ablation.SelectionError, match="layers 0 to 35"):
        ablation.resolve_selection(header, "experts-all:layers=0-36", recipes.DEFAULT_IGNORE_I)


def test_the_expected_subcommand_and_the_header_say_36_modules_and_72_tensors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = DATA / "olala_header.json"
    assert ablation.main(["expected", "--header", str(path), "--select", "experts-all"]) == 0
    counts = json.loads(capsys.readouterr().out)
    assert (counts["spec"], counts["quantized_layers"], counts["quantized_tensors"]) == (
        "experts-all",
        36,
        72,
    )
    assert counts["quantized_layers_by_type"] == {"mlp.experts": 36}
