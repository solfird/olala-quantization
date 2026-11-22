"""Wiring checks of the real task registry. Need the dataset hosts: run with ``-m network``."""

import pytest

import run_tasks

pytestmark = pytest.mark.network

EXPECTED_ITEMS = {
    "mgsm": 250 * 11,
    "humaneval_plus": 164,
    "mbpp_plus": 378,
    "gsm8k": 1319,
    "ifeval": 541,
    "aime2025": 30,
}


@pytest.mark.parametrize("name", list(EXPECTED_ITEMS))
def test_registered_task_builds_with_unique_ids_and_the_expected_size(name):
    spec = run_tasks.TASKS[name]
    task = spec.factory()
    originals, strata = run_tasks._dataset_items(task, spec.stratum_field)
    assert len(originals) == EXPECTED_ITEMS[name]
    if spec.stratum_field:
        assert strata is not None
        assert len(set(strata)) == 11
        assert None not in strata


def test_gsm8k_is_zero_shot():
    task = run_tasks.TASKS["gsm8k"].factory()
    assert "system_message" not in repr(task.solver)


def test_the_stratified_mgsm_subset_has_equal_languages():
    spec = run_tasks.TASKS["mgsm"]
    originals, strata = run_tasks._dataset_items(spec.factory(), spec.stratum_field)
    chosen = run_tasks.select_items(list(originals), strata, 11 * 50, seed=0)
    by_id = dict(zip(originals, strata or [], strict=True))
    counts = {lang: sum(by_id[i] == lang for i in chosen) for lang in set(by_id.values())}
    assert set(counts.values()) == {50}


def test_the_protocol_and_the_registry_agree():
    assert set(run_tasks.DEFAULT_TASKS) == set(EXPECTED_ITEMS)
    # PROTOCOL.md section 6: AIME 2025 = 5 samples per problem, every other task 1
    assert run_tasks.TASKS["aime2025"].epochs == 5
    for name in ("mgsm", "humaneval_plus", "mbpp_plus", "ifeval", "gsm8k"):
        assert run_tasks.TASKS[name].epochs == 1, name
