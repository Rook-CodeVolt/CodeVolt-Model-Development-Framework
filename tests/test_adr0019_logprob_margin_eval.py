"""Tests for the ADR-0019 held-out log-prob margin evaluation script
(examples/pilot-metatrainer-v2/run_adr0019_logprob_margin_eval.py).

Same scope boundary as tests/test_bounded_metatrainer_cycle_adr0018.py:
proves the script's statistics, schema/hash-checking, and gate-
verification logic, and its dry-run/--execute split, without ever
loading a real model or scoring a real pair. No test in this file calls
``.train()``, loads a real Hugging Face checkpoint, or requires the
pinned SmolLM2 snapshot to be present on the runner -- this is the
lesson from PR #100/#101 (CI has no cached model snapshot), applied
here from the start rather than discovered after a CI failure.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from codevolt_mdf.hf_local_evaluator_adapter import _hash_model_dir

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/pilot-metatrainer-v2/run_adr0019_logprob_margin_eval.py"
)


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("adr0019_runner_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_model_dir(tmp_path, name: str, content: bytes) -> tuple[Path, str]:
    """Build a tiny local stand-in checkpoint directory and return its
    (path, real content hash) via the same ``_hash_model_dir`` the script
    itself calls -- same discipline as ADR-0018's runner tests'
    ``_fake_model_dir``, reused here against this script's own checks.
    """
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "weights.bin").write_bytes(content)
    return model_dir, _hash_model_dir(model_dir)


def _patch_valid_model_paths(runner, tmp_path, monkeypatch) -> None:
    candidate_dir, candidate_hash = _fake_model_dir(tmp_path, "candidate-model", b"fake-candidate-weights")
    reference_dir, reference_hash = _fake_model_dir(tmp_path, "reference-model", b"fake-reference-weights")
    monkeypatch.setattr(runner, "CANDIDATE_MODEL_PATH", candidate_dir)
    monkeypatch.setattr(runner, "EXPECTED_CANDIDATE_MODEL_HASH", candidate_hash)
    monkeypatch.setattr(runner, "REFERENCE_MODEL_PATH", reference_dir)
    monkeypatch.setattr(runner, "EXPECTED_REFERENCE_MODEL_HASH", reference_hash)


# ---------------------------------------------------------------------------
# 1. Statistics layer: classifier on all four outcomes plus boundary cases.
# ---------------------------------------------------------------------------


def test_classify_outcome_shift_present(runner):
    assert runner.classify_outcome(lo=1.0, hi=2.0, theta=0.843) == runner.OUTCOME_SHIFT_PRESENT


def test_classify_outcome_wrong_direction(runner):
    assert (
        runner.classify_outcome(lo=-2.0, hi=-1.0, theta=0.843) == runner.OUTCOME_WRONG_DIRECTION
    )


def test_classify_outcome_no_shift_of_training_magnitude(runner):
    assert (
        runner.classify_outcome(lo=-0.5, hi=0.5, theta=0.843)
        == runner.OUTCOME_NO_SHIFT_OF_TRAINING_MAGNITUDE
    )


def test_classify_outcome_ambiguous_straddles_positive_theta(runner):
    assert (
        runner.classify_outcome(lo=0.5, hi=1.5, theta=0.843)
        == runner.OUTCOME_AMBIGUOUS_UNDERPOWERED
    )


def test_classify_outcome_ambiguous_straddles_negative_theta(runner):
    assert (
        runner.classify_outcome(lo=-1.5, hi=-0.5, theta=0.843)
        == runner.OUTCOME_AMBIGUOUS_UNDERPOWERED
    )


def test_classify_outcome_ambiguous_straddles_both_thetas(runner):
    assert (
        runner.classify_outcome(lo=-1.5, hi=1.5, theta=0.843)
        == runner.OUTCOME_AMBIGUOUS_UNDERPOWERED
    )


# -- boundary cases: lo/hi exactly equal to +-theta ---------------------------


def test_classify_outcome_boundary_lo_exactly_theta_is_not_shift_present(runner):
    # section 3 outcome 1 is a strict inequality (lo > theta); lo == theta
    # must NOT classify as shift_present.
    assert (
        runner.classify_outcome(lo=0.843, hi=1.0, theta=0.843)
        == runner.OUTCOME_AMBIGUOUS_UNDERPOWERED
    )


def test_classify_outcome_boundary_hi_exactly_neg_theta_is_not_wrong_direction(runner):
    # outcome 2 is a strict inequality (hi < -theta); hi == -theta must NOT
    # classify as wrong_direction_shift.
    assert (
        runner.classify_outcome(lo=-1.0, hi=-0.843, theta=0.843)
        == runner.OUTCOME_AMBIGUOUS_UNDERPOWERED
    )


def test_classify_outcome_boundary_lo_exactly_neg_theta_is_no_shift(runner):
    # outcome 3 uses -theta <= lo (non-strict).
    assert (
        runner.classify_outcome(lo=-0.843, hi=0.843, theta=0.843)
        == runner.OUTCOME_NO_SHIFT_OF_TRAINING_MAGNITUDE
    )


def test_classify_outcome_boundary_hi_exactly_theta_is_no_shift(runner):
    # outcome 3 uses hi <= theta (non-strict).
    assert (
        runner.classify_outcome(lo=0.0, hi=0.843, theta=0.843)
        == runner.OUTCOME_NO_SHIFT_OF_TRAINING_MAGNITUDE
    )


def test_four_outcomes_are_mutually_exclusive_and_exhaustive_by_construction(runner):
    # Sweep a grid of (lo, hi) pairs and confirm exactly one outcome fires
    # for each, and every grid point classifies to something.
    thetas = 0.843
    grid = [round(x, 3) for x in [-2.0, -1.5, -0.843, -0.5, 0.0, 0.5, 0.843, 1.5, 2.0]]
    seen = set()
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            outcome = runner.classify_outcome(lo=lo, hi=hi, theta=thetas)
            assert outcome in {
                runner.OUTCOME_SHIFT_PRESENT,
                runner.OUTCOME_WRONG_DIRECTION,
                runner.OUTCOME_NO_SHIFT_OF_TRAINING_MAGNITUDE,
                runner.OUTCOME_AMBIGUOUS_UNDERPOWERED,
            }
            seen.add(outcome)
    assert seen == {
        runner.OUTCOME_SHIFT_PRESENT,
        runner.OUTCOME_WRONG_DIRECTION,
        runner.OUTCOME_NO_SHIFT_OF_TRAINING_MAGNITUDE,
        runner.OUTCOME_AMBIGUOUS_UNDERPOWERED,
    }


# ---------------------------------------------------------------------------
# 2. Hodges-Lehmann / bootstrap-CI determinism.
# ---------------------------------------------------------------------------


def test_hodges_lehmann_median_of_symmetric_data_is_the_mean(runner):
    values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert runner.hodges_lehmann_median(values) == pytest.approx(0.0)


def test_hodges_lehmann_median_single_value(runner):
    assert runner.hodges_lehmann_median([5.0]) == pytest.approx(5.0)


def test_hodges_lehmann_median_matches_hand_computed_walsh_averages(runner):
    # values = [1, 2, 4]; Walsh averages (i<=j): 1, 1.5, 2.5, 2, 3, 4 -> sorted:
    # [1, 1.5, 2, 2.5, 3, 4] -> median of 6 values = (2 + 2.5)/2 = 2.25
    assert runner.hodges_lehmann_median([1.0, 2.0, 4.0]) == pytest.approx(2.25)


def test_bootstrap_ci_is_deterministic_given_same_seed(runner):
    values = [0.1, 0.4, -0.2, 0.9, 0.3, -0.1, 0.6, 0.2, 0.05, -0.05]
    first = runner.percentile_bootstrap_hl_ci(values, resamples=500, seed=42)
    second = runner.percentile_bootstrap_hl_ci(values, resamples=500, seed=42)
    assert first == second


def test_bootstrap_ci_differs_with_different_seed(runner):
    # Deliberately irregular float values (not evenly spaced) so two
    # different seeds' resample draws are astronomically unlikely to land
    # on the exact same percentile boundary by coincidence.
    values = [0.137, 0.482, -0.261, 0.933, 0.318, -0.104, 0.657, 0.219, 0.071, -0.058]
    first = runner.percentile_bootstrap_hl_ci(values, resamples=2000, seed=1)
    second = runner.percentile_bootstrap_hl_ci(values, resamples=2000, seed=2)
    assert first.lo != second.lo or first.hi != second.hi


def test_bootstrap_ci_default_resamples_and_seed_match_module_constants(runner):
    values = [0.5, -0.3, 0.1, 0.2, -0.4]
    result = runner.percentile_bootstrap_hl_ci(values)
    assert result.resamples == runner.BOOTSTRAP_RESAMPLES == 10_000
    assert result.seed == runner.BOOTSTRAP_SEED


def test_bootstrap_ci_lo_le_point_estimate_le_hi_for_wide_spread(runner):
    values = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
    result = runner.percentile_bootstrap_hl_ci(values, resamples=2000, seed=7)
    assert result.lo <= result.point_estimate <= result.hi


def test_percentile_bootstrap_hl_ci_does_not_mutate_shared_random_state(runner):
    import random

    random.seed(12345)
    before = random.random()
    values = [0.1, 0.2, 0.3]
    runner.percentile_bootstrap_hl_ci(values, resamples=200, seed=99)
    random.seed(12345)
    after = random.random()
    assert before == after


# ---------------------------------------------------------------------------
# 3. Wilcoxon signed-rank correctness (sanity, not authoritative per section 3).
# ---------------------------------------------------------------------------


def test_wilcoxon_all_positive_deltas_gives_low_p_value(runner):
    deltas = [1.0, 2.0, 3.0, 1.5, 2.5, 0.8, 1.2, 2.2]
    result = runner.wilcoxon_signed_rank(deltas)
    assert result.p_value < 0.05
    assert result.n_effective == len(deltas)


def test_wilcoxon_symmetric_deltas_around_zero_gives_high_p_value(runner):
    deltas = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5]
    result = runner.wilcoxon_signed_rank(deltas)
    assert result.p_value > 0.5


def test_wilcoxon_drops_zero_deltas_from_n_effective(runner):
    deltas = [1.0, -1.0, 0.0, 0.0, 2.0]
    result = runner.wilcoxon_signed_rank(deltas)
    assert result.n_effective == 3


def test_wilcoxon_empty_input_is_defined_and_non_significant(runner):
    result = runner.wilcoxon_signed_rank([])
    assert result.n_effective == 0
    assert result.p_value == 1.0


# ---------------------------------------------------------------------------
# 4. Full statistics report: refusal/counter subset breakdowns, length-
#    normalization-divergence flag, primary-vs-secondary quantity.
# ---------------------------------------------------------------------------


def _make_choice_score(runner, sum_log_prob: float, token_count: int = 4):
    return runner.ChoiceScore(
        sum_log_prob=sum_log_prob,
        mean_log_prob=sum_log_prob / token_count,
        token_count=token_count,
        renderer_id="bare_text",
    )


def _make_pair_result(
    runner,
    pair_id: str,
    direction: str,
    *,
    candidate_margin_sum: float,
    reference_margin_sum: float,
    token_count: int = 4,
) -> runner.PairMarginResult:
    # Fix rejected at 0 for both models so margin == chosen's own sum.
    candidate_chosen = _make_choice_score(runner, candidate_margin_sum, token_count)
    candidate_rejected = _make_choice_score(runner, 0.0, token_count)
    reference_chosen = _make_choice_score(runner, reference_margin_sum, token_count)
    reference_rejected = _make_choice_score(runner, 0.0, token_count)
    return runner.PairMarginResult(
        pair_id=pair_id,
        direction=direction,
        semantic_family="confabulated_recipe_refusal_dpo",
        candidate_chosen=candidate_chosen,
        candidate_rejected=candidate_rejected,
        reference_chosen=reference_chosen,
        reference_rejected=reference_rejected,
    )


def test_build_statistics_report_computes_full_and_subset_breakdowns(runner):
    pairs = [
        _make_pair_result(runner, "p1", "refusal", candidate_margin_sum=2.0, reference_margin_sum=1.0),
        _make_pair_result(runner, "p2", "refusal", candidate_margin_sum=3.0, reference_margin_sum=1.0),
        _make_pair_result(runner, "p3", "counter", candidate_margin_sum=1.5, reference_margin_sum=1.0),
        _make_pair_result(runner, "p4", "counter", candidate_margin_sum=1.2, reference_margin_sum=1.0),
    ]
    report = runner.build_statistics_report(pairs)
    assert report["n_pairs"] == 4
    assert report["full"]["authoritative"] is True
    assert report["refusal_subset"]["n"] == 2
    assert report["refusal_subset"]["authoritative"] is False
    assert report["counter_subset"]["n"] == 2
    assert report["counter_subset"]["authoritative"] is False
    assert report["theta"] == runner.THETA


def test_build_statistics_report_flags_length_normalization_divergence(runner):
    # Construct a pair where the summed delta is positive but the mean delta
    # (a differently-length-normalized quantity) is negative, by giving
    # candidate a longer chosen continuation than the reference for the same
    # raw summed shift.
    candidate_chosen = runner.ChoiceScore(
        sum_log_prob=1.0, mean_log_prob=1.0 / 20, token_count=20, renderer_id="bare_text"
    )
    candidate_rejected = runner.ChoiceScore(
        sum_log_prob=0.0, mean_log_prob=0.0, token_count=20, renderer_id="bare_text"
    )
    reference_chosen = runner.ChoiceScore(
        sum_log_prob=0.5, mean_log_prob=0.5 / 2, token_count=2, renderer_id="bare_text"
    )
    reference_rejected = runner.ChoiceScore(
        sum_log_prob=0.0, mean_log_prob=0.0, token_count=2, renderer_id="bare_text"
    )
    pair = runner.PairMarginResult(
        pair_id="p1",
        direction="refusal",
        semantic_family="x",
        candidate_chosen=candidate_chosen,
        candidate_rejected=candidate_rejected,
        reference_chosen=reference_chosen,
        reference_rejected=reference_rejected,
    )
    # delta_sum = (1.0 - 0.0) - (0.5 - 0.0) = 0.5  (positive)
    # delta_mean = (0.05 - 0.0) - (0.25 - 0.0) = -0.2 (negative)
    assert pair.delta_sum == pytest.approx(0.5)
    assert pair.delta_mean == pytest.approx(-0.2)
    report = runner.build_statistics_report([pair])
    assert report["length_normalization_divergence"] is True


def test_build_statistics_report_no_divergence_when_signs_agree(runner):
    pairs = [
        _make_pair_result(runner, "p1", "refusal", candidate_margin_sum=2.0, reference_margin_sum=1.0),
    ]
    report = runner.build_statistics_report(pairs)
    assert report["length_normalization_divergence"] is False


def test_build_statistics_report_empty_input_raises(runner):
    with pytest.raises(ValueError):
        runner.build_statistics_report([])


def test_build_statistics_report_missing_direction_subset_is_none(runner):
    pairs = [
        _make_pair_result(runner, "p1", "refusal", candidate_margin_sum=2.0, reference_margin_sum=1.0),
    ]
    report = runner.build_statistics_report(pairs)
    assert report["counter_subset"] is None


def test_pair_margin_result_primary_is_summed_not_mean(runner):
    pair = _make_pair_result(runner, "p1", "refusal", candidate_margin_sum=4.0, reference_margin_sum=1.0)
    # sum-based delta should equal 4.0 - 1.0 = 3.0 regardless of token_count
    assert pair.delta_sum == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 5. score_choice: reuses _choice_log_likelihood, does not reimplement it.
# ---------------------------------------------------------------------------


def test_score_choice_converts_mean_to_sum_using_real_token_count(runner):
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        chat_template = None

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            assert return_tensors == "pt"
            if text == "prompt":
                return {"input_ids": torch.tensor([[1, 2]])}
            # " choice" tokenizes to 3 tokens in this fake tokenizer.
            assert add_special_tokens is False
            return {"input_ids": torch.tensor([[7, 8, 9]])}

    class FakeEvaluator:
        def _choice_log_likelihood(self, model, tokenizer, prompt, choice):
            # length-normalized mean of -0.3 per token
            return -0.3, "bare_text"

    evaluator = FakeEvaluator()
    score = runner.score_choice(evaluator, object(), FakeTokenizer(), "prompt", " choice")
    assert score.token_count == 3
    assert score.mean_log_prob == pytest.approx(-0.3)
    assert score.sum_log_prob == pytest.approx(-0.9)
    assert score.renderer_id == "bare_text"


def test_score_choice_rejects_zero_token_choice(runner):
    torch = pytest.importorskip("torch")

    class FakeTokenizer2:
        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            return {"input_ids": torch.tensor([[]])}

    class FakeEvaluator:
        def _choice_log_likelihood(self, model, tokenizer, prompt, choice):
            return -0.1, "bare_text"

    with pytest.raises(SystemExit, match="tokenized to zero tokens"):
        runner.score_choice(FakeEvaluator(), object(), FakeTokenizer2(), "prompt", "")


# ---------------------------------------------------------------------------
# 6. Held-out pair set loading and schema validation.
# ---------------------------------------------------------------------------


def _pair_record(pair_id: str, direction: str, split: str = "held_out") -> dict:
    return {
        "pair_id": pair_id,
        "prompt": f"prompt-{pair_id}",
        "chosen": "a hedged, cited answer",
        "rejected": "a confident fabrication",
        "direction": direction,
        "semantic_family": "confabulated_recipe_refusal_dpo",
        "content_class": "recipe",
        "source_scope": "synthetic",
        "citations": [],
        "split": split,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_load_held_out_pairs_accepts_a_valid_20_pair_set(runner, tmp_path):
    records = [_pair_record(f"p{i}", "counter" if i < 5 else "refusal") for i in range(20)]
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    loaded = runner.load_held_out_pairs(path)
    assert len(loaded) == 20


def test_load_held_out_pairs_rejects_wrong_count(runner, tmp_path):
    records = [_pair_record(f"p{i}", "counter" if i < 5 else "refusal") for i in range(19)]
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    with pytest.raises(SystemExit, match="expected exactly 20"):
        runner.load_held_out_pairs(path)


def test_load_held_out_pairs_rejects_missing_field(runner, tmp_path):
    records = [_pair_record(f"p{i}", "counter" if i < 5 else "refusal") for i in range(20)]
    del records[0]["citations"]
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    with pytest.raises(SystemExit, match="expected exactly"):
        runner.load_held_out_pairs(path)


def test_load_held_out_pairs_rejects_train_split(runner, tmp_path):
    records = [_pair_record(f"p{i}", "counter" if i < 5 else "refusal") for i in range(19)]
    records.append(_pair_record("p19", "refusal", split="train"))
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    with pytest.raises(SystemExit, match="not 'held_out'"):
        runner.load_held_out_pairs(path)


def test_load_held_out_pairs_rejects_bad_direction(runner, tmp_path):
    records = [_pair_record(f"p{i}", "counter" if i < 5 else "refusal") for i in range(19)]
    bad = _pair_record("p19", "refusal")
    bad["direction"] = "neutral"
    records.append(bad)
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    with pytest.raises(SystemExit, match="not one of"):
        runner.load_held_out_pairs(path)


def test_load_held_out_pairs_rejects_duplicate_pair_id(runner, tmp_path):
    records = [_pair_record(f"p{i}", "counter" if i < 5 else "refusal") for i in range(19)]
    records.append(_pair_record("p0", "refusal"))
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    with pytest.raises(SystemExit, match="duplicate pair_id"):
        runner.load_held_out_pairs(path)


def test_load_held_out_pairs_rejects_low_counter_share(runner, tmp_path):
    # Only 3/20 = 15% counter, below the 20% minimum.
    records = [_pair_record(f"p{i}", "counter" if i < 3 else "refusal") for i in range(20)]
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    with pytest.raises(SystemExit, match="counter-direction share"):
        runner.load_held_out_pairs(path)


def test_load_held_out_pairs_accepts_exactly_20_percent_counter_share(runner, tmp_path):
    # Exactly 4/20 = 20% counter -- the boundary itself must be accepted
    # (section 2: "counter-direction share >= 20%").
    records = [_pair_record(f"p{i}", "counter" if i < 4 else "refusal") for i in range(20)]
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    loaded = runner.load_held_out_pairs(path)
    assert len(loaded) == 20


# ---------------------------------------------------------------------------
# 7. Registry contamination re-check.
# ---------------------------------------------------------------------------


def test_verify_registry_admits_no_contamination_passes_for_clean_registry(runner, tmp_path):
    from held_out_eval import HeldOutExclusionRegistry

    registry = HeldOutExclusionRegistry()
    pair_ids = [f"p{i}" for i in range(20)]
    registry.register_package_held_out(runner.HELD_OUT_PACKAGE_ID, pair_ids)
    registry_path = tmp_path / "registry.json"
    registry.save(registry_path)

    result = runner.verify_registry_admits_no_contamination(registry_path, pair_ids)
    assert result.package_held_out_ids[runner.HELD_OUT_PACKAGE_ID] == frozenset(pair_ids)


def test_verify_registry_admits_no_contamination_rejects_train_overlap(runner, tmp_path):
    from held_out_eval import HeldOutExclusionRegistry

    registry = HeldOutExclusionRegistry()
    pair_ids = [f"p{i}" for i in range(20)]
    # Contaminate: one held-out id already used as train data elsewhere.
    registry.register_package_train("some-other-package", ["p0"])
    registry.package_held_out_ids[runner.HELD_OUT_PACKAGE_ID] = frozenset(pair_ids)
    registry_path = tmp_path / "registry.json"
    registry.save(registry_path)

    with pytest.raises(SystemExit, match="contamination"):
        runner.verify_registry_admits_no_contamination(registry_path, pair_ids)


def test_verify_registry_admits_no_contamination_rejects_missing_package(runner, tmp_path):
    from held_out_eval import HeldOutExclusionRegistry

    registry = HeldOutExclusionRegistry()
    registry_path = tmp_path / "registry.json"
    registry.save(registry_path)

    with pytest.raises(SystemExit, match="no held-out registration"):
        runner.verify_registry_admits_no_contamination(registry_path, ["p0"])


def test_verify_registry_admits_no_contamination_rejects_id_set_mismatch(runner, tmp_path):
    from held_out_eval import HeldOutExclusionRegistry

    registry = HeldOutExclusionRegistry()
    registry.register_package_held_out(runner.HELD_OUT_PACKAGE_ID, ["p0", "p1"])
    registry_path = tmp_path / "registry.json"
    registry.save(registry_path)

    with pytest.raises(SystemExit, match="do not match"):
        runner.verify_registry_admits_no_contamination(registry_path, ["p0", "p1", "p2"])


# ---------------------------------------------------------------------------
# 8. Gate refusal and hash-mismatch refusal.
# ---------------------------------------------------------------------------


def test_load_gate_rejects_schema_mismatch(runner, tmp_path):
    gate = {"nonsense": True}
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="schema mismatch"):
        runner.load_gate(path)


def test_load_gate_rejects_script_hash_mismatch(runner, tmp_path):
    gate = {
        "schema_version": 1,
        "script_sha256": "0" * 64,  # deliberately wrong
        "held_out_package_id": runner.HELD_OUT_PACKAGE_ID,
        "held_out_pairs_hash": "1" * 64,
        "held_out_registry_hash": "2" * 64,
        "measured_max_memory_mb": 500,
        "measured_max_wall_seconds": 60,
        "approval": {},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="different script content hash"):
        runner.load_gate(path)


def test_load_gate_rejects_wrong_package_id(runner, tmp_path):
    gate = {
        "schema_version": 1,
        "script_sha256": runner._this_script_sha256(),
        "held_out_package_id": "some-other-package",
        "held_out_pairs_hash": "1" * 64,
        "held_out_registry_hash": "2" * 64,
        "measured_max_memory_mb": 500,
        "measured_max_wall_seconds": 60,
        "approval": {},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="not bound to this evaluation"):
        runner.load_gate(path)


def test_load_gate_refuses_while_held_out_placeholder_is_unset(runner, tmp_path, monkeypatch):
    # Card 1 is sealed on main, so the module-level constants are real
    # pinned hashes by default now -- monkeypatch them back to the
    # placeholder ``None`` sentinel here to prove load_gate still fails
    # closed in that state, rather than guessing a hash.
    monkeypatch.setattr(runner, "EXPECTED_HELD_OUT_PAIRS_HASH", None)
    monkeypatch.setattr(runner, "EXPECTED_HELD_OUT_REGISTRY_HASH", None)
    assert runner.EXPECTED_HELD_OUT_PAIRS_HASH is None
    assert runner.EXPECTED_HELD_OUT_REGISTRY_HASH is None
    gate = {
        "schema_version": 1,
        "script_sha256": runner._this_script_sha256(),
        "held_out_package_id": runner.HELD_OUT_PACKAGE_ID,
        "held_out_pairs_hash": "1" * 64,
        "held_out_registry_hash": "2" * 64,
        "measured_max_memory_mb": 500,
        "measured_max_wall_seconds": 60,
        "approval": {},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="card 1"):
        runner.load_gate(path)


def test_load_gate_rejects_bad_memory_ceiling_type(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EXPECTED_HELD_OUT_PAIRS_HASH", "1" * 64)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text('{"schema_version": 1}', encoding="utf-8")
    monkeypatch.setattr(runner, "HELD_OUT_REGISTRY_PATH", registry_path)
    real_registry_hash = registry_path.read_bytes()
    import hashlib

    registry_hash = hashlib.sha256(real_registry_hash).hexdigest()
    monkeypatch.setattr(runner, "EXPECTED_HELD_OUT_REGISTRY_HASH", registry_hash)
    gate = {
        "schema_version": 1,
        "script_sha256": runner._this_script_sha256(),
        "held_out_package_id": runner.HELD_OUT_PACKAGE_ID,
        "held_out_pairs_hash": "1" * 64,
        "held_out_registry_hash": registry_hash,
        "measured_max_memory_mb": -5,  # deliberately invalid
        "measured_max_wall_seconds": 60,
        "approval": {},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="measured_max_memory_mb"):
        runner.load_gate(path)


def test_load_gate_rejects_stale_registry_hash(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EXPECTED_HELD_OUT_PAIRS_HASH", "1" * 64)
    monkeypatch.setattr(runner, "EXPECTED_HELD_OUT_REGISTRY_HASH", "2" * 64)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text('{"schema_version": 1}', encoding="utf-8")
    monkeypatch.setattr(runner, "HELD_OUT_REGISTRY_PATH", registry_path)
    gate = {
        "schema_version": 1,
        "script_sha256": runner._this_script_sha256(),
        "held_out_package_id": runner.HELD_OUT_PACKAGE_ID,
        "held_out_pairs_hash": "1" * 64,
        "held_out_registry_hash": "2" * 64,
        "measured_max_memory_mb": 500,
        "measured_max_wall_seconds": 60,
        "approval": {},
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match the registry file"):
        runner.load_gate(path)


def test_verify_signed_approval_rejects_wrong_field_set(runner, tmp_path):
    approval = {"document": "x"}  # missing signature/document_sha256
    with pytest.raises(SystemExit, match="approval must contain exactly"):
        runner._verify_signed_approval(
            approval,
            script_sha256="a" * 64,
            held_out_registry_hash="b" * 64,
            measured_max_memory_mb=1.0,
            measured_max_wall_seconds=1.0,
        )


def test_verify_signed_approval_rejects_missing_document_file(runner, tmp_path):
    signature_path = tmp_path / "sig"
    signature_path.write_text("sig", encoding="utf-8")
    approval = {
        "document": str(tmp_path / "does-not-exist.json"),
        "signature": str(signature_path),
        "document_sha256": "a" * 64,
    }
    with pytest.raises(SystemExit, match="approval document/signature is missing"):
        runner._verify_signed_approval(
            approval,
            script_sha256="a" * 64,
            held_out_registry_hash="b" * 64,
            measured_max_memory_mb=1.0,
            measured_max_wall_seconds=1.0,
        )


# ---------------------------------------------------------------------------
# 9. Model hash-mismatch refusal (validate_plan) -- no real snapshot needed.
# ---------------------------------------------------------------------------


def test_validate_plan_detects_candidate_hash_mismatch(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "EXPECTED_CANDIDATE_MODEL_HASH", "2" * 64)  # deliberately wrong
    with pytest.raises(SystemExit, match="candidate checkpoint content hash mismatch"):
        runner.validate_plan()


def test_validate_plan_detects_reference_hash_mismatch(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "EXPECTED_REFERENCE_MODEL_HASH", "3" * 64)  # deliberately wrong
    with pytest.raises(SystemExit, match="reference checkpoint content hash mismatch"):
        runner.validate_plan()


def test_validate_plan_passes_with_valid_fake_checkpoints_and_reports_blockers(
    runner, tmp_path, monkeypatch
):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    result = runner.validate_plan()
    assert result["status"] == "PASS"
    assert result["scoring_called"] is False
    # Card 1 is sealed on main and the real held-out pair set/registry
    # files (matching the pinned hashes) ship in this checkout, so the
    # held-out placeholder blocker must be gone; only the still-missing
    # review gate remains.
    assert not any("card 1" in b for b in result["execution_blockers"])
    assert any("gate" in b.lower() for b in result["execution_blockers"])


def test_validate_plan_reports_placeholder_blocker_when_held_out_unset(
    runner, tmp_path, monkeypatch
):
    # Direct test of the fail-closed placeholder path: monkeypatch the
    # pinned hashes back to the ``None`` sentinel and confirm the "card
    # 1 not sealed" blocker reappears exactly as it did before sealing.
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "EXPECTED_HELD_OUT_PAIRS_HASH", None)
    monkeypatch.setattr(runner, "EXPECTED_HELD_OUT_REGISTRY_HASH", None)
    result = runner.validate_plan()
    assert result["status"] == "PASS"
    assert any("card 1" in b for b in result["execution_blockers"])
    assert any("gate" in b.lower() for b in result["execution_blockers"])


def test_validate_plan_refuses_on_mismatched_held_out_pairs_hash(
    runner, tmp_path, monkeypatch
):
    # A pinned hash that no longer matches the file on disk (tampered or
    # stale content) must be reported as a blocker, not silently ignored.
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "EXPECTED_HELD_OUT_PAIRS_HASH", "f" * 64)
    result = runner.validate_plan()
    assert result["status"] == "PASS"
    assert any("held-out pair set file is missing or hash-mismatched" in b for b in result["execution_blockers"])


def test_validate_plan_refuses_on_mismatched_held_out_registry_hash(
    runner, tmp_path, monkeypatch
):
    # Same guarantee for the exclusion-registry hash.
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "EXPECTED_HELD_OUT_REGISTRY_HASH", "f" * 64)
    result = runner.validate_plan()
    assert result["status"] == "PASS"
    assert any("held-out registry file is missing or hash-mismatched" in b for b in result["execution_blockers"])


def test_validate_plan_never_imports_torch_or_transformers(runner, tmp_path, monkeypatch):
    # Dry-run validation must stay pure-hash/schema: no model load, no
    # torch/transformers import triggered anywhere in validate_plan().
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    for name in list(sys.modules):
        if name.startswith(("torch", "transformers")):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    result = runner.validate_plan()
    assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# 10. Dry-run / --execute split at the argparse boundary.
# ---------------------------------------------------------------------------


def test_main_without_execute_flag_never_calls_execute_evaluation(runner, monkeypatch):
    calls = {"validate_plan": 0, "execute_evaluation": 0}

    def _fake_validate_plan():
        calls["validate_plan"] += 1
        return {"status": "PASS", "scoring_called": False, "execution_blockers": []}

    def _fake_execute(*args, **kwargs):
        calls["execute_evaluation"] += 1
        raise AssertionError("execute_evaluation must not be called without --execute")

    monkeypatch.setattr(runner, "validate_plan", _fake_validate_plan)
    monkeypatch.setattr(runner, "execute_evaluation", _fake_execute)
    monkeypatch.setattr(sys, "argv", ["run_adr0019_logprob_margin_eval.py"])
    exit_code = runner.main()
    assert exit_code == 0
    assert calls["validate_plan"] == 1
    assert calls["execute_evaluation"] == 0


def test_execute_without_review_gate_is_rejected(runner, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["run_adr0019_logprob_margin_eval.py", "--execute"]
    )
    with pytest.raises(SystemExit, match="requires --review-gate"):
        runner.main()


def test_execute_rejects_non_reviewed_scratch_root(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_adr0019_logprob_margin_eval.py",
            "--execute",
            "--review-gate",
            str(runner.APPROVED_REVIEW_GATE_PATH),
            "--scratch-root",
            str(tmp_path / "not-approved"),
        ],
    )
    with pytest.raises(SystemExit, match="exact reviewed scratch root"):
        runner.main()


def test_execute_rejects_non_reviewed_gate_path(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "APPROVED_EXECUTION_SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_adr0019_logprob_margin_eval.py",
            "--execute",
            "--review-gate",
            str(tmp_path / "not-approved-gate.json"),
            "--scratch-root",
            str(tmp_path / "scratch"),
        ],
    )
    with pytest.raises(SystemExit, match="exact reviewed gate path"):
        runner.main()


def test_execute_evaluation_refuses_before_loading_any_model_when_blocked(
    runner, tmp_path, monkeypatch
):
    # With the held-out set unsealed (module default), execute_evaluation
    # must refuse via validate_plan()'s blockers before ever calling
    # HFLocalCausalLMEvaluatorAdapter._get_model -- proven by monkeypatching
    # _get_model to explode if reached.
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)

    def _boom(self, path):
        raise AssertionError("must not load a model while execution is blocked")

    monkeypatch.setattr(
        runner.HFLocalCausalLMEvaluatorAdapter, "_get_model", _boom, raising=True
    )

    gate = {
        "schema_version": 1,
        "script_sha256": runner._this_script_sha256(),
        "held_out_package_id": runner.HELD_OUT_PACKAGE_ID,
        "held_out_pairs_hash": "1" * 64,
        "held_out_registry_hash": "2" * 64,
        "measured_max_memory_mb": 500,
        "measured_max_wall_seconds": 60,
        "approval": {},
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    with pytest.raises(SystemExit):
        runner.execute_evaluation(tmp_path / "scratch", gate_path)


# ---------------------------------------------------------------------------
# 11. Structural: this script's own content hash helper is self-consistent.
# ---------------------------------------------------------------------------


def test_this_script_sha256_matches_a_direct_recompute(runner):
    import hashlib

    digest = hashlib.sha256(RUNNER_PATH.read_bytes()).hexdigest()
    assert runner._this_script_sha256() == digest
