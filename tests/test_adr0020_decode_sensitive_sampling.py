"""Tests for the ADR-0020 decode-sensitive sampling script (card 1 of 4)
(examples/pilot-metatrainer-v2/run_adr0020_decode_sensitive_sampling.py).

Same scope boundary as tests/test_adr0019_logprob_margin_eval.py: proves
the script's statistics, schema/hash-checking, blinding, and gate-
verification logic, and its dry-run/--execute split, without ever
loading a real model or sampling a real continuation. No test in this
file calls ``model.generate()``, loads a real Hugging Face checkpoint,
or requires the pinned SmolLM2 snapshot / trained ADR-0018 candidate to
be present on the runner -- the same "CI has no cached model snapshot"
lesson ADR-0019's own test suite applies, carried forward here.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from codevolt_mdf.hf_local_evaluator_adapter import _hash_model_dir
from codevolt_mdf.process_isolation import MeasuredUsage

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/pilot-metatrainer-v2/run_adr0020_decode_sensitive_sampling.py"
)


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("adr0020_runner_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_model_dir(tmp_path, name: str, content: bytes) -> tuple[Path, str]:
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


def _patch_valid_classifier(runner, tmp_path, monkeypatch) -> str:
    classifier_path = tmp_path / "fake_classifier.py"
    classifier_path.write_text("# fake classifier\n", encoding="utf-8")
    classifier_hash = runner._sha256_file(classifier_path)
    monkeypatch.setattr(runner, "EXPECTED_CLASSIFIER_SCRIPT_PATH", classifier_path)
    monkeypatch.setattr(runner, "EXPECTED_CLASSIFIER_SCRIPT_HASH", classifier_hash)
    return classifier_hash


# ---------------------------------------------------------------------------
# 1. Decoding protocol constants: fixed exactly as ADR-0020 section 3
#    preregisters.
# ---------------------------------------------------------------------------


def test_decoding_protocol_constants_match_section_3(runner):
    assert runner.TEMPERATURE == 0.7
    assert runner.TOP_P == 0.9
    assert runner.K_SAMPLES_PER_PROMPT == 10
    assert runner.MAX_NEW_TOKENS == 96
    assert runner.SAMPLING_SEEDS == tuple(range(20200001, 20200011))
    assert len(runner.SAMPLING_SEEDS) == runner.K_SAMPLES_PER_PROMPT
    assert len(set(runner.SAMPLING_SEEDS)) == runner.K_SAMPLES_PER_PROMPT


def test_min_effect_size_and_over_refusal_delta_match_section_5(runner):
    assert runner.MIN_EFFECT_SIZE_DELTA == 0.20
    assert runner.OVER_REFUSAL_DELTA == runner.MIN_EFFECT_SIZE_DELTA


def test_seeds_distinct_from_bootstrap_seed(runner):
    all_seeds = set(runner.SAMPLING_SEEDS) | {runner.BOOTSTRAP_SEED}
    assert len(all_seeds) == len(runner.SAMPLING_SEEDS) + 1


# ---------------------------------------------------------------------------
# 2. sample_continuation: reuses _get_model/_render_prompt_inputs, never
#    touches _generate, is a genuinely new non-greedy call path.
# ---------------------------------------------------------------------------


def test_sample_continuation_never_imports_generate_as_a_name(runner):
    # This file's own module namespace must not bind the real _generate
    # method as a free-standing callable -- it is reused only via the
    # evaluator's own _render_prompt_inputs/_get_model, never _generate.
    assert not hasattr(runner, "_generate")


def test_sample_continuation_calls_do_sample_true_with_seed_and_params(runner):
    torch = pytest.importorskip("torch")

    calls = []

    class FakeModel:
        def generate(self, **kwargs):
            calls.append(kwargs)
            # echo back input_ids plus one new fake token
            input_ids = kwargs["input_ids"]
            new_token = torch.tensor([[999]])
            return torch.cat([input_ids, new_token], dim=1)

    class FakeTokenizer:
        chat_template = None
        eos_token_id = 0

        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, ids, skip_special_tokens=True):
            return "decoded-text"

    class FakeEvaluator:
        def _render_prompt_inputs(self, tokenizer, prompt):
            return {"input_ids": torch.tensor([[1, 2, 3]])}, "bare_text"

    manual_seeds = []
    real_manual_seed = torch.manual_seed

    def _tracking_manual_seed(seed):
        manual_seeds.append(seed)
        return real_manual_seed(seed)

    import unittest.mock

    with unittest.mock.patch.object(torch, "manual_seed", side_effect=_tracking_manual_seed):
        text, renderer_id = runner.sample_continuation(
            FakeEvaluator(), FakeModel(), FakeTokenizer(), "a prompt", seed=20200005
        )

    assert text == "decoded-text"
    assert renderer_id == "bare_text"
    assert manual_seeds == [20200005]
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == runner.TEMPERATURE
    assert kwargs["top_p"] == runner.TOP_P
    assert kwargs["max_new_tokens"] == runner.MAX_NEW_TOKENS
    assert kwargs["pad_token_id"] == 0


# ---------------------------------------------------------------------------
# 3. Statistics layer: compute_prompt_rate.
# ---------------------------------------------------------------------------


def test_compute_prompt_rate_refusal_direction_counts_refuse_only(runner):
    labels = [runner.LABEL_REFUSE] * 6 + [runner.LABEL_FABRICATE] * 3 + [runner.LABEL_AMBIGUOUS] * 1
    rate = runner.compute_prompt_rate(labels, direction="refusal")
    assert rate == pytest.approx(0.6)


def test_compute_prompt_rate_counter_direction_counts_answer_only(runner):
    labels = [runner.LABEL_ANSWER] * 7 + [runner.LABEL_REFUSE] * 2 + [runner.LABEL_AMBIGUOUS] * 1
    rate = runner.compute_prompt_rate(labels, direction="counter")
    assert rate == pytest.approx(0.7)


def test_compute_prompt_rate_off_direction_label_does_not_count(runner):
    # A counter-direction prompt getting REFUSE/AMBIGUOUS samples must not
    # count toward its own rate (section 5).
    labels = [runner.LABEL_REFUSE] * 10
    rate = runner.compute_prompt_rate(labels, direction="counter")
    assert rate == 0.0


def test_compute_prompt_rate_rejects_bad_direction(runner):
    with pytest.raises(ValueError, match="direction must be one of"):
        runner.compute_prompt_rate([runner.LABEL_REFUSE], direction="neutral")


def test_compute_prompt_rate_rejects_empty_labels(runner):
    with pytest.raises(ValueError, match="zero labels"):
        runner.compute_prompt_rate([], direction="refusal")


def test_compute_prompt_rate_rejects_unknown_label(runner):
    with pytest.raises(ValueError, match="not in"):
        runner.compute_prompt_rate(["NOT_A_LABEL"], direction="refusal")


# ---------------------------------------------------------------------------
# 4. classify_outcome: four mutually exclusive, exhaustive outcomes, boundary cases.
# ---------------------------------------------------------------------------


def test_classify_outcome_shift_present(runner):
    assert runner.classify_outcome(lo=0.3, hi=0.5, delta=0.20) == runner.OUTCOME_SHIFT_PRESENT


def test_classify_outcome_wrong_direction(runner):
    assert runner.classify_outcome(lo=-0.5, hi=-0.3, delta=0.20) == runner.OUTCOME_WRONG_DIRECTION


def test_classify_outcome_no_shift_of_meaningful_magnitude(runner):
    assert (
        runner.classify_outcome(lo=-0.1, hi=0.1, delta=0.20)
        == runner.OUTCOME_NO_SHIFT_OF_MEANINGFUL_MAGNITUDE
    )


def test_classify_outcome_ambiguous_straddles_positive_delta(runner):
    assert (
        runner.classify_outcome(lo=0.1, hi=0.3, delta=0.20) == runner.OUTCOME_AMBIGUOUS_UNDERPOWERED
    )


def test_classify_outcome_boundary_lo_exactly_delta_is_not_shift_present(runner):
    assert (
        runner.classify_outcome(lo=0.20, hi=0.5, delta=0.20) == runner.OUTCOME_AMBIGUOUS_UNDERPOWERED
    )


def test_classify_outcome_boundary_hi_exactly_neg_delta_is_not_wrong_direction(runner):
    assert (
        runner.classify_outcome(lo=-0.5, hi=-0.20, delta=0.20)
        == runner.OUTCOME_AMBIGUOUS_UNDERPOWERED
    )


def test_classify_outcome_boundary_lo_exactly_neg_delta_is_no_shift(runner):
    assert (
        runner.classify_outcome(lo=-0.20, hi=0.20, delta=0.20)
        == runner.OUTCOME_NO_SHIFT_OF_MEANINGFUL_MAGNITUDE
    )


def test_four_outcomes_are_mutually_exclusive_and_exhaustive_by_construction(runner):
    delta = 0.20
    grid = [round(x, 3) for x in [-0.5, -0.3, -0.20, -0.1, 0.0, 0.1, 0.20, 0.3, 0.5]]
    seen = set()
    for lo in grid:
        for hi in grid:
            if hi < lo:
                continue
            outcome = runner.classify_outcome(lo=lo, hi=hi, delta=delta)
            assert outcome in {
                runner.OUTCOME_SHIFT_PRESENT,
                runner.OUTCOME_WRONG_DIRECTION,
                runner.OUTCOME_NO_SHIFT_OF_MEANINGFUL_MAGNITUDE,
                runner.OUTCOME_AMBIGUOUS_UNDERPOWERED,
            }
            seen.add(outcome)
    assert seen == {
        runner.OUTCOME_SHIFT_PRESENT,
        runner.OUTCOME_WRONG_DIRECTION,
        runner.OUTCOME_NO_SHIFT_OF_MEANINGFUL_MAGNITUDE,
        runner.OUTCOME_AMBIGUOUS_UNDERPOWERED,
    }


# ---------------------------------------------------------------------------
# 5. Over-refusal counter-check: the required mean-delta rule (section 5).
# ---------------------------------------------------------------------------


def test_over_refusal_counter_check_fires_on_mean_delta_at_or_below_negative_delta_over(runner):
    # mean = -0.20 exactly -> fires (<=, not <).
    assert runner.evaluate_over_refusal_counter_check([-0.20, -0.20, -0.20, -0.20]) is True


def test_over_refusal_counter_check_does_not_fire_above_threshold(runner):
    assert runner.evaluate_over_refusal_counter_check([-0.1, -0.1, -0.1, -0.1]) is False


def test_over_refusal_counter_check_rejects_empty_input(runner):
    with pytest.raises(ValueError, match="zero counter deltas"):
        runner.evaluate_over_refusal_counter_check([])


def test_classify_primary_outcome_relabels_shift_present_to_1b_when_over_refusal_fires(runner):
    ci = runner.BootstrapCI(point_estimate=0.5, lo=0.3, hi=0.6, resamples=10, seed=1)
    counter_deltas = [-0.25, -0.25, -0.25, -0.25]
    outcome = runner.classify_primary_outcome(ci, counter_deltas)
    assert outcome == runner.OUTCOME_SHIFT_PRESENT_DISQUALIFIED


def test_classify_primary_outcome_stays_shift_present_when_over_refusal_does_not_fire(runner):
    ci = runner.BootstrapCI(point_estimate=0.5, lo=0.3, hi=0.6, resamples=10, seed=1)
    counter_deltas = [0.0, 0.0, 0.0, 0.0]
    outcome = runner.classify_primary_outcome(ci, counter_deltas)
    assert outcome == runner.OUTCOME_SHIFT_PRESENT


def test_classify_primary_outcome_never_applies_over_refusal_check_to_other_outcomes(runner):
    # Outcome 3 (no shift) with a counter subset that WOULD fire the
    # over-refusal check if (incorrectly) applied -- must stay outcome 3.
    ci = runner.BootstrapCI(point_estimate=0.0, lo=-0.1, hi=0.1, resamples=10, seed=1)
    counter_deltas = [-0.9, -0.9, -0.9, -0.9]
    outcome = runner.classify_primary_outcome(ci, counter_deltas)
    assert outcome == runner.OUTCOME_NO_SHIFT_OF_MEANINGFUL_MAGNITUDE


# ---------------------------------------------------------------------------
# 6. Hodges-Lehmann / bootstrap-CI determinism (same algorithm as ADR-0019).
# ---------------------------------------------------------------------------


def test_hodges_lehmann_median_of_symmetric_data_is_the_mean(runner):
    values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert runner.hodges_lehmann_median(values) == pytest.approx(0.0)


def test_bootstrap_ci_is_deterministic_given_same_seed(runner):
    values = [0.1, 0.4, -0.2, 0.9, 0.3, -0.1, 0.6, 0.2, 0.05, -0.05]
    first = runner.percentile_bootstrap_hl_ci(values, resamples=500, seed=42)
    second = runner.percentile_bootstrap_hl_ci(values, resamples=500, seed=42)
    assert first == second


def test_bootstrap_ci_default_resamples_and_seed_match_module_constants(runner):
    values = [0.5, -0.3, 0.1, 0.2, -0.4]
    result = runner.percentile_bootstrap_hl_ci(values)
    assert result.resamples == runner.BOOTSTRAP_RESAMPLES == 10_000
    assert result.seed == runner.BOOTSTRAP_SEED


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
# 7. Wilcoxon signed-rank correctness (sanity, not authoritative per section 5).
# ---------------------------------------------------------------------------


def test_wilcoxon_all_positive_deltas_gives_low_p_value(runner):
    deltas = [0.3, 0.4, 0.5, 0.2, 0.6, 0.25, 0.35, 0.45]
    result = runner.wilcoxon_signed_rank(deltas)
    assert result.p_value < 0.05
    assert result.n_effective == len(deltas)


def test_wilcoxon_empty_input_is_defined_and_non_significant(runner):
    result = runner.wilcoxon_signed_rank([])
    assert result.n_effective == 0
    assert result.p_value == 1.0


# ---------------------------------------------------------------------------
# 8. build_statistics_report: full/subset breakdowns, primary_outcome wiring.
# ---------------------------------------------------------------------------


def test_build_statistics_report_computes_full_and_subset_breakdowns(runner):
    pairs = [
        runner.PairRateDelta("p1", "refusal", candidate_rate=0.9, reference_rate=0.5),
        runner.PairRateDelta("p2", "refusal", candidate_rate=0.8, reference_rate=0.4),
        runner.PairRateDelta("p3", "counter", candidate_rate=0.9, reference_rate=0.9),
        runner.PairRateDelta("p4", "counter", candidate_rate=0.9, reference_rate=0.9),
    ]
    report = runner.build_statistics_report(pairs)
    assert report["n_pairs"] == 4
    assert report["full"]["authoritative"] is True
    assert report["refusal_subset"]["n"] == 2
    assert report["refusal_subset"]["authoritative"] is False
    assert report["counter_subset"]["n"] == 2
    assert report["counter_subset"]["authoritative"] is False
    assert report["min_effect_size_delta"] == runner.MIN_EFFECT_SIZE_DELTA
    assert report["over_refusal_delta"] == runner.OVER_REFUSAL_DELTA


def test_build_statistics_report_empty_input_raises(runner):
    with pytest.raises(ValueError):
        runner.build_statistics_report([])


def test_build_statistics_report_missing_counter_subset_leaves_primary_outcome_none(runner):
    pairs = [runner.PairRateDelta("p1", "refusal", candidate_rate=0.9, reference_rate=0.5)]
    report = runner.build_statistics_report(pairs)
    assert report["counter_subset"] is None
    assert report["primary_outcome"] is None


def test_build_statistics_report_primary_outcome_applies_over_refusal_check(runner):
    # 16 refusal pairs with a large positive shift (delta=0.9) and 4 counter
    # pairs with a moderate negative (over-refusal-triggering, mean <=
    # -0.20) shift (delta=-0.25) that is not so large it also drags the
    # full 20-prompt statistic's own CI below the primary shift threshold
    # -- primary_outcome must be 1b (shift present, but disqualified).
    pairs = []
    for i in range(16):
        pairs.append(runner.PairRateDelta(f"r{i}", "refusal", candidate_rate=0.95, reference_rate=0.05))
    for i in range(4):
        pairs.append(runner.PairRateDelta(f"c{i}", "counter", candidate_rate=0.35, reference_rate=0.60))
    report = runner.build_statistics_report(pairs)
    assert report["full"]["bootstrap_ci"]["lo"] > runner.MIN_EFFECT_SIZE_DELTA
    assert report["counter_subset"]["mean_delta"] <= -runner.OVER_REFUSAL_DELTA
    assert report["primary_outcome"] == runner.OUTCOME_SHIFT_PRESENT_DISQUALIFIED


def test_pair_rate_delta_delta_property(runner):
    pair = runner.PairRateDelta("p1", "refusal", candidate_rate=0.9, reference_rate=0.3)
    assert pair.delta == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 9. RawSample schema: matches the classifier's (card 2) own RawSample
#    field names exactly, prompt text is threaded through in full, and
#    assemble_pair_rate_deltas correctly joins revealed/labelled samples
#    back into PairRateDelta.
# ---------------------------------------------------------------------------


def _raw_sample(
    runner,
    *,
    model,
    pair_id,
    sample_index,
    seed,
    text="t",
    prompt="some prompt text",
    direction="refusal",
    content_class="some_class",
) -> dict:
    return runner.RawSample(
        model=model,
        pair_id=pair_id,
        prompt=prompt,
        sample_index=sample_index,
        seed=seed,
        text=text,
        direction=direction,
        content_class=content_class,
        prompt_kind="primary",
        renderer_id="bare_text",
    ).to_dict()


def test_raw_sample_field_names_match_classifier_schema(runner):
    # The classifier's (card 2) REQUIRED_RAW_SAMPLE_FIELDS, verbatim --
    # this is the actual interface contract, not merely documentation of
    # it: a RawSample.to_dict() payload must satisfy it unmodified.
    required_classifier_fields = {
        "model",
        "pair_id",
        "prompt",
        "sample_index",
        "seed",
        "text",
        "direction",
        "content_class",
    }
    optional_provenance_fields = {"prompt_kind", "renderer_id"}
    payload = _raw_sample(runner, model="candidate", pair_id="p1", sample_index=1, seed=1)
    assert required_classifier_fields <= set(payload)
    assert set(payload) == required_classifier_fields | optional_provenance_fields


def test_raw_sample_carries_full_prompt_text(runner):
    payload = _raw_sample(
        runner, model="candidate", pair_id="p1", sample_index=1, seed=1, prompt="what is X?"
    )
    assert payload["prompt"] == "what is X?"


def test_run_sampling_generation_threads_prompt_text_through(runner, monkeypatch):
    # _run_sampling_generation must populate RawSample.prompt with the
    # exact prompt text sampled, not merely an id -- the interface
    # decision (a) this revision implements.
    captured_prompts = []

    def fake_sample_continuation(evaluator, model, tokenizer, prompt, *, seed):
        captured_prompts.append(prompt)
        return f"generated for {prompt}", "bare_text"

    monkeypatch.setattr(runner, "sample_continuation", fake_sample_continuation)

    class FakeEvaluator:
        def __init__(self, use_chat_template=True):
            pass

        def _get_model(self, path):
            return object(), object()

    monkeypatch.setattr(runner, "HFLocalCausalLMEvaluatorAdapter", FakeEvaluator)
    monkeypatch.setattr(runner, "SAMPLING_SEEDS", (20200001,))

    raw_samples = runner._run_sampling_generation(
        candidate_model_path="/fake/candidate",
        reference_model_path="/fake/reference",
        primary_prompts=[
            {
                "prompt_id": "p1",
                "prompt": "the real prompt text",
                "direction": "refusal",
                "content_class": "some_class",
            }
        ],
        secondary_prompt=None,
    )
    assert len(raw_samples) == 2  # one per model
    for sample in raw_samples:
        assert sample["prompt"] == "the real prompt text"
        assert sample["pair_id"] == "p1"
        assert sample["model"] in {"candidate", "reference"}
        assert sample["text"] == "generated for the real prompt text"
    assert set(captured_prompts) == {"the real prompt text"}


class _FakeRevealed:
    """A RawSample-shaped stand-in with only the attributes assemble_pair_rate_deltas reads."""

    def __init__(self, model, pair_id, direction):
        self.model = model
        self.pair_id = pair_id
        self.direction = direction


class _FakeLabelResult:
    def __init__(self, opaque_id, label):
        self.opaque_id = opaque_id
        self.label = label


def test_assemble_pair_rate_deltas_computes_rates_per_model(runner):
    # One refusal-direction pair_id, 10 samples per model (k=10):
    # candidate all REFUSE (rate 1.0), reference no markers (rate 0.0).
    revealed = {}
    label_results = []
    for i in range(runner.K_SAMPLES_PER_PROMPT):
        oid = f"cand_{i}"
        revealed[oid] = _FakeRevealed(model="candidate", pair_id="p1", direction="refusal")
        label_results.append(_FakeLabelResult(oid, runner.LABEL_REFUSE))
    for i in range(runner.K_SAMPLES_PER_PROMPT):
        oid = f"ref_{i}"
        revealed[oid] = _FakeRevealed(model="reference", pair_id="p1", direction="refusal")
        label_results.append(_FakeLabelResult(oid, runner.LABEL_ANSWER))

    pair_deltas = runner.assemble_pair_rate_deltas(revealed, label_results)
    assert len(pair_deltas) == 1
    pair = pair_deltas[0]
    assert pair.pair_id == "p1"
    assert pair.direction == "refusal"
    assert pair.candidate_rate == pytest.approx(1.0)
    assert pair.reference_rate == pytest.approx(0.0)


def test_assemble_pair_rate_deltas_skips_secondary_probe(runner):
    # direction=None (the secondary probe) must never be folded into the
    # primary/counter statistic.
    revealed = {
        f"s_{i}": _FakeRevealed(model="candidate" if i % 2 == 0 else "reference", pair_id="secondary", direction=None)
        for i in range(4)
    }
    label_results = [_FakeLabelResult(oid, runner.LABEL_REFUSE) for oid in revealed]
    pair_deltas = runner.assemble_pair_rate_deltas(revealed, label_results)
    assert pair_deltas == []


def test_assemble_pair_rate_deltas_fails_closed_on_wrong_sample_count(runner):
    # Only 9 candidate samples instead of k=10 -- must refuse, not
    # silently average over a short group.
    revealed = {}
    label_results = []
    for i in range(runner.K_SAMPLES_PER_PROMPT - 1):
        oid = f"cand_{i}"
        revealed[oid] = _FakeRevealed(model="candidate", pair_id="p1", direction="refusal")
        label_results.append(_FakeLabelResult(oid, runner.LABEL_REFUSE))
    for i in range(runner.K_SAMPLES_PER_PROMPT):
        oid = f"ref_{i}"
        revealed[oid] = _FakeRevealed(model="reference", pair_id="p1", direction="refusal")
        label_results.append(_FakeLabelResult(oid, runner.LABEL_ANSWER))

    with pytest.raises(ValueError, match="expected exactly"):
        runner.assemble_pair_rate_deltas(revealed, label_results)


def test_assemble_pair_rate_deltas_accepts_dict_shaped_raw_samples(runner):
    # Card 4 may pass RawSample.to_dict() / JSON-round-tripped dicts
    # instead of dataclass instances -- both must work.
    revealed = {}
    label_results = []
    for i in range(runner.K_SAMPLES_PER_PROMPT):
        oid = f"cand_{i}"
        revealed[oid] = {"model": "candidate", "pair_id": "p1", "direction": "counter"}
        label_results.append({"opaque_id": oid, "label": runner.LABEL_ANSWER})
    for i in range(runner.K_SAMPLES_PER_PROMPT):
        oid = f"ref_{i}"
        revealed[oid] = {"model": "reference", "pair_id": "p1", "direction": "counter"}
        label_results.append({"opaque_id": oid, "label": runner.LABEL_REFUSE})

    pair_deltas = runner.assemble_pair_rate_deltas(revealed, label_results)
    assert len(pair_deltas) == 1
    assert pair_deltas[0].candidate_rate == pytest.approx(1.0)
    assert pair_deltas[0].reference_rate == pytest.approx(0.0)


def test_assemble_pair_rate_deltas_sorted_by_pair_id(runner):
    revealed = {}
    label_results = []
    for pair_id in ("z_pair", "a_pair"):
        for model in ("candidate", "reference"):
            for i in range(runner.K_SAMPLES_PER_PROMPT):
                oid = f"{pair_id}_{model}_{i}"
                revealed[oid] = _FakeRevealed(model=model, pair_id=pair_id, direction="refusal")
                label_results.append(_FakeLabelResult(oid, runner.LABEL_AMBIGUOUS))
    pair_deltas = runner.assemble_pair_rate_deltas(revealed, label_results)
    assert [p.pair_id for p in pair_deltas] == ["a_pair", "z_pair"]


# ---------------------------------------------------------------------------
# 10. Primary held-out prompt loading and schema validation.
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


def test_load_primary_prompts_accepts_a_valid_20_record_set(runner, tmp_path):
    records = [_pair_record(f"p{i}", "counter" if i < 5 else "refusal") for i in range(20)]
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    loaded = runner.load_primary_prompts(path)
    assert len(loaded) == 20


def test_load_primary_prompts_rejects_wrong_count(runner, tmp_path):
    records = [_pair_record(f"p{i}", "counter" if i < 5 else "refusal") for i in range(19)]
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    with pytest.raises(SystemExit, match="expected exactly 20"):
        runner.load_primary_prompts(path)


def test_load_primary_prompts_rejects_train_split(runner, tmp_path):
    records = [_pair_record(f"p{i}", "counter" if i < 5 else "refusal") for i in range(19)]
    records.append(_pair_record("p19", "refusal", split="train"))
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    with pytest.raises(SystemExit, match="not 'held_out'"):
        runner.load_primary_prompts(path)


def test_load_primary_prompts_rejects_low_counter_share(runner, tmp_path):
    records = [_pair_record(f"p{i}", "counter" if i < 3 else "refusal") for i in range(20)]
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    with pytest.raises(SystemExit, match="counter-direction share"):
        runner.load_primary_prompts(path)


def test_load_primary_prompts_rejects_duplicate_pair_id(runner, tmp_path):
    records = [_pair_record(f"p{i}", "counter" if i < 5 else "refusal") for i in range(19)]
    records.append(_pair_record("p0", "refusal"))
    path = tmp_path / "held_out_pairs.jsonl"
    _write_jsonl(path, records)
    with pytest.raises(SystemExit, match="duplicate pair_id"):
        runner.load_primary_prompts(path)


def test_load_primary_prompts_against_real_sealed_file(runner):
    # The real sealed ADR-0019 held-out set ships in this checkout; prove
    # this file's own loader accepts it exactly as the pinned hash expects.
    loaded = runner.load_primary_prompts(runner.HELD_OUT_PAIRS_PATH)
    assert len(loaded) == runner.N_PRIMARY_PROMPTS_EXPECTED
    assert all(r["direction"] in runner.ALLOWED_DIRECTIONS for r in loaded)


# ---------------------------------------------------------------------------
# 11. Secondary probe loading.
# ---------------------------------------------------------------------------


def test_load_secondary_probe_against_real_sealed_file(runner):
    probe = runner.load_secondary_probe(runner.SECONDARY_HELD_OUT_PATH, runner.SECONDARY_ITEM_ID)
    assert probe["prompt_id"] == runner.SECONDARY_ITEM_ID
    assert isinstance(probe["prompt"], str) and probe["prompt"].strip()
    assert probe["direction"] is None
    assert probe["content_class"] is None


def test_load_secondary_probe_rejects_missing_item(runner, tmp_path):
    path = tmp_path / "held_out.json"
    path.write_text(json.dumps([{"example_id": "other-item", "messages": []}]), encoding="utf-8")
    with pytest.raises(SystemExit, match="expected exactly 1"):
        runner.load_secondary_probe(path, "mtr-v2-heldout-0013")


def test_load_secondary_probe_rejects_non_list_json(runner, tmp_path):
    path = tmp_path / "held_out.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a JSON list"):
        runner.load_secondary_probe(path, "mtr-v2-heldout-0013")


def test_load_secondary_probe_rejects_item_with_no_user_turn(runner, tmp_path):
    path = tmp_path / "held_out.json"
    path.write_text(
        json.dumps(
            [
                {
                    "example_id": "mtr-v2-heldout-0013",
                    "messages": [{"role": "assistant", "content": "only an assistant turn"}],
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="no user turn"):
        runner.load_secondary_probe(path, "mtr-v2-heldout-0013")


# ---------------------------------------------------------------------------
# 12. Registry contamination re-check (identical mechanism to ADR-0019).
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
    registry.register_package_train("some-other-package", ["p0"])
    registry.package_held_out_ids[runner.HELD_OUT_PACKAGE_ID] = frozenset(pair_ids)
    registry_path = tmp_path / "registry.json"
    registry.save(registry_path)

    with pytest.raises(SystemExit, match="contamination"):
        runner.verify_registry_admits_no_contamination(registry_path, pair_ids)


def test_verify_registry_admits_no_contamination_against_real_sealed_registry(runner):
    # The real sealed ADR-0019 registry ships in this checkout; this
    # evaluation reuses it unchanged (ADR-0020 section 2 bullet 3), so
    # this file's own re-check must pass against it exactly.
    loaded = runner.load_primary_prompts(runner.HELD_OUT_PAIRS_PATH)
    pair_ids = [p["pair_id"] for p in loaded]
    result = runner.verify_registry_admits_no_contamination(runner.HELD_OUT_REGISTRY_PATH, pair_ids)
    assert result.package_held_out_ids[runner.HELD_OUT_PACKAGE_ID] == frozenset(pair_ids)


# ---------------------------------------------------------------------------
# 13. Gate refusal and hash-mismatch refusal.
# ---------------------------------------------------------------------------


def test_load_gate_rejects_schema_mismatch(runner, tmp_path):
    gate = {"nonsense": True}
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(SystemExit, match="schema mismatch"):
        runner.load_gate(path)


def _valid_gate(runner, *, memory_mb: float = 500.0, wall_seconds: float = 60.0) -> dict:
    return {
        "schema_version": 1,
        "script_sha256": runner._this_script_sha256(),
        "classifier_script_sha256": runner.EXPECTED_CLASSIFIER_SCRIPT_HASH,
        "held_out_package_id": runner.HELD_OUT_PACKAGE_ID,
        "held_out_pairs_hash": runner.EXPECTED_HELD_OUT_PAIRS_HASH,
        "held_out_registry_hash": runner.EXPECTED_HELD_OUT_REGISTRY_HASH,
        "measured_max_memory_mb": memory_mb,
        "measured_max_wall_seconds": wall_seconds,
        "approval": {},
    }


def _write_gate(tmp_path: Path, gate: dict) -> Path:
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    return path


def test_load_gate_rejects_script_hash_mismatch(runner, tmp_path):
    gate = {
        "schema_version": 1,
        "script_sha256": "0" * 64,
        "classifier_script_sha256": "9" * 64,
        "held_out_package_id": runner.HELD_OUT_PACKAGE_ID,
        "held_out_pairs_hash": "1" * 64,
        "held_out_registry_hash": "2" * 64,
        "measured_max_memory_mb": 500,
        "measured_max_wall_seconds": 60,
        "approval": {},
    }
    path = _write_gate(tmp_path, gate)
    with pytest.raises(SystemExit, match="different script content hash"):
        runner.load_gate(path)


def test_load_gate_refuses_while_classifier_placeholder_is_unset(runner, tmp_path, monkeypatch):
    # Card 2 (classifier) is now sealed and its hash is pinned (see
    # test_expected_classifier_script_hash_matches_sealed_classifier_file
    # below), but load_gate must still fail closed if that pin is ever
    # unset again -- this is a regression guard on the fail-closed branch,
    # not a statement about the current default.
    monkeypatch.setattr(runner, "EXPECTED_CLASSIFIER_SCRIPT_HASH", None)
    gate = {
        "schema_version": 1,
        "script_sha256": runner._this_script_sha256(),
        "classifier_script_sha256": "9" * 64,
        "held_out_package_id": runner.HELD_OUT_PACKAGE_ID,
        "held_out_pairs_hash": runner.EXPECTED_HELD_OUT_PAIRS_HASH,
        "held_out_registry_hash": runner.EXPECTED_HELD_OUT_REGISTRY_HASH,
        "measured_max_memory_mb": 500,
        "measured_max_wall_seconds": 60,
        "approval": {},
    }
    path = _write_gate(tmp_path, gate)
    with pytest.raises(SystemExit, match="card 2"):
        runner.load_gate(path)


def test_load_gate_rejects_classifier_hash_mismatch(runner, tmp_path, monkeypatch):
    _patch_valid_classifier(runner, tmp_path, monkeypatch)
    gate = _valid_gate(runner)
    gate["classifier_script_sha256"] = "f" * 64  # deliberately wrong
    path = _write_gate(tmp_path, gate)
    with pytest.raises(SystemExit, match="classifier script hash"):
        runner.load_gate(path)


# ---------------------------------------------------------------------------
# 13b. EXPECTED_CLASSIFIER_SCRIPT_HASH pin: proves the sealed constant
#      equals the real card-2 classifier file's content hash on disk, and
#      that load_gate's classifier-hash-check step now passes with that
#      real, unpatched pin (not a monkeypatched stand-in) once the rest of
#      the gate is otherwise valid.
# ---------------------------------------------------------------------------


def test_expected_classifier_script_hash_matches_sealed_classifier_file(runner):
    classifier_path = runner.REPO_ROOT / "examples/pilot-metatrainer-v2/adr0020_scoring_classifier.py"
    assert classifier_path.is_file()
    assert runner.EXPECTED_CLASSIFIER_SCRIPT_HASH is not None
    assert runner._sha256_file(classifier_path) == runner.EXPECTED_CLASSIFIER_SCRIPT_HASH


def test_load_gate_classifier_hash_check_passes_with_real_pinned_hash(runner, tmp_path, monkeypatch):
    # Deliberately does NOT call _patch_valid_classifier: this exercises the
    # real, unpatched EXPECTED_CLASSIFIER_SCRIPT_HASH module constant to
    # prove the pin itself -- not a fake classifier substituted in its
    # place -- is what load_gate's classifier-hash-check step now accepts.
    gate = _valid_gate(runner)
    monkeypatch.setattr(
        runner,
        "_verify_signed_approval",
        lambda approval, **kwargs: {"scope": [runner.HOST_CONTAINMENT_SCOPE]},
    )
    path = _write_gate(tmp_path, gate)
    result = runner.load_gate(path)
    assert result["classifier_script_sha256"] == runner.EXPECTED_CLASSIFIER_SCRIPT_HASH


def test_load_gate_rejects_wrong_package_id(runner, tmp_path, monkeypatch):
    _patch_valid_classifier(runner, tmp_path, monkeypatch)
    gate = _valid_gate(runner)
    gate["held_out_package_id"] = "some-other-package"
    path = _write_gate(tmp_path, gate)
    with pytest.raises(SystemExit, match="not bound to this evaluation"):
        runner.load_gate(path)


def test_load_gate_rejects_bad_memory_ceiling_type(runner, tmp_path, monkeypatch):
    _patch_valid_classifier(runner, tmp_path, monkeypatch)
    gate = _valid_gate(runner, memory_mb=-5)
    path = _write_gate(tmp_path, gate)
    with pytest.raises(SystemExit, match="measured_max_memory_mb"):
        runner.load_gate(path)


def test_load_gate_refuses_memory_ceiling_looser_than_hard_limit(runner, tmp_path, monkeypatch):
    _patch_valid_classifier(runner, tmp_path, monkeypatch)
    gate = _valid_gate(runner, memory_mb=runner.ISOLATION_MAX_MEMORY_MB + 1)
    path = _write_gate(tmp_path, gate)
    with pytest.raises(SystemExit, match="exceeds the hard ceiling"):
        runner.load_gate(path)


def test_load_gate_refuses_wall_seconds_ceiling_looser_than_hard_limit(runner, tmp_path, monkeypatch):
    _patch_valid_classifier(runner, tmp_path, monkeypatch)
    gate = _valid_gate(runner, wall_seconds=runner.ISOLATION_MAX_WALL_SECONDS + 1)
    path = _write_gate(tmp_path, gate)
    with pytest.raises(SystemExit, match="exceeds the hard ceiling"):
        runner.load_gate(path)


def test_load_gate_accepts_ceilings_exactly_at_the_hard_limit(runner, tmp_path, monkeypatch):
    _patch_valid_classifier(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "_verify_signed_approval",
        lambda approval, **kwargs: {"scope": [runner.HOST_CONTAINMENT_SCOPE]},
    )
    gate = _valid_gate(
        runner,
        memory_mb=runner.ISOLATION_MAX_MEMORY_MB,
        wall_seconds=runner.ISOLATION_MAX_WALL_SECONDS,
    )
    path = _write_gate(tmp_path, gate)
    result = runner.load_gate(path)
    assert result["measured_max_memory_mb"] == runner.ISOLATION_MAX_MEMORY_MB
    assert result["measured_max_wall_seconds"] == runner.ISOLATION_MAX_WALL_SECONDS


def test_load_gate_rejects_stale_registry_hash(runner, tmp_path, monkeypatch):
    _patch_valid_classifier(runner, tmp_path, monkeypatch)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text('{"schema_version": 1}', encoding="utf-8")
    monkeypatch.setattr(runner, "HELD_OUT_REGISTRY_PATH", registry_path)
    gate = _valid_gate(runner)
    gate["held_out_registry_hash"] = "2" * 64
    path = _write_gate(tmp_path, gate)
    with pytest.raises(SystemExit, match="does not match the registry file"):
        runner.load_gate(path)


def test_verify_signed_approval_rejects_wrong_field_set(runner, tmp_path):
    approval = {"document": "x"}
    with pytest.raises(SystemExit, match="approval must contain exactly"):
        runner._verify_signed_approval(
            approval,
            script_sha256="a" * 64,
            classifier_script_sha256="b" * 64,
            held_out_registry_hash="c" * 64,
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
            classifier_script_sha256="b" * 64,
            held_out_registry_hash="c" * 64,
            measured_max_memory_mb=1.0,
            measured_max_wall_seconds=1.0,
        )


# ---------------------------------------------------------------------------
# 14. Model hash-mismatch refusal (validate_plan) -- no real snapshot needed.
# ---------------------------------------------------------------------------


def test_validate_plan_detects_candidate_hash_mismatch(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "EXPECTED_CANDIDATE_MODEL_HASH", "2" * 64)
    with pytest.raises(SystemExit, match="candidate checkpoint content hash mismatch"):
        runner.validate_plan()


def test_validate_plan_detects_reference_hash_mismatch(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "EXPECTED_REFERENCE_MODEL_HASH", "3" * 64)
    with pytest.raises(SystemExit, match="reference checkpoint content hash mismatch"):
        runner.validate_plan()


def test_validate_plan_passes_and_reports_expected_blockers(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    # Deterministic regardless of host state: force the review-gate path to
    # a tmp_path location that does not exist, rather than relying on
    # whatever real evidence file may or may not be present on this
    # machine at the fixed absolute APPROVED_REVIEW_GATE_PATH.
    monkeypatch.setattr(runner, "APPROVED_REVIEW_GATE_PATH", tmp_path / "no-gate-here.json")
    result = runner.validate_plan()
    assert result["status"] == "PASS"
    assert result["sampling_called"] is False
    # Held-out/secondary files are real and sealed in this checkout, and
    # the classifier hash is now pinned (card 2 sealed), so those
    # blockers are absent; only the missing gate (card 3) remains, since
    # it is not patched here.
    assert not any("held-out pair set" in b for b in result["execution_blockers"])
    assert not any("secondary held-out" in b for b in result["execution_blockers"])
    assert not any("card 2" in b for b in result["execution_blockers"])
    assert any("gate" in b.lower() for b in result["execution_blockers"])


def test_validate_plan_blockers_clear_once_classifier_and_gate_patched(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    _patch_valid_classifier(runner, tmp_path, monkeypatch)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "APPROVED_REVIEW_GATE_PATH", gate_path)
    result = runner.validate_plan()
    assert result["execution_blockers"] == []


def test_validate_plan_refuses_on_mismatched_held_out_pairs_hash(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "EXPECTED_HELD_OUT_PAIRS_HASH", "f" * 64)
    result = runner.validate_plan()
    assert result["status"] == "PASS"
    assert any("primary held-out pair set file is missing or hash-mismatched" in b for b in result["execution_blockers"])


def test_validate_plan_refuses_on_mismatched_secondary_hash(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "EXPECTED_SECONDARY_HELD_OUT_HASH", "f" * 64)
    result = runner.validate_plan()
    assert result["status"] == "PASS"
    assert any("secondary held-out suite file is missing or hash-mismatched" in b for b in result["execution_blockers"])


def test_validate_plan_never_imports_torch_or_transformers(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    for name in list(sys.modules):
        if name.startswith(("torch", "transformers")):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    result = runner.validate_plan()
    assert result["status"] == "PASS"


def test_validate_plan_reports_decoding_protocol_fields(runner, tmp_path, monkeypatch):
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)
    result = runner.validate_plan()
    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.9
    assert result["k_samples_per_prompt"] == 10
    assert result["max_new_tokens"] == 96
    assert result["min_effect_size_delta"] == 0.20


# ---------------------------------------------------------------------------
# 15. Dry-run / --execute split at the argparse boundary.
# ---------------------------------------------------------------------------


def test_main_without_execute_flag_never_calls_execute_evaluation(runner, monkeypatch):
    calls = {"validate_plan": 0, "execute_evaluation": 0}

    def _fake_validate_plan():
        calls["validate_plan"] += 1
        return {"status": "PASS", "sampling_called": False, "execution_blockers": []}

    def _fake_execute(*args, **kwargs):
        calls["execute_evaluation"] += 1
        raise AssertionError("execute_evaluation must not be called without --execute")

    monkeypatch.setattr(runner, "validate_plan", _fake_validate_plan)
    monkeypatch.setattr(runner, "execute_evaluation", _fake_execute)
    monkeypatch.setattr(sys, "argv", ["run_adr0020_decode_sensitive_sampling.py"])
    exit_code = runner.main()
    assert exit_code == 0
    assert calls["validate_plan"] == 1
    assert calls["execute_evaluation"] == 0


def test_execute_without_review_gate_is_rejected(runner, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_adr0020_decode_sensitive_sampling.py", "--execute"])
    with pytest.raises(SystemExit, match="requires --review-gate"):
        runner.main()


def test_execute_rejects_non_reviewed_scratch_root(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_adr0020_decode_sensitive_sampling.py",
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
            "run_adr0020_decode_sensitive_sampling.py",
            "--execute",
            "--review-gate",
            str(tmp_path / "not-approved-gate.json"),
            "--scratch-root",
            str(tmp_path / "scratch"),
        ],
    )
    with pytest.raises(SystemExit, match="exact reviewed gate path"):
        runner.main()


def test_execute_evaluation_refuses_before_loading_any_model_when_blocked(runner, tmp_path, monkeypatch):
    # With the classifier placeholder unset (module default), execute_evaluation
    # must refuse via validate_plan()'s blockers before ever calling
    # HFLocalCausalLMEvaluatorAdapter._get_model.
    _patch_valid_model_paths(runner, tmp_path, monkeypatch)

    def _boom(self, path):
        raise AssertionError("must not load a model while execution is blocked")

    monkeypatch.setattr(runner.HFLocalCausalLMEvaluatorAdapter, "_get_model", _boom, raising=True)

    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(_valid_gate(runner)), encoding="utf-8")

    with pytest.raises(SystemExit):
        runner.execute_evaluation(tmp_path / "scratch", gate_path)


# ---------------------------------------------------------------------------
# 16. Structural: this script's own content hash helper is self-consistent.
# ---------------------------------------------------------------------------


def test_this_script_sha256_matches_a_direct_recompute(runner):
    import hashlib

    digest = hashlib.sha256(RUNNER_PATH.read_bytes()).hexdigest()
    assert runner._this_script_sha256() == digest


# ---------------------------------------------------------------------------
# 17. Isolated-process resource enforcement (this task's own instruction:
#     the PR #106 pattern applied from the start): execute_evaluation must
#     run the model-load-plus-sampling work through
#     codevolt_mdf.process_isolation.run_callable_in_isolated_process,
#     never in-process; overrun/timeout must refuse with no sample evidence
#     written; and a gate declaring looser ceilings than this script's hard
#     limits must be refused outright by load_gate before any isolation
#     budget is even built.
# ---------------------------------------------------------------------------


def _prepare_execute_evaluation_call(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "validate_plan",
        lambda: {"status": "PASS", "sampling_called": False, "execution_blockers": []},
    )
    monkeypatch.setattr(
        runner,
        "load_primary_prompts",
        lambda path: [
            {
                "pair_id": f"p{i}",
                "prompt": f"prompt-{i}",
                "direction": "counter" if i < 4 else "refusal",
                "content_class": "some_class",
            }
            for i in range(20)
        ],
    )
    monkeypatch.setattr(runner, "verify_registry_admits_no_contamination", lambda path, ids: None)
    monkeypatch.setattr(
        runner,
        "load_secondary_probe",
        lambda path, item_id: {
            "prompt_id": item_id,
            "prompt": "secondary prompt",
            "direction": None,
            "content_class": None,
        },
    )
    gate = _valid_gate(runner)
    monkeypatch.setattr(runner, "load_gate", lambda path: gate)
    return gate


def test_execute_evaluation_overrun_refuses_with_no_evidence_written(runner, tmp_path, monkeypatch):
    _prepare_execute_evaluation_call(runner, monkeypatch, tmp_path)

    def _fake_isolated_run(*, compute_fn, kwargs, budget):
        measured = MeasuredUsage(
            wall_seconds=1.0,
            cpu_seconds=1.0,
            memory_mb_peak=9999.0,
            storage_mb_used=None,
            killed_for_overrun=True,
            killed_for_timeout=False,
        )
        return None, None, measured

    monkeypatch.setattr(runner, "run_callable_in_isolated_process", _fake_isolated_run)

    scratch_root = tmp_path / "scratch"
    with pytest.raises(SystemExit, match="exceeded the resource budget"):
        runner.execute_evaluation(scratch_root, tmp_path / "gate.json")
    assert not scratch_root.exists() or not list(scratch_root.glob("adr0020_raw_samples.json"))


def test_execute_evaluation_timeout_refuses_with_no_evidence_written(runner, tmp_path, monkeypatch):
    _prepare_execute_evaluation_call(runner, monkeypatch, tmp_path)

    def _fake_isolated_run(*, compute_fn, kwargs, budget):
        measured = MeasuredUsage(
            wall_seconds=9999.0,
            cpu_seconds=1.0,
            memory_mb_peak=1.0,
            storage_mb_used=None,
            killed_for_overrun=False,
            killed_for_timeout=True,
        )
        return None, None, measured

    monkeypatch.setattr(runner, "run_callable_in_isolated_process", _fake_isolated_run)

    scratch_root = tmp_path / "scratch"
    with pytest.raises(SystemExit, match="exceeded max_wall_seconds"):
        runner.execute_evaluation(scratch_root, tmp_path / "gate.json")
    assert not scratch_root.exists() or not list(scratch_root.glob("adr0020_raw_samples.json"))


def test_execute_evaluation_child_exception_refuses_with_no_evidence_written(runner, tmp_path, monkeypatch):
    _prepare_execute_evaluation_call(runner, monkeypatch, tmp_path)

    def _fake_isolated_run(*, compute_fn, kwargs, budget):
        measured = MeasuredUsage(
            wall_seconds=1.0,
            cpu_seconds=1.0,
            memory_mb_peak=1.0,
            storage_mb_used=None,
            killed_for_overrun=False,
            killed_for_timeout=False,
        )
        return None, RuntimeError("boom"), measured

    monkeypatch.setattr(runner, "run_callable_in_isolated_process", _fake_isolated_run)

    scratch_root = tmp_path / "scratch"
    with pytest.raises(SystemExit, match="failed without producing a result"):
        runner.execute_evaluation(scratch_root, tmp_path / "gate.json")
    assert not scratch_root.exists() or not list(scratch_root.glob("adr0020_raw_samples.json"))


def test_execute_evaluation_happy_path_round_trips_raw_sample_evidence(
    runner, tmp_path, monkeypatch
):
    """Proves the isolation budget matches the gate's own ceilings and the
    raw samples the (faked) child process returns are written to disk
    unmodified and unblinded -- every field, including ``model``, present
    in the clear, per this file's single-blinding-mechanism-is-card-2's
    interface decision."""
    gate = _prepare_execute_evaluation_call(runner, monkeypatch, tmp_path)

    fake_raw_samples = [
        _raw_sample(runner, model="candidate", pair_id="p0", sample_index=1, seed=20200001, text="cand-text"),
        _raw_sample(runner, model="reference", pair_id="p0", sample_index=1, seed=20200001, text="ref-text"),
    ]
    captured_budgets = []
    captured_kwargs = []

    def _fake_isolated_run(*, compute_fn, kwargs, budget):
        captured_budgets.append(budget)
        captured_kwargs.append(kwargs)
        assert compute_fn is runner._run_sampling_generation
        measured = MeasuredUsage(
            wall_seconds=12.5,
            cpu_seconds=20.0,
            memory_mb_peak=321.0,
            storage_mb_used=0.01,
            killed_for_overrun=False,
            killed_for_timeout=False,
        )
        return fake_raw_samples, None, measured

    monkeypatch.setattr(runner, "run_callable_in_isolated_process", _fake_isolated_run)

    scratch_root = tmp_path / "scratch"
    result = runner.execute_evaluation(scratch_root, tmp_path / "gate.json")

    assert result["n_raw_samples"] == 2
    models_present = {sample["model"] for sample in result["raw_samples"]}
    assert models_present == {"candidate", "reference"}

    assert len(captured_budgets) == 1
    budget = captured_budgets[0]
    assert budget.max_memory_mb == gate["measured_max_memory_mb"]
    assert budget.max_wall_seconds == gate["measured_max_wall_seconds"]
    assert budget.max_memory_mb <= runner.ISOLATION_MAX_MEMORY_MB
    assert budget.max_wall_seconds <= runner.ISOLATION_MAX_WALL_SECONDS
    assert budget.network_policy == "offline"
    assert budget.filesystem_root == str(scratch_root)
    assert set(captured_kwargs[0]) == {
        "candidate_model_path",
        "reference_model_path",
        "primary_prompts",
        "secondary_prompt",
    }

    evidence_path = scratch_root / "adr0020_raw_samples.json"
    assert evidence_path.is_file()
    sha_path = scratch_root / "adr0020_raw_samples.json.sha256"
    assert sha_path.is_file()
    import hashlib

    assert sha_path.read_text(encoding="utf-8").strip() == hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()

    written_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    recovered_texts = set()
    for sample in written_evidence["raw_samples"]:
        recovered_texts.add((sample["model"], sample["text"]))
    assert recovered_texts == {("candidate", "cand-text"), ("reference", "ref-text")}


def test_isolation_max_cpu_seconds_is_twice_wall_ceiling(runner):
    assert runner.ISOLATION_MAX_CPU_SECONDS == runner.ISOLATION_MAX_WALL_SECONDS * 2


# ---------------------------------------------------------------------------
# 18. Structural: this file never modifies or calls hf_local_evaluator_adapter's
#     real _generate method (the task's own hard constraint).
# ---------------------------------------------------------------------------


def test_generate_method_on_the_real_adapter_class_is_unchanged_and_still_greedy(runner):
    """Import the real adapter class fresh and confirm _generate's own
    contract (do_sample=False, no temperature/top_p kwargs) still holds --
    this file must never have patched or shadowed it at import time."""
    import inspect

    source = inspect.getsource(runner.HFLocalCausalLMEvaluatorAdapter._generate)
    assert "do_sample=False" in source
    assert "temperature" not in source
    assert "top_p" not in source
