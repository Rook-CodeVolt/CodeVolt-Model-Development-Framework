"""Unit tests for ``scoring_modes.py``'s shape-validation helpers.

Stdlib-only, no model inference -- these are pure input/spec parsing
and format-checking tests, independent of which adapter (fake or real)
later calls these helpers. Also carries this module's own AST-import
test proving zero import relationship with
``trl_adapter.py``/``trainer_contract.py``, matching the pattern used
by ``test_evaluator_contract.py`` and
``test_hf_local_evaluator_adapter.py``.
"""

from __future__ import annotations

import pytest

from codevolt_mdf.evaluator_contract import HeldOutExample, InvalidInputError
from codevolt_mdf.scoring_modes import (
    check_format_conformance,
    parse_format_conformance_input,
    parse_format_spec,
    parse_multiple_choice_input,
    resolve_expected_choice,
)

# 1. Multiple-choice input parsing --------------------------------------------


def test_parse_multiple_choice_input_accepts_well_formed_shape():
    example = HeldOutExample(
        example_id="q1",
        input={"prompt": "2 + 2 = ?", "choices": ["3", "4", "5"]},
        expected="4",
    )
    prompt, choices = parse_multiple_choice_input(example)
    assert prompt == "2 + 2 = ?"
    assert choices == ["3", "4", "5"]


def test_parse_multiple_choice_input_rejects_non_dict_input():
    example = HeldOutExample(example_id="q1", input="not a dict", expected="4")
    with pytest.raises(InvalidInputError, match="must be a dict"):
        parse_multiple_choice_input(example)


def test_parse_multiple_choice_input_rejects_too_few_choices():
    example = HeldOutExample(
        example_id="q1", input={"prompt": "p", "choices": ["only-one"]}, expected="only-one"
    )
    with pytest.raises(InvalidInputError, match="at least 2"):
        parse_multiple_choice_input(example)


def test_parse_multiple_choice_input_rejects_duplicate_choices():
    example = HeldOutExample(
        example_id="q1", input={"prompt": "p", "choices": ["a", "a"]}, expected="a"
    )
    with pytest.raises(InvalidInputError, match="duplicate"):
        parse_multiple_choice_input(example)


def test_parse_multiple_choice_input_rejects_blank_choice():
    example = HeldOutExample(
        example_id="q1", input={"prompt": "p", "choices": ["a", "   "]}, expected="a"
    )
    with pytest.raises(InvalidInputError, match="non-empty string"):
        parse_multiple_choice_input(example)


# 2. Multiple-choice expected-value resolution --------------------------------


def test_resolve_expected_choice_by_text():
    assert resolve_expected_choice("b", ["a", "b", "c"]) == "b"


def test_resolve_expected_choice_by_index():
    assert resolve_expected_choice(1, ["a", "b", "c"]) == "b"


def test_resolve_expected_choice_rejects_out_of_range_index():
    with pytest.raises(InvalidInputError, match="out of range"):
        resolve_expected_choice(5, ["a", "b", "c"])


def test_resolve_expected_choice_rejects_unknown_text():
    with pytest.raises(InvalidInputError, match="not one of"):
        resolve_expected_choice("z", ["a", "b", "c"])


def test_resolve_expected_choice_rejects_bool():
    with pytest.raises(InvalidInputError, match="must not be a bool"):
        resolve_expected_choice(True, ["a", "b", "c"])


def test_resolve_expected_choice_rejects_wrong_type():
    with pytest.raises(InvalidInputError, match="must be a str"):
        resolve_expected_choice(3.14, ["a", "b", "c"])


# 3. Format-conformance input/spec parsing ------------------------------------


def test_parse_format_conformance_input_rejects_empty_prompt():
    example = HeldOutExample(example_id="q1", input="   ", expected={"format": "json"})
    with pytest.raises(InvalidInputError, match="non-empty string prompt"):
        parse_format_conformance_input(example)


def test_parse_format_spec_accepts_json():
    example = HeldOutExample(example_id="q1", input="p", expected={"format": "json"})
    spec = parse_format_spec(example)
    assert spec["format"] == "json"


def test_parse_format_spec_accepts_json_with_required_keys():
    example = HeldOutExample(
        example_id="q1",
        input="p",
        expected={"format": "json", "required_keys": ["name", "age"]},
    )
    spec = parse_format_spec(example)
    assert spec["required_keys"] == ["name", "age"]


def test_parse_format_spec_rejects_unsupported_format():
    example = HeldOutExample(example_id="q1", input="p", expected={"format": "xml"})
    with pytest.raises(InvalidInputError, match="one of"):
        parse_format_spec(example)


def test_parse_format_spec_rejects_non_dict_expected():
    example = HeldOutExample(example_id="q1", input="p", expected="not a spec")
    with pytest.raises(InvalidInputError, match="must be a dict"):
        parse_format_spec(example)


def test_parse_format_spec_rejects_invalid_regex_pattern():
    example = HeldOutExample(
        example_id="q1", input="p", expected={"format": "regex", "pattern": "("}
    )
    with pytest.raises(InvalidInputError, match="does not compile"):
        parse_format_spec(example)


def test_parse_format_spec_rejects_regex_without_pattern():
    example = HeldOutExample(example_id="q1", input="p", expected={"format": "regex"})
    with pytest.raises(InvalidInputError, match="non-empty string 'pattern'"):
        parse_format_spec(example)


def test_parse_format_spec_rejects_non_list_required_keys():
    example = HeldOutExample(
        example_id="q1", input="p", expected={"format": "json", "required_keys": "name"}
    )
    with pytest.raises(InvalidInputError, match="required_keys"):
        parse_format_spec(example)


# 4. Format-conformance checking ------------------------------------------------


def test_check_format_conformance_json_valid():
    conforms, _ = check_format_conformance('{"a": 1}', {"format": "json"})
    assert conforms is True


def test_check_format_conformance_json_invalid():
    conforms, detail = check_format_conformance("not json", {"format": "json"})
    assert conforms is False
    assert "not valid JSON" in detail


def test_check_format_conformance_json_required_keys_present():
    conforms, _ = check_format_conformance(
        '{"name": "a", "age": 1}', {"format": "json", "required_keys": ["name", "age"]}
    )
    assert conforms is True


def test_check_format_conformance_json_required_keys_missing():
    conforms, detail = check_format_conformance(
        '{"name": "a"}', {"format": "json", "required_keys": ["name", "age"]}
    )
    assert conforms is False
    assert "age" in detail


def test_check_format_conformance_json_required_keys_on_non_object():
    conforms, detail = check_format_conformance(
        "[1, 2, 3]", {"format": "json", "required_keys": ["name"]}
    )
    assert conforms is False
    assert "not a JSON object" in detail


def test_check_format_conformance_regex_match():
    conforms, _ = check_format_conformance(
        "the answer is 42", {"format": "regex", "pattern": r"\d+"}
    )
    assert conforms is True


def test_check_format_conformance_regex_no_match():
    conforms, _ = check_format_conformance(
        "no numbers here", {"format": "regex", "pattern": r"\d+"}
    )
    assert conforms is False


# 5. Structural separation from trl_adapter.py / trainer_contract.py ---------


def test_scoring_modes_module_does_not_import_trainer_side():
    import ast

    import codevolt_mdf.scoring_modes as scoring_modes_module

    with open(scoring_modes_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any("trl_adapter" in name for name in imported_modules)
    assert not any("trainer_contract" in name for name in imported_modules)
