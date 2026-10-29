"""评测集校验：结构、id 唯一、has_answer 与 gold 一致。"""
from __future__ import annotations

from src.eval.validate import validate_dataset


def make_item(**overrides) -> dict:
    item = {
        "id": "q1",
        "question": "What is X?",
        "gold_pmc_ids": ["PMC1"],
        "gold_sections": ["Abstract"],
        "has_answer": True,
    }
    item.update(overrides)
    return item


def test_valid_item_passes():
    assert validate_dataset([make_item()]) == []


def test_negative_item_passes():
    item = make_item(has_answer=False, gold_pmc_ids=[], gold_sections=[])
    assert validate_dataset([item]) == []


def test_missing_field_is_reported():
    item = make_item()
    del item["question"]

    errors = validate_dataset([item])

    assert len(errors) == 1
    assert "q1" in errors[0]


def test_duplicate_id_is_reported():
    errors = validate_dataset([make_item(), make_item()])

    assert len(errors) == 1
    assert "q1" in errors[0]


def test_has_answer_true_requires_gold():
    item = make_item(gold_sections=[])

    errors = validate_dataset([item])

    assert len(errors) == 1
    assert "q1" in errors[0]


def test_has_answer_false_rejects_gold():
    item = make_item(has_answer=False)

    errors = validate_dataset([item])

    assert len(errors) == 1
    assert "q1" in errors[0]


def test_blank_question_is_reported():
    item = make_item(question="   ")

    errors = validate_dataset([item])

    assert len(errors) == 1
    assert "q1" in errors[0]


def test_multiple_errors_are_all_reported():
    bad_blank = make_item(id="q2", question="")
    bad_gold = make_item(id="q3", gold_pmc_ids=[])

    errors = validate_dataset([bad_blank, bad_gold])

    assert len(errors) == 2
