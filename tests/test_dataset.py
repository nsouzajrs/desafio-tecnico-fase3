import json

from src.data.build_dataset import build
from src.knowledge.protocols import HOLDOUT_PROTOCOLS
from src.safety.guardrails import check_output
from src.utils import read_jsonl


def test_build_dataset_end_to_end(tmp_path):
    stats = build(out_dir=tmp_path, seed=42, max_pubmed_train=50)
    train = read_jsonl(tmp_path / "train.jsonl")
    validation = read_jsonl(tmp_path / "validation.jsonl")
    test = read_jsonl(tmp_path / "test.jsonl")

    assert stats["sizes"] == {"train": len(train), "validation": len(validation), "test": len(test)}
    assert len([e for e in train if e["meta"]["source"] == "pubmedqa"]) == 50

    # formato de chat: system, user, assistant
    for example in train + validation + test:
        assert [m["role"] for m in example["messages"]] == ["system", "user", "assistant"]
        assert example["messages"][-1]["content"].strip()


def test_no_pubmed_leakage_between_splits(tmp_path):
    build(out_dir=tmp_path, seed=42, max_pubmed_train=500)
    ids = {
        name: {e["meta"]["pubid"] for e in read_jsonl(tmp_path / f"{name}.jsonl") if e["meta"]["source"] == "pubmedqa"}
        for name in ("train", "validation", "test")
    }
    assert not ids["train"] & ids["test"]
    assert not ids["train"] & ids["validation"]
    assert not ids["validation"] & ids["test"]


def test_holdout_protocols_never_appear_in_training_data(tmp_path):
    build(out_dir=tmp_path, seed=42, max_pubmed_train=10)
    for name in ("train", "validation"):
        for example in read_jsonl(tmp_path / f"{name}.jsonl"):
            if example["meta"]["source"] != "hospital":
                continue
            text = json.dumps(example["messages"], ensure_ascii=False)
            for holdout in HOLDOUT_PROTOCOLS:
                assert holdout not in text, f"{holdout} vazou para {name}"


def test_hospital_targets_pass_the_production_guardrails(tmp_path):
    build(out_dir=tmp_path, seed=42, max_pubmed_train=10)
    for name in ("train", "test"):
        for example in read_jsonl(tmp_path / f"{name}.jsonl"):
            if example["meta"]["source"] == "hospital":
                result = check_output(example["messages"][-1]["content"], example["meta"]["context_ids"])
                assert result.allowed and not result.disclaimer_added, (example["meta"]["kind"], result.violations)


def test_expected_sources_are_in_context(tmp_path):
    build(out_dir=tmp_path, seed=42, max_pubmed_train=10)
    for example in read_jsonl(tmp_path / "train.jsonl"):
        meta = example["meta"]
        if meta["source"] == "hospital":
            assert set(meta["expected_sources"]) <= set(meta["context_ids"])


def test_build_is_deterministic(tmp_path):
    build(out_dir=tmp_path / "a", seed=7, max_pubmed_train=20)
    build(out_dir=tmp_path / "b", seed=7, max_pubmed_train=20)
    assert (tmp_path / "a" / "train.jsonl").read_text(encoding="utf-8") == (tmp_path / "b" / "train.jsonl").read_text(encoding="utf-8")
