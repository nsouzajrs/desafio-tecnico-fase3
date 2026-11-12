"""Carregamento e formatação do PubMedQA (subconjunto rotulado por especialistas, 1000 exemplos).

Fonte: https://github.com/pubmedqa/pubmedqa  (licença MIT)
Campos usados: QUESTION, CONTEXTS (lista de trechos do abstract), LONG_ANSWER
(conclusão dos autores) e final_decision (yes | no | maybe).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.config import PUBMEDQA_PATH
from src.prompts import build_messages, build_pubmed_prompt

VALID_DECISIONS = {"yes", "no", "maybe"}


def load_pubmedqa(path: str | Path = PUBMEDQA_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Baixe ori_pqal.json de "
            "https://github.com/pubmedqa/pubmedqa (pasta data/) e salve em data/raw/."
        )
    with open(path, encoding="utf-8") as file:
        raw = json.load(file)

    records = []
    for pubid, item in raw.items():
        decision = str(item.get("final_decision", "")).strip().lower()
        if decision not in VALID_DECISIONS:
            continue
        if not item.get("QUESTION") or not item.get("CONTEXTS") or not item.get("LONG_ANSWER"):
            continue
        records.append(
            {
                "pubid": pubid,
                "question": item["QUESTION"].strip(),
                "contexts": [str(c).strip() for c in item["CONTEXTS"]],
                "long_answer": item["LONG_ANSWER"].strip(),
                "decision": decision,
            }
        )
    return records


def split_pubmedqa(records: list[dict], seed: int, n_val: int = 40, n_test: int = 200) -> dict[str, list[dict]]:
    """Divide de forma reprodutível em train/validation/test (sem vazamento entre os conjuntos)."""
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return {"train": train, "validation": val, "test": test}


def to_example(record: dict) -> dict:
    """Converte um registro do PubMedQA no formato de chat usado no fine-tuning."""
    answer = f"{record['long_answer']}\n\nDecision: {record['decision']}"
    return {
        "messages": build_messages(build_pubmed_prompt(record["question"], record["contexts"]), answer),
        "meta": {
            "source": "pubmedqa",
            "kind": "pubmed_qa",
            "decision": record["decision"],
            "pubid": record["pubid"],
        },
    }
