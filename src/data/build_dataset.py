"""Pipeline de dados: carga -> anonimização -> curadoria -> split -> JSONL.

Uso:
    python -m src.data.build_dataset
    python -m src.data.build_dataset --max-pubmed-train 300 --seed 7

Saídas em data/processed/: train.jsonl, validation.jsonl, test.jsonl e stats.json.
Cada linha tem {"messages": [...], "meta": {...}}. O treino usa apenas "messages";
"meta" guarda o gabarito para a avaliação (decisão esperada, fontes esperadas etc.).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from src.config import PROCESSED_DIR, SEED
from src.data.anonymization import anonymize_messages
from src.data.pubmedqa import load_pubmedqa, split_pubmedqa, to_example
from src.data.synthetic_hospital import export_knowledge_base, generate_examples
from src.safety.guardrails import check_output
from src.utils import write_jsonl


def _fingerprint(example: dict) -> str:
    payload = json.dumps(example["messages"][1:], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def anonymize_examples(examples: list[dict], stats: Counter) -> list[dict]:
    """Etapa de ANONIMIZAÇÃO aplicada a todo conteúdo de origem hospitalar."""
    cleaned = []
    for example in examples:
        messages, counts = anonymize_messages(example["messages"])
        stats.update(counts)
        cleaned.append({**example, "messages": messages})
    return cleaned


def curate(examples: list[dict], report: Counter) -> list[dict]:
    """Etapa de CURADORIA: remove duplicados e valida a qualidade das respostas.

    - descarta duplicados (mesma pergunta/contexto/resposta);
    - descarta respostas vazias;
    - para exemplos do hospital, a resposta-alvo precisa passar nos MESMOS
      guardrails usados em produção (cita fonte existente no contexto, sem doses,
      sem linguagem prescritiva). Assim o modelo só aprende comportamento seguro.
    """
    seen: set[str] = set()
    kept = []
    for example in examples:
        answer = example["messages"][-1]["content"].strip()
        if not answer:
            report["descartado_resposta_vazia"] += 1
            continue
        fingerprint = _fingerprint(example)
        if fingerprint in seen:
            report["descartado_duplicado"] += 1
            continue
        seen.add(fingerprint)

        if example["meta"]["source"] == "hospital":
            check = check_output(answer, example["meta"]["context_ids"])
            if not check.allowed or check.disclaimer_added:
                report["descartado_guardrail"] += 1
                continue
        kept.append(example)
    return kept


def _decision_distribution(examples: list[dict]) -> dict:
    counts = Counter(e["meta"]["decision"] for e in examples if e["meta"]["source"] == "pubmedqa")
    total = sum(counts.values()) or 1
    return {k: round(v / total, 3) for k, v in counts.items()}


def build(
    out_dir: Path = PROCESSED_DIR,
    seed: int = SEED,
    max_pubmed_train: int = 500,
    n_val_pubmed: int = 40,
    n_test_pubmed: int = 200,
) -> dict:
    rng = random.Random(seed)
    anonymization_stats: Counter = Counter()
    curation_report: Counter = Counter()

    # --- PubMedQA (literatura pública: não requer anonimização) ---------------
    pubmed = split_pubmedqa(load_pubmedqa(), seed=seed, n_val=n_val_pubmed, n_test=n_test_pubmed)
    pubmed_train = [to_example(r) for r in pubmed["train"][:max_pubmed_train]]
    pubmed_val = [to_example(r) for r in pubmed["validation"]]
    pubmed_test = [to_example(r) for r in pubmed["test"]]

    # --- Dados do hospital (sintéticos): anonimizar -> curar -----------------
    synthetic = generate_examples(seed=seed)
    hospital_train = curate(anonymize_examples(synthetic["train"], anonymization_stats), curation_report)
    hospital_test = curate(anonymize_examples(synthetic["test"], anonymization_stats), curation_report)

    # validação do hospital = 10% do treino (amostra aleatória, só para acompanhar a loss)
    rng.shuffle(hospital_train)
    n_val_hospital = max(1, int(len(hospital_train) * 0.1))
    hospital_val = hospital_train[:n_val_hospital]
    hospital_train = hospital_train[n_val_hospital:]

    train = pubmed_train + hospital_train
    validation = pubmed_val + hospital_val
    test = pubmed_test + hospital_test
    rng.shuffle(train)
    rng.shuffle(validation)

    out_dir = Path(out_dir)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(validation, out_dir / "validation.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")
    export_knowledge_base()

    def by_kind(examples: list[dict]) -> dict:
        return dict(Counter(f"{e['meta']['source']}/{e['meta']['kind']}" for e in examples))

    stats = {
        "seed": seed,
        "sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "train_by_kind": by_kind(train),
        "validation_by_kind": by_kind(validation),
        "test_by_kind": by_kind(test),
        "pubmedqa_decision_distribution": {
            "train": _decision_distribution(pubmed_train),
            "test": _decision_distribution(pubmed_test),
        },
        "anonymization_replacements": dict(anonymization_stats),
        "curation": dict(curation_report),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-pubmed-train", type=int, default=500, help="limite de exemplos PubMedQA no treino")
    parser.add_argument("--n-val-pubmed", type=int, default=40)
    parser.add_argument("--n-test-pubmed", type=int, default=200)
    args = parser.parse_args(argv)

    stats = build(args.out_dir, args.seed, args.max_pubmed_train, args.n_val_pubmed, args.n_test_pubmed)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
