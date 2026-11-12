"""Avalia o modelo BASE contra o modelo FINE-TUNED no mesmo conjunto de teste.

    python -m src.evaluation.evaluate \
        --adapter /content/drive/MyDrive/fase3/run1/final_adapter \
        --output reports/eval_3b.json

Como funciona: carrega o modelo base em 4 bits UMA vez, aplica o adapter LoRA,
gera as respostas do fine-tuned e depois usa `disable_adapter()` para gerar as
respostas do modelo base (sem recarregar nada). Os dois recebem exatamente os
mesmos prompts, com decodificação gulosa (sem aleatoriedade) para comparação justa.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

from src.config import BASE_MODEL, PROCESSED_DIR, REPORTS_DIR
from src.evaluation.metrics import aggregate, score_example
from src.utils import read_jsonl, write_jsonl


def select_examples(examples: list[dict], max_pubmed: int, max_hospital: int) -> list[dict]:
    pubmed = [e for e in examples if e["meta"]["source"] == "pubmedqa"][:max_pubmed]
    hospital = [e for e in examples if e["meta"]["source"] == "hospital"][:max_hospital]
    return pubmed + hospital


def generate_texts(model, tokenizer, examples: list[dict], batch_size: int, max_new_tokens: int, adapter_enabled: bool) -> list[str]:
    import torch

    prompts = [
        tokenizer.apply_chat_template(e["messages"][:-1], tokenize=False, add_generation_prompt=True) for e in examples
    ]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # agrupa prompts de tamanho parecido
    outputs: dict[int, str] = {}
    # sem adapter carregado (modelo puro) não existe disable_adapter(): nada a desligar
    context = nullcontext() if adapter_enabled or not hasattr(model, "disable_adapter") else model.disable_adapter()

    with context, torch.inference_mode():
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[:, batch["input_ids"].shape[1] :]
            for i, text in zip(idx, tokenizer.batch_decode(new_tokens, skip_special_tokens=True)):
                outputs[i] = text.strip()
            print(f"  gerados {min(start + batch_size, len(order))}/{len(order)}", end="\r")
    print()
    return [outputs[i] for i in range(len(examples))]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--adapter", type=Path, default=None, help="pasta final_adapter; sem ela avalia só o base")
    p.add_argument("--test-file", type=Path, default=PROCESSED_DIR / "test.jsonl")
    p.add_argument("--max-pubmed", type=int, default=100)
    p.add_argument("--max-hospital", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-base", action="store_true", help="não avalia o modelo base (economiza tempo)")
    p.add_argument("--output", type=Path, default=REPORTS_DIR / "eval.json")
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # obrigatório para geração em lote

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16, token=token)
    model.eval()

    has_adapter = args.adapter is not None
    if has_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
        model.eval()

    examples = select_examples(read_jsonl(args.test_file), args.max_pubmed, args.max_hospital)
    print(f"Avaliando {len(examples)} exemplos de teste.")

    report: dict = {"model": args.model, "adapter": str(args.adapter) if args.adapter else None, "n_examples": len(examples)}
    predictions: list[dict] = []

    runs = []
    if has_adapter:
        runs.append(("finetuned", True))
    if not has_adapter or not args.skip_base:
        runs.append(("base", False))

    for name, adapter_enabled in runs:
        print(f"[{name}] gerando respostas...")
        start = time.time()
        texts = generate_texts(model, tokenizer, examples, args.batch_size, args.max_new_tokens, adapter_enabled)
        scored = [score_example(t, e) for t, e in zip(texts, examples)]
        report[name] = aggregate(scored)
        report[f"{name}_seconds"] = round(time.time() - start, 1)
        for example, text, score in zip(examples, texts, scored):
            predictions.append({"run": name, "kind": example["meta"]["kind"], "prompt": example["messages"][1]["content"], "prediction": text, "reference": example["messages"][-1]["content"], "scores": score})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(predictions, args.output.with_suffix(".predictions.jsonl"))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nRelatório salvo em {args.output}")


if __name__ == "__main__":
    main()
