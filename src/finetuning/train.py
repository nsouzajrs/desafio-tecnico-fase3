"""Fine-tuning QLoRA de um Llama com checkpoints retomáveis (feito para o Colab gratuito).

Exemplo (Colab, salvando no Google Drive):
    python -m src.finetuning.train \
        --output-dir /content/drive/MyDrive/fase3/run1 \
        --epochs 2

Conceitos:
- QLoRA: o modelo base fica congelado e quantizado em 4 bits (cabe na T4 de 16 GB);
  só treinamos pequenas matrizes LoRA (~1% dos parâmetros) em fp32.
- Checkpoints: a cada --save-steps passos o estado completo (adapter + otimizador +
  scheduler) é salvo em --output-dir. Se a sessão do Colab cair, basta rodar o mesmo
  comando: o script detecta o último checkpoint e continua de onde parou.
- Máscara de prompt: a loss é calculada só sobre a resposta (ver tokenization.py).

Observação sobre GPU: a T4 (e a GTX 1660 Ti) NÃO suportam bfloat16 nativamente,
por isso o cálculo usa float16 (fp16=True).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.config import BASE_MODEL, MAX_SEQ_LENGTH, PROCESSED_DIR, SEED
from src.finetuning.tokenization import tokenize_dataset
from src.utils import read_jsonl

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--train-file", type=Path, default=PROCESSED_DIR / "train.jsonl")
    p.add_argument("--val-file", type=Path, default=PROCESSED_DIR / "validation.jsonl")
    p.add_argument("--output-dir", type=Path, required=True, help="pasta dos checkpoints (use o Google Drive no Colab)")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8, help="batch efetivo = batch-size x grad-accum")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--max-train-samples", type=int, default=None, help="útil para um teste rápido (ex.: 64)")
    p.add_argument("--no-resume", action="store_true", help="ignora checkpoints existentes e recomeça do zero")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Imports pesados aqui dentro: assim os testes/CLI de dados não exigem torch instalado.
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available():
        raise SystemExit(
            "GPU não encontrada. No Colab: Ambiente de execução > Alterar tipo de ambiente de execução > GPU T4."
        )

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    # --- Dados -------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_raw = read_jsonl(args.train_file)
    val_raw = read_jsonl(args.val_file)
    if args.max_train_samples:
        train_raw = train_raw[: args.max_train_samples]
    train_ds, dropped_train = tokenize_dataset(train_raw, tokenizer, args.max_length)
    val_ds, dropped_val = tokenize_dataset(val_raw, tokenizer, args.max_length)
    print(
        f"Treino: {len(train_ds)} exemplos ({dropped_train} descartados por > {args.max_length} tokens) | "
        f"Validação: {len(val_ds)} ({dropped_val} descartados)"
    )

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    # --- Modelo 4-bit + LoRA -------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        token=hf_token,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Treinamento ---------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=0.3,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ListDataset(train_ds),
        eval_dataset=ListDataset(val_ds),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
    )

    last_checkpoint = None if args.no_resume else get_last_checkpoint(str(args.output_dir))
    if last_checkpoint:
        print(f">>> Retomando do checkpoint: {last_checkpoint}")
    else:
        print(">>> Começando um treino novo.")

    result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # --- Salvando o resultado final -----------------------------------------
    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))  # salva só o adapter LoRA (dezenas de MB)
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "base_model": args.model,
        "args": {k: str(v) for k, v in vars(args).items()},
        "train_metrics": result.metrics,
        "log_history": trainer.state.log_history,
        "dropped_train_examples": dropped_train,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTreino concluído. Adapter salvo em: {final_dir}")


if __name__ == "__main__":
    main()
