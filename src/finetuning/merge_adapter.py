"""(Opcional) Funde o adapter LoRA no modelo base e salva um modelo único em fp16.

Útil se você quiser converter para GGUF/Ollama ou distribuir um único diretório.
Para o fluxo normal do projeto NÃO é necessário: o assistente carrega o modelo
base em 4 bits e aplica o adapter por cima (src/assistant/llm.py).

    python -m src.finetuning.merge_adapter --adapter artifacts/adapter --output artifacts/merged
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.config import BASE_MODEL


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-model", default=None, help="padrão: lido do adapter_config.json")
    args = p.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.getenv("HF_TOKEN")
    base_name = args.base_model
    if base_name is None:
        config = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
        base_name = config.get("base_model_name_or_path", BASE_MODEL)

    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16, token=token)
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output))
    AutoTokenizer.from_pretrained(base_name, token=token).save_pretrained(str(args.output))
    print(f"Modelo fundido salvo em {args.output}")


if __name__ == "__main__":
    main()
