"""Backends de LLM intercambiáveis para o assistente.

  fake   -> LLM de mentira, determinística, para testes e demonstração do fluxo SEM GPU.
  hf     -> Llama 3.2 (+ adapter LoRA do fine-tuning) via transformers, em 4 bits se houver GPU.
  ollama -> qualquer modelo servido pelo Ollama (ex.: se o grupo converter o modelo para GGUF).

Todos devolvem um Runnable do LangChain que recebe um PromptValue e devolve texto,
então entram no pipeline LCEL (prompt | llm | parser) sem nenhuma alteração.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.runnables import Runnable, RunnableLambda

from src.config import ADAPTER_DIR, BASE_MODEL
from src.prompts import DISCLAIMER

_CHUNK_LINE = re.compile(r"^\[([A-Z0-9]+(?:-[A-Z0-9]+)*#[a-z_]+)\]\s+(.+)$", re.MULTILINE)
_PRESCRIPTION_ASK = re.compile(r"prescrev|receit|posologia|qual\s+(?:a\s+)?dose", re.IGNORECASE)


def _fake_generate(prompt_value, mode: str) -> str:
    user_prompt = prompt_value.to_messages()[-1].content
    chunks = _CHUNK_LINE.findall(user_prompt)

    if mode == "unsafe":  # simula uma LLM que erra feio: dose + sem fonte
        return "Administre 500 mg do medicamento agora e inicie antibiótico imediatamente."
    if mode == "uncited":  # simula uma LLM que responde sem citar fonte
        return f"O quadro exige atenção da equipe. {DISCLAIMER}"

    if not chunks:
        return f"Não encontrei base nos protocolos fornecidos para responder a essa pergunta. {DISCLAIMER}"

    question = user_prompt.rsplit("Pergunta:", 1)[-1]
    ids = {cid for cid, _ in chunks}
    if _PRESCRIPTION_ASK.search(question) and "PROT-PRESCRICAO-01#conduta_inicial" in ids:
        return (
            "Não posso prescrever medicamentos, doses ou condutas definitivas: a prescrição é "
            "responsabilidade exclusiva do médico responsável [PROT-PRESCRICAO-01#conduta_inicial]. "
            f"{DISCLAIMER}"
        )

    top = chunks[:2]
    body = " ".join(f"{text} [{cid}]" for cid, text in top)
    return f"Segundo os protocolos consultados: {body}\n\n{DISCLAIMER}"


def make_fake_llm(mode: str = "grounded") -> Runnable:
    """mode: grounded (padrão) | unsafe | uncited — os dois últimos servem para demonstrar os guardrails."""
    return RunnableLambda(lambda prompt_value: _fake_generate(prompt_value, mode))


class HFChatBackend:
    """Carrega o Llama (4-bit se houver GPU) + adapter LoRA e gera respostas a partir de mensagens de chat."""

    ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant"}

    def __init__(self, model_name: str, adapter_dir: str | Path | None, max_new_tokens: int = 384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.float16, token=token
            )
        else:
            print("AVISO: sem GPU. A geração em CPU é lenta; prefira --backend fake para testar o fluxo.")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, token=token)

        if adapter_dir and Path(adapter_dir).exists():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))
        elif adapter_dir:
            print(f"AVISO: adapter não encontrado em {adapter_dir}; usando o modelo base sem fine-tuning.")
        self.model.eval()

    def __call__(self, prompt_value) -> str:
        import torch

        messages = [{"role": self.ROLE_MAP[m.type], "content": m.content} for m in prompt_value.to_messages()]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def make_hf_llm(model_name: str = BASE_MODEL, adapter_dir: str | Path | None = ADAPTER_DIR, max_new_tokens: int = 384) -> Runnable:
    return RunnableLambda(HFChatBackend(model_name, adapter_dir, max_new_tokens))


def make_ollama_llm(model: str | None = None) -> Runnable:
    from langchain_ollama import ChatOllama  # pip install langchain-ollama

    return ChatOllama(model=model or os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)


def get_llm(backend: str = "fake", **kwargs) -> Runnable:
    if backend == "fake":
        return make_fake_llm(kwargs.get("mode", "grounded"))
    if backend == "hf":
        return make_hf_llm(kwargs.get("model_name", BASE_MODEL), kwargs.get("adapter_dir", ADAPTER_DIR))
    if backend == "ollama":
        return make_ollama_llm(kwargs.get("model"))
    raise ValueError(f"Backend desconhecido: {backend} (use fake, hf ou ollama)")
