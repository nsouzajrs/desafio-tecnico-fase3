"""Tokenização para fine-tuning supervisionado (SFT) com máscara de prompt.

Ideia central: o modelo só deve aprender a produzir a RESPOSTA. Por isso os
tokens do prompt (system + user) recebem label -100, que o PyTorch ignora no
cálculo da loss. Sem isso, o modelo gastaria capacidade "aprendendo a repetir"
os contextos e as perguntas.

As funções só dependem da interface do tokenizer (apply_chat_template e
__call__), o que permite testá-las sem baixar nenhum modelo.
"""

from __future__ import annotations

IGNORE_INDEX = -100


def build_labels(prompt_length: int, input_ids: list[int]) -> list[int]:
    """Labels = -100 para o prompt, e os próprios ids para a resposta."""
    return [IGNORE_INDEX] * prompt_length + list(input_ids[prompt_length:])


def tokenize_example(example: dict, tokenizer, max_length: int) -> dict | None:
    """Converte {"messages": [...]} em input_ids/attention_mask/labels.

    Retorna None se o exemplo passar de max_length (cortar a resposta no meio
    ensinaria o modelo a não terminar frases; é melhor descartar e contar).
    """
    messages = example["messages"]
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # O template completo deve começar exatamente com o prompt; senão, monta manualmente.
    if not full_text.startswith(prompt_text):
        full_text = prompt_text + messages[-1]["content"] + (tokenizer.eos_token or "")

    # add_special_tokens=False: o BOS já vem escrito no texto do chat template.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_length or len(full_ids) <= len(prompt_ids):
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": build_labels(len(prompt_ids), full_ids),
    }


def tokenize_dataset(examples: list[dict], tokenizer, max_length: int) -> tuple[list[dict], int]:
    """Tokeniza todos os exemplos; retorna (tokenizados, quantidade descartada por tamanho)."""
    tokenized, dropped = [], 0
    for example in examples:
        item = tokenize_example(example, tokenizer, max_length)
        if item is None:
            dropped += 1
        else:
            tokenized.append(item)
    return tokenized, dropped
