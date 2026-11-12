from src.finetuning.tokenization import IGNORE_INDEX, build_labels, tokenize_dataset, tokenize_example


class StubTokenizer:
    """Tokenizer de caracteres, só para testar a lógica de máscara sem baixar modelos."""

    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</s>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


EXAMPLE = {
    "messages": [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "pergunta"},
        {"role": "assistant", "content": "resposta"},
    ]
}


def test_build_labels_masks_prompt():
    assert build_labels(2, [10, 11, 12, 13]) == [IGNORE_INDEX, IGNORE_INDEX, 12, 13]


def test_only_answer_tokens_have_labels():
    item = tokenize_example(EXAMPLE, StubTokenizer(), max_length=500)
    tok = StubTokenizer()
    prompt_len = len(tok("<system>S</s><user>pergunta</s><assistant>")["input_ids"])
    assert item["labels"][:prompt_len] == [IGNORE_INDEX] * prompt_len
    answer_ids = [l for l in item["labels"] if l != IGNORE_INDEX]
    assert "".join(chr(i) for i in answer_ids) == "resposta</s>"
    assert len(item["input_ids"]) == len(item["labels"]) == len(item["attention_mask"])


def test_long_examples_are_dropped_not_truncated():
    tokenized, dropped = tokenize_dataset([EXAMPLE], StubTokenizer(), max_length=10)
    assert tokenized == [] and dropped == 1
