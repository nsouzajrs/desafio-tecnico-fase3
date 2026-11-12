"""Anonimização de dados pessoais (PII/PHI) em texto livre, com foco em padrões brasileiros.

Abordagem: regras (regex) + lista opcional de nomes conhecidos. É uma camada
de PROTEÇÃO EM PROFUNDIDADE para este projeto acadêmico; em produção seria
necessário somar NER treinado para português, revisão humana por amostragem e
validação jurídica (LGPD). Por isso o pipeline também registra quantas
substituições foram feitas em cada tipo.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# A ORDEM IMPORTA: padrões mais específicos primeiro.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("CRM", re.compile(r"\bCRM\s*[-/]?\s*[A-Z]{2}\s*[-/]?\s*\d{3,7}\b|\bCRM\s*[-/]?\s*\d{3,7}\s*[-/]?\s*[A-Z]{2}\b", re.IGNORECASE)),
    ("PRONTUARIO", re.compile(r"\b(?:prontu[aá]rio|pront\.?|registro|matr[ií]cula)\s*(?:n[ºo°.]?\s*)?[:\-]?\s*\d[\d.\-/]*", re.IGNORECASE)),
    ("RG", re.compile(r"\bRG\s*[:\-]?\s*[\d.\-xX]{5,}", re.IGNORECASE)),
    ("TELEFONE", re.compile(r"(?:\+?55\s?)?(?:\(\d{2}\)\s?|\b\d{2}\s)?9?\d{4}[-\s]\d{4}\b")),
    ("CEP", re.compile(r"\b\d{5}-\d{3}\b")),
    ("DATA", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")),
]

# Nomes próprios após títulos/marcadores (Sr., Dra., Paciente ...).
_NAME_AFTER_TITLE = re.compile(
    r"\b(?:Sr\.?|Sra\.?|Dr\.?|Dra\.?|Paciente|Pac\.?)\s+"
    r"([A-ZÀ-Ú][a-zà-ú]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÀ-Ú][a-zà-ú]+){0,3})"
)


@dataclass
class AnonymizationResult:
    text: str
    replacements: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.replacements.values())


def anonymize_text(text: str, known_names: list[str] | None = None) -> AnonymizationResult:
    """Substitui PII por marcadores como [CPF], [TELEFONE], [NOME]."""
    counts: Counter = Counter()
    result = text

    for label, pattern in _PATTERNS:
        result, n = pattern.subn(f"[{label}]", result)
        if n:
            counts[label] += n

    def _mask_name(match: re.Match) -> str:
        counts["NOME"] += 1
        return match.group(0).replace(match.group(1), "[NOME]")

    result = _NAME_AFTER_TITLE.sub(_mask_name, result)

    for name in known_names or []:
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        result, n = pattern.subn("[NOME]", result)
        if n:
            counts["NOME"] += n

    return AnonymizationResult(text=result, replacements=counts)


def anonymize_messages(messages: list[dict], known_names: list[str] | None = None) -> tuple[list[dict], Counter]:
    """Anonimiza o conteúdo das mensagens de usuário e assistente (não toca no prompt de sistema)."""
    total: Counter = Counter()
    cleaned = []
    for message in messages:
        if message["role"] == "system":
            cleaned.append(message)
            continue
        res = anonymize_text(message["content"], known_names)
        total.update(res.replacements)
        cleaned.append({**message, "content": res.text})
    return cleaned, total
