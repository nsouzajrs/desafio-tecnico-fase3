"""Guardrails de entrada e saída do assistente médico.

Princípios (alinhados ao PROT-PRESCRICAO-01 e ao enunciado do desafio):
- O assistente NUNCA prescreve: respostas com doses ou linguagem prescritiva são bloqueadas.
- Toda resposta precisa citar fontes que realmente foram fornecidas no contexto.
- Toda resposta termina com lembrete de validação humana.
As regras são determinísticas (regex) de propósito: são auditáveis e não dependem da LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.prompts import DISCLAIMER

CITATION_RE = re.compile(r"\[([A-Z0-9]+(?:-[A-Z0-9]+)*#[a-z_]+)\]")

# Dose: número + unidade de massa/volume, exceto concentrações laboratoriais (mg/dL, mL/kg/h...).
DOSE_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:mg|mcg|µg|g|ml|mL|UI)\b(?!\s?/)")
PRESCRIPTIVE_RE = re.compile(
    r"\b(prescrevo|prescreva|administre|aplique|inicie|receito|recomendo\s+(?:iniciar|administrar|prescrever))\b",
    re.IGNORECASE,
)
VALIDATION_RE = re.compile(r"valida[çc][aã]o\s+(?:do|pelo)\s+m[eé]dico", re.IGNORECASE)

PRESCRIPTION_REQUEST_RE = re.compile(
    r"\b(prescrev\w*|receit\w*|posologia|qual\s+(?:a\s+)?dose|quantos?\s+(?:mg|comprimidos|ml))\b",
    re.IGNORECASE,
)
INJECTION_RE = re.compile(
    r"(ignore\s+(?:todas\s+)?(?:as\s+)?(?:instru[çc][õo]es|regras)"
    r"|ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions"
    r"|desconsidere\s+(?:as\s+)?(?:regras|instru[çc][õo]es)"
    r"|revele\s+(?:o\s+)?(?:seu\s+)?prompt"
    r"|modo\s+desenvolvedor|jailbreak)",
    re.IGNORECASE,
)

BLOCKED_INPUT_MESSAGE = (
    "Não posso processar este pedido: ele tenta alterar as regras de segurança do assistente. "
    "Reformule a pergunta clínica ou consulte diretamente o médico responsável."
)


@dataclass
class InputCheck:
    allowed: bool
    flags: list[str] = field(default_factory=list)
    message: str | None = None


@dataclass
class OutputCheck:
    allowed: bool
    violations: list[str]
    cited_ids: list[str]
    answer: str  # resposta já com o lembrete de validação, quando faltava
    disclaimer_added: bool = False


def check_input(question: str) -> InputCheck:
    """Bloqueia tentativas de contornar regras e sinaliza pedidos de prescrição."""
    if INJECTION_RE.search(question):
        return InputCheck(allowed=False, flags=["PROMPT_INJECTION"], message=BLOCKED_INPUT_MESSAGE)
    flags = []
    if PRESCRIPTION_REQUEST_RE.search(question):
        flags.append("PEDIDO_DE_PRESCRICAO")
    return InputCheck(allowed=True, flags=flags)


def extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(CITATION_RE.findall(text)))


def check_output(answer: str, allowed_source_ids: list[str]) -> OutputCheck:
    """Valida a resposta da LLM. Qualquer violação => allowed=False (o grafo usa o fallback)."""
    violations: list[str] = []
    text = (answer or "").strip()
    if not text:
        violations.append("RESPOSTA_VAZIA")

    if DOSE_RE.search(text):
        violations.append("DOSE_DETECTADA")
    if PRESCRIPTIVE_RE.search(text):
        violations.append("LINGUAGEM_PRESCRITIVA")

    cited = extract_citations(text)
    if not cited:
        violations.append("SEM_FONTE")
    unknown = [c for c in cited if c not in allowed_source_ids]
    if unknown:
        violations.append("FONTE_NAO_FORNECIDA:" + ",".join(unknown))

    disclaimer_added = False
    if text and not VALIDATION_RE.search(text):
        text = f"{text}\n\n{DISCLAIMER}"
        disclaimer_added = True

    return OutputCheck(
        allowed=not violations,
        violations=violations,
        cited_ids=cited,
        answer=text,
        disclaimer_added=disclaimer_added,
    )
