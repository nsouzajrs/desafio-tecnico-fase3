"""Prompts compartilhados entre treino, avaliação e inferência.

Treinar e usar o modelo com EXATAMENTE o mesmo formato de prompt é essencial:
o fine-tuning ensina o modelo a responder a esse formato específico.
"""

from __future__ import annotations

from typing import Iterable

DISCLAIMER = (
    "Sugestão de apoio à decisão para validação do médico responsável; "
    "não é prescrição nem substitui o julgamento clínico."
)

SYSTEM_PROMPT = (
    "Você é um assistente virtual de apoio à decisão clínica do Hospital Modelo "
    "(ambiente de demonstração acadêmica).\n"
    "Regras:\n"
    "1. Responda somente com base nos trechos de protocolos e nos dados do paciente fornecidos.\n"
    "2. Cite a fonte no formato [ID#seção] logo após cada afirmação baseada em protocolo.\n"
    "3. Nunca prescreva medicamentos, doses ou condutas definitivas; apresente apenas o que os protocolos preveem.\n"
    "4. Se a informação não estiver nos trechos, diga que não há base nos protocolos fornecidos.\n"
    "5. Finalize lembrando que a sugestão requer validação do médico responsável.\n"
    "Para perguntas em inglês sobre literatura biomédica, responda em inglês usando apenas o contexto "
    "científico fornecido e termine com 'Decision: yes', 'Decision: no' ou 'Decision: maybe'."
)


def _fmt_number(value) -> str:
    return str(value).replace(".", ",")


def format_patient_context(patient: dict, exams: Iterable[dict] | None = None) -> str:
    """Formata o registro estruturado do paciente para entrar no prompt."""
    vitals = patient.get("vitals", {})
    lines = [
        "Dados do paciente (base estruturada):",
        f"- ID: {patient['id']} | {patient['age']} anos | {patient['sex']}",
        f"- Queixa principal: {patient['chief_complaint']}",
        f"- Condições: {', '.join(patient.get('conditions') or ['não informadas'])}",
        f"- Alergias: {', '.join(patient.get('allergies') or ['não informadas'])}",
        (
            f"- Sinais vitais: FC {vitals.get('hr')} bpm; FR {vitals.get('rr')} irpm; "
            f"Temp {_fmt_number(vitals.get('temp'))} °C; PA {vitals.get('sbp')}/{vitals.get('dbp')} mmHg; "
            f"SpO2 {vitals.get('spo2')}%; consciência: {vitals.get('consciousness')}"
        ),
    ]
    if patient.get("notes"):
        lines.append(f"- Observações: {patient['notes']}")
    if exams is not None:
        lines.append("Exames:")
        for exam in exams:
            if exam["status"] == "pendente":
                late = " [EM ATRASO]" if exam.get("overdue") else ""
                hours = f"{exam.get('pending_hours', 0):.0f}".replace(".", ",")
                lines.append(f"- {exam['name']} ({exam['priority']}): pendente há {hours} h{late}")
            else:
                critical = " [CRÍTICO]" if exam.get("critical") else ""
                lines.append(f"- {exam['name']} ({exam['priority']}): realizado — {exam.get('result')}{critical}")
    return "\n".join(lines)


def format_chunks(chunks: Iterable) -> str:
    """Lista os trechos de protocolo no formato citável: [ID#seção] texto."""
    return "\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)


def build_user_prompt(question: str, chunks: Iterable, patient_context: str | None = None) -> str:
    parts = []
    if patient_context:
        parts.append(patient_context)
    parts.append("Trechos de protocolos do hospital:\n" + format_chunks(chunks))
    parts.append(f"Pergunta: {question}")
    return "\n\n".join(parts)


def build_pubmed_prompt(question: str, contexts: list[str]) -> str:
    context_text = "\n\n".join(contexts)
    return (
        f"Scientific contexts:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the provided contexts and finish with 'Decision: yes', "
        "'Decision: no' or 'Decision: maybe'."
    )


def build_messages(user_prompt: str, answer: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages
