"""Regras determinísticas sobre exames, alertas e composição de respostas.

Este módulo tem dois usos:
1. Gerar os exemplos de treino "contextualizados com o paciente" (dataset sintético).
2. Servir como resposta de FALLBACK segura no LangGraph, quando a resposta da LLM
   é bloqueada pelos guardrails (assim o fluxo nunca devolve algo sem fonte).
"""

from __future__ import annotations

from src.knowledge.protocols import chunk_index, get_protocol
from src.prompts import DISCLAIMER

URGENT_LIMIT_H = 2.0  # exame de urgência pendente há mais que isso gera alerta
ROUTINE_LIMIT_H = 24.0  # exame de rotina pendente há mais que isso gera alerta

SEPSIS_CRITERIA_LABELS = {
    "hr": "FC > 90 bpm",
    "rr": "FR > 20 irpm",
    "temp": "temperatura > 38 °C ou < 36 °C",
    "sbp": "PA sistólica ≤ 100 mmHg",
    "consciousness": "alteração do nível de consciência",
}


def annotate_exams(exams: list[dict]) -> list[dict]:
    """Acrescenta pending_hours, limit_h e overdue aos exames pendentes."""
    annotated = []
    for exam in exams:
        item = dict(exam)
        if item["status"] == "pendente":
            hours = float(item.get("pending_hours", item.get("requested_hours_ago", 0.0)))
            limit = URGENT_LIMIT_H if item["priority"] == "urgente" else ROUTINE_LIMIT_H
            item["pending_hours"] = hours
            item["limit_h"] = limit
            item["overdue"] = hours > limit
        annotated.append(item)
    return annotated


def sepsis_screen(vitals: dict) -> list[str]:
    """Retorna os critérios de triagem de sepse atendidos (PROT-SEPSE-01#criterios)."""
    met = []
    if vitals.get("hr", 0) > 90:
        met.append("hr")
    if vitals.get("rr", 0) > 20:
        met.append("rr")
    temp = vitals.get("temp")
    if temp is not None and (temp > 38 or temp < 36):
        met.append("temp")
    if vitals.get("sbp", 999) <= 100:
        met.append("sbp")
    if vitals.get("consciousness", "alerta") != "alerta":
        met.append("consciousness")
    return met


def build_alerts(patient: dict, exams: list[dict]) -> list[dict]:
    """Gera alertas para a equipe médica a partir de exames e sinais vitais."""
    alerts: list[dict] = []
    for exam in exams:
        if exam["status"] == "pendente" and exam.get("overdue"):
            alerts.append(
                {
                    "type": "EXAME_ATRASADO",
                    "severity": "alta" if exam["priority"] == "urgente" else "media",
                    "message": (
                        f"Exame '{exam['name']}' ({exam['priority']}) pendente há "
                        f"{exam['pending_hours']:.0f} h, acima do limite de {exam['limit_h']:.0f} h "
                        "[PROT-EXAMES-01#criterios]"
                    ),
                }
            )
        if exam["status"] == "realizado" and exam.get("critical"):
            alerts.append(
                {
                    "type": "RESULTADO_CRITICO",
                    "severity": "alta",
                    "message": (
                        f"Resultado crítico: {exam['name']} — {exam.get('result')}. "
                        "Comunicar ao médico em até 30 minutos [PROT-EXAMES-01#resultados_criticos]"
                    ),
                }
            )
    met = sepsis_screen(patient.get("vitals", {}))
    if len(met) >= 2:
        labels = "; ".join(SEPSIS_CRITERIA_LABELS[m] for m in met)
        alerts.append(
            {
                "type": "TRIAGEM_SEPSE_POSITIVA",
                "severity": "alta",
                "message": (
                    f"{len(met)} critérios de triagem de sepse atendidos ({labels}). "
                    "Se houver suspeita de infecção, avisar imediatamente o médico plantonista "
                    "[PROT-SEPSE-01#criterios]"
                ),
            }
        )
    return alerts


def _exam_phrase(exam: dict) -> str:
    hours = f"{exam['pending_hours']:.0f}"
    late = ", em atraso" if exam.get("overdue") else ""
    return f"{exam['name']} ({exam['priority']}, pendente há {hours} h{late})"


def compose_patient_answer(
    patient: dict,
    exams: list[dict],
    primary_protocol_id: str | None,
) -> tuple[str, list[str]]:
    """Compõe a resposta estruturada e retorna (texto, ids dos chunks citados)."""
    index = chunk_index()
    used: list[str] = []
    vitals = patient.get("vitals", {})
    parts = [
        (
            f"Resumo do paciente {patient['id']}: {patient['age']} anos, sexo {patient['sex']}; "
            f"queixa: {patient['chief_complaint']}. Sinais vitais: FC {vitals.get('hr')} bpm, "
            f"FR {vitals.get('rr')} irpm, temperatura {str(vitals.get('temp')).replace('.', ',')} °C, "
            f"PA {vitals.get('sbp')}/{vitals.get('dbp')} mmHg, SpO2 {vitals.get('spo2')}%, "
            f"consciência {vitals.get('consciousness')}."
        )
    ]

    pending = [e for e in exams if e["status"] == "pendente"]
    if pending:
        parts.append("Exames pendentes: " + "; ".join(_exam_phrase(e) for e in pending) + ".")
    else:
        parts.append("Não há exames pendentes.")

    if any(e.get("overdue") for e in pending):
        parts.append(
            "Exames em atraso devem ser verificados pela enfermagem e comunicados ao médico "
            "responsável [PROT-EXAMES-01#criterios]."
        )
        used.append("PROT-EXAMES-01#criterios")

    critical = [e for e in exams if e["status"] == "realizado" and e.get("critical")]
    if critical:
        phrases = "; ".join(f"{e['name']} ({e['result']})" for e in critical)
        parts.append(
            f"Resultados críticos: {phrases}. Devem ser comunicados ao médico em até 30 minutos "
            "[PROT-EXAMES-01#resultados_criticos]."
        )
        used.append("PROT-EXAMES-01#resultados_criticos")

    if primary_protocol_id:
        protocol = get_protocol(primary_protocol_id)
        conduta_id = f"{primary_protocol_id}#conduta_inicial"
        parts.append(
            f"Protocolo aplicável: {protocol['title']}. Conduta inicial prevista: "
            f"{protocol['sections']['conduta_inicial']} [{conduta_id}]"
        )
        used.append(conduta_id)
        if "exames" in protocol["sections"]:
            exames_id = f"{primary_protocol_id}#exames"
            parts.append(f"Exames previstos no protocolo: {protocol['sections']['exames']} [{exames_id}]")
            used.append(exames_id)
    else:
        parts.append(
            "Não encontrei um protocolo específico para este quadro nos trechos consultados; "
            "recomenda-se consultar diretamente o médico responsável [PROT-PRESCRICAO-01#excecoes]."
        )
        used.append("PROT-PRESCRICAO-01#excecoes")

    parts.append(DISCLAIMER)
    # garante que só devolvemos ids existentes
    used = [u for u in dict.fromkeys(used) if u in index]
    return "\n\n".join(parts), used
