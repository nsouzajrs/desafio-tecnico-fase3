"""Explainability: mostra de onde veio cada afirmação da resposta.

A explicação traz (1) as fontes citadas com o trecho original, (2) fontes
recuperadas mas não citadas, (3) os campos do prontuário usados e (4) uma
checagem de "citações alucinadas" (ids citados que não estavam no contexto).
"""

from __future__ import annotations

from typing import Iterable

from src.safety.guardrails import extract_citations


def build_explanation(answer: str, chunks: Iterable, patient: dict | None = None, exams: list[dict] | None = None) -> dict:
    chunks = list(chunks)
    by_id = {c.chunk_id: c for c in chunks}
    cited = extract_citations(answer)

    sources = []
    for chunk_id in cited:
        chunk = by_id.get(chunk_id)
        sources.append(
            {
                "id": chunk_id,
                "title": chunk.title if chunk else None,
                "excerpt": chunk.text if chunk else None,
                "in_context": chunk is not None,
            }
        )

    explanation = {
        "cited_sources": sources,
        "retrieved_not_cited": [c.chunk_id for c in chunks if c.chunk_id not in cited],
        "hallucinated_citations": [s["id"] for s in sources if not s["in_context"]],
        "patient_fields_used": [],
    }
    if patient:
        used = ["id", "age", "sex", "chief_complaint", "vitals"]
        if patient.get("conditions"):
            used.append("conditions")
        explanation["patient_fields_used"] = used
    if exams is not None:
        explanation["exams_considered"] = [e["name"] for e in exams]
    return explanation


def format_explanation(explanation: dict) -> str:
    """Versão em texto, para o CLI e para o vídeo."""
    lines = ["Fontes citadas:"]
    for source in explanation["cited_sources"]:
        flag = "" if source["in_context"] else "  (ATENÇÃO: não estava no contexto)"
        lines.append(f"  - [{source['id']}] {source['title']}{flag}")
        if source["excerpt"]:
            lines.append(f"      \"{source['excerpt'][:160]}{'...' if len(source['excerpt']) > 160 else ''}\"")
    if explanation["patient_fields_used"]:
        lines.append("Campos do prontuário usados: " + ", ".join(explanation["patient_fields_used"]))
    return "\n".join(lines)
