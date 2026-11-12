"""Ferramentas LangChain (@tool) que dão ao assistente acesso aos dados estruturados e aos protocolos.

Por serem `Tool`s do LangChain, podem ser usadas tanto pelos nós do LangGraph
(como fazemos aqui, de forma determinística e auditável) quanto por um agente
com tool-calling, se o grupo quiser evoluir o projeto.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from src.assistant.retrieval import KnowledgeRetriever
from src.db.database import connect, get_exams, get_patient


def build_tools(db_path: str | Path, retriever: KnowledgeRetriever) -> dict:
    """Cria as ferramentas ligadas a um banco e a um recuperador específicos."""

    @tool
    def get_patient_record(patient_id: str) -> dict:
        """Consulta o registro estruturado de um paciente (idade, queixa, condições, alergias, sinais vitais)."""
        conn = connect(db_path)
        try:
            return get_patient(conn, patient_id) or {}
        finally:
            conn.close()

    @tool
    def list_exams(patient_id: str) -> list[dict]:
        """Lista os exames do paciente (pendentes e realizados) com horas de espera dos pendentes."""
        conn = connect(db_path)
        try:
            return get_exams(conn, patient_id)
        finally:
            conn.close()

    @tool
    def search_protocols(query: str, k: int = 4, include_templates: bool = False) -> list[dict]:
        """Busca trechos dos protocolos internos do hospital mais relevantes para a consulta."""
        hits = retriever.search(query, k=k, include_templates=include_templates)
        return [
            {"chunk_id": h.chunk.chunk_id, "title": h.chunk.title, "text": h.chunk.text, "score": round(h.score, 4)}
            for h in hits
        ]

    return {
        "get_patient_record": get_patient_record,
        "list_exams": list_exams,
        "search_protocols": search_protocols,
    }
