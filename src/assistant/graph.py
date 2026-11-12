"""Fluxo clínico automatizado e seguro com LangGraph.

        intake ──(bloqueado)──────────────────────────────┐
          │                                               │
        load_patient ──(não encontrado)───────────────────┤
          │                                               │
        check_pending_exams                               │
          │                                               │
        retrieve_protocols                                │
          │                                               │
        draft_answer (LangChain + LLM fine-tuned)         │
          │                                               │
        safety_check ──(violação)──> fallback_answer ─┐   │
          │ (ok)                                      │   │
          └──────────────────────────────────────> finalize <┘
                                                      │
                                      (há alertas?) ──> emit_alerts ──> END

Toda resposta termina com status PENDENTE_VALIDACAO_HUMANA: o assistente NUNCA
executa ações clínicas; o médico aprova ou rejeita (src/safety/logging_config.py).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.assistant.chains import build_answer_chain
from src.assistant.retrieval import KnowledgeRetriever
from src.assistant.tools import build_tools
from src.config import DB_PATH
from src.db.database import ensure_db
from src.knowledge.answers import annotate_exams, build_alerts, compose_patient_answer
from src.knowledge.protocols import Chunk, chunk_index
from src.prompts import format_patient_context
from src.safety.explainability import build_explanation
from src.safety.guardrails import check_input, check_output
from src.safety.logging_config import audit, new_request_id

STATUS_PENDING = "PENDENTE_VALIDACAO_HUMANA"
STATUS_BLOCKED = "BLOQUEADO"
STATUS_NOT_FOUND = "PACIENTE_NAO_ENCONTRADO"
_TERMINAL = {STATUS_BLOCKED, STATUS_NOT_FOUND}

_TEMPLATE_ASK = re.compile(r"modelo|laudo|receita", re.IGNORECASE)
_NON_CLINICAL_DOCS = {"PROT-EXAMES-01", "PROT-PRESCRICAO-01"}


class AssistantState(TypedDict, total=False):
    request_id: str
    patient_id: str
    question: str
    input_flags: list[str]
    patient: dict
    exams: list[dict]
    alerts: list[dict]
    chunks: list[Chunk]
    draft: str
    guardrail: dict
    explanation: dict
    final_answer: str
    status: str
    used_fallback: bool
    alerts_dispatched: int
    trace: list[str]


def _trace(state: AssistantState, node: str) -> list[str]:
    return state.get("trace", []) + [node]


def build_graph(llm, db_path: str | Path = DB_PATH, retriever: KnowledgeRetriever | None = None):
    """Compila o grafo. `llm` é qualquer Runnable (ver src/assistant/llm.py)."""
    ensure_db(db_path)
    retriever = retriever or KnowledgeRetriever()
    tools = build_tools(db_path, retriever)
    chain = build_answer_chain(llm)
    index = chunk_index()

    # ------------------------------------------------------------------ nós
    def intake(state: AssistantState) -> dict:
        rid = state.get("request_id") or new_request_id()
        check = check_input(state["question"])
        audit("request_received", rid, patient_id=state["patient_id"], question=state["question"], flags=check.flags)
        update: dict = {"request_id": rid, "input_flags": check.flags, "trace": _trace(state, "intake")}
        if not check.allowed:
            update.update(status=STATUS_BLOCKED, final_answer=check.message)
            audit("input_blocked", rid, flags=check.flags)
        return update

    def load_patient(state: AssistantState) -> dict:
        rid = state["request_id"]
        patient = tools["get_patient_record"].invoke({"patient_id": state["patient_id"]})
        audit("patient_loaded", rid, patient_id=state["patient_id"], found=bool(patient))
        update: dict = {"trace": _trace(state, "load_patient")}
        if patient:
            update["patient"] = patient
        else:
            update.update(status=STATUS_NOT_FOUND, final_answer=f"Paciente {state['patient_id']} não encontrado na base de prontuários.")
        return update

    def check_pending_exams(state: AssistantState) -> dict:
        rid = state["request_id"]
        exams = annotate_exams(tools["list_exams"].invoke({"patient_id": state["patient_id"]}))
        alerts = build_alerts(state["patient"], exams)
        audit(
            "exams_checked",
            rid,
            n_exams=len(exams),
            n_pending=sum(1 for e in exams if e["status"] == "pendente"),
            n_alerts=len(alerts),
        )
        return {"exams": exams, "alerts": alerts, "trace": _trace(state, "check_pending_exams")}

    def retrieve_protocols(state: AssistantState) -> dict:
        rid = state["request_id"]
        query = f"{state['question']} {state['patient']['chief_complaint']}"
        hits = tools["search_protocols"].invoke(
            {"query": query, "k": 4, "include_templates": bool(_TEMPLATE_ASK.search(state["question"]))}
        )
        chunks = [index[h["chunk_id"]] for h in hits]

        # contexto extra obrigatório: regras de prescrição e de exames, quando relevantes
        extra: list[str] = []
        if "PEDIDO_DE_PRESCRICAO" in state.get("input_flags", []):
            extra.append("PROT-PRESCRICAO-01#conduta_inicial")
        for alert in state.get("alerts", []):
            if alert["type"] == "EXAME_ATRASADO":
                extra.append("PROT-EXAMES-01#criterios")
            elif alert["type"] == "RESULTADO_CRITICO":
                extra.append("PROT-EXAMES-01#resultados_criticos")
        for chunk_id in dict.fromkeys(extra):
            if chunk_id not in {c.chunk_id for c in chunks}:
                chunks.append(index[chunk_id])

        audit("protocols_retrieved", rid, sources=[c.chunk_id for c in chunks], scores=[h["score"] for h in hits])
        return {"chunks": chunks, "trace": _trace(state, "retrieve_protocols")}

    def draft_answer(state: AssistantState) -> dict:
        rid = state["request_id"]
        try:
            draft = chain.invoke(
                {
                    "question": state["question"],
                    "chunks": state["chunks"],
                    "patient_context": format_patient_context(state["patient"], state["exams"]),
                }
            )
        except Exception as error:  # LLM indisponível/erro: o fallback seguro assume
            audit("llm_error", rid, error=repr(error))
            draft = ""
        audit("llm_draft", rid, draft=draft)
        return {"draft": draft, "trace": _trace(state, "draft_answer")}

    def safety_check(state: AssistantState) -> dict:
        rid = state["request_id"]
        result = check_output(state["draft"], [c.chunk_id for c in state["chunks"]])
        guardrail = {
            "allowed": result.allowed,
            "violations": result.violations,
            "cited_ids": result.cited_ids,
            "disclaimer_added": result.disclaimer_added,
        }
        audit("guardrail_check", rid, **guardrail)
        update: dict = {"guardrail": guardrail, "trace": _trace(state, "safety_check")}
        if result.allowed:
            update["final_answer"] = result.answer
        return update

    def fallback_answer(state: AssistantState) -> dict:
        """Resposta determinística (sem LLM) quando o rascunho viola os guardrails."""
        rid = state["request_id"]
        primary = next(
            (c.doc_id for c in state["chunks"] if c.doc_id.startswith("PROT-") and c.doc_id not in _NON_CLINICAL_DOCS),
            None,
        )
        body, used_ids = compose_patient_answer(state["patient"], state["exams"], primary)

        chunks = list(state["chunks"])
        known = {c.chunk_id for c in chunks}
        prefix = "Nota: a resposta do modelo não passou nas verificações de segurança; segue um resumo baseado diretamente nos protocolos.\n\n"
        if "PEDIDO_DE_PRESCRICAO" in state.get("input_flags", []):
            prefix += (
                "Não posso prescrever medicamentos, doses ou condutas definitivas: a prescrição é "
                "responsabilidade exclusiva do médico responsável [PROT-PRESCRICAO-01#conduta_inicial].\n\n"
            )
            used_ids.append("PROT-PRESCRICAO-01#conduta_inicial")
        for chunk_id in used_ids:
            if chunk_id not in known:
                chunks.append(index[chunk_id])
                known.add(chunk_id)

        result = check_output(prefix + body, [c.chunk_id for c in chunks])
        audit("fallback_used", rid, violations=state["guardrail"]["violations"], allowed=result.allowed)
        answer = result.answer if result.allowed else (
            "Não foi possível gerar uma resposta segura com fonte. Consulte diretamente o médico responsável."
        )
        return {"final_answer": answer, "chunks": chunks, "used_fallback": True, "trace": _trace(state, "fallback_answer")}

    def finalize(state: AssistantState) -> dict:
        rid = state["request_id"]
        status = state.get("status") if state.get("status") in _TERMINAL else STATUS_PENDING
        explanation = {}
        if status == STATUS_PENDING:
            explanation = build_explanation(state["final_answer"], state["chunks"], state.get("patient"), state.get("exams"))
        audit(
            "response_ready",
            rid,
            status=status,
            answer=state.get("final_answer", ""),
            sources=[s["id"] for s in explanation.get("cited_sources", [])],
            used_fallback=state.get("used_fallback", False),
            n_alerts=len(state.get("alerts", [])),
        )
        return {"status": status, "explanation": explanation, "trace": _trace(state, "finalize")}

    def emit_alerts(state: AssistantState) -> dict:
        """Dispara os alertas para a equipe médica (aqui: registro de auditoria + lista no resultado)."""
        rid = state["request_id"]
        for alert in state["alerts"]:
            audit("alert_emitted", rid, patient_id=state["patient_id"], alert_type=alert["type"], severity=alert["severity"], message=alert["message"])
        return {"alerts_dispatched": len(state["alerts"]), "trace": _trace(state, "emit_alerts")}

    # ------------------------------------------------------------- roteamento
    def after_intake(state: AssistantState) -> str:
        return "blocked" if state.get("status") == STATUS_BLOCKED else "ok"

    def after_load(state: AssistantState) -> str:
        return "missing" if state.get("status") == STATUS_NOT_FOUND else "ok"

    def after_safety(state: AssistantState) -> str:
        return "ok" if state["guardrail"]["allowed"] else "fallback"

    def after_finalize(state: AssistantState) -> str:
        return "alerts" if state.get("alerts") and state.get("status") == STATUS_PENDING else "end"

    # ------------------------------------------------------------------ grafo
    graph = StateGraph(AssistantState)
    for name, fn in [
        ("intake", intake),
        ("load_patient", load_patient),
        ("check_pending_exams", check_pending_exams),
        ("retrieve_protocols", retrieve_protocols),
        ("draft_answer", draft_answer),
        ("safety_check", safety_check),
        ("fallback_answer", fallback_answer),
        ("finalize", finalize),
        ("emit_alerts", emit_alerts),
    ]:
        graph.add_node(name, fn)

    graph.set_entry_point("intake")
    graph.add_conditional_edges("intake", after_intake, {"ok": "load_patient", "blocked": "finalize"})
    graph.add_conditional_edges("load_patient", after_load, {"ok": "check_pending_exams", "missing": "finalize"})
    graph.add_edge("check_pending_exams", "retrieve_protocols")
    graph.add_edge("retrieve_protocols", "draft_answer")
    graph.add_edge("draft_answer", "safety_check")
    graph.add_conditional_edges("safety_check", after_safety, {"ok": "finalize", "fallback": "fallback_answer"})
    graph.add_edge("fallback_answer", "finalize")
    graph.add_conditional_edges("finalize", after_finalize, {"alerts": "emit_alerts", "end": END})
    graph.add_edge("emit_alerts", END)
    return graph.compile()


def ask(app, patient_id: str, question: str, request_id: str | None = None) -> dict:
    """Atalho: executa o grafo e devolve o estado final."""
    state: dict = {"patient_id": patient_id, "question": question}
    if request_id:
        state["request_id"] = request_id
    return app.invoke(state)
