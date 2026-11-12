"""Logging estruturado para auditoria (JSON Lines) + log legível no console.

Cada evento vira UMA linha JSON em logs/audit.jsonl, com request_id para
reconstruir o caminho completo de uma consulta: pergunta -> paciente -> exames
-> fontes -> resposta -> guardrails -> decisão do médico.

Privacidade: texto livre passa pelo anonimizador ANTES de ser gravado.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import AUDIT_LOG_PATH
from src.data.anonymization import anonymize_text

_console = logging.getLogger("assistente_medico")
if not _console.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _console.addHandler(handler)
    _console.setLevel(os.getenv("LOG_LEVEL", "WARNING"))

# Campos de texto livre que devem ser anonimizados antes de gravar.
_FREE_TEXT_FIELDS = {"question", "answer", "draft", "comment"}


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def audit_path() -> Path:
    # lê a variável a cada chamada, para permitir redirecionar o log nos testes
    return Path(os.getenv("AUDIT_LOG_PATH", str(AUDIT_LOG_PATH)))


def audit(event: str, request_id: str, **fields) -> dict:
    """Grava um evento de auditoria e devolve o registro gravado."""
    safe_fields = {}
    for key, value in fields.items():
        if key in _FREE_TEXT_FIELDS and isinstance(value, str):
            value = anonymize_text(value).text
        safe_fields[key] = value

    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "request_id": request_id,
        "event": event,
        **safe_fields,
    }
    path = audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    _console.info("%s %s", event, request_id)
    return record


def record_validation(request_id: str, physician: str, decision: str, comment: str = "") -> dict:
    """Registra a decisão do médico (aprovar/rejeitar) sobre uma sugestão do assistente."""
    if decision not in {"aprovado", "rejeitado"}:
        raise ValueError("decision deve ser 'aprovado' ou 'rejeitado'")
    return audit("human_validation", request_id, physician=physician, decision=decision, comment=comment)


def read_audit(request_id: str | None = None) -> list[dict]:
    path = audit_path()
    if not path.exists():
        return []
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if request_id:
        events = [e for e in events if e["request_id"] == request_id]
    return events
