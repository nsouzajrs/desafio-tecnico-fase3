"""Demo em linha de comando do assistente médico (LangChain + LangGraph).

Exemplos:
    python app.py --list-patients
    python app.py --patient P001 --question "Quais exames estão pendentes e qual a conduta prevista?"
    python app.py --patient P001 --question "Prescreva o antibiótico" --backend fake
    python app.py --patient P005 --question "Resuma o caso" --backend hf --adapter artifacts/adapter

Backends: fake (sem GPU, para testar o fluxo) | hf (Llama + adapter) | ollama.
"""

from __future__ import annotations

import argparse
import sys

from src.assistant.graph import STATUS_PENDING, ask, build_graph
from src.assistant.llm import get_llm
from src.config import ADAPTER_DIR, AUDIT_LOG_PATH, BASE_MODEL, DB_PATH
from src.db.database import connect, ensure_db, get_patient, list_patient_ids
from src.safety.explainability import format_explanation
from src.safety.logging_config import record_validation

SEPARATOR = "-" * 72


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # acentos corretos no PowerShell

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patient", help="ID do paciente (ex.: P001)")
    p.add_argument("--question", help="pergunta do médico")
    p.add_argument("--backend", choices=["fake", "hf", "ollama"], default="fake")
    p.add_argument("--fake-mode", choices=["grounded", "unsafe", "uncited"], default="grounded", help="só para --backend fake")
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--adapter", default=str(ADAPTER_DIR))
    p.add_argument("--physician", default="medico.demo", help="identificação de quem valida a sugestão")
    p.add_argument("--no-validate", action="store_true", help="não pede a decisão do médico ao final")
    p.add_argument("--list-patients", action="store_true")
    p.add_argument("--keep-db", action="store_true", help="não repopula o banco (as horas de exames pendentes continuam crescendo)")
    args = p.parse_args(argv)

    ensure_db(DB_PATH, refresh=not args.keep_db)  # por padrão, recria os pacientes sintéticos com os tempos do cenário
    if args.list_patients:
        conn = connect(DB_PATH)
        for pid in list_patient_ids(conn):
            patient = get_patient(conn, pid)
            print(f"{pid}: {patient['age']} anos, {patient['sex']} — {patient['chief_complaint']}")
        return 0

    if not args.patient or not args.question:
        p.error("informe --patient e --question (ou use --list-patients)")

    llm = get_llm(args.backend, mode=args.fake_mode, model_name=args.model, adapter_dir=args.adapter)
    app = build_graph(llm)
    state = ask(app, args.patient, args.question)

    print(SEPARATOR)
    print(f"Requisição: {state['request_id']} | Status: {state['status']}")
    if state.get("alerts"):
        print("\nALERTAS PARA A EQUIPE MÉDICA:")
        for alert in state["alerts"]:
            print(f"  [{alert['severity'].upper()}] {alert['type']}: {alert['message']}")
    print("\nRESPOSTA DO ASSISTENTE:\n")
    print(state["final_answer"])
    if state.get("explanation"):
        print("\n" + format_explanation(state["explanation"]))
    if state.get("guardrail", {}).get("violations"):
        print(f"\nGuardrails: rascunho da LLM bloqueado ({', '.join(state['guardrail']['violations'])}); fallback seguro usado.")
    print(SEPARATOR)
    print("Fluxo executado:", " -> ".join(state["trace"]))
    print(f"Auditoria gravada em: {AUDIT_LOG_PATH}")

    if state["status"] == STATUS_PENDING and not args.no_validate and sys.stdin.isatty():
        choice = input("\nValidação humana — [a]provar, [r]ejeitar ou Enter para pular: ").strip().lower()
        if choice in {"a", "r"}:
            decision = "aprovado" if choice == "a" else "rejeitado"
            record_validation(state["request_id"], args.physician, decision)
            print(f"Decisão registrada: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
