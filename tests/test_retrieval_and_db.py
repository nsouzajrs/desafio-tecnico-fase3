from datetime import datetime, timedelta, timezone

import pytest

from src.assistant.retrieval import KnowledgeRetriever
from src.db.database import connect, get_exams, get_patient, list_patient_ids, seed
from src.knowledge.answers import annotate_exams, build_alerts, sepsis_screen
from src.knowledge.patients import PATIENTS


@pytest.fixture(scope="module")
def retriever():
    return KnowledgeRetriever()


@pytest.mark.parametrize("patient", PATIENTS, ids=[p["id"] for p in PATIENTS])
def test_retrieval_finds_expected_protocol_for_each_patient(retriever, patient):
    """Consulta genérica + queixa principal deve trazer o protocolo esperado entre os 3 primeiros."""
    query = f"Qual a conduta prevista no protocolo? {patient['chief_complaint']}"
    top_docs = [h.chunk.doc_id for h in retriever.search(query, k=3)]
    assert patient["expected_protocol"] in top_docs


def test_templates_are_excluded_unless_requested(retriever):
    assert not any(h.chunk.doc_id.startswith("TEMPL-") for h in retriever.search("laudo de radiografia de tórax", k=5))
    hits = retriever.search("modelo de laudo de radiografia de tórax", k=3, include_templates=True)
    assert hits[0].chunk.doc_id == "TEMPL-LAUDO-RX"


def test_db_seed_and_pending_hours(db_path):
    conn = connect(db_path)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    seed(conn, now=now)
    assert list_patient_ids(conn) == [p["id"] for p in PATIENTS]

    patient = get_patient(conn, "P001")
    assert patient["vitals"]["hr"] == 112 and "diabetes tipo 2" in patient["conditions"]
    assert get_patient(conn, "P999") is None

    exams = get_exams(conn, "P001", now=now + timedelta(hours=1))
    lactate = next(e for e in exams if e["name"] == "lactato")
    assert lactate["pending_hours"] == pytest.approx(4.0)  # 3 h do seed + 1 h de relógio


def test_overdue_rules_and_alerts():
    p001 = next(p for p in PATIENTS if p["id"] == "P001")
    exams = annotate_exams(p001["exams"])
    overdue = {e["name"] for e in exams if e.get("overdue")}
    assert overdue == {"lactato", "hemoculturas"}  # urgentes pendentes há 3 h (> 2 h)
    types = {a["type"] for a in build_alerts(p001, exams)}
    assert types == {"EXAME_ATRASADO", "TRIAGEM_SEPSE_POSITIVA"}

    p006 = next(p for p in PATIENTS if p["id"] == "P006")
    critical = [a for a in build_alerts(p006, annotate_exams(p006["exams"])) if a["type"] == "RESULTADO_CRITICO"]
    assert len(critical) == 2

    p003 = next(p for p in PATIENTS if p["id"] == "P003")
    late = [a for a in build_alerts(p003, annotate_exams(p003["exams"])) if a["type"] == "EXAME_ATRASADO"]
    assert len(late) == 2  # rotina pendente há 72 h (> 24 h); o de 12 h não alerta


def test_sepsis_screen_thresholds():
    assert sepsis_screen({"hr": 80, "rr": 16, "temp": 36.5, "sbp": 120, "consciousness": "alerta"}) == []
    assert len(sepsis_screen({"hr": 112, "rr": 24, "temp": 38.9, "sbp": 92, "consciousness": "confuso"})) == 5
