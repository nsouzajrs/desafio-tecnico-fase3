"""Base estruturada (SQLite) com prontuários SINTÉTICOS.

Simula as "bases de dados estruturadas (prontuários e registros)" do enunciado.
Em produção seria o banco/API do hospital; aqui usamos SQLite (biblioteca padrão,
sem instalação) para manter o projeto reproduzível.

    python -m src.db.database            # cria e popula data/hospital.db
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import DB_PATH
from src.knowledge.patients import PATIENTS

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    age INTEGER NOT NULL,
    sex TEXT NOT NULL,
    chief_complaint TEXT NOT NULL,
    conditions TEXT NOT NULL,   -- JSON
    allergies TEXT NOT NULL,    -- JSON
    vitals TEXT NOT NULL,       -- JSON
    notes TEXT
);
CREATE TABLE IF NOT EXISTS exams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    name TEXT NOT NULL,
    priority TEXT NOT NULL CHECK (priority IN ('urgente', 'rotina')),
    status TEXT NOT NULL CHECK (status IN ('pendente', 'realizado')),
    requested_at TEXT NOT NULL,  -- ISO 8601 (UTC)
    result TEXT,
    critical INTEGER NOT NULL DEFAULT 0
);
"""


def connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Abre a conexão. Use ':memory:' para um banco temporário (testes)."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def seed(conn: sqlite3.Connection, now: datetime | None = None) -> None:
    """Recria as tabelas e insere os pacientes sintéticos (idempotente)."""
    now = now or datetime.now(timezone.utc)
    conn.executescript("DROP TABLE IF EXISTS exams; DROP TABLE IF EXISTS patients;")
    conn.executescript(SCHEMA)
    for patient in PATIENTS:
        conn.execute(
            "INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                patient["id"],
                patient["display_name"],
                patient["age"],
                patient["sex"],
                patient["chief_complaint"],
                json.dumps(patient["conditions"], ensure_ascii=False),
                json.dumps(patient["allergies"], ensure_ascii=False),
                json.dumps(patient["vitals"], ensure_ascii=False),
                patient.get("notes"),
            ),
        )
        for exam in patient["exams"]:
            requested_at = now - timedelta(hours=exam["requested_hours_ago"])
            conn.execute(
                "INSERT INTO exams (patient_id, name, priority, status, requested_at, result, critical) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    patient["id"],
                    exam["name"],
                    exam["priority"],
                    exam["status"],
                    requested_at.isoformat(),
                    exam["result"],
                    int(exam["critical"]),
                ),
            )
    conn.commit()


def get_patient(conn: sqlite3.Connection, patient_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if row is None:
        return None
    patient = dict(row)
    for key in ("conditions", "allergies", "vitals"):
        patient[key] = json.loads(patient[key])
    return patient


def get_exams(conn: sqlite3.Connection, patient_id: str, now: datetime | None = None) -> list[dict]:
    """Retorna os exames do paciente; para pendentes calcula pending_hours (relógio injetável p/ testes)."""
    now = now or datetime.now(timezone.utc)
    rows = conn.execute("SELECT * FROM exams WHERE patient_id = ? ORDER BY id", (patient_id,)).fetchall()
    exams = []
    for row in rows:
        exam = dict(row)
        exam["critical"] = bool(exam["critical"])
        if exam["status"] == "pendente":
            requested = datetime.fromisoformat(exam["requested_at"])
            exam["pending_hours"] = round((now - requested).total_seconds() / 3600, 2)
        exams.append(exam)
    return exams


def list_patient_ids(conn: sqlite3.Connection) -> list[str]:
    return [r["id"] for r in conn.execute("SELECT id FROM patients ORDER BY id")]


def ensure_db(db_path: str | Path = DB_PATH, refresh: bool = False) -> Path:
    """Cria e popula o banco se ainda não existir.

    Os exames guardam a data/hora em que foram solicitados (relativa ao momento da criação do banco).
    Por isso, num banco antigo as "horas pendentes" só crescem. Com refresh=True o banco é repopulado
    e os tempos voltam aos valores do cenário (ex.: lactato do P001 pendente há 3 h).
    """
    path = Path(db_path)
    if refresh or not path.exists():
        conn = connect(path)
        seed(conn)
        conn.close()
    return path


if __name__ == "__main__":
    connection = connect()
    seed(connection)
    print(f"Banco criado em {DB_PATH} com pacientes: {', '.join(list_patient_ids(connection))}")
    connection.close()
