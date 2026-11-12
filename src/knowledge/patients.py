"""Pacientes SINTÉTICOS para demonstração (nenhum dado real).

Os campos de tempo dos exames usam "requested_hours_ago" (horas desde a
solicitação) para que o cenário seja reproduzível. A base SQLite é populada a
partir desta lista (src/db/database.py).
"""

from __future__ import annotations

PATIENTS: list[dict] = [
    {
        "id": "P001",
        "display_name": "Paciente Sintético 01",
        "age": 67,
        "sex": "M",
        "chief_complaint": "febre e confusão mental há 1 dia, com suspeita de infecção",
        "conditions": ["diabetes tipo 2", "hipertensão arterial"],
        "allergies": ["nenhuma conhecida"],
        "vitals": {"hr": 112, "rr": 24, "temp": 38.9, "sbp": 92, "dbp": 58, "spo2": 93, "consciousness": "confuso"},
        "notes": "Admitido na emergência há 1 hora.",
        "expected_protocol": "PROT-SEPSE-01",
        "exams": [
            {"name": "lactato", "priority": "urgente", "status": "pendente", "requested_hours_ago": 3.0, "result": None, "critical": False},
            {"name": "hemoculturas", "priority": "urgente", "status": "pendente", "requested_hours_ago": 3.0, "result": None, "critical": False},
            {"name": "hemograma", "priority": "urgente", "status": "realizado", "requested_hours_ago": 3.0, "result": "leucocitose", "critical": False},
            {"name": "creatinina", "priority": "urgente", "status": "realizado", "requested_hours_ago": 3.0, "result": "1,9 mg/dL (basal 1,0 mg/dL)", "critical": False},
        ],
    },
    {
        "id": "P002",
        "display_name": "Paciente Sintético 02",
        "age": 58,
        "sex": "M",
        "chief_complaint": "dor torácica opressiva há 40 minutos, com irradiação para o braço esquerdo",
        "conditions": ["hipertensão arterial", "tabagismo"],
        "allergies": ["nenhuma conhecida"],
        "vitals": {"hr": 98, "rr": 20, "temp": 36.8, "sbp": 150, "dbp": 90, "spo2": 97, "consciousness": "alerta"},
        "notes": "Chegou por demanda espontânea.",
        "expected_protocol": "PROT-DOR-TORACICA-01",
        "exams": [
            {"name": "eletrocardiograma", "priority": "urgente", "status": "realizado", "requested_hours_ago": 2.5, "result": "ritmo sinusal, sem supradesnivelamento do segmento ST", "critical": False},
            {"name": "troponina", "priority": "urgente", "status": "pendente", "requested_hours_ago": 2.5, "result": None, "critical": False},
            {"name": "radiografia de tórax", "priority": "rotina", "status": "pendente", "requested_hours_ago": 1.0, "result": None, "critical": False},
        ],
    },
    {
        "id": "P003",
        "display_name": "Paciente Sintético 03",
        "age": 54,
        "sex": "F",
        "chief_complaint": "retorno ambulatorial para seguimento de diabetes",
        "conditions": ["diabetes tipo 2"],
        "allergies": ["látex"],
        "vitals": {"hr": 76, "rr": 16, "temp": 36.5, "sbp": 128, "dbp": 82, "spo2": 98, "consciousness": "alerta"},
        "notes": "Última consulta há 6 meses.",
        "expected_protocol": "PROT-DM2-01",
        "exams": [
            {"name": "hemoglobina glicada", "priority": "rotina", "status": "pendente", "requested_hours_ago": 72.0, "result": None, "critical": False},
            {"name": "microalbuminúria", "priority": "rotina", "status": "pendente", "requested_hours_ago": 72.0, "result": None, "critical": False},
            {"name": "fundo de olho", "priority": "rotina", "status": "pendente", "requested_hours_ago": 12.0, "result": None, "critical": False},
        ],
    },
    {
        "id": "P004",
        "display_name": "Paciente Sintético 04",
        "age": 49,
        "sex": "F",
        "chief_complaint": "retorno ambulatorial de hipertensão arterial com pressão elevada na aferição",
        "conditions": ["hipertensão arterial"],
        "allergies": ["nenhuma conhecida"],
        "vitals": {"hr": 80, "rr": 16, "temp": 36.6, "sbp": 168, "dbp": 104, "spo2": 98, "consciousness": "alerta"},
        "notes": "Refere esquecimento de doses nas últimas semanas.",
        "expected_protocol": "PROT-HAS-01",
        "exams": [
            {"name": "creatinina", "priority": "rotina", "status": "realizado", "requested_hours_ago": 48.0, "result": "0,9 mg/dL", "critical": False},
            {"name": "potássio", "priority": "rotina", "status": "pendente", "requested_hours_ago": 10.0, "result": None, "critical": False},
            {"name": "eletrocardiograma", "priority": "rotina", "status": "realizado", "requested_hours_ago": 48.0, "result": "sem alterações agudas", "critical": False},
        ],
    },
    {
        "id": "P005",
        "display_name": "Paciente Sintético 05",
        "age": 72,
        "sex": "M",
        "chief_complaint": "tosse, febre e falta de ar há 3 dias, com pneumonia suspeita",
        "conditions": ["DPOC"],
        "allergies": ["nenhuma conhecida"],
        "vitals": {"hr": 104, "rr": 28, "temp": 38.2, "sbp": 108, "dbp": 66, "spo2": 90, "consciousness": "alerta"},
        "notes": "Em uso de oxigênio suplementar.",
        "expected_protocol": "PROT-PNEUMONIA-01",
        "exams": [
            {"name": "radiografia de tórax", "priority": "urgente", "status": "realizado", "requested_hours_ago": 2.0, "result": "infiltrado em lobo inferior direito", "critical": False},
            {"name": "hemograma", "priority": "urgente", "status": "pendente", "requested_hours_ago": 1.0, "result": None, "critical": False},
            {"name": "ureia", "priority": "urgente", "status": "pendente", "requested_hours_ago": 1.0, "result": None, "critical": False},
        ],
    },
    {
        "id": "P006",
        "display_name": "Paciente Sintético 06",
        "age": 80,
        "sex": "F",
        "chief_complaint": "internada com aumento da creatinina e queda da diurese",
        "conditions": ["insuficiência cardíaca", "hipertensão arterial"],
        "allergies": ["nenhuma conhecida"],
        "vitals": {"hr": 88, "rr": 18, "temp": 36.7, "sbp": 118, "dbp": 70, "spo2": 95, "consciousness": "alerta"},
        "notes": "Internada há 4 dias; diurese reduzida nas últimas 8 horas.",
        "expected_protocol": "PROT-IRA-01",
        "exams": [
            {"name": "creatinina", "priority": "urgente", "status": "realizado", "requested_hours_ago": 3.0, "result": "2,4 mg/dL (basal 1,1 mg/dL)", "critical": True},
            {"name": "potássio", "priority": "urgente", "status": "realizado", "requested_hours_ago": 3.0, "result": "6,1 mEq/L", "critical": True},
            {"name": "gasometria", "priority": "urgente", "status": "pendente", "requested_hours_ago": 4.0, "result": None, "critical": False},
        ],
    },
]


def get_patient_seed(patient_id: str) -> dict | None:
    for patient in PATIENTS:
        if patient["id"] == patient_id:
            return patient
    return None
