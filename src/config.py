"""Configuração central do projeto (caminhos e hiperparâmetros).

Tudo pode ser sobrescrito por variáveis de ambiente, o que facilita rodar o
mesmo código localmente (Windows/PowerShell) e no Google Colab (com Drive).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Dados -----------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
PUBMEDQA_PATH = RAW_DIR / "ori_pqal.json"

# --- Banco estruturado (prontuários sintéticos) -----------------------------
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "hospital.db"))

# --- Modelo -----------------------------------------------------------------
# Llama 3.2 3B Instruct é o ponto de partida (cabe na T4 do Colab com QLoRA).
# Para testes rápidos, use "meta-llama/Llama-3.2-1B-Instruct".
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", ROOT / "artifacts" / "adapter"))

# --- Fine-tuning (valores padrão; podem ser sobrescritos via CLI) ----------
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1536"))
SEED = int(os.getenv("SEED", "42"))

# --- Logs / auditoria -------------------------------------------------------
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))
AUDIT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", LOG_DIR / "audit.jsonl"))

# --- Relatórios de avaliação ------------------------------------------------
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT / "reports"))
