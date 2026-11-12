"""Demonstra a etapa de anonimização com dados FICTÍCIOS (nenhum é de pessoa real).

    python scripts/demo_anonymization.py

Por que existe: os dados sintéticos do projeto já são gerados sem dados pessoais,
então no `stats.json` a contagem de substituições aparece vazia. Este script mostra
o anonimizador funcionando sobre um texto que PARECE um prontuário real, com CPF,
telefone, e-mail, datas e nomes inventados. É o que usamos para provar o requisito
de "anonimização" no relatório e no vídeo.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.anonymization import anonymize_text  # noqa: E402

SAMPLE = (
    "Paciente Maria Aparecida da Silva, prontuário 123456, CPF 123.456.789-09, "
    "nascida em 14/03/1958, telefone (11) 91234-5678, e-mail maria.silva@exemplo.com. "
    "Atendida pela Dra. Fernanda Lima (CRM-SP 123456) com dor torácica há 2 horas. "
    "Retorno marcado para 20/11/2026. Contato do filho: 11 98765-4321."
)

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # acentos corretos no PowerShell

    result = anonymize_text(SAMPLE)
    print("ANTES:\n", SAMPLE, "\n", sep="")
    print("DEPOIS:\n", result.text, "\n", sep="")
    print("Substituições por tipo:", json.dumps(dict(result.replacements), ensure_ascii=False))
    print(f"Total: {result.total}")
