from src.data.anonymization import anonymize_messages, anonymize_text


def test_masks_brazilian_identifiers():
    text = (
        "Paciente João da Silva, CPF 123.456.789-09, RG 12.345.678-9, prontuário nº 889123. "
        "Contato: (21) 99999-1234, joao.silva@email.com, CEP 20040-020. Consulta em 12/03/2025."
    )
    result = anonymize_text(text)
    for leaked in ("João", "123.456.789-09", "12.345.678-9", "889123", "99999-1234", "joao.silva@email.com", "20040-020", "12/03/2025"):
        assert leaked not in result.text
    for tag in ("[NOME]", "[CPF]", "[RG]", "[PRONTUARIO]", "[TELEFONE]", "[EMAIL]", "[CEP]", "[DATA]"):
        assert tag in result.text
    assert result.total >= 8


def test_masks_doctor_name_and_crm():
    result = anonymize_text("Avaliado pela Dra. Maria Souza (CRM-RJ 123456).")
    assert "Maria" not in result.text and "123456" not in result.text


def test_known_names_are_masked_case_insensitive():
    result = anonymize_text("Falei com carlos pereira ontem.", known_names=["Carlos Pereira"])
    assert "carlos" not in result.text.lower()


def test_clinical_values_are_preserved():
    text = "PA 180/120 mmHg, creatinina 0,3 mg/dL em 48 horas, SpO2 93%, FC 112 bpm."
    assert anonymize_text(text).text == text


def test_system_prompt_is_not_modified():
    messages = [
        {"role": "system", "content": "Dr. House regras"},
        {"role": "user", "content": "CPF 123.456.789-09"},
        {"role": "assistant", "content": "ok"},
    ]
    cleaned, counts = anonymize_messages(messages)
    assert cleaned[0]["content"] == "Dr. House regras"
    assert "[CPF]" in cleaned[1]["content"]
    assert counts["CPF"] == 1
