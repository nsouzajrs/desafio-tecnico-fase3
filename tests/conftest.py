import pytest


@pytest.fixture(autouse=True)
def isolated_audit_log(tmp_path, monkeypatch):
    """Cada teste grava a auditoria em um arquivo temporário (nunca em logs/ do projeto)."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(path))
    return path


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "hospital_test.db"
