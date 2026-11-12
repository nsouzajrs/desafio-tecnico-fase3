"""Exporta o diagrama do fluxo LangGraph em Mermaid (renderiza direto no GitHub).

    python scripts/export_graph.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.assistant.graph import build_graph  # noqa: E402
from src.assistant.llm import make_fake_llm  # noqa: E402

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        app = build_graph(make_fake_llm(), db_path=Path(tmp) / "diagram.db")
        mermaid = app.get_graph().draw_mermaid()
    output = Path(__file__).resolve().parent.parent / "docs" / "diagrama_fluxo.mmd"
    output.write_text(mermaid, encoding="utf-8")
    print(mermaid)
    print(f"\nSalvo em {output}")
