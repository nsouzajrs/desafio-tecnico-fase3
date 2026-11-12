"""Recuperação de trechos de protocolos (RAG) com TF-IDF.

Escolha deliberada: TF-IDF (scikit-learn) roda offline, é rápido, determinístico
e fácil de explicar. Para uma base maior, troque por embeddings + FAISS/Chroma
mantendo a mesma interface (search -> lista de Chunk com score).
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.knowledge.protocols import Chunk, build_chunks


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float


class KnowledgeRetriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or build_chunks()
        self._vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform([c.search_text for c in self.chunks])

    def search(self, query: str, k: int = 4, min_score: float = 0.02, include_templates: bool = False) -> list[Retrieved]:
        """Retorna até k trechos mais similares à consulta, do mais para o menos relevante."""
        scores = linear_kernel(self._vectorizer.transform([query]), self._matrix).ravel()
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked:
            chunk = self.chunks[i]
            if not include_templates and chunk.doc_id.startswith("TEMPL-"):
                continue
            if scores[i] < min_score:
                break
            results.append(Retrieved(chunk=chunk, score=float(scores[i])))
            if len(results) == k:
                break
        return results
