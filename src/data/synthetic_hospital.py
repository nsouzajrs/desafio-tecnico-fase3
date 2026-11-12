"""Geração do dataset SINTÉTICO do hospital (protocolos, FAQs, modelos, recusas e casos de paciente).

Por que sintético? Os dados internos reais de um hospital são sigilosos (LGPD).
Aqui criamos exemplos no MESMO formato que dados reais teriam, de modo que o
pipeline (anonimização -> curadoria -> fine-tuning) seja idêntico ao de produção.

O que o fine-tuning realmente ensina com estes dados:
  1. responder SOMENTE com base nos trechos fornecidos (RAG);
  2. citar a fonte no formato [ID#seção];
  3. recusar pedidos de prescrição e perguntas sem base;
  4. sempre lembrar a validação humana.

Os protocolos em HOLDOUT_PROTOCOLS ficam só no conjunto de teste, para medir
generalização para protocolos nunca vistos no treino.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.config import SYNTHETIC_DIR
from src.knowledge.answers import annotate_exams, compose_patient_answer
from src.knowledge.patients import PATIENTS
from src.knowledge.protocols import (
    HOLDOUT_PROTOCOLS,
    PROTOCOLS,
    SECTION_LABELS,
    TEMPLATES,
    Chunk,
    build_chunks,
    chunk_index,
    get_protocol,
)
from src.prompts import DISCLAIMER, build_messages, build_user_prompt, format_patient_context

GENERIC_QUESTIONS = [
    "Segundo o protocolo de {topic}, qual é o conteúdo sobre {label}?",
    "O que o protocolo de {topic} define para {label}?",
]
SPECIFIC_QUESTIONS = {
    "objetivo": [
        "Qual é o objetivo do protocolo de {topic}?",
        "Para que serve o protocolo de {topic}?",
    ],
    "criterios": [
        "Quais critérios o protocolo de {topic} usa para identificar o quadro?",
        "Como reconhecer o quadro segundo o protocolo de {topic}?",
    ],
    "conduta_inicial": [
        "Qual é a conduta inicial no protocolo de {topic}?",
        "O que a equipe deve fazer primeiro segundo o protocolo de {topic}?",
    ],
    "exames": [
        "Quais exames o protocolo de {topic} recomenda?",
        "Que exames devem ser considerados no protocolo de {topic}?",
    ],
    "escalonamento": [
        "Quando escalonar ou acionar outras equipes no protocolo de {topic}?",
        "Em que situações o protocolo de {topic} pede escalonamento?",
    ],
}

REFUSAL_QUESTIONS = [
    "Prescreva o tratamento para {topic}.",
    "Qual dose de medicamento devo administrar no caso de {topic}?",
    "Pode receitar o remédio para o paciente com {topic}?",
]

PATIENT_QUESTIONS = [
    "Quais exames estão pendentes para este paciente e qual protocolo se aplica?",
    "Resuma o caso do paciente e a conduta prevista no protocolo.",
    "Há alertas para a equipe médica neste caso?",
]

OUT_OF_SCOPE_TOPICS_TRAIN = [
    "cetoacidose diabética",
    "intoxicação por monóxido de carbono",
    "crise asmática",
    "transfusão de sangue",
    "apendicite aguda",
]
OUT_OF_SCOPE_TOPICS_TEST = ["queimaduras extensas", "parada cardiorrespiratória", "vacinação infantil"]

# (pergunta, id do protocolo, seção, resposta). A resposta sempre cita a fonte.
FAQS = [
    ("Quem valida as sugestões do assistente virtual?", "PROT-PRESCRICAO-01", "conduta_inicial",
     "As sugestões são apoio à decisão e devem ser validadas pelo médico responsável antes de qualquer ação; o assistente não prescreve nem define condutas definitivas [PROT-PRESCRICAO-01#conduta_inicial]."),
    ("O assistente pode prescrever medicamentos?", "PROT-PRESCRICAO-01", "conduta_inicial",
     "Não. O assistente nunca prescreve medicamentos, doses ou condutas definitivas; ele apresenta o que os protocolos preveem para validação do médico responsável [PROT-PRESCRICAO-01#conduta_inicial]."),
    ("Os atendimentos feitos com o assistente ficam registrados?", "PROT-PRESCRICAO-01", "auditoria",
     "Sim. Todo uso é registrado em log de auditoria, com a pergunta, as fontes consultadas e a decisão de validação do médico [PROT-PRESCRICAO-01#auditoria]."),
    ("O que fazer quando o assistente não tem base nos protocolos?", "PROT-PRESCRICAO-01", "excecoes",
     "O assistente informa a limitação e orienta a consulta direta ao médico responsável [PROT-PRESCRICAO-01#excecoes]."),
    ("Em quanto tempo resultados críticos devem ser comunicados ao médico?", "PROT-EXAMES-01", "resultados_criticos",
     "Em até 30 minutos [PROT-EXAMES-01#resultados_criticos]."),
    ("Quando um exame de urgência é considerado em atraso?", "PROT-EXAMES-01", "criterios",
     "Quando está pendente há mais de 2 horas; para exames de rotina, o limite é de 24 horas [PROT-EXAMES-01#criterios]."),
    ("Em quanto tempo o eletrocardiograma deve ser feito na dor torácica aguda?", "PROT-DOR-TORACICA-01", "conduta_inicial",
     "Em até 10 minutos da chegada do paciente, com monitorização e comunicação imediata ao médico [PROT-DOR-TORACICA-01#conduta_inicial]."),
    ("Com que frequência a hemoglobina glicada deve ser medida no diabetes tipo 2?", "PROT-DM2-01", "exames",
     "A cada 3 a 6 meses [PROT-DM2-01#exames]."),
    ("Quais sinais indicam triagem positiva para sepse?", "PROT-SEPSE-01", "criterios",
     "Suspeita de infecção associada a pelo menos dois sinais: frequência cardíaca acima de 90 bpm, frequência respiratória acima de 20 irpm, temperatura acima de 38 °C ou abaixo de 36 °C, pressão sistólica igual ou inferior a 100 mmHg ou alteração do nível de consciência [PROT-SEPSE-01#criterios]."),
    ("O que registrar primeiro em um caso suspeito de AVC?", "PROT-AVC-01", "conduta_inicial",
     "O horário exato de início dos sintomas (ou do último momento em que o paciente foi visto bem), além de medir a glicemia capilar e acionar o médico e a neurologia [PROT-AVC-01#conduta_inicial]."),
    ("Como avaliar a gravidade de uma pneumonia adquirida na comunidade?", "PROT-PNEUMONIA-01", "criterios",
     "Pelo escore CURB-65, que considera confusão, ureia elevada, frequência respiratória igual ou superior a 30, pressão arterial baixa e idade igual ou superior a 65 anos [PROT-PNEUMONIA-01#criterios]."),
    ("A partir de qual pressão arterial o hipertenso deve ser encaminhado ao médico no mesmo dia?", "PROT-HAS-01", "escalonamento",
     "Quando a pressão arterial for igual ou superior a 180/120 mmHg com sintomas como cefaleia intensa, dor torácica ou alteração neurológica [PROT-HAS-01#escalonamento]."),
]


def _example(messages: list[dict], kind: str, expected: list[str], context: list[Chunk], **extra) -> dict:
    return {
        "messages": messages,
        "meta": {
            "source": "hospital",
            "kind": kind,
            "expected_sources": expected,
            "context_ids": [c.chunk_id for c in context],
            **extra,
        },
    }


def _make_context(rng: random.Random, targets: list[Chunk], pool: list[Chunk], n_distractors: int) -> list[Chunk]:
    """Contexto = trechos-alvo + distratores aleatórios, embaralhados (ensina o modelo a escolher a fonte certa)."""
    target_docs = {t.doc_id for t in targets}
    candidates = [c for c in pool if c.doc_id not in target_docs]
    distractors = rng.sample(candidates, k=min(n_distractors, len(candidates)))
    context = targets + distractors
    rng.shuffle(context)
    return context


def _protocol_text_answer(chunk: Chunk) -> str:
    return f"{chunk.text} [{chunk.chunk_id}]\n\n{DISCLAIMER}"


def generate_examples(seed: int = 42) -> dict[str, list[dict]]:
    """Gera exemplos sintéticos separados em train (protocolos vistos) e test (protocolos em HOLDOUT)."""
    rng = random.Random(seed)
    index = chunk_index()
    all_chunks = build_chunks()
    train_pool = [c for c in all_chunks if c.doc_id not in HOLDOUT_PROTOCOLS]
    out: dict[str, list[dict]] = {"train": [], "test": []}

    # 1) Perguntas sobre seções de protocolos (RAG com citação de fonte)
    for protocol in PROTOCOLS:
        split = "test" if protocol["id"] in HOLDOUT_PROTOCOLS else "train"
        pool = all_chunks if split == "test" else train_pool
        for section in protocol["sections"]:
            chunk = index[f"{protocol['id']}#{section}"]
            label = SECTION_LABELS.get(section, section)
            templates = SPECIFIC_QUESTIONS.get(section, GENERIC_QUESTIONS)
            for template in templates:
                question = template.format(topic=protocol["topic"], label=label)
                n_variants = 2 if split == "train" else 1
                for _ in range(n_variants):
                    context = _make_context(rng, [chunk], pool, n_distractors=2)
                    out[split].append(
                        _example(
                            build_messages(build_user_prompt(question, context), _protocol_text_answer(chunk)),
                            "protocol_qa",
                            [chunk.chunk_id],
                            context,
                            protocol=protocol["id"],
                        )
                    )

    # 2) FAQs de médicos (treino)
    for question, pid, section, answer in FAQS:
        chunk = index[f"{pid}#{section}"]
        for _ in range(2):
            context = _make_context(rng, [chunk], train_pool, n_distractors=2)
            out["train"].append(
                _example(
                    build_messages(build_user_prompt(question, context), f"{answer}\n\n{DISCLAIMER}"),
                    "faq",
                    [chunk.chunk_id],
                    context,
                    protocol=pid,
                )
            )

    # 3) Modelos de laudos e receita (apenas templates com campos a preencher)
    for template in TEMPLATES:
        chunk = index[f"{template['id']}#modelo"]
        for question in (
            f"Qual é o modelo interno de {template['title']}?",
            f"Preciso do modelo de {template['title']}. Pode mostrar?",
        ):
            context = _make_context(rng, [chunk], train_pool, n_distractors=2)
            answer = f"Segue o modelo interno [{chunk.chunk_id}]:\n\n{chunk.text}\n\n{DISCLAIMER}"
            out["train"].append(
                _example(build_messages(build_user_prompt(question, context), answer), "template", [chunk.chunk_id], context)
            )

    # 4) Recusa de prescrição (o assistente nunca prescreve)
    guard = index["PROT-PRESCRICAO-01#conduta_inicial"]
    for protocol in PROTOCOLS:
        if protocol["id"] in {"PROT-PRESCRICAO-01", "PROT-EXAMES-01"}:
            continue
        split = "test" if protocol["id"] in HOLDOUT_PROTOCOLS else "train"
        pool = all_chunks if split == "test" else train_pool
        conduta = index[f"{protocol['id']}#conduta_inicial"]
        for template in REFUSAL_QUESTIONS:
            question = template.format(topic=protocol["topic"])
            context = _make_context(rng, [guard, conduta], pool, n_distractors=1)
            answer = (
                "Não posso prescrever medicamentos, doses ou condutas definitivas: a prescrição é "
                f"responsabilidade exclusiva do médico responsável [{guard.chunk_id}]. "
                f"O que o protocolo de {protocol['topic']} prevê para a conduta inicial: "
                f"{conduta.text} [{conduta.chunk_id}]\n\n{DISCLAIMER}"
            )
            out[split].append(
                _example(
                    build_messages(build_user_prompt(question, context), answer),
                    "refusal",
                    [guard.chunk_id, conduta.chunk_id],
                    context,
                    protocol=protocol["id"],
                )
            )

    # 5) Perguntas sem base nos trechos (o assistente admite a limitação)
    excecoes = index["PROT-PRESCRICAO-01#excecoes"]
    for split, topics, pool in (
        ("train", OUT_OF_SCOPE_TOPICS_TRAIN, train_pool),
        ("test", OUT_OF_SCOPE_TOPICS_TEST, all_chunks),
    ):
        for topic in topics:
            question = f"Qual é o protocolo do hospital para {topic}?"
            context = _make_context(rng, [excecoes], pool, n_distractors=2)
            answer = (
                "Não encontrei base nos protocolos fornecidos para responder a essa pergunta. "
                f"Recomendo consultar diretamente o médico responsável [{excecoes.chunk_id}].\n\n{DISCLAIMER}"
            )
            out[split].append(
                _example(build_messages(build_user_prompt(question, context), answer), "out_of_scope", [excecoes.chunk_id], context)
            )

    # 6) Casos de paciente (registro estruturado + protocolos + exames pendentes)
    for patient in PATIENTS:
        split = "test" if patient["expected_protocol"] in HOLDOUT_PROTOCOLS else "train"
        pool = all_chunks if split == "test" else train_pool
        exams = annotate_exams(patient["exams"])
        answer, used_ids = compose_patient_answer(patient, exams, patient["expected_protocol"])
        used = [index[i] for i in used_ids]
        patient_context = format_patient_context(patient, exams)
        for question in PATIENT_QUESTIONS:
            context = _make_context(rng, used, pool, n_distractors=2)
            out[split].append(
                _example(
                    build_messages(build_user_prompt(question, context, patient_context), answer),
                    "patient",
                    used_ids,
                    context,
                    protocol=patient["expected_protocol"],
                    patient_id=patient["id"],
                )
            )

    return out


def export_knowledge_base(directory: Path | None = None) -> None:
    """Exporta protocolos, modelos e pacientes sintéticos em JSON (entregável: 'dataset de exemplo')."""
    directory = Path(directory or SYNTHETIC_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "protocols.json").write_text(json.dumps(PROTOCOLS, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "templates.json").write_text(json.dumps(TEMPLATES, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "patients.json").write_text(json.dumps(PATIENTS, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    export_knowledge_base()
    data = generate_examples()
    print({split: len(items) for split, items in data.items()})
