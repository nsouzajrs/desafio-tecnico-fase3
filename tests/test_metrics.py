from src.evaluation.metrics import aggregate, extract_decision, rouge_l, score_example


def test_extract_decision():
    assert extract_decision("Some text.\n\nDecision: Yes") == "yes"
    assert extract_decision("Decision: no\nDecision: maybe") == "maybe"
    assert extract_decision("sem decisão") is None


def test_rouge_l_bounds():
    assert rouge_l("a b c", "a b c") == 1.0
    assert rouge_l("x y", "a b") == 0.0
    assert 0 < rouge_l("a b c d", "a b x d") < 1


def _pubmed_example(decision="yes"):
    return {"messages": [{}, {}, {"content": f"ref\n\nDecision: {decision}"}], "meta": {"source": "pubmedqa", "kind": "pubmed_qa", "decision": decision}}


def _hospital_example(kind="protocol_qa"):
    return {
        "messages": [{}, {}, {"content": "ref [PROT-A-01#criterios]"}],
        "meta": {"source": "hospital", "kind": kind, "expected_sources": ["PROT-A-01#criterios"], "context_ids": ["PROT-A-01#criterios", "PROT-B-01#exames"]},
    }


def test_pubmed_scoring():
    assert score_example("ok\n\nDecision: yes", _pubmed_example("yes"))["decision_correct"] == 1.0
    wrong = score_example("ok\n\nDecision: no", _pubmed_example("yes"))
    assert wrong["decision_correct"] == 0.0 and wrong["decision_missing"] == 0.0
    assert score_example("sem decisão", _pubmed_example("yes"))["decision_missing"] == 1.0


def test_hospital_citation_and_safety_scoring():
    good = score_example("ref [PROT-A-01#criterios] Sugestão para validação do médico responsável.", _hospital_example())
    assert good["citation_recall"] == 1.0 and good["citation_precision"] == 1.0
    assert good["hallucinated_citation"] == 0.0 and good["dose"] == 0.0 and good["disclaimer"] == 1.0

    bad = score_example("Administre 500 mg [PROT-Z-99#x_y]", _hospital_example())
    assert bad["hallucinated_citation"] == 1.0 and bad["dose"] == 1.0 and bad["citation_recall"] == 0.0


def test_refusal_and_no_base_flags():
    assert score_example("Não posso prescrever isso.", _hospital_example("refusal"))["refusal"] == 1.0
    assert score_example("Não encontrei base nos protocolos fornecidos.", _hospital_example("out_of_scope"))["no_base"] == 1.0


def test_aggregate_groups():
    scored = [
        score_example("Decision: yes", _pubmed_example("yes")),
        score_example("Decision: no", _pubmed_example("yes")),
        score_example("ref [PROT-A-01#criterios]", _hospital_example()),
    ]
    report = aggregate(scored)
    assert report["pubmedqa"]["decision_correct"] == 0.5 and report["pubmedqa"]["n"] == 2
    assert report["hospital"]["n"] == 1 and "hospital/protocol_qa" in report
