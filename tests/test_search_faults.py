import pytest

from chatbot.retriever import TfidfFaultRetriever
from chatbot.tools import ToolRunner


@pytest.fixture(scope="module")
def retriever(data):
    return TfidfFaultRetriever(data.faults)


def top(retriever, q, **kw):
    hits = retriever.search(q, **kw)
    return hits[0]["Fault_ID"] if hits else None


def test_synonym_from_keywords(retriever):
    # "ماتور سخن" و "بخار طالع" موجودين في Keywords_AR بتاع F1
    assert top(retriever, "ماتور سخن وبخار طالع من الكبوت") == "F1"


def test_egyptian_phrasing(retriever):
    assert top(retriever, "العربية مش بتدور خالص") == "F2"
    assert top(retriever, "الفرامل بتزن لما بدوس") == "F3"
    assert top(retriever, "التكييف مبيبردش والهوا سخن") == "F4"


def test_spelling_variants_still_match(retriever):
    assert top(retriever, "الكاوتش مفروش") == "F5"
    assert top(retriever, "الكاوتـش مفروش") == "F5"  # بتطويل


def test_category_filter(retriever):
    hits = retriever.search("العربية بتسخن", category="كهرباء")
    assert all(h["Category"] == "كهرباء" for h in hits)


def test_results_have_scores_sorted(retriever):
    hits = retriever.search("العربية بتسخن", limit=3)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True) and 0 < scores[0] <= 1


def test_example_and_unchecked_rows_never_returned(retriever):
    for q in ["عطل مثال", "عطل لسه متراجعش"]:
        assert all(h["Fault_ID"] not in ("F6", "F7") for h in retriever.search(q, limit=5))


def test_tool_no_confident_match_for_offtopic(data, retriever):
    runner = ToolRunner(data.cars, retriever)
    out = runner.run("search_faults", {"query": "عايز اعرف سعر الدولار النهاردة"})
    assert out["status"] == "no_confident_match"


def test_tool_ok_for_real_symptom(data, retriever):
    runner = ToolRunner(data.cars, retriever)
    out = runner.run("search_faults", {"query": "العربية بتسخن"})
    assert out["status"] == "ok" and out["results"][0]["Fault_ID"] == "F1"
    assert out["results"][0]["Severity"] == "خطير"
