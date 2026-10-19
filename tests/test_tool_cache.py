from chatbot.retriever import TfidfFaultRetriever
from chatbot.tools import ToolRunner


class CountingRetriever(TfidfFaultRetriever):
    calls = 0

    def search(self, *a, **kw):
        type(self).calls += 1
        return super().search(*a, **kw)


def test_repeated_fault_search_is_served_from_cache(data):
    CountingRetriever.calls = 0
    runner = ToolRunner(data.cars, CountingRetriever(data.faults))
    first = runner.run("search_faults", {"query": "العربية بتسخن"})
    second = runner.run("search_faults", {"query": "العربية بتسخن"})
    assert first == second and CountingRetriever.calls == 1
    runner.run("search_faults", {"query": "الفرامل بتزن"})
    assert CountingRetriever.calls == 2


def test_cache_ignores_argument_order_and_is_not_poisoned_by_mutation(data):
    runner = ToolRunner(data.cars, TfidfFaultRetriever(data.faults))
    a = runner.run("search_cars", {"max_price": 700000, "brand": "toyota"})
    a["cars"].clear()  # الجلسة عدّلت النتيجة
    b = runner.run("search_cars", {"brand": "toyota", "max_price": 700000})
    assert [c["Car_ID"] for c in b["cars"]] == ["C2"]


def test_cache_is_bounded_and_errors_are_not_cached(data):
    runner = ToolRunner(data.cars, TfidfFaultRetriever(data.faults), cache_size=2)
    for p in (1, 2, 3):
        runner.run("search_cars", {"max_price": p})
    assert len(runner._cache) == 2
    for _ in range(2):
        try:
            runner.run("nope", {})
        except ValueError:
            pass
    assert len(runner._cache) == 2
