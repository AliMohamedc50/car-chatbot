from chatbot.retriever import search_cars


def ids(result):
    return [c["Car_ID"] for c in result["cars"]]


def test_price_range(data):
    r = search_cars(data.cars, min_price=500_000, max_price=1_000_000)
    assert set(ids(r)) == {"C1", "C2"}
    assert not r["relaxed"]


def test_default_sort_cheapest_first(data):
    r = search_cars(data.cars, max_price=2_000_000)
    prices = [c["Price_EGP"] for c in r["cars"]]
    assert prices == sorted(prices)


def test_arabic_brand(data):
    assert set(ids(search_cars(data.cars, brand="تويوتا"))) == {"C1", "C2"}


def test_english_brand_case_insensitive(data):
    assert set(ids(search_cars(data.cars, brand="TOYOTA"))) == {"C1", "C2"}


def test_arabic_model_with_alef_variants(data):
    # "النترا" مكتوبة بأشكال مختلفة لازم تتطابق
    assert ids(search_cars(data.cars, model="النترا")) == ["C3"]
    assert ids(search_cars(data.cars, model="elantra")) == ["C3"]


def test_required_features(data):
    r = search_cars(data.cars, required_features=["Sunroof", "Leather_Seats"])
    assert ids(r) == ["C3"]


def test_condition_transmission_fuel(data):
    assert set(ids(search_cars(data.cars, condition="مستعمل"))) == {"C2", "C4", "C5"}
    assert set(ids(search_cars(data.cars, transmission="automatic"))) == {"C1", "C3", "C4"}
    assert set(ids(search_cars(data.cars, transmission="مانيوال"))) == {"C2", "C5"}
    assert len(ids(search_cars(data.cars, fuel_type="بنزين"))) == 5


def test_fuel_consumption_and_year(data):
    assert set(ids(search_cars(data.cars, max_fuel_consumption=6.2, min_year=2020))) == {"C5"}


def test_limit_and_count(data):
    r = search_cars(data.cars, limit=2)
    assert r["returned"] == 2 and r["count"] == 5


def test_sort_by_fuel_economy(data):
    assert ids(search_cars(data.cars, sort_by="fuel_economy"))[0] == "C2"


def test_relaxed_price_fallback(data):
    # أرخص عربية 450,000؛ ميزانية 400,000 => بعد +15% (460,000) تظهر C5
    r = search_cars(data.cars, max_price=400_000)
    assert r["relaxed"] is True
    assert ids(r) == ["C5"]
    assert "15%" in r["relaxed_note"]


def test_relaxed_drops_features(data):
    # مفيش عربية فيها فتحة سقف وكمان أقل من 700k، حتى مع +15% => بنشيل المميزات
    r = search_cars(data.cars, max_price=700_000, required_features=["Sunroof"])
    assert r["relaxed"] is True
    assert set(ids(r)) == {"C2", "C5"}
    assert "المميزات" in r["relaxed_note"]


def test_nothing_found_even_after_relaxing(data):
    r = search_cars(data.cars, brand="Ferrari")
    assert r["count"] == 0 and r["cars"] == [] and r["relaxed"] is False
    assert r["relaxed_note"]


def test_output_is_json_serializable(data):
    import json
    json.dumps(search_cars(data.cars), ensure_ascii=False)
