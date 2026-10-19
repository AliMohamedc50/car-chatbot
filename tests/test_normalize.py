from chatbot.normalize import normalize_ar, split_keywords


def test_alef_variants_unified():
    assert normalize_ar("أحمد إبراهيم آلة") == "احمد ابراهيم اله"


def test_ya_and_ta_marbuta():
    assert normalize_ar("مستشفى مدرسة") == "مستشفي مدرسه"


def test_diacritics_and_tatweel_removed():
    assert normalize_ar("مَـحَـرِّك") == "محرك"


def test_arabic_digits_to_western():
    assert normalize_ar("سعر ١٢٣٤٥٠٠") == "سعر 1234500"


def test_punctuation_case_and_spaces():
    assert normalize_ar("  Toyota،  COROLLA!! ") == "toyota corolla"


def test_none_and_nan():
    assert normalize_ar(None) == ""
    assert normalize_ar(float("nan")) == ""


def test_split_keywords_both_commas():
    assert split_keywords("بتسخن، الحرارة عالية, ماتور سخن") == ["بتسخن", "الحراره عاليه", "ماتور سخن"]
