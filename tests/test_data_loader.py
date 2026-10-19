import pandas as pd
import pytest

from chatbot.data_loader import DataError, load_data


def test_only_valid_cars_kept(data):
    # C6 (Data_Check=CHECK) و C7 (مثال) و C8 (Brand فاضي) لازم يتشالوا
    assert sorted(data.cars["Car_ID"]) == ["C1", "C2", "C3", "C4", "C5"]


def test_only_valid_faults_kept(data):
    assert sorted(data.faults["Fault_ID"]) == ["F1", "F2", "F3", "F4", "F5"]


def test_price_is_numeric(data):
    assert pd.api.types.is_numeric_dtype(data.cars["Price_EGP"])


def test_normalized_helper_columns(data):
    row = data.cars[data.cars["Car_ID"] == "C3"].iloc[0]
    assert row["_model_ar"] == "النترا"
    assert row["_brand_en"] == "hyundai"


def test_missing_file_gives_arabic_error(tmp_path):
    with pytest.raises(DataError, match="مش موجود"):
        load_data(tmp_path / "nope.xlsx")


def test_missing_column_names_sheet_and_column(tmp_path, excel_path):
    bad = tmp_path / "bad.xlsx"
    cars = pd.read_excel(excel_path, "Cars").drop(columns=["Price_EGP"])
    faults = pd.read_excel(excel_path, "Faults")
    with pd.ExcelWriter(bad) as xw:
        cars.to_excel(xw, sheet_name="Cars", index=False)
        faults.to_excel(xw, sheet_name="Faults", index=False)
    with pytest.raises(DataError) as e:
        load_data(bad)
    assert "Cars" in str(e.value) and "Price_EGP" in str(e.value)


def test_missing_sheet(tmp_path, excel_path):
    bad = tmp_path / "nosheet.xlsx"
    with pd.ExcelWriter(bad) as xw:
        pd.read_excel(excel_path, "Cars").to_excel(xw, sheet_name="Cars", index=False)
    with pytest.raises(DataError, match="Faults"):
        load_data(bad)


def test_file_signature_changes_when_file_changes(tmp_path):
    """مفتاح الكاش (mtime) لازم يتغير لما الملف يتعدّل، عشان تحديثات الزملاء تظهر."""
    import os

    from chatbot.data_loader import file_signature
    f = tmp_path / "x.xlsx"
    f.write_bytes(b"a")
    before = file_signature(f)
    os.utime(f, (before + 100, before + 100))
    assert file_signature(f) != before
    assert file_signature(tmp_path / "missing.xlsx") == 0.0
