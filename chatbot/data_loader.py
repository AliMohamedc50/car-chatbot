"""
data_loader.py - قراءة ملف الإكسيل وتنضيفه.

القواعد (من المشروع):
  * اقرأ الأعمدة بالاسم (header) مش بالترتيب.
  * شيل الصفوف اللي Brand (في Cars) أو Category (في Faults) فاضي.
  * خلّي بس الصفوف اللي Data_Check == "OK".
  * تجاهل صفوف المثال اللي في Notes بتاعها "مثال للشكل فقط".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .normalize import normalize_ar

log = logging.getLogger("carbot.data")

EXAMPLE_MARK = "مثال للشكل فقط"

CARS_REQUIRED = [
    "Car_ID", "Brand", "Brand_AR", "Model", "Model_AR", "Year", "Condition", "Price_EGP",
    "Body_Type", "Engine_CC", "Horsepower", "Transmission", "Fuel_Type",
    "Fuel_Consumption_L100km", "Seats", "Sunroof", "Rear_Camera", "Cruise_Control",
    "Leather_Seats", "Bluetooth_Screen", "Other_Features", "Notes", "Data_Check", "Search_Text",
]
FAULTS_REQUIRED = [
    "Fault_ID", "Category", "Component", "Symptom_AR", "Keywords_AR", "Warning_Light",
    "Possible_Cause", "Solution", "Severity", "Safe_To_Drive", "Est_Cost_EGP_Min",
    "Est_Cost_EGP_Max", "Applies_To", "Notes", "Data_Check", "Search_Text",
]


class DataError(Exception):
    """خطأ في ملف الداتا - الرسالة بالعربي عشان تتعرض للمستخدم."""


@dataclass
class LoadedData:
    cars: pd.DataFrame
    faults: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def file_signature(path) -> float:
    """آخر تعديل للملف (mtime). بنستخدمه في مفتاح الكاش عشان الداتا تتحدّث لما الملف يتغير."""
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def _read_sheet(xls: pd.ExcelFile, sheet: str, required: list[str]) -> pd.DataFrame:
    if sheet not in xls.sheet_names:
        raise DataError(f"الشيت '{sheet}' مش موجود في ملف الإكسيل. الشيتات الموجودة: {xls.sheet_names}")
    df = xls.parse(sheet)
    df.columns = [str(c).strip() for c in df.columns]  # شيل المسافات من أسماء الأعمدة
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"الشيت '{sheet}' ناقصه الأعمدة دي: {', '.join(missing)}")
    return df


def _clean_rows(df: pd.DataFrame, key_col: str, sheet: str, warnings: list[str]) -> pd.DataFrame:
    """طبّق قواعد الفلترة الثلاثة بالترتيب."""
    n0 = len(df)
    # 1) العمود الأساسي (Brand / Category) مش فاضي
    has_key = df[key_col].notna() & df[key_col].astype(str).str.strip().ne("")
    df = df[has_key]
    # 2) Data_Check == OK
    ok = df["Data_Check"].astype(str).str.strip().str.upper().eq("OK")
    skipped = int((~ok).sum())
    df = df[ok]
    # 3) تجاهل صفوف المثال
    is_example = df["Notes"].astype(str).str.contains(EXAMPLE_MARK, regex=False, na=False)
    df = df[~is_example]
    if skipped:
        warnings.append(f"{sheet}: تم تجاهل {skipped} صف لأن Data_Check مش OK.")
    log.info("%s: %d صف قبل التنضيف -> %d بعد التنضيف", sheet, n0, len(df))
    return df.reset_index(drop=True)


def _num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")


def load_data(path) -> LoadedData:
    """اقرأ وظبّط الشيتين Cars و Faults. بيرمي DataError برسالة عربي لو فيه مشكلة."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"ملف الداتا مش موجود: {path.name}. حطه جنب app.py.")
    try:
        xls = pd.ExcelFile(path)
    except Exception as exc:  # ملف تالف أو مش إكسيل
        raise DataError(f"مقدرتش أفتح ملف الإكسيل ({path.name}): {exc}") from exc

    warnings: list[str] = []

    # ---------- Cars ----------
    cars = _clean_rows(_read_sheet(xls, "Cars", CARS_REQUIRED), "Brand", "Cars", warnings)
    _num(cars, ["Year", "Price_EGP", "Engine_CC", "Horsepower", "Fuel_Consumption_L100km", "Seats"])
    bad_price = int(cars["Price_EGP"].isna().sum())
    if bad_price:
        warnings.append(f"Cars: {bad_price} عربية من غير سعر صالح (هتتجاهل في البحث بالسعر).")
    # أعمدة مطبّعة مخفية بنستخدمها في البحث (بتبدأ بـ _)
    for col, new in [("Brand", "_brand_en"), ("Brand_AR", "_brand_ar"), ("Model", "_model_en"),
                     ("Model_AR", "_model_ar"), ("Body_Type", "_body"), ("Condition", "_cond"),
                     ("Transmission", "_trans"), ("Fuel_Type", "_fuel")]:
        cars[new] = cars[col].map(normalize_ar)

    # ---------- Faults ----------
    faults = _clean_rows(_read_sheet(xls, "Faults", FAULTS_REQUIRED), "Category", "Faults", warnings)
    _num(faults, ["Est_Cost_EGP_Min", "Est_Cost_EGP_Max"])
    faults["_category"] = faults["Category"].map(normalize_ar)

    if cars.empty:
        warnings.append("تنبيه: شيت Cars فاضي بعد التنضيف.")
    if faults.empty:
        warnings.append("تنبيه: شيت Faults فاضي بعد التنضيف.")
    return LoadedData(cars=cars, faults=faults, warnings=warnings)
