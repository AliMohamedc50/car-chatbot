"""
retriever.py - البحث في الداتا.

1) search_cars: فلترة بـ pandas (سعر، ماركة، مواصفات...) + بدائل أقرب لو مفيش نتايج.
2) FaultRetriever: واجهة صغيرة (interface) للبحث في الأعطال.
   التنفيذ الحالي TfidfFaultRetriever (خفيف، من غير موديلات تقيلة).
   لو عايزين نستخدم embeddings بعدين، نكتب كلاس جديد بنفس الدالة search() ونبدّله من غير ما نلمس باقي الكود.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import normalize_ar, split_keywords

YES = "نعم"
FEATURE_COLUMNS = ["Sunroof", "Rear_Camera", "Cruise_Control", "Leather_Seats", "Bluetooth_Screen"]
SORT_OPTIONS = ["price_asc", "price_desc", "year_desc", "fuel_economy", "horsepower_desc"]
PRICE_RELAX = 0.15  # بنوسّع السعر 15% لو مفيش نتايج

# الأعمدة اللي بنرجّعها للموديل/الواجهة (من غير الأعمدة الداخلية)
CAR_OUTPUT_COLUMNS = [
    "Car_ID", "Brand", "Brand_AR", "Model", "Model_AR", "Year", "Condition", "Price_EGP",
    "Body_Type", "Engine_CC", "Horsepower", "Transmission", "Fuel_Type",
    "Fuel_Consumption_L100km", "Seats", *FEATURE_COLUMNS, "Other_Features", "Notes",
]
FAULT_OUTPUT_COLUMNS = [
    "Fault_ID", "Category", "Component", "Symptom_AR", "Warning_Light", "Possible_Cause",
    "Solution", "Severity", "Safe_To_Drive", "Est_Cost_EGP_Min", "Est_Cost_EGP_Max",
    "Applies_To", "Notes",
]

# مرادفات للقيم اللي ممكن المستخدم يكتبها بشكل مختلف عن الداتا
_TRANS_GROUPS = [
    ["automatic", "auto", "اوتوماتيك", "اتوماتيك", "اوتوماتيكي", "اوتماتيك"],
    ["manual", "مانيوال", "مانوال", "يدوي", "عادي"],
]
_FUEL_GROUPS = [
    ["بنزين", "benzine", "benzin", "petrol", "gasoline"],
    ["سولار", "ديزل", "diesel"],
    ["كهرباء", "كهربا", "كهربائي", "electric"],
    ["هايبرد", "هجين", "hybrid"],
    ["غاز", "cng", "غاز طبيعي"],
]
_COND_GROUPS = [["جديد", "new", "زيرو", "0 كم"], ["مستعمل", "used", "second hand"]]


def _normalized(groups: list[list[str]]) -> list[list[str]]:
    """طبّع كل كلمة في مجموعات المرادفات (عشان تتقارن مع الداتا المطبّعة)."""
    return [[normalize_ar(w) for w in g] for g in groups]


_TRANS_GROUPS = _normalized(_TRANS_GROUPS)
_FUEL_GROUPS = _normalized(_FUEL_GROUPS)
_COND_GROUPS = _normalized(_COND_GROUPS)


# ============================================================
# 1) البحث عن عربيات
# ============================================================
def _synonym_mask(col: pd.Series, query: str, groups: list[list[str]]) -> pd.Series:
    """هل القيمة في العمود بتطابق الاستعلام (أو أي مرادف له)؟"""
    q = normalize_ar(query)
    words = [q]
    for g in groups:
        if q in g:
            words = g
            break
    return col.map(lambda v: any(w and w in v for w in words))


def _name_mask(df: pd.DataFrame, query: str, cols: list[str]) -> pd.Series:
    """مطابقة اسم ماركة/موديل مع العمودين الإنجليزي والعربي (مطبّعين). بنقبل الاتجاهين:
    'كورولا' جوّه 'كورولا' أو عمود الماركة جوّه جملة المستخدم."""
    q = normalize_ar(query)
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask |= df[c].map(lambda v: bool(v) and (q in v or v in q))
    return mask


def _filter_cars(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    """طبّق كل الفلاتر اللي المستخدم حددها."""
    m = pd.Series(True, index=df.index)
    if c.get("min_price") is not None:
        m &= df["Price_EGP"] >= c["min_price"]
    if c.get("max_price") is not None:
        m &= df["Price_EGP"] <= c["max_price"]
    if c.get("min_year") is not None:
        m &= df["Year"] >= c["min_year"]
    if c.get("min_seats") is not None:
        m &= df["Seats"] >= c["min_seats"]
    if c.get("max_fuel_consumption") is not None:
        m &= df["Fuel_Consumption_L100km"] <= c["max_fuel_consumption"]
    if c.get("brand"):
        m &= _name_mask(df, c["brand"], ["_brand_en", "_brand_ar"])
    if c.get("model"):
        m &= _name_mask(df, c["model"], ["_model_en", "_model_ar"])
    if c.get("body_type"):
        q = normalize_ar(c["body_type"])
        m &= df["_body"].map(lambda v: bool(v) and (q in v or v in q))
    if c.get("condition"):
        m &= _synonym_mask(df["_cond"], c["condition"], _COND_GROUPS)
    if c.get("transmission"):
        m &= _synonym_mask(df["_trans"], c["transmission"], _TRANS_GROUPS)
    if c.get("fuel_type"):
        m &= _synonym_mask(df["_fuel"], c["fuel_type"], _FUEL_GROUPS)
    for feat in c.get("required_features") or []:
        if feat in FEATURE_COLUMNS:
            m &= df[feat].astype(str).str.strip().eq(YES)
    return df[m]


def _sort_cars(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if sort_by == "price_desc":
        return df.sort_values("Price_EGP", ascending=False)
    if sort_by == "year_desc":
        return df.sort_values(["Year", "Price_EGP"], ascending=[False, True])
    if sort_by == "fuel_economy":
        return df.sort_values("Fuel_Consumption_L100km", na_position="last")
    if sort_by == "horsepower_desc":
        return df.sort_values("Horsepower", ascending=False, na_position="last")
    return df.sort_values("Price_EGP")  # الافتراضي: الأرخص الأول


def _py(v):
    """حوّل قيم pandas/numpy لقيم بايثون عادية (NaN -> None) عشان تتحوّل JSON."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
        return int(v) if v.is_integer() else round(v, 2)
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 2)
    return v


def _rows_to_dicts(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    return [{c: _py(row[c]) for c in cols} for _, row in df[cols].iterrows()]


def search_cars(cars: pd.DataFrame, limit: int = 5, sort_by: str = "price_asc", **criteria) -> dict:
    """
    ابحث في الداتا. لو مفيش نتايج بنجرب خطوتين بالترتيب:
      1) نوسّع السعر 15%   2) نشيل المميزات الاختيارية (required_features)
    ولو لقينا نتايج بعد التوسيع بنعلّم النتيجة relaxed=True ونقول إيه اللي اتغيّر.
    """
    limit = max(1, min(int(limit or 5), 10))
    sort_by = sort_by if sort_by in SORT_OPTIONS else "price_asc"
    crit = {k: v for k, v in criteria.items() if v not in (None, "", [])}

    def finish(found: pd.DataFrame, relaxed: bool, note: str | None) -> dict:
        found = _sort_cars(found, sort_by)
        return {
            "count": int(len(found)),
            "returned": int(min(len(found), limit)),
            "relaxed": relaxed,
            "relaxed_note": note,
            "cars": _rows_to_dicts(found.head(limit), CAR_OUTPUT_COLUMNS),
        }

    found = _filter_cars(cars, crit)
    if not found.empty:
        return finish(found, False, None)

    # --- خطوة 1: وسّع السعر ---
    relaxed_crit = dict(crit)
    notes: list[str] = []
    if crit.get("max_price") is not None or crit.get("min_price") is not None:
        if crit.get("max_price") is not None:
            relaxed_crit["max_price"] = crit["max_price"] * (1 + PRICE_RELAX)
        if crit.get("min_price") is not None:
            relaxed_crit["min_price"] = crit["min_price"] * (1 - PRICE_RELAX)
        notes.append("وسّعنا نطاق السعر حوالي 15%")
        found = _filter_cars(cars, relaxed_crit)
        if not found.empty:
            return finish(found, True, "مفيش نتايج مطابقة تمامًا، فده أقرب بدائل: " + " و".join(notes))

    # --- خطوة 2: شيل المميزات الاختيارية ---
    if crit.get("required_features"):
        relaxed_crit["required_features"] = []
        notes.append("شلنا المميزات الاختيارية المطلوبة")
        found = _filter_cars(cars, relaxed_crit)
        if not found.empty:
            return finish(found, True, "مفيش نتايج مطابقة تمامًا، فده أقرب بدائل: " + " و".join(notes))

    return finish(found, False, "مفيش عربيات في الداتا بتطابق الطلب حتى بعد توسيع البحث.")


# ============================================================
# 2) البحث في الأعطال
# ============================================================
CONFIDENCE_THRESHOLD = 0.25  # أقل من كده = "no_confident_match"
KEYWORD_BOOST = 0.40  # زيادة لو كلمة مفتاحية من Keywords_AR ظهرت في كلام المستخدم


class FaultRetriever(ABC):
    """واجهة البحث في الأعطال - أي تنفيذ جديد (embeddings مثلاً) لازم يطبّقها."""

    @abstractmethod
    def search(self, query: str, category: str | None = None, limit: int = 3) -> list[dict]:
        """رجّع قايمة أعطال مرتبة بالأعلى تشابهًا، كل عنصر فيه 'score' من 0 لـ 1."""


class TfidfFaultRetriever(FaultRetriever):
    """
    TF-IDF على نص كل عطل. بنستخدم نوعين من الـ features:
      - char n-grams (2-5): بتتحمّل أخطاء الإملاء واختلاف الصيغ (بتسخن / سخنت).
      - words: كلمات كاملة.
    + "دفعة" (boost) لو كلمة من Keywords_AR (المرادفات المصرية) موجودة في سؤال المستخدم.
    """

    def __init__(self, faults: pd.DataFrame):
        self.faults = faults.reset_index(drop=True)
        text_cols = ["Search_Text", "Keywords_AR", "Symptom_AR", "Component", "Warning_Light"]
        self._docs = [
            " ".join(normalize_ar(row[c]) for c in text_cols) for _, row in self.faults.iterrows()
        ]
        self._keywords = [split_keywords(k) for k in self.faults["Keywords_AR"]]
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        self._word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        if self._docs:
            self._char_m = self._char.fit_transform(self._docs)
            self._word_m = self._word.fit_transform(self._docs)

    def _keyword_hits(self, idx: int, q: str) -> int:
        return sum(1 for k in self._keywords[idx] if len(k) >= 3 and k in q)

    def search(self, query: str, category: str | None = None, limit: int = 3) -> list[dict]:
        q = normalize_ar(query)
        if not q or not self._docs:
            return []
        # الـ vectors بتبقى L2-normalized تلقائيًا، فالضرب النقطي = cosine similarity
        sim_char = (self._char_m @ self._char.transform([q]).T).toarray().ravel()
        sim_word = (self._word_m @ self._word.transform([q]).T).toarray().ravel()
        scores = 0.6 * sim_char + 0.4 * sim_word

        for i in range(len(scores)):
            hits = self._keyword_hits(i, q)
            if hits:
                scores[i] += KEYWORD_BOOST + 0.1 * min(hits - 1, 2)
        scores = np.clip(scores, 0.0, 1.0)

        if category:
            cq = normalize_ar(category)
            allowed = self.faults["_category"].map(lambda v: bool(v) and (cq in v or v in cq)).to_numpy()
            scores = np.where(allowed, scores, 0.0)

        order = np.argsort(-scores)[: max(1, limit)]
        out = []
        for i in order:
            if scores[i] <= 0:
                continue
            row = {c: _py(self.faults.iloc[i][c]) for c in FAULT_OUTPUT_COLUMNS}
            row["score"] = round(float(scores[i]), 3)
            out.append(row)
        return out
