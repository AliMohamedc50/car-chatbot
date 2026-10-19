"""
normalize.py - تطبيع النص العربي (Arabic text normalization).

ليه محتاجينه؟ لأن الناس بتكتب نفس الكلمة بأشكال مختلفة (أحمد / احمد، مدرسة / مدرسه).
لو طبّعنا النص اللي في الداتا واللي المستخدم بيكتبه بنفس الطريقة، المقارنة بتبقى أدق.
"""
import re

# التشكيل (فتحة، ضمة، كسرة، شدة، سكون ...) + الألف الخنجرية
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"  # ـ (التطويل)

# الأرقام العربية والفارسية -> أرقام إنجليزية
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# توحيد الحروف: أإآ -> ا ، ى -> ي ، ة -> ه
_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})

# أي حاجة مش حرف/رقم/مسافة (علامات الترقيم) بنحولها لمسافة
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_ar(text) -> str:
    """رجّع النص مطبّع: حروف صغيرة، بدون تشكيل/تطويل/ترقيم، وأرقام إنجليزية."""
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN
        return ""
    s = str(text).lower()
    s = _DIACRITICS.sub("", s).replace(_TATWEEL, "")
    s = s.translate(_LETTERS).translate(_DIGITS)
    s = _PUNCT.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def split_keywords(text) -> list[str]:
    """قسّم عمود الكلمات المفتاحية (مفصولة بفاصلة عربية أو إنجليزية) لقايمة مطبّعة."""
    if text is None or (isinstance(text, float) and text != text):
        return []
    parts = re.split(r"[,،;؛/\n]+", str(text))
    return [k for k in (normalize_ar(p) for p in parts) if k]
