"""
agent.py - اللوب الأساسي (agent loop) اللي بيكلّم الموديل عن طريق llm.py.

الفكرة:
  1) نبعت رسالة المستخدم (وصورة لو فيه) للموديل.
  2) لو الموديل طلب أدوات (function calls) ننفّذها ونرجّعله النتايج.
  3) نكرر لحد ما الموديل يرد بنص نهائي من غير ما يطلب أدوات.
الملف ده مش بيعرف أي حاجة عن Gemini؛ كل التعامل مع المزوّد جوّه llm.py.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

from PIL import Image, ImageOps

from .llm import LLMClient, LLMError, ToolResult, friendly_error
from .prompts import DEFAULT_IMAGE_QUESTION
from .tools import ToolRunner

log = logging.getLogger("carbot.agent")

MAX_IMAGE_SIDE = 1568  # أقصى طول لأي ضلع للصورة
MAX_HISTORY_TURNS = 6  # بنبعت آخر 6 تبادلات بس (توفير في الحد المجاني)


@dataclass
class AgentReply:
    text: str
    cars: list[dict] = field(default_factory=list)  # عربيات تتعرض كروت
    relaxed_note: str | None = None
    is_error: bool = False


# ---------------- الصور ----------------
def prepare_image(raw: bytes, max_side: int = MAX_IMAGE_SIDE) -> bytes:
    """صغّر الصورة (لو كبيرة) وحوّلها JPEG. بيرمي ValueError برسالة عربي لو الملف مش صورة."""
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)  # اظبط اتجاه الصورة بتاعة الموبايل
        img = img.convert("RGB")
    except Exception as exc:
        raise ValueError("مقدرتش أقرا الصورة دي. جرّب صورة JPG أو PNG تانية.") from exc
    img.thumbnail((max_side, max_side))  # بيحافظ على النسبة ومبيكبّرش الصورة
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def make_thumbnail(jpeg: bytes, side: int = 320) -> bytes:
    """صورة مصغّرة بنحفظها في تاريخ المحادثة للعرض بس."""
    img = Image.open(io.BytesIO(jpeg))
    img.thumbnail((side, side))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=70)
    return out.getvalue()


# ---------------- السياق ----------------
def build_context(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """حوّل تاريخ الواجهة لرسائل API (نص بس) واقتصّ آخر max_turns تبادل.
    التبادلات اللي حصل فيها خطأ بنشيلها عشان متلوّثش السياق."""
    pairs, i = [], 0
    while i < len(history) - 1:
        u, a = history[i], history[i + 1]
        if u["role"] == "user" and a["role"] == "assistant" and not a.get("is_error"):
            user_text = ("[المستخدم أرفق صورة] " if u.get("thumb") else "") + u["content"]
            pairs.append([{"role": "user", "content": user_text},
                          {"role": "assistant", "content": a["content"]}])
            i += 2
        else:
            i += 1
    return [m for pair in pairs[-max_turns:] for m in pair]


# ---------------- الـ Agent ----------------
class CarAgent:
    def __init__(self, llm: LLMClient, runner: ToolRunner, max_iterations: int = 6):
        self.llm = llm
        self.runner = runner
        self.max_iterations = max_iterations

    def reply(self, history: list[dict], user_text: str, image_jpeg: bytes | None = None) -> AgentReply:
        """history: تاريخ المحادثة قبل الرسالة الحالية. image_jpeg: صورة جاهزة (بعد prepare_image)."""
        user_text = (user_text or "").strip() or (DEFAULT_IMAGE_QUESTION if image_jpeg else "")
        cars: list[dict] = []
        relaxed_note = None
        try:
            session = self.llm.new_session(build_context(history))
            turn = session.send(user_text, image_jpeg)
            for _ in range(self.max_iterations):
                if not turn.tool_calls:
                    return AgentReply(turn.text or "معرفتش أطلّع رد. جرّب تسأل بطريقة تانية.",
                                      cars=cars, relaxed_note=relaxed_note)
                # الموديل طلب أداة (أو أكتر): ننفّذها ونرجّع كل النتايج مرة واحدة
                results = []
                for call in turn.tool_calls:
                    try:
                        out, is_err = self.runner.run(call.name, call.args), False
                    except Exception as exc:  # مش بنوقّف المحادثة بسبب أداة فشلت
                        log.exception("tool failed")
                        out, is_err = {"error": str(exc)}, True
                    if call.name == "search_cars" and "cars" in out:
                        cars, relaxed_note = out["cars"], out.get("relaxed_note")
                    results.append(ToolResult(call=call, output=out, is_error=is_err))
                turn = session.send_tool_results(results)
            return AgentReply("الطلب محتاج خطوات كتير. جرّب تسأل سؤال أبسط.", is_error=True)
        except LLMError as exc:  # أخطاء معروفة (مشغول، مفتاح غلط...) رسالتها عربي جاهزة
            return AgentReply(exc.user_message, is_error=True)
        except Exception as exc:
            log.exception("agent failed")
            return AgentReply(friendly_error(exc), is_error=True)
