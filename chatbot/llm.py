"""
llm.py - الملف الوحيد اللي بيكلّم مزوّد الذكاء الاصطناعي (حاليًا Google Gemini).

باقي المشروع (agent.py, tools.py, app.py) ميعرفش حاجة عن Gemini؛ بيتعامل بس مع الواجهة الصغيرة دي:

    llm = GeminiClient(...)                 # أو أي كلاس تاني بيطبّق LLMClient
    session = llm.new_session(history)      # محادثة جديدة بالتاريخ السابق (نص بس)
    turn = session.send("سؤال", image)      # بيرجّع LLMTurn: نص و/أو طلبات أدوات
    turn = session.send_tool_results([...]) # بنرجّع نتايج الأدوات ونكمّل

لو عايزين نبدّل المزوّد بعدين: نكتب كلاس جديد بنفس الدوال وخلاص، من غير ما نلمس الـ agent.

ملاحظات مهمة عن Gemini (من الوثائق الرسمية ai.google.dev):
  * بنستخدم Function Calling اليدوي (بنعطّل automatic_function_calling) عشان نتحكم في الأدوات بنفسنا.
  * Gemini 3 بيرجّع "thought signatures" جوه رد الموديل، ولازم نرجّع Content الرد كامل زي ما هو
    في الطلب اللي بعده، عشان كده بنحتفظ بالـ Content الأصلي ومنفككوش.
  * عدد/ترتيب/id بتاع function_response لازم يطابق طلبات الأدوات السابقة.
"""
from __future__ import annotations

import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

log = logging.getLogger("carbot.llm")

BUSY_MESSAGE = "الخدمة مشغولة، جرب بعد دقيقة."
RETRYABLE_CODES = {429, 503}  # 429 = تجاوز الحد (Rate limit) ، 503 = الخدمة محمّلة


# ============================================================
# أنواع البيانات المحايدة (مش مربوطة بمزوّد معيّن)
# ============================================================
@dataclass
class ToolCall:
    """طلب أداة من الموديل."""
    name: str
    args: dict
    id: str | None = None  # Gemini ممكن يديله id، ولازم نرجّعه مع النتيجة


@dataclass
class ToolResult:
    """نتيجة تنفيذ أداة، بنرجّعها للموديل."""
    call: ToolCall
    output: dict
    is_error: bool = False


@dataclass
class LLMTurn:
    """رد واحد من الموديل: نص نهائي، أو طلبات أدوات (أو الاتنين)."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""


class LLMError(Exception):
    """أي خطأ من المزوّد. user_message رسالة عربي جاهزة للعرض للمستخدم."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


class LLMBusyError(LLMError):
    """الخدمة مشغولة أو الحد المجاني خلص (HTTP 429/503) حتى بعد إعادة المحاولة."""


# ============================================================
# الواجهة (Interface)
# ============================================================
class LLMSession(ABC):
    """محادثة واحدة (سؤال المستخدم + لفّات الأدوات لحد الرد النهائي)."""

    @abstractmethod
    def send(self, text: str, image_jpeg: bytes | None = None) -> LLMTurn:
        """ابعت رسالة المستخدم (واختياريًا صورة JPEG)."""

    @abstractmethod
    def send_tool_results(self, results: list[ToolResult]) -> LLMTurn:
        """رجّع نتايج كل الأدوات اللي الموديل طلبها في اللفّة اللي فاتت."""


class LLMClient(ABC):
    @abstractmethod
    def new_session(self, history: list[dict]) -> LLMSession:
        """history: قايمة {"role": "user"|"assistant", "content": str} (نص بس، بعد الاقتصاص)."""


# ============================================================
# رسائل الأخطاء بالعربي
# ============================================================
def friendly_error(exc: Exception) -> str:
    """حوّل أي استثناء لرسالة عربي مفهومة (من غير تفاصيل تقنية)."""
    if isinstance(exc, LLMError):
        return exc.user_message
    code = getattr(exc, "code", None)
    message = str(getattr(exc, "message", "") or exc).lower()
    if isinstance(code, int):
        if code in RETRYABLE_CODES:
            return BUSY_MESSAGE
        if code in (401, 403) or (code == 400 and "api key" in message):
            return "مفتاح الـ API مش صالح. راجع الإعدادات (GEMINI_API_KEY)."
        if code == 404:
            return "اسم الموديل (GEMINI_MODEL) مش صحيح. راجع الإعدادات."
        return "الخدمة رجّعت خطأ مؤقت. جرّب تاني بعد شوية."
    if isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__name__ in (
            "ConnectError", "ReadTimeout", "ConnectTimeout", "TransportError", "RemoteProtocolError"):
        return "مش قادر أوصل للخدمة. اتأكد من النت وجرّب تاني."
    return "حصلت مشكلة غير متوقعة. حاول تاني."


def _suggested_delay(exc: Exception, default: float) -> float:
    """لو Google قالت "استنى X ثانية" (retryDelay جوه تفاصيل الخطأ) نحترمها، بحد أقصى 15 ثانية."""
    try:
        m = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s['\"]", str(getattr(exc, "details", "")))
        if m:
            return min(max(float(m.group(1)), 1.0), 15.0)
    except Exception:  # pragma: no cover - أي مشكلة في القراءة نتجاهلها ونستخدم القيمة الافتراضية
        pass
    return default


# ============================================================
# تنفيذ Gemini
# ============================================================
class GeminiClient(LLMClient):
    """
    tools: قايمة تعريفات الأدوات بصيغة محايدة: {"name", "description", "parameters": JSON Schema}
    client: اختياري، لتمرير عميل وهمي في الاختبارات (من غير API key).
    """

    def __init__(self, api_key: str | None, model: str, tools: list[dict], system_prompt: str,
                 max_output_tokens: int = 2048, thinking_level: str | None = None,
                 timeout_s: float = 60.0, retry_delay_s: float = 5.0, sleep=time.sleep, client=None):
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model
        self.retry_delay_s = retry_delay_s
        self._sleep = sleep
        self._client = client or genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

        declarations = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                                  parameters_json_schema=t["parameters"]) for t in tools]
        cfg = dict(
            system_instruction=system_prompt,
            tools=[types.Tool(function_declarations=declarations)],
            # نعطّل التنفيذ التلقائي: الـ agent loop بتاعنا هو اللي بينفّذ الأدوات
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            # حد أقصى للرد (في Gemini 3 بيشمل كمان tokens التفكير، فمتخلّيهوش صغير قوي)
            max_output_tokens=max_output_tokens,
            temperature=0.3,
        )
        if thinking_level:  # اختياري: MINIMAL / LOW / MEDIUM / HIGH
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level.upper())
        self._config = types.GenerateContentConfig(**cfg)

    # ---- استدعاء API مع إعادة محاولة واحدة عند 429/503 ----
    def _generate(self, contents):
        from google.genai import errors

        for attempt in (1, 2):
            try:
                return self._client.models.generate_content(
                    model=self.model, contents=contents, config=self._config)
            except errors.APIError as exc:
                if exc.code in RETRYABLE_CODES:
                    if attempt == 1:
                        delay = _suggested_delay(exc, self.retry_delay_s) * (1 + random.random() * 0.3)
                        log.warning("Gemini HTTP %s - هعيد المحاولة بعد %.1f ثانية", exc.code, delay)
                        self._sleep(delay)
                        continue
                    raise LLMBusyError(BUSY_MESSAGE) from exc  # فشلت المحاولة التانية كمان
                log.error("Gemini API error %s: %s", exc.code, exc)
                raise LLMError(friendly_error(exc)) from exc
            except Exception as exc:  # مشاكل شبكة وغيرها
                log.exception("Gemini call failed")
                raise LLMError(friendly_error(exc)) from exc
        raise LLMBusyError(BUSY_MESSAGE)  # pragma: no cover

    def new_session(self, history: list[dict]) -> LLMSession:
        return _GeminiSession(self, history)


class _GeminiSession(LLMSession):
    def __init__(self, owner: GeminiClient, history: list[dict]):
        self.owner, self.t = owner, owner._types
        # Gemini بيسمّي دور الموديل "model" مش "assistant"
        self.contents = [
            self.t.Content(role="model" if m["role"] == "assistant" else "user",
                           parts=[self.t.Part(text=m["content"])])
            for m in history
        ]

    def send(self, text: str, image_jpeg: bytes | None = None) -> LLMTurn:
        parts = []
        if image_jpeg:  # الصورة قبل النص (الترتيب اللي Google بتنصح بيه)
            parts.append(self.t.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg"))
        parts.append(self.t.Part(text=text))
        self.contents.append(self.t.Content(role="user", parts=parts))
        return self._step()

    def send_tool_results(self, results: list[ToolResult]) -> LLMTurn:
        parts = [
            self.t.Part(function_response=self.t.FunctionResponse(
                id=r.call.id, name=r.call.name,
                response={"error": r.output.get("error", "tool failed")} if r.is_error else {"result": r.output}))
            for r in results
        ]
        self.contents.append(self.t.Content(role="user", parts=parts))
        return self._step()

    def _step(self) -> LLMTurn:
        resp = self.owner._generate(self.contents)
        cand = resp.candidates[0] if getattr(resp, "candidates", None) else None
        if cand is None or cand.content is None or not cand.content.parts:
            # رد فاضي (محجوب أو الحد الأقصى للـ tokens خلص في التفكير)
            return LLMTurn(finish_reason=str(getattr(cand, "finish_reason", "EMPTY")))
        # مهم: نحتفظ بالـ Content الأصلي كامل (فيه thought signatures)
        self.contents.append(cand.content)
        text, calls = [], []
        for p in cand.content.parts:
            if getattr(p, "function_call", None):
                fc = p.function_call
                calls.append(ToolCall(name=fc.name, args=dict(fc.args or {}), id=fc.id))
            elif getattr(p, "text", None) and not getattr(p, "thought", False):
                text.append(p.text)
        return LLMTurn(text="".join(text).strip(), tool_calls=calls,
                       finish_reason=str(getattr(cand, "finish_reason", "")))
