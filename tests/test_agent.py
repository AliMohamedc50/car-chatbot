"""اختبار لوب الـ agent بـ LLM وهمي (من غير API key ولا إنترنت)."""
import io

import pytest
from PIL import Image

from chatbot.agent import CarAgent, build_context, prepare_image
from chatbot.llm import (BUSY_MESSAGE, LLMBusyError, LLMClient, LLMSession, LLMTurn, ToolCall)
from chatbot.retriever import TfidfFaultRetriever
from chatbot.tools import ToolRunner


class FakeSession(LLMSession):
    """بيرجّع لفّات محضّرة بالترتيب (LLMTurn أو استثناء) ويسجّل اللي اتبعت."""
    def __init__(self, turns):
        self.turns, self.sent = list(turns), []

    def send(self, text, image_jpeg=None):
        self.sent.append(("send", text, image_jpeg))
        return self._next()

    def send_tool_results(self, results):
        self.sent.append(("results", results))
        return self._next()

    def _next(self):
        t = self.turns.pop(0)
        if isinstance(t, Exception):
            raise t
        return t


class FakeLLM(LLMClient):
    def __init__(self, turns):
        self.session, self.histories = FakeSession(turns), []

    def new_session(self, history):
        self.histories.append(history)
        return self.session


def calls(*items):
    return LLMTurn(tool_calls=[ToolCall(name=n, args=a, id=f"id{i}") for i, (n, a) in enumerate(items)])


def final(text):
    return LLMTurn(text=text, finish_reason="STOP")


def make_agent(data, turns):
    runner = ToolRunner(data.cars, TfidfFaultRetriever(data.faults))
    llm = FakeLLM(turns)
    return CarAgent(llm, runner), llm


def test_tool_loop_returns_cars_and_final_text(data):
    agent, llm = make_agent(data, [calls(("search_cars", {"max_price": 700000})), final("دي ترشيحاتي")])
    r = agent.reply([], "عايز عربية لحد 700 الف")
    assert r.text == "دي ترشيحاتي" and not r.is_error
    assert {c["Car_ID"] for c in r.cars} == {"C2", "C5"}
    kind, results = llm.session.sent[1]  # تاني إرسال لازم يكون نتايج الأدوات
    assert kind == "results" and results[0].call.name == "search_cars" and "cars" in results[0].output


def test_fault_flow_has_no_cars(data):
    agent, _ = make_agent(data, [calls(("search_faults", {"query": "بتسخن"})), final("تحذير")])
    r = agent.reply([], "العربية بتسخن")
    assert r.text == "تحذير" and r.cars == []


def test_parallel_tool_calls_answered_together(data):
    agent, llm = make_agent(data, [
        calls(("search_cars", {"brand": "kia"}), ("search_faults", {"query": "بتسخن"})), final("تمام")])
    assert agent.reply([], "x").text == "تمام"
    _, results = llm.session.sent[1]
    assert [r.call.name for r in results] == ["search_cars", "search_faults"]  # نتيجة لكل طلب


def test_unknown_tool_is_reported_not_crashing(data):
    agent, llm = make_agent(data, [calls(("hack", {})), final("تمام")])
    assert agent.reply([], "x").text == "تمام"
    assert llm.session.sent[1][1][0].is_error is True


def test_busy_error_shows_friendly_arabic_message(data):
    agent, _ = make_agent(data, [LLMBusyError(BUSY_MESSAGE)])
    r = agent.reply([], "x")
    assert r.is_error and r.text == "الخدمة مشغولة، جرب بعد دقيقة."


def test_unexpected_error_becomes_arabic_message(data):
    agent, _ = make_agent(data, [RuntimeError("boom")])
    r = agent.reply([], "x")
    assert r.is_error and "حاول" in r.text


def test_empty_model_reply_gets_fallback_text(data):
    agent, _ = make_agent(data, [LLMTurn(text="", finish_reason="MAX_TOKENS")])
    r = agent.reply([], "x")
    assert r.text and not r.is_error


def test_endless_tool_loop_is_stopped(data):
    agent, _ = make_agent(data, [calls(("search_cars", {}))] * 10)
    r = agent.reply([], "x")
    assert r.is_error and "خطوات" in r.text


def test_image_is_passed_to_the_llm_with_default_question(data):
    agent, llm = make_agent(data, [final("ok")])
    agent.reply([], "", image_jpeg=b"\xff\xd8fake")
    _, text, image = llm.session.sent[0]
    assert image == b"\xff\xd8fake" and text  # فيه سؤال افتراضي


def test_history_is_trimmed_before_reaching_the_llm(data):
    hist = []
    for i in range(10):
        hist += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    agent, llm = make_agent(data, [final("ok")])
    agent.reply(hist, "new")
    assert len(llm.histories[0]) == 12 and llm.histories[0][0]["content"] == "q4"


def test_build_context_default_is_six_turns_and_skips_errors():
    hist = []
    for i in range(15):
        hist += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    hist += [{"role": "user", "content": "bad"}, {"role": "assistant", "content": "err", "is_error": True}]
    ctx = build_context(hist)
    assert len(ctx) == 12 and ctx[0]["content"] == "q9" and all(m["content"] != "bad" for m in ctx)


def test_build_context_marks_image_turns():
    hist = [{"role": "user", "content": "ايه ده", "thumb": b"x"}, {"role": "assistant", "content": "لمبة"}]
    assert build_context(hist)[0]["content"].startswith("[المستخدم أرفق صورة]")


def test_prepare_image_resizes_and_converts():
    buf = io.BytesIO()
    Image.new("RGBA", (4000, 2000), (255, 0, 0, 128)).save(buf, "PNG")
    out = Image.open(io.BytesIO(prepare_image(buf.getvalue())))
    assert max(out.size) == 1568 and out.format == "JPEG"


def test_prepare_image_rejects_garbage():
    with pytest.raises(ValueError):
        prepare_image(b"not an image")
