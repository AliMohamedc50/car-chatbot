"""اختبار GeminiClient بعميل google-genai وهمي (من غير API key ولا إنترنت)."""
import pytest
from google.genai import errors, types

from chatbot.llm import (BUSY_MESSAGE, GeminiClient, LLMBusyError, LLMError, ToolCall, ToolResult,
                         friendly_error)
from chatbot.tools import TOOL_SCHEMAS


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class FakeGenai:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


def resp(*parts, finish="STOP"):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=list(parts)), finish_reason=finish)])


def api_error(code, message="x", status="ERR"):
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": message, "status": status}})


def make(outcomes, sleeps=None):
    fake = FakeGenai(outcomes)
    sleeps = sleeps if sleeps is not None else []
    client = GeminiClient(None, "gemini-3.5-flash-lite", TOOL_SCHEMAS, "SYS", max_output_tokens=777,
                          client=fake, sleep=sleeps.append)
    return client, fake, sleeps


def test_config_has_tools_limits_and_manual_function_calling():
    client, fake, _ = make([resp(types.Part(text="hi"))])
    client.new_session([]).send("سؤال")
    call = fake.models.calls[0]
    cfg = call["config"]
    assert call["model"] == "gemini-3.5-flash-lite"
    assert cfg.system_instruction == "SYS" and cfg.max_output_tokens == 777
    assert cfg.automatic_function_calling.disable is True
    assert [d.name for d in cfg.tools[0].function_declarations] == ["search_cars", "search_faults"]


def test_history_roles_are_mapped_and_text_returned():
    client, fake, _ = make([resp(types.Part(text="أهلاً"))])
    turn = client.new_session([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]).send("c")
    roles = [c.role for c in fake.models.calls[0]["contents"]]
    assert roles == ["user", "model", "user"] and turn.text == "أهلاً" and not turn.tool_calls


def test_image_is_sent_as_inline_jpeg_part_before_text():
    client, fake, _ = make([resp(types.Part(text="ok"))])
    client.new_session([]).send("إيه ده؟", image_jpeg=b"\xff\xd8jpegbytes")
    parts = fake.models.calls[0]["contents"][-1].parts
    assert parts[0].inline_data.mime_type == "image/jpeg" and parts[0].inline_data.data == b"\xff\xd8jpegbytes"
    assert parts[1].text == "إيه ده؟"


def test_function_call_round_trip_keeps_thought_signature_and_ids():
    fc_part = types.Part(function_call=types.FunctionCall(name="search_cars", args={"max_price": 5}, id="abc"),
                         thought_signature=b"SIG")
    client, fake, _ = make([resp(fc_part), resp(types.Part(text="خلصت"))])
    session = client.new_session([])
    turn = session.send("عايز عربية")
    assert turn.tool_calls == [ToolCall(name="search_cars", args={"max_price": 5}, id="abc")]

    final = session.send_tool_results([ToolResult(turn.tool_calls[0], {"count": 1})])
    assert final.text == "خلصت"
    sent = fake.models.calls[1]["contents"]
    assert sent[-2].role == "model" and sent[-2].parts[0].thought_signature == b"SIG"  # الرد الأصلي زي ما هو
    fr = sent[-1].parts[0].function_response
    assert (fr.id, fr.name, fr.response) == ("abc", "search_cars", {"result": {"count": 1}})


def test_tool_error_is_reported_to_model():
    fc = types.Part(function_call=types.FunctionCall(name="x", args={}, id="1"))
    client, fake, _ = make([resp(fc), resp(types.Part(text="ok"))])
    s = client.new_session([])
    t = s.send("q")
    s.send_tool_results([ToolResult(t.tool_calls[0], {"error": "bad"}, is_error=True)])
    assert fake.models.calls[1]["contents"][-1].parts[0].function_response.response == {"error": "bad"}


def test_thought_parts_are_not_shown_to_user():
    client, _, _ = make([resp(types.Part(text="تفكير داخلي", thought=True), types.Part(text="الرد"))])
    assert client.new_session([]).send("q").text == "الرد"


def test_empty_or_blocked_reply_returns_empty_turn():
    blocked = types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="SAFETY")])
    client, _, _ = make([blocked])
    turn = client.new_session([]).send("q")
    assert turn.text == "" and not turn.tool_calls and "SAFETY" in turn.finish_reason


def test_429_retries_once_with_backoff_then_succeeds():
    client, fake, sleeps = make([api_error(429, status="RESOURCE_EXHAUSTED"), resp(types.Part(text="تمام"))])
    assert client.new_session([]).send("q").text == "تمام"
    assert len(fake.models.calls) == 2 and len(sleeps) == 1 and sleeps[0] >= 5.0


def test_429_twice_gives_friendly_busy_message():
    client, fake, sleeps = make([api_error(429), api_error(429)])
    with pytest.raises(LLMBusyError) as e:
        client.new_session([]).send("q")
    assert e.value.user_message == "الخدمة مشغولة، جرب بعد دقيقة." == BUSY_MESSAGE
    assert len(fake.models.calls) == 2 and len(sleeps) == 1  # محاولة واحدة بس زيادة


def test_503_is_also_retried():
    client, fake, _ = make([api_error(503), resp(types.Part(text="ok"))])
    assert client.new_session([]).send("q").text == "ok" and len(fake.models.calls) == 2


def test_retry_delay_hint_from_google_is_respected():
    err = errors.ClientError(429, {"error": {"code": 429, "message": "q", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "9s"}]}})
    client, _, sleeps = make([err, resp(types.Part(text="ok"))])
    client.new_session([]).send("q")
    assert 9.0 <= sleeps[0] <= 9.0 * 1.3 + 0.01


def test_bad_api_key_is_not_retried_and_message_mentions_key():
    client, fake, sleeps = make([api_error(400, "API key not valid. Please pass a valid API key.")])
    with pytest.raises(LLMError) as e:
        client.new_session([]).send("q")
    assert "GEMINI_API_KEY" in e.value.user_message and len(fake.models.calls) == 1 and not sleeps


def test_unknown_model_message():
    assert "GEMINI_MODEL" in friendly_error(api_error(404))


def test_network_error_becomes_friendly_llm_error():
    client, _, _ = make([ConnectionError("down")])
    with pytest.raises(LLMError, match="النت"):
        client.new_session([]).send("q")
