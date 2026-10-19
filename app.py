"""
app.py - واجهة Streamlit لشات بوت العربيات (عربي RTL، لهجة مصرية).

ترتيب الملف:
  1) إعدادات وأدوات مساعدة (secrets، CSS، رد صوتي)
  2) تحميل الداتا والـ agent (مع كاش)
  3) بوابة الباسورد
  4) الشريط الجانبي (إرشادات، صوت، صور)
  5) عرض المحادثة + معالجة الرسالة الجديدة
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from chatbot.agent import CarAgent, make_thumbnail, prepare_image
from chatbot.data_loader import DataError, file_signature, load_data
from chatbot.llm import GeminiClient
from chatbot.prompts import DEFAULT_IMAGE_QUESTION, SYSTEM_PROMPT
from chatbot.retriever import TfidfFaultRetriever
from chatbot.tools import TOOL_SCHEMAS, ToolRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("carbot.app")

st.set_page_config(page_title="مساعد العربيات", page_icon="🚗", layout="wide")

APP_DIR = Path(__file__).parent


# ============================================================
# 1) إعدادات
# ============================================================
def get_secret(name: str, default=None):
    """اقرأ إعداد من st.secrets (على Streamlit Cloud) أو من متغيرات البيئة (على جهازك)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # مفيش ملف secrets أصلاً
        pass
    return os.environ.get(name, default)


# gemini-3.5-flash-lite: موديل Flash-Lite مستقر، مجاني في الـ free tier، بيدعم الأدوات والصور
# (اتأكدنا من صفحة الموديل والأسعار في ai.google.dev)
MODEL = get_secret("GEMINI_MODEL", "gemini-3.5-flash-lite")
THINKING_LEVEL = get_secret("GEMINI_THINKING_LEVEL")     # اختياري: MINIMAL / LOW / MEDIUM / HIGH
MAX_MESSAGES = int(get_secret("MAX_MESSAGES", 20))       # أقصى عدد رسايل في الجلسة (الحد المجاني محدود)
MAX_IMAGE_MB = float(get_secret("MAX_IMAGE_MB", 5))      # أقصى حجم لصورة مرفوعة
MAX_TOKENS = int(get_secret("MAX_TOKENS", 2048))         # أقصى طول للرد (بيشمل tokens التفكير)
DATA_FILE = APP_DIR / str(get_secret("DATA_FILE", "cars_chatbot_data.xlsx"))

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
html, body, .stApp, [data-testid="stSidebar"], .stMarkdown, .stChatMessage, button, input, textarea {
    font-family: 'Cairo', sans-serif !important;
}
.stApp, [data-testid="stSidebar"], [data-testid="stChatMessage"], .stMarkdown, .stAlert {
    direction: rtl; text-align: right;
}
[data-testid="stChatMessage"] { direction: rtl; }
[data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }
.stMarkdown ul, .stMarkdown ol { padding-right: 1.2rem; padding-left: 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# 2) الرد الصوتي (Text-to-Speech) عن طريق المتصفح
# ============================================================
def speak_widget(text: str, autoplay: bool) -> None:
    """زرار 🔊 بيقرا الرد بصوت عربي باستخدام speechSynthesis بتاع المتصفح (من غير سيرفر)."""
    clean = re.sub(r"[*_`#>|~\[\]]", "", text)            # شيل رموز الماركداون
    clean = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", clean)  # شيل الإيموجي
    # أمان: الرد جاي من الموديل، فنهرّب كل علامة < عشان مفيش حاجة تقدر تقفل وسم <script>
    payload = json.dumps(clean).replace("<", "\\u003c")
    html = f"""
    <div style="direction:rtl;font-family:Cairo,sans-serif">
      <button id="b" style="padding:4px 12px;border-radius:8px;border:1px solid #ccc;cursor:pointer">🔊 اسمع الرد</button>
      <span id="m" style="color:#b00020;font-size:13px"></span>
    </div>
    <script>
      const text = {payload};
      const msg = document.getElementById('m');
      function speak() {{
        if (!('speechSynthesis' in window)) {{
          msg.textContent = ' المتصفح ده مش بيدعم الصوت. استخدم Google Chrome.'; return;
        }}
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.lang = 'ar-EG';
        const v = window.speechSynthesis.getVoices().find(v => v.lang.toLowerCase().startsWith('ar'));
        if (v) u.voice = v;
        window.speechSynthesis.speak(u);
      }}
      document.getElementById('b').onclick = speak;
      if ({str(autoplay).lower()}) setTimeout(speak, 300);
    </script>"""
    if hasattr(st, "iframe"):          # الإصدارات الجديدة (components.html اتلغت)
        st.iframe(html, height=48)
    else:                               # إصدارات أقدم
        components.html(html, height=48)


# ============================================================
# 3) تحميل الداتا (بكاش مرتبط بتاريخ تعديل الملف)
# ============================================================
@st.cache_resource(show_spinner="بحمّل بيانات العربيات...")
def load_runtime(path: str, mtime: float):
    """mtime في المعاملات: لما الملف يتغير، المفتاح يتغير والكاش يتحدّث."""
    data = load_data(path)
    return data, ToolRunner(data.cars, TfidfFaultRetriever(data.faults))


@st.cache_resource
def get_llm(api_key: str, model: str, max_tokens: int, thinking_level: str | None):
    """عميل Gemini واحد مشترك (الإعدادات في المعاملات عشان الكاش يتحدّث لو اتغيّرت)."""
    return GeminiClient(api_key, model, TOOL_SCHEMAS, SYSTEM_PROMPT,
                        max_output_tokens=max_tokens, thinking_level=thinking_level)


# ============================================================
# 4) بوابة الباسورد (بتظهر بس لو APP_PASSWORD متحدد)
# ============================================================
password = get_secret("APP_PASSWORD")
if password and not st.session_state.get("authed"):
    st.title("🚗 مساعد العربيات")
    entered = st.text_input("الباسورد", type="password")
    if entered:
        if entered == str(password):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("الباسورد غلط.")
    st.stop()

# ============================================================
# 5) تجهيز الداتا والمفتاح
# ============================================================
try:
    data, runner = load_runtime(str(DATA_FILE), file_signature(DATA_FILE))
except DataError as exc:
    st.error(f"في مشكلة في ملف البيانات: {exc}")
    st.stop()

api_key = get_secret("GEMINI_API_KEY")
if not api_key:
    st.error("مفتاح GEMINI_API_KEY مش متحدد. ضيفه في secrets أو في متغيرات البيئة (شوف README).")
    st.stop()
agent = CarAgent(get_llm(api_key, MODEL, MAX_TOKENS, THINKING_LEVEL), runner)

# حالة الجلسة
ss = st.session_state
ss.setdefault("messages", [])      # تاريخ المحادثة
ss.setdefault("user_msgs", 0)      # عداد الرسايل (حماية التكلفة)
ss.setdefault("img_key", 0)        # بنغيّره عشان نفضّي أدوات رفع الصور بعد الإرسال
ss.setdefault("speak_next", False)  # هل نقرا آخر رد تلقائي؟

# ============================================================
# 6) الشريط الجانبي
# ============================================================
with st.sidebar:
    st.title("🚗 مساعد العربيات")
    st.markdown(
        "**🔎 بحث عن عربية**  \nمثال: *عايز عربية أوتوماتيك لحد مليون جنيه فيها كاميرا*\n\n"
        "**🛠️ فهم عطل**  \nمثال: *العربية بتسخن وفيه بخار طالع من الكبوت*"
    )
    if st.button("🗑️ محادثة جديدة", use_container_width=True):
        ss.messages, ss.speak_next = [], False
        st.rerun()

    st.divider()
    voice_on = st.toggle("🔊 رد صوتي", value=False, help="بيقرا الرد بصوت عالي (الأفضل على Chrome)")
    st.caption("🎤 تحدّث بدل ما تكتب (الصوت بيشتغل على Google Chrome):")
    try:
        from streamlit_mic_recorder import speech_to_text
        voice_text = speech_to_text(language="ar-EG", start_prompt="🎤 ابدأ الكلام", stop_prompt="⏹️ خلّصت",
                                    just_once=True, use_container_width=True, key="stt")
    except Exception:  # الحزمة مش متسطّبة أو المتصفح مش داعم
        voice_text = None
        st.warning("الميكروفون مش متاح. استخدم Google Chrome أو اكتب سؤالك.")

    st.divider()
    st.caption("📷 ارفع صورة (لمبة تحذير، تسريب، جزء متضرر...)")
    up_file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"], key=f"up_{ss.img_key}",
                               label_visibility="collapsed")
    with st.expander("أو صوّر بالكاميرا"):
        cam_file = st.camera_input("صوّر", key=f"cam_{ss.img_key}", label_visibility="collapsed")
    pending_image = cam_file or up_file
    analyze_clicked = False
    if pending_image:
        st.image(pending_image, width=160)
        st.caption("الصورة هتتبعت مع رسالتك الجاية، أو اضغط تحليل.")
        analyze_clicked = st.button("🔍 حلّل الصورة", use_container_width=True)

    if data.warnings:
        with st.expander("⚠️ ملاحظات على البيانات"):
            for w in data.warnings:
                st.write("• " + w)
    st.caption(f"الرسايل المتبقية في الجلسة: {max(0, MAX_MESSAGES - ss.user_msgs)}")


# ============================================================
# 7) عرض كروت العربيات
# ============================================================
FEATURE_LABELS = {"Sunroof": "فتحة سقف", "Rear_Camera": "كاميرا خلفية", "Cruise_Control": "فتيس كروز",
                  "Leather_Seats": "فرش جلد", "Bluetooth_Screen": "شاشة بلوتوث"}


def render_car_cards(cars: list[dict]) -> None:
    for start in range(0, len(cars), 3):  # 3 كروت في الصف
        cols = st.columns(3)
        for col, car in zip(cols, cars[start:start + 3]):
            with col, st.container(border=True):
                title = f"{car['Brand_AR'] or car['Brand']} {car['Model_AR'] or car['Model']} {car['Year'] or ''}"
                price = f"{car['Price_EGP']:,} جنيه" if car["Price_EGP"] is not None else "السعر غير متاح"
                st.markdown(f"**{title}**  \n💰 **{price}** · {car['Condition'] or ''}")
                specs = [car["Body_Type"], car["Transmission"], car["Fuel_Type"],
                         f"{car['Engine_CC']} سي سي" if car["Engine_CC"] else None,
                         f"{car['Horsepower']} حصان" if car["Horsepower"] else None,
                         f"{car['Fuel_Consumption_L100km']} لتر/100كم" if car["Fuel_Consumption_L100km"] else None,
                         f"{car['Seats']} مقاعد" if car["Seats"] else None]
                st.caption(" · ".join(str(s) for s in specs if s))
                feats = [label for key, label in FEATURE_LABELS.items() if car.get(key) == "نعم"]
                if feats:
                    st.markdown("✅ " + " ✅ ".join(feats))
                if car.get("Other_Features"):
                    st.caption(str(car["Other_Features"]))
                st.caption(f"كود: {car['Car_ID']}")


# ============================================================
# 8) عرض المحادثة
# ============================================================
if not ss.messages:
    st.info("أهلاً! 👋 أنا مساعد العربيات. اسألني عن عربية تناسب ميزانيتك، أو احكيلي عن عطل (أو ابعت صورة له).")

last_idx = len(ss.messages) - 1
for i, m in enumerate(ss.messages):
    with st.chat_message(m["role"]):
        if m.get("thumb"):
            st.image(m["thumb"], width=200)
        st.markdown(m["content"])
        if m.get("relaxed_note"):
            st.info(m["relaxed_note"])
        if m.get("cars"):
            render_car_cards(m["cars"])
        # زرار القراءة الصوتية على آخر رد بس
        if m["role"] == "assistant" and i == last_idx and not m.get("is_error"):
            speak_widget(m["content"], autoplay=ss.speak_next and voice_on)
            ss.speak_next = False

# ============================================================
# 9) رسالة جديدة (كتابة / صوت / تحليل صورة)
# ============================================================
limit_reached = ss.user_msgs >= MAX_MESSAGES
if limit_reached:
    st.warning("وصلت للحد الأقصى من الرسايل في الجلسة دي. اعمل ريفرش للصفحة لو محتاج تكمل.")

typed = st.chat_input("اكتب سؤالك هنا...", disabled=limit_reached)
user_text = typed or voice_text or (DEFAULT_IMAGE_QUESTION if analyze_clicked else None)

if user_text and not limit_reached:
    image_jpeg, thumb = None, None
    if pending_image:
        if pending_image.size > MAX_IMAGE_MB * 1024 * 1024:
            st.error(f"الصورة كبيرة قوي. الحد الأقصى {MAX_IMAGE_MB:g} ميجا.")
            st.stop()
        try:
            image_jpeg = prepare_image(pending_image.getvalue())
            thumb = make_thumbnail(image_jpeg)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    with st.chat_message("user"):
        if thumb:
            st.image(thumb, width=200)
        st.markdown(user_text)
    with st.chat_message("assistant"):
        with st.spinner("بدوّر لك..."):
            reply = agent.reply(ss.messages, user_text, image_jpeg)

    ss.messages.append({"role": "user", "content": user_text, "thumb": thumb})
    ss.messages.append({"role": "assistant", "content": reply.text, "cars": reply.cars,
                        "relaxed_note": reply.relaxed_note, "is_error": reply.is_error})
    ss.user_msgs += 1
    ss.speak_next = True
    if pending_image:
        ss.img_key += 1  # يفضّي أدوات رفع الصور
    st.rerun()
