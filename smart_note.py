# smart_note.py


import streamlit as st
import requests
import json
from datetime import datetime
import os
from typing import Optional, Dict, Any

# إعدادات الصفحة

st.set_page_config(
page_title=“النوت الذكي المحسن”,
page_icon=“📝”,
layout=“wide”,
initial_sidebar_state=“expanded”
)

# إضافة CSS مخصص للتصميم

st.markdown(”””

<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .feature-box {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86C1;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D5EDDA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28A745;
    }
    .rtl {
        direction: rtl;
        text-align: right;
    }
</style>

“””, unsafe_allow_html=True)

# الشريط الجانبي للإعدادات

st.sidebar.title(“⚙️ الإعدادات”)

# اختيار مقدم الخدمة

provider = st.sidebar.selectbox(
“اختر مقدم خدمة الذكاء الاصطناعي:”,
[“OpenAI”, “Ollama (محلي)”, “Hugging Face”]
)

# إعدادات مختلفة حسب المقدم

if provider == “OpenAI”:
api_key = st.sidebar.text_input(“مفتاح OpenAI API:”, type=“password”)
model = st.sidebar.selectbox(“النموذج:”, [“gpt-4”, “gpt-3.5-turbo”])
elif provider == “Ollama (محلي)”:
ollama_url = st.sidebar.text_input(“رابط Ollama:”, value=“http://localhost:11434”)
model = st.sidebar.text_input(“اسم النموذج:”, value=“llama2”)
elif provider == “Hugging Face”:
hf_token = st.sidebar.text_input(“Hugging Face Token:”, type=“password”)
model = st.sidebar.selectbox(“النموذج:”, [“microsoft/DialoGPT-medium”, “facebook/blenderbot-400M-distill”])

# إعدادات التخصيص

st.sidebar.subheader(“🎨 خيارات التخصيص”)
organization_style = st.sidebar.selectbox(
“نمط التنظيم:”,
[“نقاط رئيسية”, “قوائم مرقمة”, “فقرات منظمة”, “جدول”, “خريطة ذهنية”]
)

temperature = st.sidebar.slider(“مستوى الإبداع:”, 0.1, 1.0, 0.7)
max_tokens = st.sidebar.slider(“الطول الأقصى:”, 100, 2000, 500)

# العنوان الرئيسي

st.markdown(’<h1 class="main-header">📝 النوت الذكي المحسن</h1>’, unsafe_allow_html=True)
st.markdown(’<div class="rtl">أدخل ملاحظاتك أو أفكارك، وسأرتبها وأحسنها باستخدام الذكاء الاصطناعي!</div>’, unsafe_allow_html=True)

# وظائف مساعدة

def save_to_session_state(key: str, value: Any):
“”“حفظ البيانات في حالة الجلسة”””
if ‘history’ not in st.session_state:
st.session_state.history = []
st.session_state[key] = value

def get_ai_response_openai(prompt: str, api_key: str, model: str) -> Optional[str]:
“”“استدعاء OpenAI API”””
try:
import openai
openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content
except Exception as e:
    st.error(f"خطأ في استدعاء OpenAI: {str(e)}")
    return None

def get_ai_response_ollama(prompt: str, url: str, model: str) -> Optional[str]:
“”“استدعاء Ollama API”””
try:
response = requests.post(
f”{url}/api/generate”,
json={
“model”: model,
“prompt”: prompt,
“stream”: False,
“options”: {
“temperature”: temperature,
“num_predict”: max_tokens
}
},
timeout=30
)
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        st.error(f"خطأ في الاستجابة: {response.status_code}")
        return None
except Exception as e:
    st.error(f"خطأ في الاتصال مع Ollama: {str(e)}")
    return None

def get_ai_response_huggingface(prompt: str, token: str, model: str) -> Optional[str]:
“”“استدعاء Hugging Face API”””
try:
headers = {“Authorization”: f”Bearer {token}”}
data = {“inputs”: prompt, “parameters”: {“max_length”: max_tokens, “temperature”: temperature}}
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers,
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:

sara, [8/13/2025 12:33 AM]
return result[0].get("generated_text", "")
        return str(result)
    else:
        st.error(f"خطأ في الاستجابة: {response.status_code}")
        return None
except Exception as e:
    st.error(f"خطأ في الاتصال مع Hugging Face: {str(e)}")
    return None

def create_enhanced_prompt(user_input: str, style: str) -> str:
“”“إنشاء prompt محسن”””
style_instructions = {
“نقاط رئيسية”: “رتب المعلومات في شكل نقاط رئيسية واضحة ومختصرة”,
“قوائم مرقمة”: “رتب المعلومات في قائمة مرقمة منطقية”,
“فقرات منظمة”: “اكتب المعلومات في فقرات منظمة ومترابطة”,
“جدول”: “نظم المعلومات في شكل جدول إذا كان ذلك مناسباً”,
“خريطة ذهنية”: “رتب المعلومات بشكل هرمي كخريطة ذهنية”
}
return f"""
المهمة: تنظيم وتحسين الملاحظات التالية

الملاحظات الأصلية:
{user_input}

التعليمات:
1. {style_instructions.get(style, "رتب المعلومات بشكل واضح ومنطقي")}
2. أضف عناوين فرعية إذا لزم الأمر
3. تأكد من وضوح النص وسهولة قراءته
4. حافظ على المعنى الأصلي
5. أضف أي معلومات مفيدة إضافية إذا كان ذلك مناسباً

النمط المطلوب: {style}

اكتب النتيجة باللغة العربية:
"""

# الواجهة الرئيسية

col1, col2 = st.columns([2, 1])

with col1:
# منطقة إدخال الملاحظات
st.subheader(“✏️ إدخال الملاحظات”)
user_input = st.text_area(
“اكتب ملاحظاتك هنا:”,
height=200,
placeholder=“مثال: اجتماع اليوم، أهداف المشروع، أفكار جديدة…”
)
# خيارات إضافية
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    process_btn = st.button("🚀 ترتيب الملاحظات", type="primary")

with col_btn2:
    clear_btn = st.button("🗑️ مسح النص")

with col_btn3:
    save_btn = st.button("💾 حفظ في التاريخ")

with col2:
# منطقة المعلومات والإحصائيات
st.subheader(“📊 معلومات النص”)
if user_input:
word_count = len(user_input.split())
char_count = len(user_input)
st.metric(“عدد الكلمات”, word_count)
st.metric(“عدد الأحرف”, char_count)
# نصائح سريعة
st.markdown("""
<div class="feature-box">
    <h4>💡 نصائح للاستخدام الأمثل:</h4>
    <ul>
        <li>اكتب أفكارك بحرية</li>
        <li>لا تقلق بشأن التنظيم</li>
        <li>استخدم الكلمات المفتاحية</li>
        <li>اذكر السياق إذا لزم الأمر</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# معالجة الأزرار

if clear_btn:
st.rerun()

if process_btn:
if not user_input.strip():
st.warning(“⚠️ الرجاء إدخال ملاحظات أولاً.”)
else:
# التحقق من صحة الإعدادات
config_valid = True
error_msg = “”
    if provider == "OpenAI" and not api_key:
        config_valid = False
        error_msg = "يرجى إدخال مفتاح OpenAI API"
    elif provider == "Hugging Face" and not hf_token:
        config_valid = False
        error_msg = "يرجى إدخال Hugging Face Token"
    
    if config_valid:
        with st.spinner("🤖 جاري معالجة ملاحظاتك..."):
            # إنشاء الـ prompt
            enhanced_prompt = create_enhanced_prompt(user_input, organization_style)
            
            # استدعاء الـ API المناسب
            output_text = None
            
            if provider == "OpenAI":
                output_text = get_ai_response_openai(enhanced_prompt, api_key, model)
            elif provider == "Ollama (محلي)":
                output_text = get_ai_response_ollama(enhanced_prompt, ollama_url, model)
            elif provider == "Hugging Face":
                output_text = get_ai_response_huggingface(enhanced_prompt, hf_token, model)
            
            if output_text:
                # عرض النتيجة
                st.markdown("---")
                st.subheader("📄 الملاحظات المرتبة:")
                st.markdown(f'<div class="success-box rtl">{output_text}</div>', unsafe_allow_html=True)
                
                # حفظ في التاريخ
                if 'history' not in st.session_state:
                    st.session_state.history = []
                
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "original": user_input,
                    "processed": output_text,
                    "style": organization_style,
                    "provider": provider
                })
                
                # أزرار إضافية للنتيجة

sara, [8/13/2025 12:33 AM]
col_result1, col_result2, col_result3 = st.columns(3)
                
                with col_result1:
                    if st.button("📋 نسخ النتيجة"):
                        st.success("تم نسخ النص!")
                
                with col_result2:
                    # زر التحميل
                    st.download_button(
                        "📥 تحميل النتيجة",
                        data=output_text,
                        file_name=f"ملاحظات_مرتبة_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col_result3:
                    if st.button("🔄 معالجة مرة أخرى"):
                        st.rerun()
    else:
        st.error(f"❌ {error_msg}")

# عرض التاريخ

if ‘history’ in st.session_state and st.session_state.history:
st.markdown(”—”)
st.subheader(“📚 تاريخ الملاحظات”)
# خيار لإظهار/إخفاء التاريخ
show_history = st.checkbox("إظهار التاريخ")

if show_history:
    for i, item in enumerate(reversed(st.session_state.history[-5:])):  # آخر 5 عناصر
        with st.expander(f"📝 {item['timestamp']} - {item['style']}"):
            st.markdown("**النص الأصلي:**")
            st.text(item['original'][:200] + "..." if len(item['original']) > 200 else item['original'])
            st.markdown("**النتيجة:**")
            st.markdown(item['processed'])
            st.caption(f"المقدم: {item['provider']}")

# تذييل التطبيق

st.markdown(”—”)
st.markdown(
“””
<div style="text-align: center; color: #7F8C8D;">
🚀 تم تطوير النوت الذكي المحسن باستخدام Streamlit<br>
💡 يدعم متعدد مقدمي خدمات الذكاء الاصطناعي
</div>
“””,
unsafe_allow_html=True
)
