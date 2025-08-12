# smart_note.py

import streamlit as st
import requests

# عنوان واجهة التطبيق
st.set_page_config(page_title="النوت الذكي", page_icon="📝")

st.title("📝 النوت الذكي")
st.write("أدخل ملاحظاتك أو أفكارك، وسأرتبها لك!")

# مدخلات المستخدم
user_input = st.text_area("✏️ اكتب ملاحظاتك هنا:", height=150)

# زر التوليد
if st.button("ترتيب الملاحظات"):
    if user_input.strip():
        # استدعاء API من Ollama أو OpenAI
        # مثال باستخدام OpenAI API
        import openai
        openai.api_key = "ضع_مفتاحك_هنا"

        prompt = f"""
        هذه هي ملاحظاتي: {user_input}
        رجاءً رتبها وصغها في شكل نص منظم وواضح.
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        output_text = response["choices"][0]["message"]["content"]
        st.subheader("📄 الملاحظات المرتبة:")
        st.write(output_text)
    else:
        st.warning("الرجاء إدخال ملاحظات أولاً.")
