import streamlit as st
import requests

# Streamlit UI
st.title("📝 Tiny Transformer Text Generator")
st.write("Enter a prompt, and the model will predict the next character.")

user_input = st.text_input("Enter text:", "")

if st.button("Generate"):
    if len(user_input) < 10:
        st.warning("⚠️ Input must be at least 10 characters long.")
    else:
        response = requests.post("http://127.0.0.1:8000/generate/", json={"prompt": user_input})
        generated_text = response.json()["generated_text"]
        st.success(f"📝 Generated Text: {generated_text}")
