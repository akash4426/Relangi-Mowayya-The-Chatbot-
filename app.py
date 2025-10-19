import streamlit as st
import google.generativeai as genai
import os
import random

# ======================
# 1️⃣ APP SETUP
# ======================
st.set_page_config(page_title="🎓 Relangi Mowayya – Your EduFriend", page_icon="🤖", layout="centered")
st.title("🎓 Relangi Mowayya – Andharivadu")

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("⚠️ Gemini key not found. Running in local (offline) mode!")

# ======================
# 2️⃣ SESSION STATE INIT
# ======================
if "history" not in st.session_state:
    st.session_state.history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ======================
# 3️⃣ YOUR CREATIVE PARTS 🎨
# ======================

# --- (a) Emotion Detection ---
def detect_intent(message: str):
    msg = message.lower()
    if any(x in msg for x in ["exam", "study", "project", "subject", "college", "assignment"]):
        return "education"
    elif any(x in msg for x in ["sad", "tired", "demotivate", "depressed", "feel down", "bore"]):
        return "motivation"
    elif any(x in msg for x in ["friend", "chat", "fun", "love", "miss", "life"]):
        return "friendly"
    else:
        return "general"


# --- (b) Personality Core ---
PERSONA_BASE = """
You are Mowayya — a Tenglish-speaking (Telugu+English) student helper and close friend.
You talk like Relangi Mowayya from the movie *Seethamma Vakitlo Sirimalle Chettu*.
You mix humor, warmth, and emotion. You’re never too formal.
"""

PERSONALITY_MODS = {
    "education": "Explain things in an easy Tenglish style, like explaining to a friend during group study.But explain things in detail and should be easier to understand for the user.",
    "motivation": "Use Telugu-style motivation like a best friend cheering up another.",
    "friendly": "Be playful, tease a bit, but always be caring.",
    "general": "Just chat casually like a Telugu college student.Try to generate fun and engaging responses.And make the user. feel comfortable while chatting with you.",
}


# --- (c) Smart Prompt Builder ---
def build_prompt(user_message, intent):
    persona = PERSONA_BASE + "\n" + PERSONALITY_MODS.get(intent, "")
    # Keep last 5 chats only
    context_history = st.session_state.chat_history[-5:]

    if context_history:
        hist = ""
        for m in context_history:
            u = m.get("user", "")
            a = m.get("assistant", "")
            if u or a:
                hist += f"User: {u}\nAssistant: {a}\n"
        persona += "\nHere is our recent conversation:\n" + hist + "\n"

    persona += f"User: {user_message}\nAssistant:"
    return persona


# --- (d) Local Fallback Brain ---
def local_brain(message, intent):
    """Your custom local responses if Gemini key not available"""
    replies = {
        "education": [
            "Arey chadivey raa, concept chinna chinna steps lo explain chestha 😄",
            "Exam ante bayam padaku, focus cheyyu concept paina!",
        ],
        "motivation": [
            "Arey nuvvu chaala capable raa bujji 💪 just believe in yourself!",
            "Break teesuko ra babu, recharge avvu and then oka jet speed lo start cheyyu 🚀",
        ],
        "friendly": [
            "Hahaha 😂 nenu untanu raa neetho! em new news cheppu?",
            "Sarey ra coffee tagukunda relax avvu ☕",
        ],
        "general": [
            "Cheppu raa, em chesthunav ippudu?",
            "Life bagane nadusthundi kada raa?",
        ],
    }
    return random.choice(replies.get(intent, replies["general"]))


# --- (e) Gemini Call + Post Processing ---
def call_gemini(prompt: str, intent: str):
    if not GEMINI_API_KEY:
        # Offline fallback
        return local_brain(prompt, intent)

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Add Tenglish flavor (your signature style)
        suffixes = ["Manishi ante ne manchodu","Ey ra bonchesava","raa 😎", "bujji ❤️", "machaa 😂", "anna 💪", "le ra cheer up ☀️"]
        if not text.endswith(tuple(["!", ".", "?", "😅", "😂", "😎"])):
            text += "..."
        text += " " + random.choice(suffixes)
        return text
    except Exception as e:
        return f"😅 Error vachhindi raa: {e}"


# ======================
# 4️⃣ CHAT UI
# ======================
user_input = st.chat_input("Type your message raa...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"user": user_input})

    # detect user intent
    intent = detect_intent(user_input)

    # build prompt
    prompt = build_prompt(user_input, intent)

    # call model (your hybrid logic)
    reply = call_gemini(prompt, intent)

    st.session_state.history.append({"role": "assistant", "content": reply})
    st.session_state.chat_history[-1]["assistant"] = reply

# ======================
# 5️⃣ DISPLAY CHAT
# ======================
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
