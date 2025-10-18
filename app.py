import streamlit as st
import google.generativeai as genai
import os
from datetime import datetime

# ======================
# 1️⃣ SETUP SECTION
# ======================
st.set_page_config(page_title="🎓 Mowayya – Your EduFriend", page_icon="🤝", layout="centered")
st.title("🎓 Mowayya – Your EduFriend")

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.warning("⚠️ Please set your GEMINI_API_KEY as an environment variable.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize session states at very top (before anything else uses them)
if "history" not in st.session_state:
    st.session_state.history = []  # UI chat history

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Context memory for prompt

# ======================
# 2️⃣ PERSONALITY SETUP
# ======================
PERSONA_BASE = """
You are Mowayya — a friendly Telugu-English (Tenglish) speaking assistant.
You talk like Relangi Mowayya from the Telugu movie *Seethamma Vakitlo Sirimalle Chettu*.
You help students with education-related stuff (like exams, coding, projects, motivation) and also talk like a good friend.
Be funny, caring, and chill — use Telugu+English mix naturally.
Never be too formal.
"""

PERSONALITY_MODS = {
    "education": "Focus more on explaining study-related things in easy Tenglish.",
    "motivation": "Encourage like a true friend. Use Telugu touch and fun words.",
    "friendly": "Keep it casual, emotional, supportive like a close buddy."
}

# ======================
# 3️⃣ HELPER FUNCTIONS
# ======================
def detect_intent(message: str):
    msg = message.lower()
    if any(x in msg for x in ["exam", "study", "project", "doubt", "subject"]):
        return "education"
    elif any(x in msg for x in ["sad", "tired", "motivate", "depressed", "help"]):
        return "motivation"
    else:
        return "friendly"


def build_prompt(user_message, intent, include_history=True):
    persona = PERSONA_BASE + "\n" + PERSONALITY_MODS.get(intent, "")

    # Safely include chat history
    if include_history and st.session_state.get("chat_history"):
        history_text = ""
        for m in st.session_state.chat_history:
            u = m.get("user", "")
            a = m.get("assistant", "")
            if u or a:
                history_text += f"User: {u}\nAssistant: {a}\n"
        persona += "\nHere is our recent chat:\n" + history_text + "\n"

    persona += f"User: {user_message}\nAssistant:"
    return persona


def call_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        return f"😅 Sorry raa, konchem issue vachhindi: {e}"


# ======================
# 4️⃣ UI CHAT HANDLING
# ======================
user_input = st.chat_input("Type your message raa...")

if user_input:
    # Append to UI history
    st.session_state.history.append({"role": "user", "content": user_input})
    # Add to chat history for prompt context
    st.session_state.chat_history.append({"user": user_input})

    # Detect intent
    intent = detect_intent(user_input)

    # Build the prompt (after chat_history safely exists)
    prompt = build_prompt(user_input, intent, include_history=True)

    # Call Gemini
    assistant_reply = call_gemini(prompt)

    # Save assistant reply in both histories
    st.session_state.history.append({"role": "assistant", "content": assistant_reply})
    st.session_state.chat_history[-1]["assistant"] = assistant_reply

# ======================
# 5️⃣ DISPLAY CHAT
# ======================
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
