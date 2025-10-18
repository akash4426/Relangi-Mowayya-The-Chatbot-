# app.py
"""
Mowayya - Your EduFriend
Gemini (Google GenAI) + Streamlit demo with visible student-side logic:
- intent detection
- local knowledge (notes.json)
- persona & prompt templates
- response post-processing (Mowayya flavour)
- persistent chat history (chat_history.json)
"""

import os
import json
import random
import time
from pathlib import Path
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

# —— Load .env (optional) — put GOOGLE_API_KEY there or set env var directly
load_dotenv()

# Try to import the google genai client (new unified SDK)

# CONFIG
apikey = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=apikey)
MODEL_NAME = genai.GenerativeModel("gemini-2.5-flash")# change as needed

# Files
NOTES_FILE = Path("notes.json")
CHAT_HISTORY_FILE = Path("chat_history.json")

# ========== Helper: Ensure local notes exist ==========
default_notes = {
    "os_concepts": "Operating System basics: scheduler decides which process runs. For exam: know scheduling algorithms (FCFS, SJF, RoundRobin) with pros/cons and a simple example each.",
    "db_normalization": "Normalization: 1NF, 2NF, 3NF. Remove repeating groups, remove partial dependency, remove transitive dependency. Example: Student-Course tables -> separate enrollment table.",
    "git_basics": "Git basics: git init, git add ., git commit -m 'msg', git branch, git checkout, git merge, git push origin main. For beginners - always commit small changes."
}

if not NOTES_FILE.exists():
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(default_notes, f, indent=2, ensure_ascii=False)

# ========== Utility functions ==========

def load_notes():
    with open(NOTES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_chat_history(history_item: dict):
    # append one-line json entries (simple)
    CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_item, ensure_ascii=False) + "\n")

# Intent detection (simple rule-based) — your logic visible here
def detect_intent(message: str):
    m = message.lower()
    if any(x in m for x in ["exam", "study", "subject", "marks", "question", "clarify", "concept"]):
        return "study_help"
    if any(x in m for x in ["sad", "depressed", "stress", "tension", "bore", "upset", "lonely"]):
        return "motivation"
    if any(x in m for x in ["resume", "intern", "job", "career", "placement", "interview"]):
        return "career"
    if any(x in m for x in ["git", "github", "push", "commit", "branch"]):
        return "git_help"
    return "general"

# Local knowledge lookup (your own notes)
def lookup_local_notes(query: str):
    notes = load_notes()
    for k, v in notes.items():
        if k.replace("_", " ") in query.lower() or k in query.lower():
            return k, v
        # also check if any keyword in key matches
        if any(word in query.lower() for word in k.split("_")):
            return k, v
    # also try simple keyword matching inside values
    for k, v in notes.items():
        if any(word in query.lower() for word in v.lower().split()[:5]):
            return k, v
    return None, None

# Persona & prompt templates (your prompt-engineering logic)
PERSONA_BASE = """
You are 'Mowayya' — a caring, funny Telugu uncle (like Relangi Mowayya).
Speak in natural Tenglish (mix of Telugu + English).
Be supportive, give simple examples, and keep it short when user asks exam help.
If the user is stressed, motivate with warmth and a short actionable tip.
Use emoji sometimes and a friendly tone. Avoid being robotic.
"""

PERSONALITY_MODS = {
    "study_help": "Focus on teaching the concept step-by-step. Give exam tips and a tiny example.",
    "motivation": "Be inspiring, give a small calming breathing tip or study-break suggestion, and encourage the user.",
    "career": "Give practical career steps (small actionable items). Use respectful but friendly tone.",
    "git_help": "Give simple git commands and a short workflow example for beginners.",
    "general": "Chat casually, answer helpfully."
}

def build_prompt(user_message: str, intent: str, include_history: list = None):
    persona = PERSONA_BASE + "\n" + PERSONALITY_MODS.get(intent, "") + "\n\n"
    # include small chat history for context (optional, we keep lightweight)
    if include_history:
        history_text = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in include_history[-6:]])
        persona += "Here is chat history:\n" + history_text + "\n\n"
    persona += f"User: {user_message}\nAssistant:"
    return persona

# Post-processing: add mowayya flavour (your visible logic)
def add_mowayya_flavour(text: str):
    tails = ["😄", "😉", "💪", "🔥"]
    # make sure not to add duplicate emoji if already present
    if text.strip() and not any(t in text for t in tails):
        text = text.strip() + " " + random.choice(tails)
    return text

# Gemini call wrapper (uses google-genai SDK if available)
def call_gemini(prompt: str, model: str = MODEL_NAME, max_output_tokens: int = 512):
    if genai is None:
        raise RuntimeError("Gemini SDK not available (genai import failed).")
    # Two SDK styles: new (google.genai) or old (google.generativeai)
    try:
        if GENAI_SDK == "new":
            # new unified SDK: from google import genai
            client = genai.Client(api_key=API_KEY) if API_KEY else genai.Client()
            # generate content using models.generate_content
            response = client.models.generate_content(
                model=model,
                text=prompt,  # some SDKs accept 'text' or 'contents' - this is general; if error, catch below
                max_output_tokens=max_output_tokens
            )
            # response may contain .text or .output[0].content
            if hasattr(response, "text") and response.text:
                return response.text
            # fallback to raw
            return str(response)
        else:
            # older style: google.generativeai
            # Example: model = genai.GenerativeModel('gemini-1.5-flash'); response = model.generate_content(prompt)
            model_obj = genai.GenerativeModel(model)
            resp = model_obj.generate_content(prompt)
            # resp.text expected
            if hasattr(resp, "text"):
                return resp.text
            return str(resp)
    except Exception as e:
        # Bubble up error message for debugging; caller will fallback to local notes
        raise RuntimeError(f"Gemini call error: {e}")

# ========== Streamlit UI ==========

st.set_page_config(page_title="Mowayya - Your EduFriend", page_icon="🎓", layout="centered")
st.title("🎓 Mowayya – Your EduFriend")
st.markdown("**Relangi Mowayya style lo Tenglish cheyyadam — help + friend combo.**")

# session state for messages (short-term memory in UI)
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"user":..., "assistant":...}

# Show stored chat history (persistent simple)
if st.sidebar.button("Show saved chats (file)"):
    if CHAT_HISTORY_FILE.exists():
        st.sidebar.markdown("**Saved chat entries (last 50 lines)**")
        lines = CHAT_HISTORY_FILE.read_text(encoding="utf-8").strip().splitlines()[-50:]
        for ln in lines:
            try:
                d = json.loads(ln)
                st.sidebar.markdown(f"- **User:** {d.get('user')[:80]}  \n  **Mowayya:** {d.get('assistant')[:120]}...")
            except Exception:
                st.sidebar.text(ln)
    else:
        st.sidebar.write("No saved chat file yet.")

# Input
user_input = st.chat_input("Adige ra babu — type chey (Tenglish okay!)")

# Option: show intent & local-notes flow toggle for transparency (to show reviewers your logic)
show_debug = st.sidebar.checkbox("Show internal logic (intent, local note match)", value=True)

# Display chat history
for msg in st.session_state.history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

if user_input:
    # show user's message
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 1) Pre-processing: detect intent & check local notes
    intent = detect_intent(user_input)
    note_key, note_text = lookup_local_notes(user_input)

    if show_debug:
        st.write(f"_Intent detected:_ **{intent}**")
        if note_key:
            st.write(f"_Local note matched:_ **{note_key}**")

    # 2) If local note found and intent is study_help or git_help, prefer local note first
    assistant_reply = None
    used_local = False
    if note_text and intent in ("study_help", "git_help", "general"):
        assistant_reply = f"Hi ra! Nenu local note choosanu about *{note_key}*:\n\n{note_text}\n\nIf you want more details, ask and nenu expand chestha."
        assistant_reply = add_mowayya_flavour(assistant_reply)
        used_local = True

    # 3) Else call Gemini (but build prompt with your persona & recent history)
    if not assistant_reply:
        prompt = build_prompt(user_input, intent, include_history=st.session_state.history)
        try:
            # call Gemini
            gemini_resp = call_gemini(prompt)
            assistant_reply = gemini_resp.strip()
            assistant_reply = add_mowayya_flavour(assistant_reply)
        except Exception as e:
            # fallback if Gemini fails
            assistant_reply = ("Ayy babu, Gemini konchem problem avutondi kani nenu help chestha. " 
                               "Ikkada small offline tip istunnanu:\n\n")
            # if any local note available, include it
            if note_text:
                assistant_reply += f"{note_text}\n\n"
            assistant_reply += "Otherwise try rephrasing or ask short specific question. Nenu ikkada unna."
            assistant_reply = add_mowayya_flavour(assistant_reply)
            # Log error to console (developer)
            st.error(f"Gemini call failed - falling back to local logic. Error: {e}")

    # 4) Post-processing done — display and save
    st.session_state.history.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    # 5) Persist chat to file (your visible logic)
    try:
        save_chat_history({"timestamp": int(time.time()), "user": user_input, "assistant": assistant_reply})
    except Exception as e:
        # don't fail user flow just for file write
        print("Could not save chat history:", e)

    # Done
