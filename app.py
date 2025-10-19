import streamlit as st
import google.generativeai as genai
import os
import random
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile

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
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None  # store uploaded document embeddings

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
    "education": "Explain things in an easy Tenglish style, like explaining to a friend during group study. Be detailed and friendly.",
    "motivation": "Use Telugu-style motivation like a best friend cheering up another.",
    "friendly": "Be playful, tease a bit, but always be caring.",
    "general": "Chat casually like a Telugu college student, with humor and comfort.",
}

# --- (c) Smart Prompt Builder ---
def build_prompt(user_message, intent):
    persona = PERSONA_BASE + "\n" + PERSONALITY_MODS.get(intent, "")
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

# --- (d) Local Brain (offline mode) ---
def local_brain(message, intent):
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
        return local_brain(prompt, intent)
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()
        suffixes = ["Manishi ante ne manchodu","Ey ra bonchesava","raa 😎", "bujji ❤️", "machaa 😂", "anna 💪", "le ra cheer up ☀️"]
        if not text.endswith(tuple(["!", ".", "?", "😅", "😂", "😎"])):
            text += "..."
        text += " " + random.choice(suffixes)
        return text
    except Exception as e:
        return f"😅 Error vachhindi raa: {e}"

# ======================
# 4️⃣ FILE UPLOAD FEATURE 📂
# ======================
st.subheader("📚 Upload a file for study/explanation (PDF or TXT)")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

if uploaded_file:
    with st.spinner("Reading and indexing your file..."):
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(uploaded_file.read())
        temp_path = temp.name

        # Extract text
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        else:
            loader = TextLoader(temp_path)

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectorstore = FAISS.from_documents(split_docs, embeddings)

        st.success("✅ File processed! Now you can ask anything from it.")

# ======================
# 5️⃣ CHAT UI
# ======================
user_input = st.chat_input("Type your message raa...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"user": user_input})
    intent = detect_intent(user_input)

    # If file uploaded → use RAG
    if st.session_state.vectorstore:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = GoogleGenerativeAI(model="gemini-pro")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
        response = qa_chain({"question": user_input})
        reply = response["answer"]
        reply += " 😎 (Based on your uploaded file!)"
    else:
        prompt = build_prompt(user_input, intent)
        reply = call_gemini(prompt, intent)

    st.session_state.history.append({"role": "assistant", "content": reply})
    st.session_state.chat_history[-1]["assistant"] = reply

# ======================
# 6️⃣ DISPLAY CHAT
# ======================
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
