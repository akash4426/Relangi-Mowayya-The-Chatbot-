import streamlit as st
import google.generativeai as genai
import os
import random
from PyPDF2 import PdfReader  # (NEW) For reading PDFs
import numpy as np          # (NEW) For vector math

# ======================
# 1️⃣ APP SETUP (MODIFIED)
# ======================
# Using wide layout for better readability with sidebar
st.set_page_config(page_title="🎓 Relangi Mowayya", page_icon="🤖", layout="wide")
st.title("🎓 Relangi Mowayya – Andharivadu")

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("⚠️ Gemini key not found. Running in local (offline) mode! PDF features will be disabled.")

# ======================
# 2️⃣ SESSION STATE INIT (MODIFIED)
# ======================
if "history" not in st.session_state:
    st.session_state.history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# (NEW) For storing PDF data
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None # This will hold {'text': chunk, 'embedding': vector}
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None


# ======================
# 3️⃣ (NEW) PDF RAG HELPER FUNCTIONS
# ======================

def get_pdf_text(pdf_doc):
    """Extracts text from an uploaded PDF file."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text() or "" # Add 'or ""' to handle None
        return text
    except Exception as e:
        st.error(f"Error reading PDF raa: {e}")
        return None

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def get_embeddings(chunks):
    """Generates embeddings for a list of text chunks."""
    if not GEMINI_API_KEY:
        st.warning("API key ledu raa, embeddings generate cheyalenu.")
        return None
    try:
        # Using a model optimized for retrieval
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Embeddings create chesetappudu error: {e}")
        return None

def store_pdf_data(pdf_file):
    """Main function to process PDF and store data in session state."""
    with st.spinner("Mowayya PDF ni chaduvuthunnadu... 🧠"):
        raw_text = get_pdf_text(pdf_file)
        if not raw_text:
            st.session_state.pdf_data = None
            st.error("Ee PDF lo text em ledu raa.")
            return

        chunks = get_text_chunks(raw_text)
        if not chunks:
            st.session_state.pdf_data = None
            st.error("Text ni chunks ga break cheyalekapoya.")
            return

        embeddings = get_embeddings(chunks)
        if not embeddings:
            st.session_state.pdf_data = None
            st.error("Embeddings generate avvaledu.")
            return

        # Store as a list of dictionaries
        st.session_state.pdf_data = [
            {"text": chunk, "embedding": np.array(embedding)}
            for chunk, embedding in zip(chunks, embeddings)
        ]
        st.session_state.processed_file_name = pdf_file.name
        st.success(f"Done! '{pdf_file.name}' lo {len(chunks)} chunks ni load chesa ra. 👍")

def find_relevant_chunks(query, k=3):
    """Finds top-k relevant chunks from stored PDF data."""
    if not st.session_state.pdf_data or not GEMINI_API_KEY:
        return "" # Return empty context if no PDF or no key

    try:
        # 1. Embed the user's query
        query_embedding_result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY" # Note the different task_type
        )
        query_vec = np.array(query_embedding_result['embedding'])

        # 2. Calculate cosine similarity
        similarities = []
        for item in st.session_state.pdf_data:
            chunk_vec = item['embedding']
            dot_product = np.dot(query_vec, chunk_vec)
            norm_query = np.linalg.norm(query_vec)
            norm_chunk = np.linalg.norm(chunk_vec)
            # Avoid division by zero
            if norm_query == 0 or norm_chunk == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_query * norm_chunk)
            similarities.append(similarity)

        # 3. Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1] # Get top k and reverse

        # 4. Format as context string
        context = "--- Relevant PDF Context ---\n"
        for idx in top_k_indices:
            context += st.session_state.pdf_data[idx]['text'] + "\n---\n"
        return context

    except Exception as e:
        st.error(f"Context vethiketappudu error: {e}")
        return "" # Return empty on error


# ======================
# 4️⃣ YOUR CREATIVE PARTS 🎨 (Renumbered)
# ======================

# --- (a) Emotion Detection ---
def detect_intent(message: str):
    # ... (Your function is perfect, no changes) ...
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
    # (MODIFIED) Added instruction to use PDF context if available
    "education": "Explain things in an easy Tenglish style, like explaining to a friend. But explain in detail. **If PDF context is provided, base your answer primarily on that context.**",
    "motivation": "Use Telugu-style motivation like a best friend cheering up another.",
    "friendly": "Be playful, tease a bit, but always be caring.",
    "general": "Just chat casually. **If the user asks about the PDF, use the provided context to answer.**",
}

# --- (c) Smart Prompt Builder (MODIFIED) ---
def build_prompt(user_message, intent, pdf_context): # Added pdf_context
    persona = PERSONA_BASE + "\n" + PERSONALITY_MODS.get(intent, "")

    # (NEW) Add PDF context if it exists
    if pdf_context:
        persona += f"\nImportant: Nuvvu answer cheyyadaniki ikkada konchem context undi (This is from the user's PDF). Use this heavily for your answer:\n{pdf_context}\n"

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
    # ... (Your function is perfect, no changes) ...
    replies = {
        "education": [
            "Arey chadivey raa, concept chinna chinna steps lo explain chestha 😄 (Note: API key ledu, PDF chadavadam kudarledu)",
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
        # (MODIFIED) Let's use a more advanced model if available, like 1.5 Flash
        # If you don't have access, it will default to a model that works
        model = genai.GenerativeModel("gemini-2.5-flash") # 1.5 Flash is great for RAG
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Add Tenglish flavor (your signature style)
        suffixes = ["Manishi ante ne manchodu","Ey ra bonchesava","raa 😎", "bujji ❤️", "machaa 😂", "anna 💪", "le ra cheer up ☀️"]
        if not text.endswith(tuple(["!", ".", "?", "😅", "😂", "😎"])):
            text += "..."
        text += " " + random.choice(suffixes)
        return text
    except Exception as e:
        # Check for 1.5 Flash error and fallback
        if "gemini-1.5-flash" in str(e):
             st.warning("Gemini 1.5 Flash access ledu, falling back to 1.0 Pro...")
             model = genai.GenerativeModel("gemini-1.0-pro") # Fallback model
             response = model.generate_content(prompt)
             text = response.text.strip()
             return text + " " + random.choice(suffixes) # Add suffix here too
        
        return f"😅 Error vachhindi raa: {e}"


# ======================
# 5️⃣ (NEW) SIDEBAR FOR PDF UPLOAD
# ======================
with st.sidebar:
    st.header("Upload PDF 📄")
    uploaded_file = st.file_uploader("Nee PDF ikkada upload chey raa", type="pdf")
    
    if uploaded_file:
        # Process file only if it's new
        if st.session_state.processed_file_name != uploaded_file.name:
            store_pdf_data(uploaded_file)
    
    if st.session_state.get("pdf_data"):
        st.success(f"✅ Ready to chat about '{st.session_state.processed_file_name}'!")
        if st.button("Clear PDF Memory"):
            st.session_state.pdf_data = None
            st.session_state.processed_file_name = None
            st.rerun()


# ======================
# 6️⃣ CHAT UI & DISPLAY (MODIFIED)
# ======================

# (MODIFIED) Removed the columns
# Display Chat
with st.container(height=600): # Fixed height container for chat
    for chat in st.session_state.history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

# Chat Input
user_input = st.chat_input("Type your message raa...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"user": user_input})

    # detect user intent
    intent = detect_intent(user_input)

    # (NEW) Find relevant PDF context
    context = find_relevant_chunks(user_input)

    # build prompt (pass the context)
    prompt = build_prompt(user_input, intent, context)

    # call model
    with st.chat_message("assistant"):
        with st.spinner("Mowayya alochisthunnadu... 🤔"):
            reply = call_gemini(prompt, intent)
            st.markdown(reply)

    st.session_state.history.append({"role": "assistant", "content": reply})
    st.session_state.chat_history[-1]["assistant"] = reply
    
    # Rerun to show the latest chat message immediately
    st.rerun()

# (REMOVED) The 'with col2:' block is gone.