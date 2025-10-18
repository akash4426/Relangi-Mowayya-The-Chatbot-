# 🎓 Mowayya – Your EduFriend

A friendly Tenglish-speaking study buddy built with **Gemini (Google GenAI)** + **Streamlit**.

## ✨ Features
- Friendly Relangi Mowayya style communication
- Intent detection (study, motivation, career, git, general)
- Local knowledge base (`notes.json`)
- Persona prompt templates
- Post-processing with emojis and Telugu flavour
- Persistent chat history (`chat_history.json`)
- Debug mode to show logic layers

## 🧠 Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **LLM:** Google Gemini API
- **Local Intelligence:** Rule-based intent & keyword matching

## 🚀 Run locally
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"
streamlit run app.py
