# ========================= backend/stt.py =========================

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def speech_to_text(audio_bytes: bytes):
    response = client.audio.transcriptions.create(
        file=("speech.wav", audio_bytes),
        model="whisper-large-v3-turbo",
        response_format="verbose_json"
    )

    return {
        "text": response.text.strip(),
        "language": response.language
    }