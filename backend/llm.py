# ========================= backend/llm.py =========================

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

CONFIDENCE_THRESHOLD = 0.85

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "kn": "Kannada",
    "ml": "Malayalam",
}


def build_prompt(prediction: dict, chat_history=None, language="en"):
    crop = prediction.get("predicted_crop", "")
    disease = prediction.get("predicted_disease", "")
    confidence = prediction.get("confidence", 0)

    language_name = SUPPORTED_LANGUAGES.get(language, "English")

    system_prompt = f"""
You are an expert agricultural assistant.
Respond ONLY in {language_name}.
Use <strong> for titles and <br> for line breaks.
No markdown or bullet symbols.
"""

    if "healthy" in disease.lower():
        user_prompt = f"""
Crop: {crop}
Status: Healthy
Confidence: {confidence}
"""
    elif confidence >= CONFIDENCE_THRESHOLD:
        user_prompt = f"""
Crop: {crop}
Disease: {disease}
Confidence: {confidence}
"""
    else:
        user_prompt = "The disease prediction is uncertain."

    messages = [{"role": "system", "content": system_prompt.strip()}]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_prompt.strip()})
    return messages


def generate_cure(prediction: dict, chat_history=None, language="en"):
    messages = build_prompt(prediction, chat_history, language)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.25,
        max_tokens=700,
    )

    return {
        "response": response.choices[0].message.content.strip()
    }