
# backend/llm.py

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Confidence threshold for disease prediction
CONFIDENCE_THRESHOLD = 0.85

# Supported languages for the LLM response
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


def generate_response(prediction: dict, chat_history=None, language="en"):
    """
    Generates a response from the LLM based on the prediction and chat history.
    """
    crop = prediction.get("predicted_crop", "")
    disease = prediction.get("predicted_disease", "")
    confidence = prediction.get("confidence", 0)
    crop_uncertain = prediction.get("crop_uncertain", False)

    language_name = SUPPORTED_LANGUAGES.get(language, "English")

    # System prompt that instructs the LLM on how to behave
    system_prompt = f"""
You are an expert agricultural assistant providing guidance to farmers.
Respond ONLY in {language_name}.
Your answer must be formatted in HTML using only <strong> and <br> tags.
Do not use any other HTML tags, markdown, or bullet points.
"""

    # User prompt that provides the context for the LLM
    if crop_uncertain:
        user_prompt = """
The crop identification is uncertain. Please inform the user that they should upload a clearer image of the leaf for accurate identification and advice.
"""
    elif "healthy" in disease.lower():
        user_prompt = f"""
The {crop} leaf appears to be healthy. Briefly reassure the user and recommend best practices for maintaining crop health.
"""
    else:
        user_prompt = f"""
A {crop} leaf shows symptoms of {disease} with a confidence of {confidence:.2f}.
Explain the disease, its symptoms, and provide a step-by-step treatment plan.
Keep the language simple and clear for a farmer.
"""

    messages = [{"role": "system", "content": system_prompt.strip()}]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_prompt.strip()})

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.25,
            max_tokens=700,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback response in case of an API error
        return "<strong>Error:</strong> Unable to get a response from the AI assistant at this time."

