# backend/app.py

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import uuid

from backend.inference import predict
from backend.llm import generate_response
from backend.stt import speech_to_text

app = FastAPI(title="CropGuard AI")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Note: In-memory storage is not suitable for production.
# Replace with a persistent solution like a database or session management.
SESSION_PREDICTION = {}
SESSION_CHAT = []


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/guide", response_class=HTMLResponse)
def guide(request: Request):
    return templates.TemplateResponse("guide.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(
    request: Request,
    crop: str = Form(...),
    image: UploadFile = File(...)
):
    ext = os.path.splitext(image.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    image_path = os.path.join(UPLOAD_DIR, filename)

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    prediction = predict(image_path)
    SESSION_PREDICTION["data"] = prediction
    SESSION_CHAT.clear()

    llm_response = generate_response(prediction)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "crop": prediction["predicted_crop"],
            "disease": prediction["predicted_disease"],
            "confidence": int(prediction["confidence"] * 100),
            "image_url": f"/uploads/{filename}",
            "llm_response": llm_response,
            "chat_context": [],
            "languages": list(SESSION_PREDICTION.get("supported_languages", {}).values())
        }
    )


@app.post("/regenerate")
async def regenerate(payload: dict):
    if "data" not in SESSION_PREDICTION:
        return JSONResponse({"error": "No prediction context found"}, status_code=400)

    language = payload.get("language", "en")
    llm_response = generate_response(
        SESSION_PREDICTION["data"],
        language=language
    )
    return {"response": llm_response}


@app.post("/chat")
async def chat(payload: dict):
    if "data" not in SESSION_PREDICTION:
        return JSONResponse({"reply": "No prediction context available"}, status_code=400)

    message = payload.get("message", "")
    language = payload.get("language", "en")

    # Append user message to chat history
    SESSION_CHAT.append({"role": "user", "content": message})

    # Generate assistant response
    llm_response = generate_response(
        prediction=SESSION_PREDICTION["data"],
        chat_history=SESSION_CHAT,
        language=language
    )

    # Append assistant response to chat history
    SESSION_CHAT.append({"role": "assistant", "content": llm_response})

    return {"reply": llm_response}


@app.post("/stt")
async def speech_input(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    return speech_to_text(audio_bytes)
