# backend/app.py

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import uuid

from backend.inference import predict
from backend.llm import generate_cure
from backend.stt import speech_to_text

app = FastAPI(title="CropGuard AI")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

    confidence_pct = int(prediction["confidence"] * 100)

    if prediction["crop_uncertain"]:
        llm_response = (
            "<strong>Crop Identification Uncertain:</strong><br>"
            "The AI is not confident about the crop type.<br><br>"
            "<strong>What You Can Do:</strong><br>"
            "Please upload a clearer image with the full leaf visible "
            "or confirm the crop manually."
        )
    else:
        llm = generate_cure(prediction)
        llm_response = llm["response"]

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "crop": prediction["predicted_crop"],
            "disease": prediction["predicted_disease"],
            "confidence": confidence_pct,
            "image_url": f"/uploads/{filename}",
            "llm_response": llm_response,
            "chat_context": []
        }
    )


@app.post("/chat")
async def chat(payload: dict):
    if "data" not in SESSION_PREDICTION:
        return JSONResponse({"reply": "No prediction context available"}, status_code=400)

    if SESSION_PREDICTION["data"].get("crop_uncertain"):
        return {
            "reply": "Crop type is unclear. Please upload a clearer image before asking treatment questions."
        }

    message = payload.get("message", "")
    language = payload.get("language", "en")

    SESSION_CHAT.append({"role": "user", "content": message})

    llm = generate_cure(
        prediction=SESSION_PREDICTION["data"],
        chat_history=SESSION_CHAT,
        language=language
    )

    SESSION_CHAT.append({"role": "assistant", "content": llm["response"]})

    return {"reply": llm["response"]}


@app.post("/stt")
async def speech_input(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    return speech_to_text(audio_bytes)