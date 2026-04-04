"""
 Main FastAPI Application
"""
import time
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil, os, tempfile

from src.models import (
    AnalyzeRequest, AnalyzeResponse,
    VoiceAnalyzeResponse, HealthResponse,
    ConversationHistoryResponse
)
from src.llm_client import analyze_prompt_with_llm
from src.guardrails import validate_prompt, detect_injection, filter_output
from src.memory import ConversationMemory
from src.voice import transcribe_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



app = FastAPI(
    title="AI Prompt Analyzer",
    description="Analyzes and optimizes prompts for better LLM responses",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory rate limiting 
_rate_limit: dict = {}
RATE_LIMIT_MAX = 20       # max requests
RATE_LIMIT_WINDOW = 60    # per minute

memory_store = ConversationMemory()


def check_rate_limit(client_ip: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    requests = _rate_limit.get(client_ip, [])
    requests = [t for t in requests if t > window_start]
    if len(requests) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    requests.append(now)
    _rate_limit[client_ip] = requests


@app.get("/", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", message="AI Prompt Analyzer is running 🚀")


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_prompt(req: AnalyzeRequest, request: Request):
    """
    Analyze and enhance a text prompt.
    - Detects prompt injection
    - Runs Chain-of-Thought reasoning
    - Returns structured analysis with improved prompt
    """
    client_ip = request.client.host
    check_rate_limit(client_ip)

    # --- Guardrails: Input Validation ---
    validation = validate_prompt(req.prompt)
    if not validation.is_valid:
        raise HTTPException(status_code=400, detail=validation.error_message)

    # --- Guardrails: Injection Detection ---
    injection_result = detect_injection(req.prompt)

    # --- Memory: get conversation context ---
    history = memory_store.get_history(req.user_id)

    # --- LLM Analysis ---
    try:
        llm_result = analyze_prompt_with_llm(req.prompt, history)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {str(e)}")

    # --- Guardrails: Output Filtering ---
    filtered_prompt = filter_output(llm_result.improved_prompt)

    # --- Memory: Save interaction ---
    memory_store.add_message(req.user_id, "user", req.prompt)
    memory_store.add_message(req.user_id, "assistant", filtered_prompt)

    return AnalyzeResponse(
        is_injection=injection_result.is_injection,
        injection_details=injection_result.details,
        issues=llm_result.issues,
        suggestions=llm_result.suggestions,
        improved_prompt=filtered_prompt,
        reasoning_steps=llm_result.reasoning_steps,
        context_used=len(history) > 0
    )


@app.post("/analyze/voice", response_model=VoiceAnalyzeResponse)
async def analyze_voice(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = "default"
):
    """
    Transcribe voice audio then analyze the transcribed prompt.
    Accepts: wav, mp3, m4a, ogg files.
    """
    client_ip = request.client.host
    check_rate_limit(client_ip)

    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp4", "audio/ogg", "audio/x-wav"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {file.content_type}")

    # Save temp file
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        transcribed_text = transcribe_audio(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Now run the same analysis on the transcribed text
    validation = validate_prompt(transcribed_text)
    if not validation.is_valid:
        raise HTTPException(status_code=400, detail=validation.error_message)

    injection_result = detect_injection(transcribed_text)
    history = memory_store.get_history(user_id)

    try:
        llm_result = analyze_prompt_with_llm(transcribed_text, history)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {str(e)}")

    filtered_prompt = filter_output(llm_result.improved_prompt)
    memory_store.add_message(user_id, "user", transcribed_text)
    memory_store.add_message(user_id, "assistant", filtered_prompt)

    return VoiceAnalyzeResponse(
        transcribed_text=transcribed_text,
        is_injection=injection_result.is_injection,
        injection_details=injection_result.details,
        issues=llm_result.issues,
        suggestions=llm_result.suggestions,
        improved_prompt=filtered_prompt,
        reasoning_steps=llm_result.reasoning_steps,
        context_used=len(history) > 0
    )


@app.get("/history/{user_id}", response_model=ConversationHistoryResponse)
def get_history(user_id: str):
    """Get conversation history for a user."""
    history = memory_store.get_history(user_id)
    return ConversationHistoryResponse(user_id=user_id, messages=history)


@app.delete("/history/{user_id}")
def clear_history(user_id: str):
    """Clear conversation history for a user."""
    memory_store.clear_history(user_id)
    return {"message": f"History cleared for user {user_id}"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})