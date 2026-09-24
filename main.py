import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from faster_whisper import WhisperModel

# Model size: tiny | base | small | medium | large-v3
# Use "tiny" or "base" on small Koyeb instances.
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "25"))

model: Optional[WhisperModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Load the model once at startup (int8 keeps RAM usage low on CPU)
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    yield


app = FastAPI(title="Whisper Transcription API", lifespan=lifespan)


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_SIZE}


# Plain `def` so FastAPI runs the blocking transcription in a thread pool
@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),  # e.g. "ar", "en"; empty = auto-detect
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    suffix = os.path.splitext(file.filename or "")[1] or ".audio"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        if os.path.getsize(tmp_path) > MAX_FILE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413, detail=f"File too large (max {MAX_FILE_MB} MB)"
            )

        segments, info = model.transcribe(
            tmp_path,
            language=language or None,
            vad_filter=True,
            beam_size=5,
        )

        results = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
        ]

        return {
            "language": info.language,
            "duration": round(info.duration, 2),
            "text": " ".join(r["text"] for r in results),
            "segments": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
