"""
FastAPI service wrapping the readscore analysis pipeline.

Internal service — not exposed publicly.
POST /analyze  accepts multipart form-data: text, lang (optional), audio (file)
GET  /health   liveness check

Port and model are configured via environment variables:
  PORT          — set by Railway automatically (takes priority)
  PYTHON_PORT   — local/compose fallback (default 8000)
  WHISPER_MODEL — Whisper model size (default "base")
"""

import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

app = FastAPI(title="readscore-python-service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    text: str = Form(...),
    lang: Optional[str] = Form("auto"),
    audio: UploadFile = File(...),
):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if lang not in ("en", "ru", "he", "auto"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language '{lang}'. Use en, ru, he, or auto",
        )

    # Write uploaded audio to a temp file so readscore can open it by path
    suffix = os.path.splitext(audio.filename or "recording.wav")[1] or ".wav"
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        content = await audio.read()
        os.write(temp_fd, content)
        os.close(temp_fd)

        from readscore.report import (
            EvaluationConfig,
            convert_to_serializable,
            evaluate_reading,
        )

        config = EvaluationConfig(whisper_model=WHISPER_MODEL)
        report = evaluate_reading(temp_path, text, config, lang=lang)
        report = convert_to_serializable(report)
        return JSONResponse(content=report)

    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Missing dependency: {exc}. "
            "Ensure faster-whisper, librosa, and soundfile are installed.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    # Railway injects PORT; PYTHON_PORT is the local/compose fallback from .env.
    port = int(os.getenv("PORT") or os.getenv("PYTHON_PORT") or 8000)
    uvicorn.run(app, host="0.0.0.0", port=port)
