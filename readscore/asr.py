"""
ASR (Automatic Speech Recognition) module using Whisper.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings

# Suppress warnings during import
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class TranscriptionWord:
    """A transcribed word with timing and confidence."""
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""
    text: str
    words: List[TranscriptionWord]
    duration: float
    language: str


def load_audio(audio_path: str) -> Tuple:
    """
    Load audio file and return as numpy array.

    Supports wav, mp3, m4a via pydub/ffmpeg or soundfile.
    """
    import numpy as np

    audio_path = os.path.abspath(audio_path)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = os.path.splitext(audio_path)[1].lower()

    # Try soundfile first for wav
    if ext == ".wav":
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert to mono
            return audio.astype(np.float32), sr
        except Exception:
            pass

    # Fall back to pydub for other formats
    try:
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_file(audio_path)
        # Convert to mono and get samples
        audio_segment = audio_segment.set_channels(1)
        samples = np.array(audio_segment.get_array_of_samples())
        sr = audio_segment.frame_rate
        # Normalize to float32 [-1, 1]
        if audio_segment.sample_width == 2:
            audio = samples.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:
            audio = samples.astype(np.float32) / 2147483648.0
        else:
            audio = samples.astype(np.float32) / 256.0
        return audio, sr
    except ImportError:
        raise ImportError(
            "pydub is required for non-wav formats. "
            "Install with: pip install pydub"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    language: Optional[str] = None
) -> TranscriptionResult:
    """
    Transcribe audio using Whisper with word-level timestamps.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        device: Device to run on (cpu, cuda)
        language: Language code (e.g., 'en') or None for auto-detect

    Returns:
        TranscriptionResult with text, words, and timing info
    """
    # Try faster-whisper first
    try:
        return _transcribe_faster_whisper(audio_path, model_size, device, language)
    except ImportError:
        pass

    # Fall back to openai-whisper
    try:
        return _transcribe_openai_whisper(audio_path, model_size, device, language)
    except ImportError:
        raise ImportError(
            "No Whisper implementation found. Install one of:\n"
            "  pip install faster-whisper  (recommended)\n"
            "  pip install openai-whisper"
        )


def _transcribe_faster_whisper(
    audio_path: str,
    model_size: str,
    device: str,
    language: Optional[str]
) -> TranscriptionResult:
    """Transcribe using faster-whisper."""
    from faster_whisper import WhisperModel

    # Load model
    compute_type = "int8" if device == "cpu" else "float16"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Transcribe with word timestamps
    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=True
    )

    # Collect words
    words = []
    full_text_parts = []

    for segment in segments:
        full_text_parts.append(segment.text)
        if segment.words:
            for word in segment.words:
                words.append(TranscriptionWord(
                    word=word.word.strip(),
                    start=word.start,
                    end=word.end,
                    confidence=word.probability if hasattr(word, 'probability') else 0.9
                ))

    return TranscriptionResult(
        text=" ".join(full_text_parts).strip(),
        words=words,
        duration=info.duration,
        language=info.language
    )


def _transcribe_openai_whisper(
    audio_path: str,
    model_size: str,
    device: str,
    language: Optional[str]
) -> TranscriptionResult:
    """Transcribe using openai-whisper."""
    import whisper
    import numpy as np

    # Load model
    model = whisper.load_model(model_size, device=device)

    # Load audio
    audio = whisper.load_audio(audio_path)
    duration = len(audio) / whisper.audio.SAMPLE_RATE

    # Transcribe
    options = {"language": language} if language else {}
    result = model.transcribe(
        audio,
        word_timestamps=True,
        **options
    )

    # Collect words
    words = []
    for segment in result["segments"]:
        if "words" in segment:
            for word in segment["words"]:
                words.append(TranscriptionWord(
                    word=word["word"].strip(),
                    start=word["start"],
                    end=word["end"],
                    confidence=word.get("probability", 0.9)
                ))

    return TranscriptionResult(
        text=result["text"].strip(),
        words=words,
        duration=duration,
        language=result.get("language", "en")
    )


def get_word_data(result: TranscriptionResult) -> Tuple[List[str], List[Tuple[float, float]], List[float]]:
    """
    Extract word data from transcription result.

    Returns:
        Tuple of (words, timestamps, confidences)
    """
    words = [w.word for w in result.words]
    timestamps = [(w.start, w.end) for w in result.words]
    confidences = [w.confidence for w in result.words]
    return words, timestamps, confidences
