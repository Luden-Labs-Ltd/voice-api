"""
Report generation module - main entry point for evaluation.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Union


def convert_to_serializable(obj: Any) -> Any:
    """
    Recursively convert non-JSON-serializable objects to standard Python types.

    Handles:
    - numpy.bool_ -> bool
    - numpy.integer -> int
    - numpy.floating -> float
    - numpy.ndarray -> list
    - pathlib.Path -> str
    - unknown objects -> str(object)
    """
    # Import numpy only if available
    try:
        import numpy as np
        HAS_NUMPY = True
    except ImportError:
        HAS_NUMPY = False

    # Handle None
    if obj is None:
        return None

    # Handle numpy types if numpy is available
    if HAS_NUMPY:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return [convert_to_serializable(item) for item in obj.tolist()]

    # Handle standard Python types
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]

    # Fallback: convert unknown objects to string
    return str(obj)

from .asr import transcribe_audio, get_word_data
from .normalize import tokenize, normalize_word, detect_language
from .align import levenshtein_align
from .fluency import analyze_fluency, FluencyConfig
from .prosody import analyze_prosody
from .pronunciation import analyze_pronunciation
from .prosody_punct import analyze_punctuation_prosody, PunctuationConfig
from .punctuation_pauses import analyze_punctuation_pauses, PauseConfig, get_pause_for_prosody_event


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    language: Optional[str] = None
    fluency: FluencyConfig = None
    punctuation: PunctuationConfig = None
    pauses: PauseConfig = None

    def __post_init__(self):
        if self.fluency is None:
            self.fluency = FluencyConfig()
        if self.punctuation is None:
            self.punctuation = PunctuationConfig()
        if self.pauses is None:
            self.pauses = PauseConfig()

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluationConfig":
        fluency_config = None
        if "fluency" in d:
            fluency_config = FluencyConfig.from_dict(d["fluency"])

        punctuation_config = None
        if "punctuation" in d:
            punctuation_config = PunctuationConfig.from_dict(d["punctuation"])

        pauses_config = None
        if "pauses" in d or "punctuation_pause_ranges" in d:
            pauses_config = PauseConfig.from_dict(d)

        return cls(
            whisper_model=d.get("whisper_model", "base"),
            whisper_device=d.get("whisper_device", "cpu"),
            language=d.get("language"),
            fluency=fluency_config,
            punctuation=punctuation_config,
            pauses=pauses_config
        )

    @classmethod
    def from_file(cls, path: str) -> "EvaluationConfig":
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


def _get_audio_duration(audio_path: str) -> float:
    """Return audio duration (seconds) without running ASR transcription."""
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return float(info.duration)
    except Exception:
        pass
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(audio_path)
        return len(seg) / 1000.0
    except Exception:
        return 0.0


def evaluate_reading(
    audio_path: str,
    reference_text: str,
    config: Optional[EvaluationConfig] = None,
    lang: Optional[str] = None,
    transcript: Optional[str] = None,
    transcript_words: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a spoken audio reading against reference text.

    Args:
        audio_path: Path to audio file (wav, mp3, m4a)
        reference_text: Reference text that was supposed to be read
        config: Optional configuration
        lang: Language code ('en', 'ru', 'he', 'auto', or None for config default)

    Returns:
        Dictionary with evaluation results matching the output schema
    """
    if config is None:
        config = EvaluationConfig()

    # Determine language
    lang_requested = lang or config.language or "auto"

    # Auto-detect language from text if needed
    if lang_requested == "auto" or lang_requested is None:
        lang_used = detect_language(reference_text)
    else:
        lang_used = lang_requested

    # Validate inputs
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not reference_text or not reference_text.strip():
        raise ValueError("Reference text cannot be empty")

    # Tokenize reference text with language-aware normalization
    ref_words = tokenize(reference_text, lang=lang_used)

    if not ref_words:
        raise ValueError("Reference text contains no words after normalization")

    # Step 1: Obtain transcription
    #
    # Two paths:
    #   a) External transcript provided (e.g. OpenAI gpt-4o-transcribe via Node API)
    #      → skip Whisper, use provided words + timestamps, default confidence = 0.9
    #   b) No external transcript → run local Whisper ASR (original behaviour)
    #
    if transcript is not None:
        # Path a: use pre-transcribed text from OpenAI
        if transcript_words:
            # Word-level timestamps available (OpenAI verbose_json)
            hyp_words = [w.get("word", "") for w in transcript_words]
            timestamps = [
                (float(w.get("start", 0.0)), float(w.get("end", 0.0)))
                for w in transcript_words
            ]
            duration = float(transcript_words[-1].get("end", 0.0)) if transcript_words else 0.0
        else:
            # No word timestamps — split text, generate placeholder timing
            hyp_words = transcript.split()
            timestamps = [(0.0, 0.0)] * len(hyp_words)
            duration = _get_audio_duration(audio_path)

        # OpenAI does not provide per-word confidence; use a high default so
        # that the Hebrew leniency system does not penalise OpenAI mismatches.
        confidences = [0.9] * len(hyp_words)
        raw_transcript = transcript
        asr_source = "openai"
    else:
        # Path b: local Whisper ASR
        asr_language = lang_used if lang_used != "auto" else None
        transcription = transcribe_audio(
            audio_path,
            model_size=config.whisper_model,
            device=config.whisper_device,
            language=asr_language
        )
        hyp_words, timestamps, confidences = get_word_data(transcription)
        duration = transcription.duration
        raw_transcript = " ".join(hyp_words)
        asr_source = "whisper"

    # Normalize hypothesis words for alignment
    word_data = []
    for i, word in enumerate(hyp_words):
        normalized = normalize_word(word, lang=lang_used)
        if normalized:
            ts = timestamps[i] if i < len(timestamps) else (0, 0)
            conf = confidences[i] if i < len(confidences) else 0.9
            word_data.append((normalized, ts, conf))

    if word_data:
        hyp_words_final = [w[0] for w in word_data]
        timestamps_final = [w[1] for w in word_data]
        confidences_final = [w[2] for w in word_data]
    else:
        hyp_words_final = []
        timestamps_final = []
        confidences_final = []

    # Step 2: Align reference and hypothesis
    alignment = levenshtein_align(
        ref_words,
        hyp_words_final,
        timestamps_final,
        confidences_final,
        lang=lang_used,
    )

    # Step 3: Analyze fluency
    fluency = analyze_fluency(
        timestamps_final,
        duration,
        config.fluency
    )

    # Step 4: Analyze prosody (with contours for punctuation analysis)
    prosody = analyze_prosody(audio_path, return_contours=True)

    # Step 5: Analyze pronunciation
    pronunciation = analyze_pronunciation(alignment, lang=lang_used)

    # Step 6: Analyze punctuation pauses (timing at punctuation boundaries)
    alignment_list = alignment.to_dict()["alignment"]
    pause_analysis = analyze_punctuation_pauses(
        reference_text=reference_text,
        alignment=alignment_list,
        config=config.pauses
    )

    # Step 7: Analyze punctuation-specific prosody (pitch/energy for ? and !)
    prosody_punct = None
    if prosody.contours is not None:
        prosody_punct = analyze_punctuation_prosody(
            reference_text=reference_text,
            alignment=alignment_list,
            pitch_times=prosody.contours.pitch_times_sec,
            pitch_values=prosody.contours.pitch_hz,
            energy_times=prosody.contours.energy_times_sec,
            energy_values=prosody.contours.energy_rms,
            config=config.punctuation
        )

        # Enrich prosody_punctuation events with pause data
        for event in prosody_punct.events:
            pause_data = get_pause_for_prosody_event(
                pause_analysis, event.punct, event.ref_word
            )
            if pause_data:
                event.features.update(pause_data)

    # Build report
    report = {
        "input": {
            "audio": os.path.basename(audio_path),
            "text_len_words": len(ref_words),
            "duration_sec": round(duration, 2),
            "reference_text": reference_text,  # Include for UI rendering
            "lang_requested": lang_requested,
            "lang_used": lang_used,
            "asr_source": asr_source,
        },
        "accuracy": alignment.to_dict(),
        "fluency_speed": fluency.to_dict(),
        "prosody": prosody.to_dict(),
        "pronunciation_quality": pronunciation.to_dict(),
        "punctuation_pauses": pause_analysis.to_dict()
    }

    # Add punctuation prosody if available
    if prosody_punct is not None:
        report["prosody_punctuation"] = prosody_punct.to_dict()

    # Build ASR diagnostics (raw transcript for debugging Hebrew)
    report["asr_diagnostics"] = {
        "raw_transcript": raw_transcript,
        "asr_source": asr_source,
        "word_count_ref": len(ref_words),
        "word_count_asr": len(hyp_words_final),
        "avg_confidence": round(sum(confidences_final) / len(confidences_final), 4) if confidences_final else None,
    }

    return report


def generate_report_json(
    audio_path: str,
    reference_text: str,
    output_path: Optional[str] = None,
    config: Optional[EvaluationConfig] = None,
    lang: Optional[str] = None
) -> str:
    """
    Generate JSON report and optionally save to file.

    Args:
        audio_path: Path to audio file
        reference_text: Reference text
        output_path: Optional path to save JSON report
        config: Optional configuration
        lang: Language code ('en', 'ru', 'he', 'auto')

    Returns:
        JSON string of the report
    """
    report = evaluate_reading(audio_path, reference_text, config, lang=lang)

    # Convert all values to JSON-serializable types (handles numpy types)
    report = convert_to_serializable(report)

    json_str = json.dumps(report, indent=2)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

    return json_str
