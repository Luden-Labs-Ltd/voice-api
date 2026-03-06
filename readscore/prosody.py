"""
Prosody analysis module.

Analyzes pitch (F0), energy, and rhythm characteristics.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Constants for pitch extraction
PITCH_FMIN = 50.0   # Minimum F0 in Hz (covers low male voices)
PITCH_FMAX = 400.0  # Maximum F0 in Hz (covers high female voices)
PITCH_HOP_SEC = 0.010  # 10ms hop for pitch frames


@dataclass
class PitchStats:
    """Pitch (F0) statistics."""
    mean: float
    std: float
    min: float
    max: float
    range: float


@dataclass
class EnergyStats:
    """Energy/RMS statistics."""
    mean: float
    std: float


@dataclass
class ProsodyContours:
    """Time series data for pitch and energy contours."""
    pitch_times_sec: List[float]
    pitch_hz: List[Optional[float]]  # F0 in Hz, None for unvoiced
    energy_times_sec: List[float]
    energy_rms: List[float]
    sample_rate: int

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "pitch_times_sec": self.pitch_times_sec,
            "pitch_hz": self.pitch_hz,
            "energy_times_sec": self.energy_times_sec,
            "energy_rms": self.energy_rms
        }


@dataclass
class ProsodyResult:
    """Result of prosody analysis."""
    score_0_100: float
    f0_hz: PitchStats
    energy: EnergyStats
    flags: List[str]
    notes: List[str]
    contours: Optional[ProsodyContours] = None

    def to_dict(self) -> dict:
        result = {
            "score_0_100": round(self.score_0_100, 1),
            "f0_hz": {
                "mean": round(self.f0_hz.mean, 2),
                "std": round(self.f0_hz.std, 2),
                "min": round(self.f0_hz.min, 2),
                "max": round(self.f0_hz.max, 2)
            },
            "energy": {
                "mean": round(self.energy.mean, 6),
                "std": round(self.energy.std, 6)
            },
            "flags": self.flags,
            "notes": self.notes
        }
        # Include contours if available
        if self.contours is not None:
            result["contours"] = self.contours.to_dict()
        return result


def analyze_prosody(
    audio_path: str,
    sample_rate: int = 16000,
    return_contours: bool = False
) -> ProsodyResult:
    """
    Analyze prosody characteristics of audio.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate for analysis
        return_contours: If True, include pitch/energy time series in result

    Returns:
        ProsodyResult with pitch, energy, and quality assessment
    """
    import numpy as np

    flags = []
    notes = []
    contours = None

    # Load and preprocess audio
    try:
        audio, sr = _load_audio_for_analysis(audio_path, sample_rate)
    except Exception as e:
        return _empty_prosody_result(f"Failed to load audio: {e}")

    if len(audio) == 0:
        return _empty_prosody_result("Empty audio")

    audio_duration = len(audio) / sr

    # Extract pitch (F0) with explicit hop length
    pitch_hop_samples = int(PITCH_HOP_SEC * sr)
    try:
        f0, voiced_flag, pitch_times = _extract_pitch(audio, sr, pitch_hop_samples)
    except Exception as e:
        notes.append(f"Pitch extraction failed: {e}")
        f0 = np.array([])
        voiced_flag = np.array([])
        pitch_times = np.array([])

    # Extract energy
    energy_hop_samples = 512
    try:
        energy, energy_times = _extract_energy(audio, sr, energy_hop_samples)
    except Exception as e:
        notes.append(f"Energy extraction failed: {e}")
        energy = np.array([])
        energy_times = np.array([])

    # Build contours if requested
    if return_contours:
        # Convert pitch values: 0/NaN -> None for JSON clarity
        pitch_hz_list = []
        for val in f0:
            if val > 0 and np.isfinite(val):
                pitch_hz_list.append(float(val))
            else:
                pitch_hz_list.append(None)

        contours = ProsodyContours(
            pitch_times_sec=[float(t) for t in pitch_times] if len(pitch_times) > 0 else [],
            pitch_hz=pitch_hz_list,
            energy_times_sec=[float(t) for t in energy_times] if len(energy_times) > 0 else [],
            energy_rms=[float(e) for e in energy] if len(energy) > 0 else [],
            sample_rate=sr
        )

    # Calculate pitch statistics (voiced frames only)
    if len(f0) > 0:
        voiced_f0 = f0[(f0 > 0) & np.isfinite(f0)]
        if len(voiced_f0) > 0:
            f0_stats = PitchStats(
                mean=float(np.mean(voiced_f0)),
                std=float(np.std(voiced_f0)),
                min=float(np.min(voiced_f0)),
                max=float(np.max(voiced_f0)),
                range=float(np.max(voiced_f0) - np.min(voiced_f0))
            )
        else:
            f0_stats = PitchStats(0, 0, 0, 0, 0)
            flags.append("no_voiced_frames")
    else:
        f0_stats = PitchStats(0, 0, 0, 0, 0)
        flags.append("pitch_extraction_failed")

    # Calculate energy statistics
    if len(energy) > 0:
        energy_stats = EnergyStats(
            mean=float(np.mean(energy)),
            std=float(np.std(energy))
        )
    else:
        energy_stats = EnergyStats(0, 0)

    # Analyze prosody quality
    score, prosody_flags, prosody_notes = _analyze_prosody_quality(
        f0_stats, energy_stats
    )
    flags.extend(prosody_flags)
    notes.extend(prosody_notes)

    return ProsodyResult(
        score_0_100=score,
        f0_hz=f0_stats,
        energy=energy_stats,
        flags=flags,
        notes=notes,
        contours=contours
    )


def _load_audio_for_analysis(audio_path: str, target_sr: int) -> Tuple:
    """Load and resample audio for analysis."""
    import numpy as np

    # Try librosa first
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio.astype(np.float32), sr
    except ImportError:
        pass

    # Fall back to our generic loader
    from .asr import load_audio
    audio, sr = load_audio(audio_path)

    # Resample if needed
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except ImportError:
            # Simple resampling without librosa
            ratio = target_sr / sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)
            sr = target_sr

    return audio, sr


def _extract_pitch(audio, sr: int, hop_length: int) -> Tuple:
    """
    Extract pitch using librosa or parselmouth.

    Returns:
        Tuple of (f0_values, voiced_flags, time_points)
    """
    import numpy as np

    # Calculate time points based on hop length
    n_frames = 1 + (len(audio) - 1) // hop_length
    times = np.arange(n_frames) * hop_length / sr

    # Try librosa pyin first with explicit parameters
    try:
        import librosa
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=PITCH_FMIN,
            fmax=PITCH_FMAX,
            sr=sr,
            hop_length=hop_length,
            fill_na=0.0  # Fill unvoiced with 0
        )
        # Ensure f0 is finite
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        # Ensure times matches f0 length
        times = np.arange(len(f0)) * hop_length / sr
        return f0, voiced_flag, times
    except ImportError:
        pass
    except Exception as e:
        # librosa.pyin can fail on some audio, try fallback
        pass

    # Try parselmouth
    try:
        import parselmouth
        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = snd.to_pitch(
            time_step=hop_length / sr,
            pitch_floor=PITCH_FMIN,
            pitch_ceiling=PITCH_FMAX
        )
        f0 = pitch.selected_array['frequency']
        voiced_flag = f0 > 0
        times = pitch.xs()
        return f0, voiced_flag, times
    except ImportError:
        pass
    except Exception:
        pass

    # Basic autocorrelation method as fallback
    return _basic_pitch_extraction(audio, sr, hop_length)


def _basic_pitch_extraction(audio, sr: int, hop_length: int) -> Tuple:
    """Basic pitch extraction using autocorrelation."""
    import numpy as np

    frame_length = int(0.025 * sr)  # 25ms frames

    f0_values = []
    voiced_flags = []
    times = []

    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        t = i / sr
        times.append(t)

        # Simple autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]

        # Find first peak after initial decay
        min_lag = int(sr / PITCH_FMAX)
        max_lag = int(sr / PITCH_FMIN)

        if max_lag > len(corr):
            max_lag = len(corr) - 1
        if min_lag >= max_lag:
            f0_values.append(0)
            voiced_flags.append(False)
            continue

        peak_idx = min_lag + np.argmax(corr[min_lag:max_lag])

        if corr[peak_idx] > 0.3 * corr[0]:  # Voiced threshold
            f0 = sr / peak_idx
            f0_values.append(f0)
            voiced_flags.append(True)
        else:
            f0_values.append(0)
            voiced_flags.append(False)

    return np.array(f0_values), np.array(voiced_flags), np.array(times)


def _extract_energy(audio, sr: int, hop_length: int) -> Tuple:
    """
    Extract RMS energy.

    Returns:
        Tuple of (energy_values, time_points)
    """
    import numpy as np

    frame_length = 2048

    # Try librosa
    try:
        import librosa
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        times = np.arange(len(rms)) * hop_length / sr
        return rms, times
    except ImportError:
        pass

    # Manual RMS calculation
    rms_values = []
    times = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        rms_values.append(rms)
        times.append(i / sr)

    return np.array(rms_values), np.array(times)


def _analyze_prosody_quality(
    f0_stats: PitchStats,
    energy_stats: EnergyStats
) -> Tuple[float, List[str], List[str]]:
    """Analyze prosody quality and generate score."""
    score = 100.0
    flags = []
    notes = []

    # Check for monotone speech (low F0 variance)
    if f0_stats.mean > 0:
        # Coefficient of variation for pitch
        cv = f0_stats.std / f0_stats.mean if f0_stats.mean > 0 else 0

        if cv < 0.05:
            flags.append("monotone")
            notes.append("Very low pitch variation - speech sounds monotone")
            score -= 30
        elif cv < 0.10:
            flags.append("low_variation")
            notes.append("Below average pitch variation")
            score -= 15
        elif cv > 0.40:
            flags.append("over_exaggerated")
            notes.append("Very high pitch variation - may sound over-exaggerated")
            score -= 20
        elif cv > 0.30:
            notes.append("High pitch variation - expressive speech")
            score = min(100, score + 5)

        # Check pitch range
        if f0_stats.range < 30:
            flags.append("narrow_range")
            notes.append("Narrow pitch range")
            score -= 10
        elif f0_stats.range > 200:
            notes.append("Wide pitch range - good expressiveness")

    # Check energy variation
    if energy_stats.mean > 0:
        energy_cv = energy_stats.std / energy_stats.mean if energy_stats.mean > 0 else 0

        if energy_cv < 0.1:
            flags.append("flat_energy")
            notes.append("Very consistent energy - may lack emphasis variation")
            score -= 10
        elif energy_cv > 0.8:
            flags.append("inconsistent_energy")
            notes.append("Highly variable energy - may sound uneven")
            score -= 15

    # Ensure score is in valid range
    score = max(0, min(100, score))

    if not flags and not notes:
        notes.append("Prosody characteristics within normal range")

    return score, flags, notes


def _empty_prosody_result(reason: str) -> ProsodyResult:
    """Return empty prosody result for error cases."""
    return ProsodyResult(
        score_0_100=0,
        f0_hz=PitchStats(0, 0, 0, 0, 0),
        energy=EnergyStats(0, 0),
        flags=["error"],
        notes=[reason],
        contours=None
    )
