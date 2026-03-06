# ReadScore

Evaluate spoken audio reading against reference text with 4 independent metrics.

## Features

- **Accuracy**: Word-level alignment with WER, insertions, deletions, substitutions
- **Fluency/Speed**: WPM, pause distribution, configurable reference ranges
- **Prosody**: Pitch (F0) and energy analysis, monotone/over-exaggeration detection
- **Pronunciation Quality**: ASR confidence-based quality estimation

## Installation

```bash
# Basic install
pip install -e .

# With all dependencies (recommended)
pip install -e ".[full]"

# Minimal install with faster-whisper only
pip install -e ".[whisper,audio]"

# For web frontend (includes Flask server)
pip install -e ".[full]"
# or just the server dependencies:
pip install flask flask-cors
```

### Dependencies

- **ASR**: `faster-whisper` (recommended) or `openai-whisper`
- **Audio**: `librosa`, `soundfile`, `pydub`
- **System**: FFmpeg (for mp3/m4a support via pydub)

Install FFmpeg:
- Windows: `choco install ffmpeg` or download from https://ffmpeg.org
- macOS: `brew install ffmpeg`
- Linux: `apt install ffmpeg`

## Usage

### Web Frontend

A simple browser-based interface for recording and analyzing your reading.

```bash
# 1. Install server dependencies
pip install flask flask-cors

# 2. Start the server
python server.py

# 3. Open in browser
# Navigate to http://localhost:5000
```

Then:
1. Enter reference text in the text area
2. Click "Start Recording" and read the text aloud
3. Click "Stop Recording" when done
4. Click "Analyze" to get your results

### CLI

```bash
# Basic usage
readscore --text "The quick brown fox jumps over the lazy dog." --audio recording.wav

# With text file
readscore --text reference.txt --audio recording.wav --out report.json

# With configuration
readscore --text reference.txt --audio recording.wav --config config.json

# Use larger model for better accuracy
readscore --text reference.txt --audio recording.mp3 --model small --device cuda
```

### Python API

```python
from readscore import evaluate_reading

report = evaluate_reading(
    audio_path="recording.wav",
    reference_text="The quick brown fox jumps over the lazy dog."
)

print(f"WER: {report['accuracy']['wer']}")
print(f"WPM: {report['fluency_speed']['wpm']}")
print(f"Prosody Score: {report['prosody']['score_0_100']}")
print(f"Pronunciation Score: {report['pronunciation_quality']['score_0_100']}")
```

### Configuration

Create a `config.json` to customize reference ranges:

```json
{
  "whisper_model": "base",
  "whisper_device": "cpu",
  "language": "en",
  "fluency": {
    "wpm_min": 110,
    "wpm_max": 170,
    "pause_thresholds_ms": [250, 500, 1000, 2000]
  }
}
```

## Output Schema

```json
{
  "input": {
    "audio": "recording.wav",
    "text_len_words": 9,
    "duration_sec": 3.5
  },
  "accuracy": {
    "wer": 0.0,
    "counts": {"ins": 0, "del": 0, "sub": 0, "correct": 9},
    "alignment": [
      {"ref": "the", "hyp": "the", "tag": "ok", "t0": 0.0, "t1": 0.3, "conf": 0.95}
    ]
  },
  "fluency_speed": {
    "wpm": 154.3,
    "avg_word_dur_sec": 0.28,
    "total_dur_sec": 3.5,
    "pauses": {
      "count": 2,
      "total_duration_sec": 0.4,
      "mean": 0.2,
      "p50": 0.18,
      "p90": 0.25,
      "buckets_ms": {"250": 0, "500": 0, "1000": 0, "2000": 0}
    },
    "range": {"wpm_min": 110, "wpm_max": 170},
    "score_0_100": 92.5,
    "within_range": true,
    "notes": []
  },
  "prosody": {
    "score_0_100": 85.0,
    "f0_hz": {"mean": 180.5, "std": 35.2, "min": 120.0, "max": 280.0},
    "energy": {"mean": 0.045, "std": 0.012},
    "flags": [],
    "notes": ["Prosody characteristics within normal range"]
  },
  "pronunciation_quality": {
    "score_0_100": 95.0,
    "signals": {
      "asr_avg_conf": 0.92,
      "substitution_severity": 0.0,
      "low_confidence_word_ratio": 0.0
    },
    "notes": ["Pronunciation appears clear and accurate"]
  }
}
```

## Metrics Details

### Accuracy (WER)

- Uses Levenshtein alignment between reference and ASR transcription
- Tags each word as: `ok`, `sub` (substitution), `ins` (insertion), `del` (deletion)
- WER = (insertions + deletions + substitutions) / reference_word_count

### Fluency/Speed

- **WPM**: Words per minute based on speaking duration
- **Pause detection**: Gaps between words > 50ms
- **Score**: Based on WPM within range + pause characteristics

### Prosody

- **Pitch (F0)**: Extracted using librosa.pyin or parselmouth
- **Energy**: RMS energy analysis
- **Flags**: `monotone`, `low_variation`, `over_exaggerated`, `flat_energy`

### Pronunciation Quality

Pragmatic baseline using:
- ASR word confidence scores
- Substitution severity (edit distance between ref/hyp words)
- Deletion ratio as proxy for unclear speech

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=readscore --cov-report=term-missing
```

## Environment Variables

Configuration is centralised in a `.env` file for local development and set
directly in each service's environment for Railway.

### Quick start (local)

```bash
cp .env.example .env
# Edit .env if you want non-default values, then:
docker compose up --build
```

### Variable reference

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `NODE_PORT` | `3000` | node-api | Port the Node API listens on locally |
| `PORT` | — | node-api, python-service | Injected automatically by Railway; takes priority over `NODE_PORT` / `PYTHON_PORT` |
| `PYTHON_SERVICE_URL` | `http://python-service:8000` | node-api | URL node-api uses to reach the Python service |
| `PYTHON_PORT` | `8000` | python-service | Port the Python service listens on locally |
| `WHISPER_MODEL` | `base` | python-service | Whisper model size (`tiny` → `large-v3`) |
| `LOG_LEVEL` | `info` | node-api | Fastify log level (`trace` `debug` `info` `warn` `error`) |

### Configuring for Railway

Railway injects `PORT` into every service automatically — you do **not** need to
set `NODE_PORT` or `PYTHON_PORT` there.

Variables to set in each Railway service:

**node-api**
```
PYTHON_SERVICE_URL=http://<python-service-private-domain>:<port>
LOG_LEVEL=info
```

**python-service**
```
WHISPER_MODEL=base
```

`PYTHON_SERVICE_URL` must point to the private network address Railway assigns to
the python-service (shown in the service's "Networking → Private" tab).

## License

MIT
