"""
Minimal Flask server for ReadScore frontend.

Usage:
    python server.py

Then open http://localhost:5000 in your browser.
"""

import os
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for localhost development


@app.route('/')
def index():
    """Serve the frontend HTML file."""
    return send_from_directory('.', 'index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze audio against reference text.

    Expects:
        - text: Reference text (form field)
        - lang: Language code - 'en', 'ru', 'he', or 'auto' (form field, optional)
        - audio: WAV audio file (file upload)

    Returns:
        JSON report with accuracy, fluency, prosody analysis.
    """
    # Validate request
    if 'text' not in request.form:
        return jsonify({'error': 'Missing "text" field'}), 400

    if 'audio' not in request.files:
        return jsonify({'error': 'Missing "audio" file'}), 400

    text = request.form['text'].strip()
    lang = request.form.get('lang', 'auto')  # Default to auto-detect
    audio_file = request.files['audio']

    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400

    if not audio_file.filename:
        return jsonify({'error': 'No audio file provided'}), 400

    # Validate language
    if lang not in ('en', 'ru', 'he', 'auto'):
        return jsonify({'error': f'Invalid language: {lang}. Use en, ru, he, or auto'}), 400

    # Save audio to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    try:
        # Write uploaded audio to temp file
        audio_file.save(temp_path)
        os.close(temp_fd)

        # Import readscore modules
        from readscore.report import evaluate_reading, convert_to_serializable, EvaluationConfig

        # Create config (using defaults)
        config = EvaluationConfig()

        # Run evaluation with language parameter
        report = evaluate_reading(temp_path, text, config, lang=lang)

        # Convert numpy types to JSON-serializable
        report = convert_to_serializable(report)

        return jsonify(report)

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except ImportError as e:
        return jsonify({
            'error': f'Missing dependency: {e}',
            'hint': 'Run: pip install faster-whisper librosa soundfile'
        }), 500
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("=" * 50)
    print("ReadScore Server")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
