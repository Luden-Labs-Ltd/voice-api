"""
ReadScore - Evaluate spoken audio reading against reference text.

Provides 4 independent metrics:
1. Accuracy (WER, alignment)
2. Fluency/Speed (WPM, pauses)
3. Prosody (pitch, energy analysis)
4. Pronunciation Quality
"""

__version__ = "0.1.0"

from .report import evaluate_reading, convert_to_serializable

__all__ = ["evaluate_reading", "convert_to_serializable", "__version__"]
