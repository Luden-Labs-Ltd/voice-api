"""
CLI entrypoint for ReadScore.
"""

import argparse
import json
import sys
import os


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="readscore",
        description="Evaluate spoken audio reading against reference text"
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Reference text: either a string or path to .txt file"
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file (wav, mp3, m4a)"
    )

    parser.add_argument(
        "--out",
        help="Output path for JSON report (default: stdout)"
    )

    parser.add_argument(
        "--config",
        help="Path to configuration JSON file"
    )

    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference (default: cpu)"
    )

    parser.add_argument(
        "--lang",
        choices=["en", "ru", "he", "auto"],
        default="auto",
        help="Language code: en (English), ru (Russian), he (Hebrew), auto (detect). Default: auto"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )

    args = parser.parse_args()

    # Load reference text
    reference_text = _load_reference_text(args.text)

    # Validate audio path
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Load config
    from .report import EvaluationConfig

    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        config = EvaluationConfig.from_file(args.config)
    else:
        config = EvaluationConfig()

    # Override config with CLI args
    config.whisper_model = args.model
    config.whisper_device = args.device

    # Run evaluation
    try:
        from .report import generate_report_json

        print(f"Evaluating: {args.audio}", file=sys.stderr)
        print(f"Reference text: {len(reference_text.split())} words", file=sys.stderr)
        print(f"Language: {args.lang}", file=sys.stderr)
        print(f"Using model: {config.whisper_model} on {config.whisper_device}", file=sys.stderr)

        json_report = generate_report_json(
            args.audio,
            reference_text,
            args.out,
            config,
            lang=args.lang
        )

        if args.out:
            print(f"Report saved to: {args.out}", file=sys.stderr)
        else:
            print(json_report)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("\nInstall required dependencies:", file=sys.stderr)
        print("  pip install faster-whisper librosa soundfile", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)


def _load_reference_text(text_arg: str) -> str:
    """Load reference text from string or file."""
    # Check if it's a file path
    if os.path.exists(text_arg):
        with open(text_arg, 'r', encoding='utf-8') as f:
            return f.read().strip()

    # Check if it looks like a file path but doesn't exist
    if text_arg.endswith('.txt'):
        print(f"Warning: {text_arg} looks like a file path but doesn't exist. "
              f"Treating as literal text.", file=sys.stderr)

    return text_arg


if __name__ == "__main__":
    main()
