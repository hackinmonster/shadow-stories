"""CLI entrypoint for shadow-narrate.

Usage:
    shadow-narrate --voice "the dragon roars" --shadow "wings spread wide"
"""
from __future__ import annotations

import argparse
import sys

from shadow_stories.gemini.client import GeminiClientError
from shadow_stories.narrate import narrate


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        prog="shadow-narrate",
        description="Live shadow puppet narrator powered by Gemini.",
    )
    parser.add_argument("--voice", required=True, help="Voice input from the child")
    parser.add_argument("--shadow", required=True, help="Shadow puppet motion description")
    args = parser.parse_args()

    try:
        text = narrate(voice_input=args.voice, shadow_input=args.shadow)
        print(text)
    except GeminiClientError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
