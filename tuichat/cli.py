import argparse
import asyncio
from pathlib import Path

from . import __version__
from .app import ChatApp
from .backends import create_backend
from .context import DEFAULT_MAX_CHARS


def build_parser():
    parser = argparse.ArgumentParser(prog="tuichat", description="Terminal chat for local and cloud language models.")
    parser.add_argument("-b", "--backend", choices=["ollama", "openai", "echo"], default="ollama",
                        help="ollama (local, default), openai (or any OpenAI-compatible server), echo (offline demo)")
    parser.add_argument("-m", "--model", help="model name (default depends on the backend)")
    parser.add_argument("--base-url", help="OpenAI-compatible server URL (openai backend only; also OPENAI_BASE_URL)")
    parser.add_argument("--system", help="system prompt: instructions the model gets before the conversation")
    parser.add_argument("--system-file", type=Path, help="read the system prompt from a text file")
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CHARS, metavar="CHARS",
                        help=f"leave the oldest messages out of a request above this many characters (default {DEFAULT_MAX_CHARS}; 0 = never)")
    parser.add_argument("--version", action="version", version=f"tuichat {__version__}")
    return parser


def resolve_system(parser, args):
    """The system prompt from --system or --system-file (not both)."""
    if args.system and args.system_file:
        parser.error("use either --system or --system-file, not both")
    if args.system_file:
        try:
            text = args.system_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            parser.error(f"could not read {args.system_file}")
        if not text.strip():
            parser.error(f"{args.system_file} is empty")
        return text
    return args.system or ""


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    system = resolve_system(parser, args)
    backend = create_backend(args.backend, model=args.model, base_url=args.base_url)
    try:
        asyncio.run(ChatApp(backend, system=system, max_context=max(0, args.max_context)).run())
    except KeyboardInterrupt:
        pass
    print("Goodbye! 👋")
