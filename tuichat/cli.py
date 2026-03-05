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
    parser.add_argument("--timeout", type=float, default=120.0, metavar="SECONDS",
                        help="give up when the model sends nothing for this long (default 120; 0 = wait for ever)")
    parser.add_argument("--temperature", type=float, metavar="T", help="sampling temperature, 0 to 2 (default: the model's own)")
    parser.add_argument("--num-ctx", type=int, metavar="TOKENS", help="context window in tokens (ollama only)")
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


def check_options(parser, args):
    """Reject values that make no sense before anything starts."""
    if args.timeout < 0:
        parser.error("--timeout cannot be negative")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if args.num_ctx is not None:
        if args.backend != "ollama":
            parser.error("--num-ctx only applies to the ollama backend")
        if args.num_ctx < 256:
            parser.error("--num-ctx must be at least 256")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    system = resolve_system(parser, args)
    check_options(parser, args)
    backend = create_backend(args.backend, model=args.model, base_url=args.base_url,
                             timeout=args.timeout or None, temperature=args.temperature, num_ctx=args.num_ctx)
    try:
        asyncio.run(ChatApp(backend, system=system, max_context=max(0, args.max_context)).run())
    except KeyboardInterrupt:
        pass
    print("Goodbye! 👋")
