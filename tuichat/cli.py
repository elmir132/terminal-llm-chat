import argparse
import asyncio

from . import __version__
from .app import ChatApp
from .backends import create_backend


def build_parser():
    parser = argparse.ArgumentParser(prog="tuichat", description="Terminal chat for local and cloud language models.")
    parser.add_argument("-b", "--backend", choices=["ollama", "openai", "echo"], default="ollama",
                        help="ollama (local, default), openai (or any OpenAI-compatible server), echo (offline demo)")
    parser.add_argument("-m", "--model", help="model name (default depends on the backend)")
    parser.add_argument("--base-url", help="OpenAI-compatible server URL (openai backend only; also OPENAI_BASE_URL)")
    parser.add_argument("--version", action="version", version=f"tuichat {__version__}")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    backend = create_backend(args.backend, model=args.model, base_url=args.base_url)
    try:
        asyncio.run(ChatApp(backend).run())
    except KeyboardInterrupt:
        pass
    print("Goodbye! 👋")
