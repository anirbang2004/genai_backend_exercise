#!/usr/bin/env python3
"""
GenAI Backend Engineering Exercise - CLI entry point.

Usage:
    python app.py ingest <docs_directory>
    python app.py ask "<question>"

Run `python app.py --help` or `python app.py <command> --help` for details.
"""

import argparse
import sys

from src.loader import DocumentDirectoryNotFoundError, NoSupportedDocumentsError
from src.pipeline import run_ask, run_ingest
from src.vector_store import IndexNotFoundError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Ask questions over a local collection of .txt/.md documents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Read documents, build, and persist the local index."
    )
    ingest_parser.add_argument(
        "directory", help="Path to a directory containing .txt/.md files."
    )

    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question against the previously built index."
    )
    ask_parser.add_argument("question", help="The question to ask, in quotes.")

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    try:
        result = run_ingest(args.directory)
    except DocumentDirectoryNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except NoSupportedDocumentsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Ingested {result.documents_loaded} document(s) from "
        f"'{result.source_directory}' into {result.chunks_indexed} chunk(s)."
    )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    try:
        from src.llm import get_default_llm

        answer = run_ask(args.question, llm=get_default_llm())
    except IndexNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    sources = ", ".join(answer.sources) if answer.sources else "None"
    print(f"Answer: {answer.text}")
    print(f"Sources: {sources}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # argparse handles invalid usage (exit code 2)

    if args.command == "ingest":
        return cmd_ingest(args)
    if args.command == "ask":
        return cmd_ask(args)

    # Unreachable because `required=True` above, but kept as a safety net.
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
