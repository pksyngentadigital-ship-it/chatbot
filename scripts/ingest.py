#!/usr/bin/env python3
"""Load a grower-feedback workbook into Pinecone from the command line.

This exists so the only path that can wipe the index does not have to be a
password-protected button on a public URL. It runs the same
vog.ingest.run_ingestion the app used, so the 20 ingestion-correctness
tests cover this exactly as they covered the panel.

    # See what a workbook would produce, without a key and without writing:
    python scripts/ingest.py feedback.xlsx --dry-run

    # Add to whatever is already in the index:
    python scripts/ingest.py feedback.xlsx

    # Replace everything (destructive; asks first):
    python scripts/ingest.py feedback.xlsx --purge

Reads PINECONE_API_KEY from the environment or a .env file next to the
repo root.

Exit codes:
    0  ingested (or dry run completed)
    1  nothing ingested, or the workbook could not be read
    2  cancelled at the confirmation prompt
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vog.ingest import run_ingestion  # noqa: E402


def _load_env() -> None:
    """python-dotenv if it is installed, else parse .env directly — this
    script should not need a dependency to read a two-line file."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return
    except ImportError:
        pass
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _confirm(prompt: str) -> bool:
    """Anything other than a typed "yes" is a no.

    Catching EOFError rather than trusting isatty(): under Git Bash on
    Windows isatty() reported a terminal even with stdin closed, so the
    guard fell through to input() and died with a traceback instead of
    refusing — the one path where failing loudly and safely matters most.
    """
    try:
        return input(f"{prompt} [type 'yes' to continue] ").strip().lower() == "yes"
    except (EOFError, KeyboardInterrupt):
        print('\n' + "No confirmation received. Re-run with --yes "
              "if you are certain.", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workbook", help="Path to the .xlsx file")
    ap.add_argument("--purge", action="store_true",
                    help="Delete everything in the index before writing. Required "
                         "after any change to how products, crops or categories are "
                         "detected: records already stored keep their old tags, and "
                         "re-ingesting alone does not repair them.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and report without writing. Needs no API key.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the confirmation prompt for --purge.")
    args = ap.parse_args()

    path = Path(args.workbook).expanduser()
    if not path.is_file():
        print(f"No such file: {path}", file=sys.stderr)
        return 1

    _load_env()
    api_key = os.getenv("PINECONE_API_KEY")
    if not args.dry_run and not api_key:
        print("PINECONE_API_KEY is not set. Put it in a .env file at the repo "
              "root, or export it, or use --dry-run to parse without writing.",
              file=sys.stderr)
        return 1

    if args.purge and not args.yes:
        print(f"\n--purge will DELETE every record currently in the index "
              f"before writing {path.name}. This cannot be undone.")
        if not _confirm("Continue?"):
            print("Cancelled. Nothing was changed.")
            return 2

    print(f"\nReading {path.name} ({path.stat().st_size / 1024:.0f} KB)...")
    try:
        result = run_ingestion(path.read_bytes(), api_key,
                               purge_first=args.purge, dry_run=args.dry_run)
    except ValueError as e:
        # Nothing usable in the workbook — the message names what was skipped.
        print(f"\n{e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nIngestion failed: {e}", file=sys.stderr)
        return 1

    verb = "would ingest" if args.dry_run else "Ingested"
    print(f"\n{verb} {result['total_records']} records "
          f"(run {result['ingest_run']}).")

    if result["summary"]:
        print("\n  Records by month:")
        for period, count in sorted(result["summary"].items()):
            print(f"    {period:<20} {count:>6}")

    # Skips used to be silent, so a workbook could half-load while still
    # reporting success. They are the most useful thing on screen.
    if result["skipped"]:
        print(f"\n  {len(result['skipped'])} thing(s) skipped:")
        for item in result["skipped"]:
            print(f"    [{item['reason']}] {item['sheet']}")
            print(f"      {item['detail']}")
    else:
        print("\n  Nothing skipped.")

    if args.dry_run:
        print("\nDry run — nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
