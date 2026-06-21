#!/usr/bin/env python3
"""Scrape fresh stats from vlr.gg and rebuild the Chroma index."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve_python() -> str:
    """Prefer the project .venv over whatever shell python is active."""
    for candidate in (
        ROOT / ".venv" / "bin" / "python",
        ROOT / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def main():
    scrape = ROOT / "scrapers" / "scrape_stats.py"
    chemistry = ROOT / "scrapers" / "scrape_chemistry.py"
    python = resolve_python()
    venv_py = (
        ROOT / ".venv" / "bin" / "python",
        ROOT / ".venv" / "Scripts" / "python.exe",
    )
    if any(python == str(p) for p in venv_py if p.exists()):
        print(f"Using project venv: {python}")
    elif any(p.exists() for p in venv_py):
        print(f"Warning: your shell python may have broken packages ({sys.executable})")
        venv_hint = next(str(p) for p in venv_py if p.exists())
        print(f"Run instead: {venv_hint} scripts/refresh_data.py")
        print(f"Or: source .venv/bin/activate && pip install -r requirements.txt\n")
    else:
        print(f"Using: {python}")
        print("Tip: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt\n")

    print("\n=== Step 1: Scraping vlr.gg stats ===")
    subprocess.run([python, str(scrape)], check=True, cwd=ROOT)

    print("\n=== Step 2: Scraping co-play chemistry (VCT match lineups) ===")
    subprocess.run([python, str(chemistry), "6"], check=True, cwd=ROOT)

    print("\n=== Step 3: Building vector index ===")
    subprocess.run([python, "-m", "artemis.build_index"], check=True, cwd=ROOT)

    print("\nDone. Restart the server to use fresh data.")


if __name__ == "__main__":
    main()
