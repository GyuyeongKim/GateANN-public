#!/usr/bin/env python3
"""Generate all figures for the GateANN paper.

Usage:
    python3 scripts/generate_all_figures.py          # generate all
    python3 scripts/generate_all_figures.py pareto    # generate only matching scripts
"""

import subprocess
import sys
import os
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
SCRIPTS = sorted(SCRIPTS_DIR.glob("plot_*.py"))

def main():
    filter_keyword = sys.argv[1] if len(sys.argv) > 1 else None

    if filter_keyword:
        scripts = [s for s in SCRIPTS if filter_keyword in s.name]
        if not scripts:
            print(f"No scripts matching '{filter_keyword}'")
            sys.exit(1)
    else:
        scripts = SCRIPTS

    print(f"=== Generating {len(scripts)} figure scripts ===\n")

    failed = []
    for script in scripts:
        print(f"[RUN] {script.name} ... ", end="", flush=True)
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True,
            cwd=str(SCRIPTS_DIR.parent),
        )
        if result.returncode == 0:
            print("OK")
            if result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    print(f"      {line}")
        else:
            print("FAILED")
            failed.append(script.name)
            if result.stderr.strip():
                for line in result.stderr.strip().split("\n"):
                    print(f"      {line}")

    print(f"\n=== Done: {len(scripts) - len(failed)}/{len(scripts)} succeeded ===")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
