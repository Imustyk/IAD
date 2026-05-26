#!/usr/bin/env python3
"""Run the IAD Streamlit UI locally (port 8501 by default).

Do not run this while Docker Streamlit is mapped to the same host port.
Docker Compose defaults to host port 8502; use that stack OR this script, not both on one port.

Usage:
    python scripts/run_streamlit.py
    python scripts/run_streamlit.py --port 8501
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="IAD Streamlit UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        f"--server.address={args.host}",
        f"--server.port={args.port}",
        "--server.headless=true",
        "--server.runOnSave=false",
        "--server.fileWatcherType=none",
        "--server.enableXsrfProtection=false",
        "--server.enableCORS=false",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
