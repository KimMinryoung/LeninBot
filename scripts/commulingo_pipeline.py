#!/usr/bin/env python3
"""Durable CommuLingo work queue; defaults to draft-only execution."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from commulingo_pipeline.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
