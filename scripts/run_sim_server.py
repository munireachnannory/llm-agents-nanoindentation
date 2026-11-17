#!/usr/bin/env python3
"""Launch the nanoindent instrument simulator (Flask server)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from instrument_control.sim_server import main

if __name__ == "__main__":
    main()
