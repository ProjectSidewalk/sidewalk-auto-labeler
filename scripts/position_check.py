#!/usr/bin/env python
"""Shim: the position check lives at the repo root (position_check.py) so main.py and
send_to_ps.py can import it. This keeps `python scripts/position_check.py ...` working.
Runs the root module by path — never `import position_check` — so this file, which shares
its name, can sit on sys.path (the tests put scripts/ there) without shadowing it."""
import runpy
import sys
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'position_check.py'), run_name='__main__')
    sys.exit(0)
