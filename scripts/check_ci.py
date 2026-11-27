# generated from causaliq@c7917d9e7fa8466b07548514cdf37d494b861321 src/causaliq/devtools/templates.py - do not edit
"""Run the CausalIQ quality gates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))

from causaliq_devtools.gate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
