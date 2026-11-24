# generated from causaliq@49a27b66b20d src/causaliq/devtools/templates.py - do not edit
"""Run the CausalIQ quality gates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))

from causaliq_devtools.gate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
