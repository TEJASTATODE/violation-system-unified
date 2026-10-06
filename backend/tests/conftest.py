"""Make backend modules importable when pytest runs from the repository root."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
