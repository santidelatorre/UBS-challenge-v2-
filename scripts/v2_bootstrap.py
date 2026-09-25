"""Pin research entry points to this checkout, regardless of editable installs."""
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from ubs_recurrence.data import ROOT

if ROOT.resolve() != WORKSPACE:
    raise RuntimeError(f"A different ubs_recurrence checkout was already imported: {ROOT}")
