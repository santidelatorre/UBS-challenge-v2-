"""Subprocess entry points must not depend on a global editable installation."""
import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize("script",["v2_pipeline.py","v2_experiments.py"])
def test_entrypoint_pins_workspace(script):
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy();env.pop("PYTHONPATH",None)
    run=subprocess.run([sys.executable,"-X","utf8",str(root/"scripts"/script),"--help"],cwd=root,env=env,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert "usage:" in run.stdout
