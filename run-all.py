import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS = (
    ("scripts/convergence-plots.py",),
    ("scripts/synthetic-plots.py",),
    ("scripts/real-data-diagnostics.py", "--sarcos"),
    ("scripts/real-data-diagnostics.py", "--naval"),
)


for command in COMMANDS:
    subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
