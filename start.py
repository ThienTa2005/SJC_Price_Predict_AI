"""Launch locally, in Docker, or on Render using the platform PORT."""
import os
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(root / "app.py"),
        "--server.address=0.0.0.0", "--server.port=" + os.environ.get("PORT", "8501"),
        "--server.headless=true"], cwd=root))

