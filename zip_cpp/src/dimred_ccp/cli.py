import os, subprocess
from pathlib import Path

def serve():
    port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
    addr = os.environ.get("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    app_path = Path(__file__).parent / "app.py"
    cmd = ["streamlit", "run", str(app_path), "--server.port", str(port), "--server.address", str(addr)]
    raise SystemExit(subprocess.call(cmd))
