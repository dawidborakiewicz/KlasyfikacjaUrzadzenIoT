import atexit
import os
import signal
import subprocess
import sys
from pathlib import Path

_mlflow_proc = None
_mlflow_log_f = None

def _as_posix(p: Path) -> str:
    return p.resolve().as_posix()

def start_mlflow_server(host="127.0.0.1", port=8080):
    global _mlflow_proc, _mlflow_log_f

    if _mlflow_proc and _mlflow_proc.poll() is None:
        return _mlflow_proc

    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "mlflow.db"
    artifacts_dir = base_dir / "artifacts"
    logs_dir = base_dir / "logs"
    log_path = logs_dir / "mlflow.log"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    backend_store_uri = f"sqlite:///{_as_posix(db_path)}"

    artifacts_destination = f"file:{_as_posix(artifacts_dir)}"

    _mlflow_log_f = open(log_path, "a", encoding="utf-8")

    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--host", host,
        "--port", str(port),
        "--backend-store-uri", backend_store_uri,
        "--artifacts-destination", artifacts_destination,
        "--serve-artifacts",
    ]

    _mlflow_proc = subprocess.Popen(
        cmd,
        stdout=_mlflow_log_f,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        cwd=str(base_dir),
    )

    def cleanup():
        global _mlflow_proc, _mlflow_log_f
        if _mlflow_proc and _mlflow_proc.poll() is None:
            try:
                _mlflow_proc.send_signal(signal.CTRL_BREAK_EVENT)
                _mlflow_proc.wait(timeout=10)
            except Exception:
                try:
                    _mlflow_proc.kill()
                except Exception:
                    pass
        try:
            if _mlflow_log_f:
                _mlflow_log_f.close()
        except Exception:
            pass

    atexit.register(cleanup)
    return _mlflow_proc
