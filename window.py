import os
import sys
import time
import socket
import subprocess
import ctypes
from pathlib import Path

# 1. Enregistrer l'AppUserModelID pour que Windows dissocie le processus de Python et utilise l'icône de Dubb-in dans la barre des tâches
APP_ID = "DubbIn.Studio.AIVocalApp.v2"
try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
except Exception:
    pass

import webview

BASE = Path(__file__).resolve().parent

def is_port_open(port=8501):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex(('127.0.0.1', port)) == 0
    except Exception:
        return False

def main():
    server_proc = None
    if not is_port_open(8501):
        python_exe = str(BASE / ".venv" / "Scripts" / "python.exe")
        app_py = str(BASE / "app.py")
        cmd = [
            python_exe, "-m", "streamlit", "run", app_py,
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--ui.hideTopBar", "true",
            "--client.toolbarMode", "minimal"
        ]
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        sub_env = os.environ.copy()
        sub_env["PYTHONIOENCODING"] = "utf-8"
        sub_env["PYTHONUTF8"] = "1"
        server_proc = subprocess.Popen(
            cmd,
            cwd=str(BASE),
            env=sub_env,
            creationflags=flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        for _ in range(60):
            if is_port_open(8501):
                break
            time.sleep(0.3)

    icon_path = str(BASE / "icon.ico")
    
    # Création de la fenêtre native autonome Dubb-in
    window = webview.create_window(
        title="Dubb-in",
        url="http://127.0.0.1:8501",
        width=1440,
        height=900,
        resizable=True,
        min_size=(1024, 680),
        background_color="#08090d",
        text_select=True,
        zoomable=True,
    )

    def on_closed():
        if server_proc:
            try:
                server_proc.terminate()
            except Exception:
                pass

    window.events.closed += on_closed
    webview.start(icon=icon_path if os.path.exists(icon_path) else None, private_mode=False)

if __name__ == "__main__":
    main()
