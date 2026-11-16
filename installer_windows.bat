@echo off
setlocal enabledelayedexpansion
title Dubb-in AI Studio - Windows Installer & Environment Setup
cd /d "%~dp0"

echo ===============================================================================
echo                DUBB-IN AI STUDIO - AUTOMATED SETUP WIZARD
echo ===============================================================================
echo.

:: 1. FFmpeg PATH Configuration & Detection
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin" (
    set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin;%PATH%"
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [!] ERREUR CRITIQUE : FFmpeg est absent du PATH systeme.
    echo     Veuillez installer FFmpeg (ex: via winget install Gyan.FFmpeg) puis relancer.
    echo.
    pause
    exit /b 1
)

where ffprobe >nul 2>nul
if errorlevel 1 (
    echo [!] ERREUR CRITIQUE : ffprobe est absent du PATH systeme.
    echo.
    pause
    exit /b 1
)
echo [OK] FFmpeg et FFprobe detectes avec succes.

:: 2. Python 3.10 Detection & Virtualenv Creation
py -3.10 -V >nul 2>nul
if errorlevel 1 (
    python -V 2>nul | findstr /R "3\.10\." >nul
    if errorlevel 1 (
        echo [!] ERREUR : Python 3.10 est requis (compatibilite PyTorch 2.2 / XTTS-v2).
        echo     Veuillez installer Python 3.10 (64-bit) depuis python.org.
        echo.
        pause
        exit /b 1
    ) else (
        set "PY_CMD=python"
    )
) else (
    set "PY_CMD=py -3.10"
)

if not exist ".venv" (
    echo [*] Creation de l'environnement virtuel Python 3.10 (.venv)...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [!] ERREUR lors de la creation du venv.
        pause
        exit /b 1
    )
)
echo [OK] Environnement virtuel pret (.venv).

:: 3. Core Packaging Tools
echo [*] Mise a jour de pip, setuptools et wheel...
.venv\Scripts\python.exe -m pip install --quiet --upgrade pip "setuptools<70" wheel
if errorlevel 1 (
    echo [!] ERREUR lors de la mise a jour de pip.
    pause
    exit /b 1
)

:: 4. PyTorch Hardware Acceleration Detection (CUDA vs CPU)
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo [*] GPU NVIDIA detecte : Installation de PyTorch 2.2.2 avec acceleration CUDA 11.8...
    .venv\Scripts\python.exe -m pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 --index-url https://download.pytorch.org/whl/cu118
    .venv\Scripts\python.exe -m pip install --quiet onnxruntime-gpu>=1.16.0
) else (
    echo [*] Aucun GPU NVIDIA detecte : Installation de PyTorch 2.2.2 en mode CPU standard...
    .venv\Scripts\python.exe -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    .venv\Scripts\python.exe -m pip install --quiet onnxruntime>=1.16.0
)

:: 5. Project Dependencies
echo [*] Installation des dependances (Whisper, XTTS-v2, Demucs, Webview, Streamlit)...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo [!] AVERTISSEMENT : Verifiez les messages ci-dessus en cas de dependance manquante.
)

:: 6. Directory Structure Setup
if not exist "voices\fingerprints" mkdir "voices\fingerprints"
if not exist "projects\lipsync_exports" mkdir "projects\lipsync_exports"
if not exist "lipsync\models" mkdir "lipsync\models"
if not exist "models" mkdir "models"

:: 7. Automated Model Downloader (Wav2Lip ONNX)
if not exist "lipsync\models\wav2lip_gan.onnx" (
    if not exist "models\wav2lip_gan.onnx" (
        echo [*] Telechargement du modele de Lip-Sync optimise (wav2lip_gan.onnx, ~145 Mo)...
        powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('https://huggingface.co/bluefoxcreation/Wav2lip-Onnx/resolve/main/wav2lip_gan.onnx', 'lipsync\models\wav2lip_gan.onnx')"
        if exist "lipsync\models\wav2lip_gan.onnx" (
            echo [OK] Modele wav2lip_gan.onnx telecharge avec succes dans lipsync\models.
        ) else (
            echo [!] Note : Le modele wav2lip_gan.onnx sera telecharge automatiquement au premier lancement du module Lip-Sync.
        )
    )
)

echo.
echo ===============================================================================
echo                INSTALLATION ET CONFIGURATION TERMINEES !
echo ===============================================================================
echo.
echo Pour lancer Dubb-in :
echo   - Option 1 : Double-cliquez sur "Dubb-in.exe" (Application autonome)
echo   - Option 2 : .venv\Scripts\python.exe window.py
echo   - Option 3 : .venv\Scripts\streamlit.exe run app.py
echo.
set /p LAUNCH="Voulez-vous lancer Dubb-in maintenant ? (O/N) : "
if /i "%LAUNCH%"=="O" (
    if exist "Dubb-in.exe" (
        start "" "Dubb-in.exe"
    ) else (
        start "" .venv\Scripts\python.exe window.py
    )
)
exit /b 0
