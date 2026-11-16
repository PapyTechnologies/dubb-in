# Changelog

All notable changes to **Dubb-in** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-11-16

### Initial Public Release 🚀

#### Added
- **Core Orchestrator (`engine.py`)**:
  - Multi-stage pipeline integrating transcription, translation, voice synthesis, vocal separation, and lip synchronization.
  - Faster-Whisper automated speech recognition with Voice Activity Detection (VAD) and timestamp preservation.
  - Deep-Translator integration supporting 10 languages (FR, EN, ES, DE, IT, PT, JA, ZH, AR, RU).
  - Coqui XTTS-v2 zero-shot voice cloning engine with pitch matching and energy preservation.
  - Demucs stem separation for background music and ambient sound extraction.
- **Biometric Vocal Learning & `.dvf` Fingerprints (`voice_learning.py`)**:
  - Proprietary Dubb-in Voice Fingerprint (`.dvf`) architecture combining 512-dimensional XTTS speaker latents, F0 fundamental pitch statistics, and harmonic spectral profiles.
  - Guided 3-sample vocal recording studio with interactive multilingual teleprompter.
  - Biometric certification metrics (F0 median, harmonic clarity index, SNR estimation).
- **AI Lip-Sync Studio (`lipsync/` & `lipsync_ui.py`)**:
  - Optimized ONNX Wav2Lip model runner with YuNet lightweight face detection.
  - Numba-accelerated temporal EMA smoothing and cosine feather blending.
  - Deep Learning Face container (`.dlf`) for reproducible visual sync packaging.
  - Seamless fallback and on-demand model downloader from Hugging Face.
- **User Interface (`app.py` & `theme.css`)**:
  - Streamlit wide layout with Windows 11 Fluent dark acrylic styling.
  - Real-time hardware diagnostic monitor (GPU VRAM detection, CPU load, DirectML/CUDA status).
  - Multi-tab navigation: Studio Doublage, Empreinte Vocale (.dvf), Projets & Bibliothèque, Diagnostic Matériel, Soutien & Dons.
  - Dynamic multilingual interface localization (`i18n.py`).
- **Native Desktop Shell (`window.py` & `Dubb-in.exe`)**:
  - PyWebview standalone window wrapper offering zero-browser native desktop experience.
  - Standalone batch installer (`installer_windows.bat`) with automated FFmpeg and PyTorch CUDA configuration.
- **Documentation & Community**:
  - Complete English (`README.md`) and French (`README.fr.md`) guides with hardware requirements and architecture breakdown.
  - GitHub CI workflows, issue templates, and pull request guidelines.

[0.1.0]: https://github.com/PapyTechnologies/dubb-in/releases/tag/v0.1.0
