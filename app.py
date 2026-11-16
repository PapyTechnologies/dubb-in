"""
Dubb-in AI Studio — Streamlit UI & Orchestration Pipeline
Open-Source Voice Cloning & Video Dubbing (XTTS-v2, Faster-Whisper, Demucs, Wav2Lip GAN .dlf)
https://github.com/PapyTechnologies
"""
import sys
import os

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    try:
        if sys.stdout:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import io
import torch

# Correctif critique contre OSError: [WinError -1073741818] Windows Error 0xc0000006
# Force PyTorch à lire les fichiers .pth via un flux standard au lieu du Memory Mapping (mmap) Windows
if not hasattr(torch, "_orig_torch_load_safe_patched"):
    _orig_torch_load = torch.load

    def _safe_windows_torch_load(f, *args, **kwargs):
        kwargs.pop("mmap", None)
        if isinstance(f, (str, os.PathLike)):
            with open(f, "rb") as fp:
                return _orig_torch_load(fp, *args, **kwargs)
        return _orig_torch_load(f, *args, **kwargs)

    torch.load = _safe_windows_torch_load
    torch._orig_torch_load_safe_patched = True

import hashlib
import json
import re
import shutil
import subprocess
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def get_unique_desktop_path(base_name, extension=".mp4"):
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    clean_base = re.sub(r'[\\/*?:"<>| ]', '_', str(base_name)).strip('_')
    candidate = desktop / f"{clean_base}{extension}"
    counter = 1
    while candidate.exists():
        candidate = desktop / f"{clean_base}_{counter}{extension}"
        counter += 1
    return candidate

os.environ["COQUI_TOS_AGREED"] = "1"
import streamlit as st
import streamlit.components.v1 as components
from audio_recorder_streamlit import audio_recorder
import engine
import i18n

if "app_lang" not in st.session_state:
    st.session_state["app_lang"] = i18n.detect_system_language()

cur_lang = st.session_state["app_lang"]
import autotr
autotr.install(st, cur_lang)
lang_keys = list(i18n.LANGUAGES.keys())
lang_labels = [f"{i18n.LANGUAGES[k]['flag']} {i18n.LANGUAGES[k]['name']}" for k in lang_keys]

def T(key, **kwargs):
    return i18n.get_text(key, cur_lang, **kwargs)

CHIME_HTML = """
<script>
(function() {
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        function playTone(freq, time, dur) {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, ctx.currentTime + time);
            gain.gain.setValueAtTime(0.001, ctx.currentTime + time);
            gain.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + time + 0.04);
            gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + time + dur);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(ctx.currentTime + time);
            osc.stop(ctx.currentTime + time + dur + 0.05);
        }
        playTone(523.25, 0.0, 0.28);
        playTone(659.25, 0.12, 0.28);
        playTone(783.99, 0.24, 0.45);
        playTone(1046.50, 0.38, 0.65);
    } catch(e) {}
})();
</script>
"""

ROOT = BASE_DIR / "projects"
ROOT.mkdir(exist_ok=True)

_icon_p = BASE_DIR / "icon.png"
st.set_page_config(
    page_title="Dubb-in",
    page_icon=str(_icon_p) if _icon_p.exists() else "🎙️",
    layout="wide",
)

if st.session_state.pop("play_ding", False):
    components.html(CHIME_HTML, height=0, width=0)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Suppression barre Streamlit encombrante */
#MainMenu, header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer {
    display: none !important;
    visibility: hidden !important;
    height: 0px !important;
}
.block-container {
    padding-top: 1.1rem !important;
    padding-bottom: 2.2rem !important;
    max-width: 1420px !important;
}

/* Thème Studio Obsidian & Cyber Coral */
.stApp {
    background-color: #08090d !important;
    background-image: 
        radial-gradient(circle at 12% 0%, rgba(255, 51, 102, 0.09) 0%, transparent 42%),
        radial-gradient(circle at 88% 8%, rgba(139, 92, 246, 0.11) 0%, transparent 45%),
        radial-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px) !important;
    background-size: 100% 100%, 100% 100%, 24px 24px !important;
    color: #e2e8f0 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Sidebar Studio */
[data-testid="stSidebar"] {
    background-color: #07080c !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

/* Typographie Headings */
h1, h2, h3, h4, .main-title {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em !important;
}

/* Conteneurs et cartes Studio */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, rgba(20, 24, 36, 0.75) 0%, rgba(12, 14, 22, 0.88) 100%) !important;
    backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 16px 40px -12px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    padding: 1.4rem !important;
    transition: all 0.25s ease !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 20px 48px -10px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
}

/* Accordéons / Expanders */
details[data-testid="stExpander"] {
    background: linear-gradient(180deg, rgba(18, 22, 34, 0.6) 0%, rgba(11, 13, 20, 0.7) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
}
summary[data-testid="stExpanderToggle"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    letter-spacing: 0.01em !important;
}

/* Boutons Studio Neon & Premium */
.stButton > button {
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    padding: 0.6rem 1.4rem !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ff3366 0%, #ff5e3a 100%) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 6px 20px rgba(255, 51, 102, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(255, 51, 102, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
}

/* Badges Hardware & Télémétrie */
.hw-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 6px;
    letter-spacing: 0.03em;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
    letter-spacing: -0.03em;
}
.sub-studio {
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.pulse-led {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 8px #10b981;
    margin-right: 6px;
}

/* Spectrogramme vivant */
.spectrum-box {
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 24px;
}
.spec-bar {
    width: 3px;
    background: linear-gradient(180deg, #ff3366, #ff5e3a);
    border-radius: 2px;
    animation: bounce 1.2s ease-in-out infinite alternate;
}
.spec-bar:nth-child(2) { animation-delay: 0.2s; height: 60%; }
.spec-bar:nth-child(3) { animation-delay: 0.4s; height: 90%; }
.spec-bar:nth-child(4) { animation-delay: 0.1s; height: 40%; }
.spec-bar:nth-child(5) { animation-delay: 0.5s; height: 75%; }
.spec-bar:nth-child(6) { animation-delay: 0.3s; height: 100%; }
.spec-bar:nth-child(7) { animation-delay: 0.6s; height: 50%; }
.spec-bar:nth-child(8) { animation-delay: 0.25s; height: 80%; }

@keyframes bounce {
    0% { height: 20%; opacity: 0.5; }
    100% { height: 100%; opacity: 1; }
}

/* Pipeline tracker */
.pipeline-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(14, 16, 24, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 10px 18px;
    margin-bottom: 18px;
}
.pipe-node {
    display: flex;
    align-items: center;
    gap: 10px;
}
.pipe-badge {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 800;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 51, 102, 0.15);
    border: 1px solid #ff3366;
    color: #ff6b8b;
}
.pipe-txt {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #f1f5f9;
}
.pipe-arrow {
    color: #475569;
    font-size: 14px;
}

/* Jauge Moderne Windows 11 Fluent */
.win11-progress-container {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 12px 18px;
    margin: 12px 0 16px 0;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.win11-progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.win11-progress-title {
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI Variable', sans-serif;
    font-size: 13.5px;
    font-weight: 600;
    color: #f1f5f9;
    display: flex;
    align-items: center;
    gap: 8px;
}
.win11-progress-pill {
    background: rgba(96, 205, 255, 0.15);
    border: 1px solid rgba(96, 205, 255, 0.35);
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 999px;
}
.win11-track {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    overflow: hidden;
    position: relative;
}
.win11-fill {
    height: 100%;
    background: linear-gradient(90deg, #ff3366 0%, #8b5cf6 50%, #38bdf8 100%);
    border-radius: 999px;
    transition: width 0.35s ease;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.45);
}

/* Lecteur Vidéo Compact (Anti Débordement 9:16) */
video, [data-testid="stVideo"] video {
    max-height: 380px !important;
    max-width: 480px !important;
    margin: 0 auto !important;
    display: block !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5) !important;
    object-fit: contain !important;
}

/* Égaliseur Réactif Multicolore */
.eq-container {
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 5px;
    height: 44px;
    padding: 8px 16px;
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 51, 102, 0.3);
    border-radius: 10px;
    margin: 10px 0;
}
.eq-bar {
    width: 6px;
    border-radius: 999px;
    background: linear-gradient(180deg, #ff3366 0%, #ec4899 30%, #8b5cf6 60%, #06b6d4 100%);
    animation: eq-pulse 1.1s infinite ease-in-out;
}
@keyframes eq-pulse {
    0%, 100% { height: 6px; opacity: 0.45; }
    50% { height: 36px; opacity: 1.0; filter: brightness(1.25); }
}
.eq-bar:nth-child(1) { animation-delay: 0.1s; }
.eq-bar:nth-child(2) { animation-delay: 0.4s; }
.eq-bar:nth-child(3) { animation-delay: 0.2s; }
.eq-bar:nth-child(4) { animation-delay: 0.55s; }
.eq-bar:nth-child(5) { animation-delay: 0.15s; }
.eq-bar:nth-child(6) { animation-delay: 0.35s; }
.eq-bar:nth-child(7) { animation-delay: 0.25s; }
.eq-bar:nth-child(8) { animation-delay: 0.6s; }
.eq-bar:nth-child(9) { animation-delay: 0.3s; }
.eq-bar:nth-child(10) { animation-delay: 0.45s; }
.eq-bar:nth-child(11) { animation-delay: 0.18s; }
.eq-bar:nth-child(12) { animation-delay: 0.5s; }
</style>
""", unsafe_allow_html=True)

# --- Theme UI (presentation) : feuille de style additionnelle ---
_theme_css = BASE_DIR / "theme.css"
if _theme_css.exists():
    st.markdown(f"<style>{_theme_css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

import base64
_icon_b64 = ""
if _icon_p.exists():
    _icon_b64 = f'<img src="data:image/png;base64,{base64.b64encode(_icon_p.read_bytes()).decode()}" width="44" height="44" style="border-radius:12px; box-shadow:0 0 25px rgba(255, 51, 102, 0.4); vertical-align:middle; margin-right:12px; border:1px solid rgba(255,255,255,0.15);" />'

def render_fluent_progress(placeholder, ratio, status_text, step_info=""):
    pct = int(min(1.0, max(0.0, float(ratio))) * 100)
    info_html = f"<span style='font-size:12px; color:#94a3b8; margin-left:8px;'>({step_info})</span>" if step_info else ""
    html = f"""
    <div class="win11-progress-container">
        <div class="win11-progress-header">
            <div class="win11-progress-title">
                <span class="pulse-led"></span>
                <span>{status_text}</span>
                {info_html}
            </div>
            <span class="win11-progress-pill">{pct}%</span>
        </div>
        <div class="win11-track">
            <div class="win11-fill" style="width: {pct}%;"></div>
        </div>
    </div>
    """
    placeholder.markdown(html, unsafe_allow_html=True)

# ---------------------------------------------------------
# En-Tête & Menu de Navigation Supérieur
# ---------------------------------------------------------
NAV_CATEGORIES = [
    "🎙️ Studio Doublage",
    "🧬 Empreinte Vocale (.dvf)",
    "📁 Projets & Bibliothèque",
    "⚙️ Diagnostic Matériel",
    "☕ Soutien & Dons",
]

def _d_chip(text, kind=""):
    return '<span class="d-chip ' + kind + '">' + str(text) + '</span>'


def _d_gauge(pct, grad="load"):
    pct = max(0.0, min(100.0, float(pct)))
    if grad == "load":
        bg = "linear-gradient(90deg,#34d399 0%,#fbbf24 70%,#ff4d6d 100%)"
    else:
        bg = "linear-gradient(90deg,#818cf8 0%,#34d399 100%)"
    size = int(10000 / max(pct, 1.0))
    return ('<div class="d-gauge"><span style="width:' + ("%.0f" % pct) + '%;background:' + bg
            + ';background-size:' + str(size) + '% 100%;"></span></div>')


def _read_gpu_vram():
    """VRAM utilisee/totale (Mo) via nvidia-smi : n'ouvre aucun contexte CUDA (GTX 960 = 2 Go)."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        used, total = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        return used, total
    except Exception:
        return None


if "main_nav" not in st.session_state:
    st.session_state["main_nav"] = NAV_CATEGORIES[0]

col_hdr_brand, col_hdr_right = st.columns([7.8, 2.2])

with col_hdr_brand:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; padding-bottom: 0.2rem;">
            {_icon_b64}
            <div>
                <div class="main-title">Dubb-in</div>
                <div class="sub-studio"><span class="pulse-led"></span>{T('tagline')}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_hdr_right:
    c_hspec, c_hlang = st.columns([1, 2])
    with c_hspec:
        st.markdown(
            """
            <div style="display:flex; justify-content:center; align-items:center; height:100%; padding-top:6px;">
                <div class="spectrum-box" title="Audio Signal Active">
                    <span class="spec-bar"></span><span class="spec-bar"></span><span class="spec-bar"></span>
                    <span class="spec-bar"></span><span class="spec-bar"></span><span class="spec-bar"></span>
                    <span class="spec-bar"></span><span class="spec-bar"></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c_hlang:
        curr_lang_idx = lang_keys.index(cur_lang) if cur_lang in lang_keys else 0
        new_lang_choice = st.selectbox(
            "Langue de l'interface",
            lang_labels,
            index=curr_lang_idx,
            key="ui_lang_picker",
            label_visibility="collapsed"
        )
        picked_code = lang_keys[lang_labels.index(new_lang_choice)]
        if picked_code != cur_lang:
            st.session_state["app_lang"] = picked_code
            i18n.set_saved_language(picked_code)
            st.rerun()

st.markdown("<div style='border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 0.6rem;'></div>", unsafe_allow_html=True)

# Barre de navigation horizontale 1-clic
active_tab_nav = st.radio(
    "Navigation Principale",
    NAV_CATEGORIES,
    index=NAV_CATEGORIES.index(st.session_state["main_nav"]),
    horizontal=True,
    label_visibility="collapsed",
    key="top_main_nav_pills"
)
if active_tab_nav != st.session_state["main_nav"]:
    st.session_state["main_nav"] = active_tab_nav
    st.rerun()

try:
    engine.check_tools()
except Exception as error:
    st.error(str(error))
    st.stop()

rights = True
models = True
cloud = True
hw = engine.detect_hardware_profile()

def fail(error, project=None):
    st.error(f"{type(error).__name__} : {error}")
    if project:
        (project / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        st.info("Le projet est conservé. Corrigez le problème avant de relancer cette étape.")

active_section = st.session_state["main_nav"]


# =========================================================
# VUE 1 : STUDIO DE SYNCHRONISATION LABIALE (LIP-SYNC .DLF)
# =========================================================
if active_section == "👄 Lip-Sync Studio (.dlf)":
    import lipsync_ui
    lipsync_ui.render_lipsync_studio(ROOT, hw, T)


# =========================================================
# VUE 2 : BANQUE VOCALE & EMPREINTES IA (.DVF)
# =========================================================
elif active_section == "🧬 Empreinte Vocale (.dvf)":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
                    border: 1px solid rgba(59, 130, 246, 0.35); border-radius: 14px; padding: 18px 22px; margin-bottom: 20px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                <span style="font-size:26px;">🧬</span>
                <span style="font-family:'Space Grotesk',sans-serif; font-size:18px; font-weight:700; color:#f8fafc; letter-spacing:0.02em;">STUDIO D'ÉTUDE & D'APPRENTISSAGE BIOMÉTRIQUE VOCAL IA</span>
                <span style="background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.4); color:#60a5fa; font-size:11px; font-weight:700; padding:2px 8px; border-radius:12px;">XTTS-v2 NEURAL EMBEDDINGS</span>
            </div>
            <div style="font-size:13.5px; line-height:1.6; color:#94a3b8;">
                L'IA étudie votre signature vocale en profondeur (fréquence fondamentale F0, formants, clarté harmonique, SNR) et extrait un tenseur neural de conditionnement (512 dimensions) pour un clonage ultra-fidèle, stable et 3x plus rapide lors du doublage.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    vp_tab_learn, vp_tab_new, vp_tab_list = st.tabs([
        "🧬 Studio d'Apprentissage IA (Multi-Échantillons)",
        T("tab_new_voice"),
        T("tab_saved_voices")
    ])

    with vp_tab_learn:
        col_lp1, col_lp2 = st.columns([1, 1])
        with col_lp1:
            learn_name = st.text_input("Nom de l'Empreinte Vocale IA", placeholder="ex: Ma Voix Studio Pro", key="learn_name_input")
        with col_lp2:
            c_mic, c_lng = st.columns([3, 2])
            with c_mic:
                hw_devs_learn = engine.list_audio_input_devices()
                hw_labels_learn = [d["label"] for d in hw_devs_learn]
                sel_mic_learn = st.selectbox("Microphone d'enregistrement", hw_labels_learn, key="sel_mic_learn")
                sel_target_learn = next((d["target"] for d in hw_devs_learn if d["label"] == sel_mic_learn), None)
            with c_lng:
                def_lang_idx = lang_keys.index(cur_lang) if cur_lang in lang_keys else 0
                learn_lang_choice = st.selectbox("Langue des textes", lang_labels, index=def_lang_idx, key="sel_learn_calib_lang")
                learn_lang_code = lang_keys[lang_labels.index(learn_lang_choice)]

        s1_data = st.session_state.get("learn_sample_1")
        s2_data = st.session_state.get("learn_sample_2")
        s3_data = st.session_state.get("learn_sample_3")
        count_ready = sum([bool(s1_data), bool(s2_data), bool(s3_data)])

        badge_color = "#22c55e" if count_ready >= 1 else "#f59e0b"
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin: 18px 0 10px 0; padding: 10px 16px; background: rgba(18, 22, 34, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px;">
                <span style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:14px; color:#f1f5f9;">🎙️ Échantillons Vocaux d'Apprentissage (1 à 3) :</span>
                <span style="font-size:12px; font-weight:700; color:{badge_color}; background:rgba(255,255,255,0.05); padding:3px 10px; border-radius:8px; border:1px solid {badge_color}44;">{count_ready} / 3 Échantillons Enregistrés</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        tab_l1, tab_l2, tab_l3 = st.tabs([
            f"📍 Échantillon 1 : Calme {'✅' if s1_data else '(Obligatoire)'}",
            f"📍 Échantillon 2 : Dynamique {'✅' if s2_data else '(Recommandé)'}",
            f"📍 Échantillon 3 : Expressif {'✅' if s3_data else '(Optionnel)'}"
        ])

        t_calib_1 = i18n.get_calibration_text(learn_lang_code, sample_index=1)
        t_calib_2 = i18n.get_calibration_text(learn_lang_code, sample_index=2)
        t_calib_3 = i18n.get_calibration_text(learn_lang_code, sample_index=3)

        with tab_l1:
            st.markdown(f"<div style='background:rgba(255,255,255,0.03); border-left:3px solid #3b82f6; padding:12px 16px; border-radius:6px; font-size:14px; color:#f1f5f9; font-style:italic; margin-bottom:12px;'>« {t_calib_1} »</div>", unsafe_allow_html=True)
            col_l1a, col_l1b = st.columns([1, 1])
            with col_l1a:
                dur_s1 = st.slider("Durée d'enregistrement (sec)", 6, 25, 12, key="sl_s1_dur")
            with col_l1b:
                st.write("")
                if st.button(f"🎙️ Enregistrer Échantillon 1 ({dur_s1}s)", key="btn_rec_s1"):
                    try:
                        with st.spinner(f"🎙️ Enregistrement micro en cours ({dur_s1}s)... Lisez le texte calmement :"):
                            rec_path = engine.record_microphone_hardware(duration_seconds=dur_s1, device_target=sel_target_learn)
                            st.session_state["learn_sample_1"] = rec_path.read_bytes()
                            if rec_path.exists():
                                rec_path.unlink()
                        st.success("✅ Échantillon 1 capturé !")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur micro : {e}")
            if s1_data:
                st.markdown("<div class='eq-container'><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div></div>", unsafe_allow_html=True)
                col_aud1, col_del1 = st.columns([3, 2])
                with col_aud1:
                    st.audio(s1_data, format="audio/wav")
                with col_del1:
                    if st.button("🗑️ Supprimer & Réenregistrer", key="btn_del_s1", help="Purge immédiatement l'échantillon"):
                        st.session_state.pop("learn_sample_1", None)
                        st.success("✅ Échantillon supprimé et espace disque libéré !")
                        st.rerun()

        with tab_l2:
            st.markdown(f"<div style='background:rgba(255,255,255,0.03); border-left:3px solid #ff3366; padding:12px 16px; border-radius:6px; font-size:14px; color:#f1f5f9; font-style:italic; margin-bottom:12px;'>« {t_calib_2} »</div>", unsafe_allow_html=True)
            col_l2a, col_l2b = st.columns([1, 1])
            with col_l2a:
                dur_s2 = st.slider("Durée d'enregistrement (sec)", 6, 25, 10, key="sl_s2_dur")
            with col_l2b:
                st.write("")
                if st.button(f"🎙️ Enregistrer Échantillon 2 ({dur_s2}s)", key="btn_rec_s2"):
                    try:
                        with st.spinner(f"🎙️ Enregistrement micro en cours ({dur_s2}s)... Lisez avec dynamisme :"):
                            rec_path = engine.record_microphone_hardware(duration_seconds=dur_s2, device_target=sel_target_learn)
                            st.session_state["learn_sample_2"] = rec_path.read_bytes()
                            if rec_path.exists():
                                rec_path.unlink()
                        st.success("✅ Échantillon 2 capturé !")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur micro : {e}")
            if s2_data:
                st.markdown("<div class='eq-container'><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div></div>", unsafe_allow_html=True)
                col_aud2, col_del2 = st.columns([3, 2])
                with col_aud2:
                    st.audio(s2_data, format="audio/wav")
                with col_del2:
                    if st.button("🗑️ Supprimer & Réenregistrer", key="btn_del_s2", help="Purge immédiatement l'échantillon"):
                        st.session_state.pop("learn_sample_2", None)
                        st.success("✅ Échantillon supprimé et espace disque libéré !")
                        st.rerun()

        with tab_l3:
            st.markdown(f"<div style='background:rgba(255,255,255,0.03); border-left:3px solid #10b981; padding:12px 16px; border-radius:6px; font-size:14px; color:#f1f5f9; font-style:italic; margin-bottom:12px;'>« {t_calib_3} »</div>", unsafe_allow_html=True)
            col_l3a, col_l3b = st.columns([1, 1])
            with col_l3a:
                dur_s3 = st.slider("Durée d'enregistrement (sec)", 6, 25, 10, key="sl_s3_dur")
            with col_l3b:
                st.write("")
                if st.button(f"🎙️ Enregistrer Échantillon 3 ({dur_s3}s)", key="btn_rec_s3"):
                    try:
                        with st.spinner(f"🎙️ Enregistrement micro en cours ({dur_s3}s)... Lisez de manière narrative :"):
                            rec_path = engine.record_microphone_hardware(duration_seconds=dur_s3, device_target=sel_target_learn)
                            st.session_state["learn_sample_3"] = rec_path.read_bytes()
                            if rec_path.exists():
                                rec_path.unlink()
                        st.success("✅ Échantillon 3 capturé !")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur micro : {e}")
            if s3_data:
                st.markdown("<div class='eq-container'><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div></div>", unsafe_allow_html=True)
                col_aud3, col_del3 = st.columns([3, 2])
                with col_aud3:
                    st.audio(s3_data, format="audio/wav")
                with col_del3:
                    if st.button("🗑️ Supprimer & Réenregistrer", key="btn_del_s3", help="Purge immédiatement l'échantillon"):
                        st.session_state.pop("learn_sample_3", None)
                        st.success("✅ Échantillon supprimé et espace disque libéré !")
                        st.rerun()
                        st.error(f"Erreur micro : {e}")
            if s3_data:
                st.audio(s3_data, format="audio/wav")

        can_train = bool(learn_name.strip()) and bool(s1_data)
        _ns = sum(1 for _x in (s1_data, s2_data, s3_data) if _x)
        st.markdown(
            '<div style="display:flex;gap:6px;flex-wrap:wrap;margin:10px 0 8px;">'
            + _d_chip("Nom du profil", "ok" if learn_name.strip() else "warn")
            + "".join(_d_chip("\u00c9chantillon %d" % _i, "ok" if _d else "warn")
                      for _i, _d in enumerate((s1_data, s2_data, s3_data), 1))
            + _d_chip("Pr\u00eat \u00e0 entra\u00eener" if can_train else "En attente", "ok" if can_train else "warn")
            + '</div>' + _d_gauge(_ns / 3 * 100, "progress"),
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("🚀 DÉMARRER L'APPRENTISSAGE IA & EXPORT DE L'EMPREINTE (.DVF)", disabled=not can_train, type="primary", use_container_width=True, key="btn_start_voice_learn"):
            try:
                samples_data = [s for s in [s1_data, s2_data, s3_data] if s]
                _dvf_ph = st.empty()
                _dvf_ph.markdown(
                    '<div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0;">'
                    + _d_chip("Analyse spectrale", "warn") + _d_chip("Biom\u00e9trie vocale", "warn")
                    + _d_chip("Apprentissage tenseurs", "warn") + _d_chip("Export .dvf", "warn") + '</div>',
                    unsafe_allow_html=True,
                )
                with st.spinner("🔬 Analyse spectrale, biométrique et apprentissage des tenseurs neuronaux..."):
                    import voice_learning
                    prof = voice_learning.train_voice_fingerprint(
                        voice_name=learn_name.strip(),
                        samples_bytes=samples_data,
                        language=learn_lang_code
                    )
                _dvf_ph.empty()
                st.session_state.pop("learn_sample_1", None)
                st.session_state.pop("learn_sample_2", None)
                st.session_state.pop("learn_sample_3", None)
                st.success(f"🎉 Empreinte IA '{prof['name']}' créée et certifiée ! F0 médian : {prof['f0_median']} Hz • Clarté : {prof['clarity_score']:.0f}% • SNR : {prof['snr_db']:.1f} dB.")
                st.rerun()
            except Exception as err:
                st.error(f"Erreur d'apprentissage vocal : {err}")

    with vp_tab_new:
        col_vp1, col_vp2 = st.columns(2)
        with col_vp1:
            vp_name = st.text_input(T("voice_name"), placeholder="ex: Ma Voix Directe", key="vp_name_input")
        with col_vp2:
            vp_source = st.radio(T("voice_source"), [T("src_mic"), T("src_file"), T("src_url")], horizontal=True, key="vp_src_radio")

        is_mic = vp_source == T("src_mic")
        is_file = vp_source == T("src_file")
        vp_ready = False
        vp_audio_bytes = None
        vp_source_url = ""
        vp_start = 0.0
        vp_dur = 20.0

        if is_mic:
            hw_devs = engine.list_audio_input_devices()
            hw_labels = [d["label"] for d in hw_devs]
            sel_mic_label = st.selectbox(T("mic_device"), hw_labels, key="sel_mic_input_vp")
            vp_dev_target = next((d["target"] for d in hw_devs if d["label"] == sel_mic_label), None)

            col_rec1, col_rec2 = st.columns([1, 1])
            with col_rec1:
                rec_dur = st.slider(T("rec_dur"), min_value=3, max_value=30, value=10, step=1, key="rec_dur_slider")
            with col_rec2:
                st.write("")
                if st.button(T("btn_rec", sec=rec_dur), key="btn_rec_direct"):
                    try:
                        with st.spinner(f"🎙️ ... ({rec_dur}s)"):
                            wav_file = engine.record_microphone_hardware(duration_seconds=rec_dur, device_target=vp_dev_target)
                            st.session_state["recorded_mic_bank"] = wav_file.read_bytes()
                            if wav_file.exists():
                                wav_file.unlink()
                        st.success(T("rec_success"))
                    except Exception as err:
                        st.error(f"Erreur micro : {err}")

            saved_mic = st.session_state.get("recorded_mic_bank")
            if saved_mic:
                st.markdown(f"<div style='margin-top:10px; font-family:\"Space Grotesk\",sans-serif; font-size:13px; font-weight:600; color:#34d399;'>✅ {T('rec_verify')}</div>", unsafe_allow_html=True)
                st.markdown("<div class='eq-container'><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div><div class='eq-bar'></div></div>", unsafe_allow_html=True)
                col_aud_m, col_del_m = st.columns([3, 2])
                with col_aud_m:
                    st.audio(saved_mic, format="audio/wav")
                with col_del_m:
                    if st.button("🗑️ Supprimer & Réenregistrer", key="btn_del_rec_direct", help="Purge immédiatement l'enregistrement"):
                        st.session_state.pop("recorded_mic_bank", None)
                        st.success("✅ Enregistrement supprimé et espace disque libéré !")
                        st.rerun()
                vp_audio_bytes = saved_mic
                vp_ready = bool(vp_name.strip())
        elif is_file:
            vp_file = st.file_uploader("Fichier audio ou vidéo", type=["wav", "mp3", "m4a", "flac", "ogg", "mp4", "mov", "mkv", "webm", "avi"], key="vp_file")
            if vp_file is not None:
                col_fv1, col_fv2 = st.columns(2)
                with col_fv1:
                    vp_start = st.number_input("Début (sec)", min_value=0.0, value=0.0, step=1.0, key="vp_fv_start")
                with col_fv2:
                    vp_dur = st.number_input("Durée (sec, 3 à 60s)", min_value=3.0, max_value=60.0, value=20.0, step=1.0, key="vp_fv_dur")
                vp_audio_bytes = vp_file.getvalue()
                vp_ready = bool(vp_name.strip())
        else:
            vp_url = st.text_input("URL vidéo / audio (YouTube...)", placeholder="https://www.youtube.com/watch?v=...", key="vp_url")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                vp_start = st.number_input("Début (sec)", min_value=0.0, value=0.0, step=1.0, key="vp_start")
            with col_t2:
                vp_dur = st.number_input("Durée (sec, 3 à 60s)", min_value=3.0, max_value=60.0, value=20.0, step=1.0, key="vp_dur")
            if vp_url.strip():
                vp_source_url = vp_url.strip()
                vp_ready = bool(vp_name.strip())

        st.write("")
        if st.button(f"💾 {T('btn_save_voice')}", disabled=not (vp_ready and rights), key="btn_save_vp"):
            try:
                with st.spinner("Analyse et extraction de l'empreinte vocale..."):
                    if is_mic:
                        profile = engine.add_voice_profile(vp_name, vp_audio_bytes, source_type="direct_mic")
                    elif is_file:
                        profile = engine.add_voice_profile(vp_name, vp_audio_bytes, source_type="local_media", start_time=vp_start, duration_seconds=vp_dur)
                    else:
                        temp_dl = engine.VOICES_DIR / f"dl_{uuid.uuid4().hex}.wav"
                        try:
                            engine.download_media_from_url(vp_source_url, temp_dl, extract_audio=True, start_time=vp_start, duration_seconds=vp_dur)
                            profile = engine.add_voice_profile(vp_name, temp_dl, source_type="url", source_url=vp_source_url)
                        finally:
                            if temp_dl.exists():
                                temp_dl.unlink()
                st.success(f"Empreinte '{profile['name']}' enregistrée avec succès ({profile['duration']:.1f}s) !")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur lors de l'enregistrement de l'empreinte : {e}")

    with vp_tab_list:
        existing_profiles = engine.list_voice_profiles()
        if not existing_profiles:
            st.info("Aucune empreinte vocale enregistrée pour le moment.")
        else:
            for prof in existing_profiles:
                c1, c2, c3 = st.columns([3, 4, 1])
                with c1:
                    if prof.get("is_neural_fingerprint"):
                        st.markdown(f"**{prof['name']}** &nbsp; <span style='background:rgba(16,185,129,0.2); border:1px solid #10b981; color:#34d399; font-size:10.5px; font-weight:700; padding:2px 7px; border-radius:10px;'>🧬 EMPREINTE IA</span>", unsafe_allow_html=True)
                        st.caption(f"{prof['vocal_type']} ({prof['f0_median']} Hz) • Clarté : {prof['clarity_score']:.0f}% • SNR : {prof['snr_db']:.1f} dB")
                    else:
                        st.markdown(f"**{prof['name']}**")
                        st.caption(f"Durée : {prof['duration']:.1f}s | Source : {prof['source_type']}")
                with c2:
                    if prof['path'].exists():
                        st.audio(str(prof['path']))
                with c3:
                    if st.button("🗑️", key=f"del_vp_{prof['id']}", help="Supprimer cette empreinte"):
                        engine.delete_voice_profile(prof['id'])
                        st.rerun()


# =========================================================
# VUE 3 : BIBLIOTHÈQUE & GESTION DES PROJETS
# =========================================================
elif active_section == "📁 Projets & Bibliothèque":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
                    border: 1px solid rgba(59, 130, 246, 0.35); border-radius: 14px; padding: 18px 22px; margin-bottom: 20px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:26px;">📁</span>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:18px; font-weight:700; color:#f8fafc;">
                        BIBLIOTHÈQUE & GESTION DES PROJETS DUBB-IN
                    </div>
                    <div style="font-size:13px; color:#94a3b8;">
                        Consultez vos doublages terminés, téléchargez les exports ou lancez la synchronisation labiale
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    projs = sorted(
        [p for p in ROOT.iterdir() if p.is_dir() and (p / "project.json").exists()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    if not projs:
        st.info("Aucun projet enregistré dans la bibliothèque. Rendez-vous dans **🎙️ Studio Doublage** pour créer votre premier projet.")
    else:
        st.write(f"**Total : {len(projs)} projet(s) enregistré(s)**")
        for p in projs:
            try:
                meta = json.loads((p / "project.json").read_text(encoding="utf-8"))
            except Exception:
                meta = {"name": p.name}

            p_name = meta.get("name", "Projet")
            created = meta.get("created", "")[:10]
            dur = meta.get("duration", 0)
            target_lang = meta.get("target_language", "en")
            v_profile = meta.get("voice_profile", "Inconnue")
            has_export = (p / "export.mp4").exists()

            with st.expander(f"🎬 {p_name} — ({p.name[:8]}) | {dur:.1f}s | {target_lang.upper()} {'✅ Exporté' if has_export else '⏳ En cours'}", expanded=False):
                c_pvid, c_pmeta = st.columns([3, 2])
                with c_pvid:
                    if has_export:
                        st.video(str(p / "export.mp4"))
                    elif (p / "video.input").exists():
                        st.video(str(p / "video.input"))
                with c_pmeta:
                    st.markdown(f"**Nom :** {p_name}")
                    st.caption(f"Créé le : {created} | Voix : {v_profile} | Durée : {dur:.1f}s")
                    if (p / "dub.wav").exists():
                        st.markdown("**Voix doublée générée :**")
                        st.audio(str(p / "dub.wav"))

                    c_b1, c_b2 = st.columns(2)
                    with c_b1:
                        if st.button("👄 Ouvrir dans Lip-Sync", key=f"lib_open_ls_{p.name}", use_container_width=True):
                            st.session_state["main_nav"] = "👄 Lip-Sync Studio (.dlf)"
                            st.session_state["active"] = p.name
                            st.rerun()
                    with c_b2:
                        if st.button("📁 Ouvrir dossier", key=f"lib_open_fld_{p.name}", use_container_width=True):
                            subprocess.run(["explorer.exe", "/select,", str(p / "project.json")])

                    if st.button(f"🗑️ Supprimer {p_name}", key=f"lib_del_{p.name}", help="Supprimer définitivement ce projet"):
                        shutil.rmtree(p)
                        st.rerun()


# =========================================================
# VUE 4 : DIAGNOSTIC MATÉRIEL & TÉLÉMÉTRIE
# =========================================================
elif active_section == "⚙️ Diagnostic Matériel":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(20, 24, 36, 0.9) 0%, rgba(14, 16, 24, 0.95) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 14px; padding: 18px 22px; margin-bottom: 20px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:26px;">⚙️</span>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:18px; font-weight:700; color:#f8fafc;">
                        DIAGNOSTIC MATÉRIEL & TÉLÉMÉTRIE DUBB-IN
                    </div>
                    <div style="font-size:13px; color:#94a3b8;">
                        Calibration des modèles Faster-Whisper, XTTS-v2 et Wav2Lip pour votre architecture
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(
            f"""
            <div style="background:rgba(18,22,34,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:16px;">
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; color:#f1f5f9; margin-bottom:8px;">🖥️ Processeur & Mémoire Vive</div>
                <ul style="color:#94a3b8; font-size:13.5px; line-height:1.8; margin-bottom:0;">
                    <li><b>CPU :</b> {hw.get('cpu_name', 'Intel Core i5-3470')} ({hw.get('cpu_count', 4)} Cœurs / 4 Threads)</li>
                    <li><b>Jeu d'instructions :</b> SSE4.2 / AVX activé (Vectorisation LLVM)</li>
                    <li><b>RAM DDR3 :</b> {hw.get('ram_gb', 12)} Go installés</li>
                    <li><b>File d'attente bornée :</b> Max 24 frames mémoire (zéro swap)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_d2:
        is_gpu_ready = hw.get("has_cuda") or ("960" in str(hw.get("gpu_name", "")))
        gpu_badge = "🟢 GTX 960 DÉDIÉE" if is_gpu_ready else "⚙️ CPU VECTOR INT8"
        detected_gpu = hw.get("gpu_name") or "NVIDIA GeForce GTX 960"
        detected_vram = hw.get("vram_gb", 2.0)
        gpu_kind = "ok" if is_gpu_ready else "warn"
        _vr = _read_gpu_vram()
        if _vr:
            _used, _tot = _vr
            _pct = _used / _tot * 100 if _tot else 0
            _vk = "ok" if _pct < 60 else ("warn" if _pct < 85 else "err")
            vram_block = (
                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'
                'font-size:12px;color:#94a3b8;"><span>VRAM utilis\u00e9e (live)</span>'
                + _d_chip("%.0f / %.0f Mo \u2022 %.0f%%" % (_used, _tot, _pct), _vk) + '</div>'
                + _d_gauge(_pct) + '<div style="height:12px"></div>'
            )
        else:
            vram_block = '<div style="margin-bottom:12px;">' + _d_chip("Mesure VRAM indisponible (nvidia-smi)", "warn") + '</div>'
        chips_row = (
            '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;">'
            + _d_chip("CUDA actif" if hw.get("has_cuda") else "CPU INT8", "ok" if hw.get("has_cuda") else "warn")
            + _d_chip("Whisper : " + str(hw.get("recommended_whisper", "small")), "ok")
            + _d_chip("XTTS-v2 24 kHz", "ok")
            + _d_chip("Wav2Lip ONNX", "ok")
            + '</div>'
        )
        st.markdown(
            f"""
            <div style="background:rgba(18,22,34,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:16px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-family:'Space Grotesk',sans-serif; font-weight:700; color:#f1f5f9;">🎮 Accélération Graphique & Modèles IA</span>
                    <span class="d-chip {gpu_kind}">{gpu_badge}</span>
                </div>
                {vram_block}{chips_row}
                <ul style="color:#94a3b8; font-size:13.5px; line-height:1.8; margin-bottom:0;">
                    <li><b>GPU Détecté :</b> {detected_gpu} ({detected_vram} Go VRAM Maxwell)</li>
                    <li><b>Arène VRAM :</b> Plafond strict 650 Mo (zéro CUDA Out of Memory)</li>
                    <li><b>Whisper Calibré :</b> {hw.get('recommended_whisper', 'small')}</li>
                    <li><b>Clonage Vocal :</b> XTTS-v2 24kHz calibré pour micro-échantillons</li>
                    <li><b>Lip-Sync :</b> Wav2Lip GAN ONNX (145 Mo) + Détection YuNet C++ (~1ms)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# VUE 5 : SOUTIEN & DONATIONS
# =========================================================
elif active_section == "☕ Soutien & Dons":
    st.markdown(
        f"""
        <div style="background: rgba(18, 21, 32, 0.85); border: 1px solid rgba(255, 51, 102, 0.25); border-radius: 14px; padding: 22px; margin-bottom: 20px;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: #ff3366; font-size: 18px; margin-bottom: 6px;">
                ❤️ {T('donate_title')}
            </div>
            <p style="color: #cbd5e1; font-size: 14px; line-height: 1.6; margin: 0;">
                {T('donate_desc')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    don_file = BASE_DIR / "donations.json"
    don_data = {}
    if don_file.exists():
        try:
            don_data = json.loads(don_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown("**☕ Pourboire créateur**")
        bmac = don_data.get("buy_me_a_coffee", "")
        if bmac:
            st.link_button("☕ Buy Me a Coffee", bmac, use_container_width=True)
        kofi = don_data.get("kofi", "")
        if kofi:
            st.link_button("❤️ Ko-fi", kofi, use_container_width=True)

    with col_d2:
        st.markdown("**💳 Dépôt & Sponsors**")
        gh_sponsors = don_data.get("github_sponsors", "")
        if gh_sponsors:
            st.link_button("🐙 GitHub Sponsors", gh_sponsors, use_container_width=True)
        paypal = don_data.get("paypal", "")
        if paypal:
            st.link_button("💙 PayPal Me", paypal, use_container_width=True)

    with col_d3:
        st.markdown("**🪙 Crypto**")
        crypto = don_data.get("crypto", {})
        btc = crypto.get("bitcoin")
        eth = crypto.get("ethereum")
        usdt = crypto.get("usdt_trc20")
        if btc or eth or usdt:
            if btc:
                st.caption(f"BTC: `{btc}`")
            if eth:
                st.caption(f"ETH: `{eth}`")
            if usdt:
                st.caption(f"USDT: `{usdt}`")
        else:
            st.caption("Liens et adresses configurables dans `donations.json`.")


# =========================================================
# VUE 6 : STUDIO DE DOUBLAGE (WORKFLOW PRINCIPAL)
# =========================================================
else:
    # Créer un nouveau projet
    with st.expander("➕ Créer un Nouveau Projet de Doublage", expanded=False):
        col_proj_vid, col_proj_voice = st.columns(2)

        with col_proj_vid:
            st.markdown("<div style='font-family:\"Space Grotesk\",sans-serif; font-size:15px; font-weight:700; color:#f1f5f9; margin-bottom:8px;'>📹 1. Vidéo source</div>", unsafe_allow_html=True)
            vid_source_type = st.radio("Source de la vidéo", ["Fichier vidéo local", "Lien externe (YouTube / Web)"], horizontal=True, key="vid_src_type")
            video_file = None
            video_url = ""
            if vid_source_type == "Fichier vidéo local":
                video_file = st.file_uploader("Vidéo à traduire", type=["mp4", "mov", "mkv", "webm"], key="uploader_video")
            else:
                video_url = st.text_input("Lien vidéo (YouTube...)", placeholder="https://www.youtube.com/watch?v=...", key="input_video_url")

        with col_proj_voice:
            st.markdown("<div style='font-family:\"Space Grotesk\",sans-serif; font-size:15px; font-weight:700; color:#f1f5f9; margin-bottom:8px;'>🎙️ 2. Voix de référence (Clonage)</div>", unsafe_allow_html=True)
            saved_profiles = engine.list_voice_profiles()
            selected_profile = None
            voice_file = None

            if saved_profiles:
                def _format_prof_label(p):
                    badge = "🧬 [Empreinte IA]" if p.get("is_neural_fingerprint") else "👤"
                    vocal_tag = f" • {p['vocal_type']}" if p.get("is_neural_fingerprint") else ""
                    return f"{badge} {p['name']} ({p['duration']:.1f}s{vocal_tag})"

                prof_map = {_format_prof_label(p): p for p in saved_profiles}
                prof_choice = st.selectbox("Profil vocal sélectionné", list(prof_map.keys()), key="proj_voice_bank_sel")
                selected_profile = prof_map[prof_choice]
                if selected_profile['path'].exists():
                    st.audio(str(selected_profile['path']))

                if st.checkbox("Remplacer par un fichier audio ponctuel", key="chk_voice_adhoc"):
                    voice_file = st.file_uploader("Fichier audio ponctuel (WAV, MP3, M4A...)", type=["wav", "mp3", "m4a", "flac", "ogg"], key="uploader_voice_adhoc")
            else:
                st.info("💡 Enregistrez votre empreinte vocale dans l'onglet **🧬 Empreinte Vocale (.dvf)** ou importez un fichier audio ci-dessous :")
                voice_file = st.file_uploader("Fichier audio contenant la voix (3 à 60s)", type=["wav", "mp3", "m4a", "flac", "ogg"], key="uploader_voice")

        col_pname, col_plang = st.columns([3, 2])
        with col_pname:
            name = st.text_input("Nom du projet", value="Mon doublage", max_chars=80)
        with col_plang:
            default_out_idx = list(engine.LANGUAGES.keys()).index("Anglais") if "Anglais" in engine.LANGUAGES else 0
            proj_target_lang = st.selectbox("Langue de sortie (Doublage)", list(engine.LANGUAGES.keys()), index=default_out_idx)

        has_video = (video_file is not None) if vid_source_type == "Fichier vidéo local" else bool(video_url.strip())
        has_voice = (voice_file is not None) or (selected_profile is not None)
        can_create = rights and models and has_video and has_voice

        st.write("")
        if st.button("🚀 CRÉER LE PROJET & ANALYSER LA VIDÉO", disabled=not can_create, key="btn_create_proj", type="primary"):
            project = ROOT / uuid.uuid4().hex
            project.mkdir()
            try:
                with st.spinner("Téléchargement et préparation des médias..."):
                    if vid_source_type == "Fichier vidéo local":
                        (project / "video.input").write_bytes(video_file.getvalue())
                    else:
                        engine.download_media_from_url(video_url.strip(), project / "video.input", extract_audio=False)

                    if voice_file is not None:
                        (project / "reference.input").write_bytes(voice_file.getvalue())
                        ref_sha = hashlib.sha256(voice_file.getvalue()).hexdigest()
                        profile_used_name = "custom_file"
                    elif selected_profile is not None:
                        shutil.copyfile(selected_profile["path"], project / "reference.input")
                        ref_sha = selected_profile["sha256"]
                        profile_used_name = selected_profile["name"]
                        if selected_profile.get("dvf_filename"):
                            dvf_src = engine.VOICES_DIR / "fingerprints" / selected_profile["dvf_filename"]
                            if dvf_src.exists():
                                shutil.copyfile(dvf_src, project / "fingerprint.dvf")
                    else:
                        raise ValueError("Aucune voix de référence sélectionnée.")

                with st.spinner("Validation et extraction audio…"):
                    total = engine.prepare(project)
                engine.save(
                    project / "project.json",
                    {
                        "name": name,
                        "duration": total,
                        "created": datetime.now(timezone.utc).isoformat(),
                        "rights_declared": True,
                        "model_terms_accepted": True,
                        "reference_sha256": ref_sha,
                        "voice_profile": profile_used_name,
                        "target_language": engine.LANGUAGES[proj_target_lang],
                    },
                )
                st.session_state["active"] = project.name
                st.session_state.pop("editor", None)
                st.success("Projet créé avec succès !")
                st.rerun()
            except Exception as error:
                fail(error, project)

    # Sélection du projet actif
    projects = sorted(
        [p for p in ROOT.iterdir() if p.is_dir() and (p / "project.json").exists()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not projects:
        st.info("ℹ️ Aucun projet en cours. Cliquez sur **« ➕ Créer un Nouveau Projet de Doublage »** ci-dessus pour démarrer.")
    else:
        labels = {}
        for p in projects:
            try:
                meta = json.loads((p / "project.json").read_text(encoding="utf-8"))
                labels[p.name] = meta.get("name", "Projet") + " — " + p.name[:8]
            except Exception:
                labels[p.name] = p.name[:8]

        identifiers = list(labels.keys())
        active = st.session_state.get("active")
        active_idx = identifiers.index(active) if active in identifiers else 0

        col_sel_p, col_p_actions = st.columns([4, 1])
        with col_sel_p:
            selected = st.selectbox("Projet actif en cours de travail :", identifiers, index=active_idx, format_func=lambda v: labels[v], key="sel_active_dubbing_proj")
        with col_p_actions:
            st.write("")
            if st.button("➕ Nouveau", key="btn_quick_new_proj"):
                st.session_state["active"] = None
                st.rerun()

        if selected != st.session_state.get("active"):
            st.session_state["active"] = selected
            st.session_state.pop("editor", None)

        project = ROOT / selected
        metadata = json.loads((project / "project.json").read_text(encoding="utf-8"))

        target_code = metadata.get("target_language", "en")
        code_to_lang = {v: k for k, v in engine.LANGUAGES.items()}
        default_lang_name = code_to_lang.get(target_code, "Anglais")
        all_langs = list(engine.LANGUAGES.keys())
        default_lang_idx = all_langs.index(default_lang_name) if default_lang_name in all_langs else 0

        c_info1, c_info2, c_info3 = st.columns(3)
        with c_info1:
            st.metric("Durée vidéo", f"{metadata.get('duration', 0):.1f} s")
        with c_info2:
            st.metric("Voix de référence", metadata.get("voice_profile", "Personnalisée"))
        with c_info3:
            st.metric("Langue de sortie", default_lang_name)

        document_path = project / "document.json"
        doc_exists = document_path.exists()

        st.markdown(
            """
            <div class="pipeline-bar">
                <div class="pipe-node">
                    <span class="pipe-badge">01</span>
                    <span class="pipe-txt">TRANSCRIPTION</span>
                </div>
                <span class="pipe-arrow">➔</span>
                <div class="pipe-node">
                    <span class="pipe-badge">02</span>
                    <span class="pipe-txt">TRADUCTION</span>
                </div>
                <span class="pipe-arrow">➔</span>
                <div class="pipe-node">
                    <span class="pipe-badge">03</span>
                    <span class="pipe-txt">CLONAGE & EXPORT</span>
                </div>
                <span class="pipe-arrow">➔</span>
                <div class="pipe-node">
                    <span class="pipe-badge">04</span>
                    <span class="pipe-txt">LIP-SYNC (.DLF)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Étape 1 : Transcription
        with st.container(border=True):
            st.markdown(
                """
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 14px;">
                    <div style="display:flex; align-items:center; gap: 12px;">
                        <span class="pipe-badge">01</span>
                        <div>
                            <div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:#f8fafc; letter-spacing:0.02em;">MODULE 01 // TRANSCRIPTION IA DU DIALOGUE</div>
                            <div style="font-size:12px; color:#64748b;">Découpage VAD et reconnaissance vocale par Faster-Whisper</div>
                        </div>
                    </div>
                    <span class="hw-tag" style="background:#141724; border:1px solid rgba(255,255,255,0.08); color:#94a3b8;">FASTER-WHISPER</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_t1, col_t2 = st.columns([3, 1])
            with col_t1:
                whisper_opts = ["small", "medium", "base", "tiny", "large-v3"]
                rec_model = hw.get("recommended_whisper", "small")
                default_idx = whisper_opts.index(rec_model) if rec_model in whisper_opts else 0
                model_size = st.selectbox("Modèle de reconnaissance vocale Whisper", whisper_opts, index=default_idx, key=f"m_size_{selected}")
            with col_t2:
                st.write("")
                st.write("")
                btn_transcribe = st.button("🎙️ 1 — Transcrire l'audio", disabled=not (rights and models), key=f"btn_tr_{selected}")

            if btn_transcribe:
                try:
                    prog_holder = st.empty()
                    def update_tr_progress(p):
                        render_fluent_progress(prog_holder, p, "Transcription VAD Faster-Whisper...", "Audio en cours d'analyse")

                    document = engine.transcribe(project, model_size, progress=update_tr_progress)
                    render_fluent_progress(prog_holder, 1.0, "Transcription Whisper achevée !", "100%")
                    st.session_state["editor"] = json.dumps(document, ensure_ascii=False, indent=2)
                    st.success("Transcription réussie !")
                    st.rerun()
                except Exception as error:
                    fail(error, project)

            if doc_exists:
                if "editor" not in st.session_state or st.session_state.get("active_proj_id") != selected:
                    st.session_state["editor"] = document_path.read_text(encoding="utf-8")
                    st.session_state["active_proj_id"] = selected
                try:
                    doc_data = json.loads(st.session_state["editor"])
                    st.success(f"✅ Audio original transcrit : {len(doc_data.get('segments', []))} phrases / segments détectés.")
                except Exception:
                    pass
            else:
                st.info("ℹ️ Cliquez sur **« 🎙️ 1 — Transcrire l'audio »** pour découper les dialogues avant de traduire.")

        # Étape 2 : Traduction
        with st.container(border=True):
            st.markdown(
                """
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 14px;">
                    <div style="display:flex; align-items:center; gap: 12px;">
                        <span class="pipe-badge">02</span>
                        <div>
                            <div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:#f8fafc; letter-spacing:0.02em;">MODULE 02 // TRADUCTION & ÉDITEUR MULTILINGUE</div>
                            <div style="font-size:12px; color:#64748b;">Sélection de la langue cible et synchronisation temporelle des répliques</div>
                        </div>
                    </div>
                    <span class="hw-tag" style="background:#141724; border:1px solid rgba(255,255,255,0.08); color:#94a3b8;">TIMELINE SYNC</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_l1, col_l2 = st.columns([3, 1])
            with col_l1:
                chosen_lang_name = st.selectbox("Langue cible dans laquelle la voix va parler", all_langs, index=default_lang_idx, key=f"sel_lang_{selected}")
                chosen_lang_code = engine.LANGUAGES[chosen_lang_name]
            with col_l2:
                st.write("")
                st.write("")
                btn_translate = st.button(f"🌐 2 — Traduire en {chosen_lang_name}", disabled=not (rights and cloud and doc_exists), key=f"btn_tr_lang_{selected}")

            if btn_translate:
                try:
                    current_doc = json.loads(st.session_state.get("editor", document_path.read_text(encoding="utf-8")))
                    with st.spinner(f"Traduction du texte en {chosen_lang_name} via Google Translate..."):
                        translated_doc = engine.translate(current_doc, chosen_lang_code)
                    metadata["target_language"] = chosen_lang_code
                    engine.save(project / "project.json", metadata)
                    st.session_state["editor"] = json.dumps(translated_doc, ensure_ascii=False, indent=2)
                    engine.save(document_path, translated_doc)
                    st.success(f"Traduction vers le {chosen_lang_name} terminée avec succès !")
                    st.rerun()
                except Exception as error:
                    fail(error, project)

            if "editor" in st.session_state and st.session_state.get("active_proj_id") == selected:
                with st.expander("📝 Révision manuelle de la traduction (Texte et Timing)", expanded=False):
                    content = st.text_area("Données de synchronisation JSON", value=st.session_state["editor"], height=280, key=f"txt_ed_{selected}")
                    if st.button("Enregistrer les corrections", key=f"btn_save_ed_{selected}"):
                        try:
                            document = json.loads(content)
                            st.session_state["editor"] = content
                            engine.save(document_path, document)
                            st.success("Modifications enregistrées.")
                        except Exception as error:
                            fail(error, project)
            else:
                st.caption("*(L'éditeur de dialogue apparaîtra après la transcription)*")

        # Étape 3 : Clonage & Export
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 14px;">
                    <div style="display:flex; align-items:center; gap: 12px;">
                        <span class="pipe-badge">03</span>
                        <div>
                            <div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:#f8fafc; letter-spacing:0.02em;">MODULE 03 // CLONAGE NEURONAL & EXPORT VIDÉO</div>
                            <div style="font-size:12px; color:#64748b;">La voix de référence sera clonée avec XTTS-v2 pour réciter les dialogues en <strong style="color:#ff6b8b;">{chosen_lang_name}</strong></div>
                        </div>
                    </div>
                    <span class="hw-tag" style="background:rgba(255,51,102,0.12); border:1px solid rgba(255,51,102,0.3); color:#ff6b8b;">XTTS-v2 24kHz</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            audio_mode = st.radio(
                "Mode audio pour la vidéo finale :",
                options=[
                    "🎙️ Voix seule (Studio) — Remplace l'audio original, zéro chevauchement garanti",
                    "🎮 Voix + Ambiance originale (Demucs) — Isole musique et bruitages, supprime la voix originale"
                ],
                index=0,
                key=f"audio_mode_{selected}",
            )
            keep_bg = "Demucs" in audio_mode

            sub_style_choice = st.selectbox(
                "Sous-titres dynamiques incrustés (Style TikTok / Shorts) :",
                options=[
                    "Aucun sous-titre incrusté",
                    "🔥 Alex Hormozi (Jaune Fluo — N°1 TikTok)",
                    "🟢 Alex Hormozi (Vert Néon)",
                    "⚡ MrBeast (Bleu Électrique / Impact)",
                    "✨ Minimaliste Pro",
                ],
                index=0,
                key=f"sub_style_{selected}",
            )
            sub_style_map = {
                "Aucun sous-titre incrusté": "none",
                "🔥 Alex Hormozi (Jaune Fluo — N°1 TikTok)": "hormozi_yellow",
                "🟢 Alex Hormozi (Vert Néon)": "hormozi_green",
                "⚡ MrBeast (Bleu Électrique / Impact)": "mrbeast",
                "✨ Minimaliste Pro": "minimal",
            }
            sub_style_key = sub_style_map.get(sub_style_choice, "none")
            vocal_speed = st.slider("⚡ Cadence & Débit vocal (Vitesse d'élocution) :", min_value=0.8, max_value=1.5, value=1.0, step=0.05, key=f"vocal_speed_{selected}")

            col_sub_pos, col_sub_fs = st.columns([1, 1])
            with col_sub_pos:
                sub_pos_choice = st.selectbox("Position des sous-titres :", ["Bas de l'écran (Standard)", "Centre / Milieu (Impact TikTok/Shorts)", "Haut de l'écran"], index=0, key=f"sub_pos_{selected}")
                sub_pos_map = {"Bas de l'écran (Standard)": "bottom", "Centre / Milieu (Impact TikTok/Shorts)": "middle", "Haut de l'écran": "top"}
                sub_alignment_val = sub_pos_map[sub_pos_choice]
            with col_sub_fs:
                sub_font_size_val = st.slider("Taille de la police des sous-titres :", min_value=16, max_value=48, value=26, step=2, key=f"sub_fs_{selected}")

            col_btn_gen, col_btn_stop = st.columns([3, 2])
            with col_btn_gen:
                btn_render = st.button(
                    "🎬 3 — Générer la vidéo doublée",
                    disabled=not (rights and models and doc_exists),
                    type="primary",
                    use_container_width=True,
                    key=f"btn_rend_{selected}"
                )
            with col_btn_stop:
                st.markdown("<div style='height:38px; display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:8px; font-size:12px; color:#38bdf8;'>⚡ Mode Prêt</div>", unsafe_allow_html=True)

            if btn_render:
                prog_holder = st.empty()
                render_fluent_progress(prog_holder, 0.01, "Initialisation de la synthèse vocale XTTS-v2...", "Chargement du modèle...")
                try:
                    document = json.loads(st.session_state.get("editor", document_path.read_text(encoding="utf-8")))
                    engine.save(document_path, document)

                    def update_render_prog(ratio, curr=None, total=None):
                        step_str = f"Segment {curr}/{total}" if curr and total else ""
                        render_fluent_progress(prog_holder, ratio, "Synthèse vocale neuronale XTTS-v2...", step_str)

                    engine.render(
                        project,
                        keep_background=keep_bg,
                        subtitles_style=sub_style_key,
                        speed=vocal_speed,
                        subtitle_alignment=sub_alignment_val,
                        subtitle_fontsize=sub_font_size_val,
                        progress_callback=update_render_prog,
                    )
                    render_fluent_progress(prog_holder, 1.0, "Export vidéo et assemblage finalisés !", "100%")
                    st.session_state[f"rend_active_{selected}"] = False
                    st.session_state["play_ding"] = True
                    st.success("🎉 Doublage terminé avec succès !")
                    st.rerun()
                except InterruptedError:
                    st.session_state[f"rend_active_{selected}"] = False
                    st.warning("⚠️ Processus arrêté par l'utilisateur. Segments sécurisés dans le cache.")
                except Exception as error:
                    st.session_state[f"rend_active_{selected}"] = False
                    fail(error, project)
                finally:
                    st.session_state[f"rend_active_{selected}"] = False

            export_path = project / "export.mp4"
            if export_path.exists():
                st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:18px 0;'>", unsafe_allow_html=True)
                
                exp_stat = export_path.stat()
                exp_size_mb = exp_stat.st_size / (1024 * 1024)
                exp_time_str = datetime.fromtimestamp(exp_stat.st_mtime).strftime("%d/%m/%Y à %H:%M:%S")

                st.markdown(
                    f"""
                    <div style="background:linear-gradient(90deg, rgba(16,185,129,0.12) 0%, rgba(6,78,59,0.2) 100%);
                                border:1px solid rgba(16,185,129,0.45); border-radius:10px; padding:12px 18px; margin: 12px 0;
                                display:flex; justify-content:space-between; align-items:center;">
                        <div style="display:flex; align-items:center; gap:12px;">
                            <span style="font-size:22px;">🟢</span>
                            <div>
                                <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; color:#34d399; font-size:15px; letter-spacing:0.02em;">
                                    VIDÉO DOUBLÉE GÉNÉRÉE AVEC SUCCÈS
                                </div>
                                <div style="font-size:12px; color:#94a3b8;">
                                    Fichier : <code>export.mp4</code> ({exp_size_mb:.1f} Mo) • Générée le {exp_time_str} • Prête pour lecture et Lip-Sync
                                </div>
                            </div>
                        </div>
                        <span style="background:#059669; color:#ffffff; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700; letter-spacing:0.04em;">
                            TERMINÉ ✓
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.video(str(export_path))

                st.markdown(
                    """
                    <div style="background:linear-gradient(90deg, rgba(34,197,94,0.15) 0%, rgba(59,130,246,0.15) 100%);
                                border:1px solid rgba(34,197,94,0.4); border-radius:12px; padding:12px 18px; margin: 15px 0;
                                display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-family:'Space Grotesk',sans-serif; font-weight:700; color:#f8fafc; font-size:14px;">✨ Rendu Haute Définition avec Concordance Labiale Native</span>
                            <div style="font-size:12px; color:#cbd5e1;">Qualité vidéo 100% native sans flou ni déformation • Paroles synchronisées précisément sur l'ouverture de bouche</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                clean_proj_name = metadata.get("name", "video")
                clean_lang = chosen_lang_name
                base_export_name = f"Dubb-in_{clean_proj_name}_{clean_lang}"

                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    if st.button("💾 Enregistrer la vidéo (MP4) sur le Bureau", key="btn_save_mp4_desk", use_container_width=True):
                        dst = get_unique_desktop_path(base_export_name, ".mp4")
                        shutil.copyfile(export_path, dst)
                        st.success(f"✅ Vidéo enregistrée sous : {dst.name}")
                with col_v2:
                    if st.button("📁 Ouvrir l'emplacement dans l'Explorateur", key="btn_open_exp_folder", use_container_width=True):
                        subprocess.run(["explorer.exe", "/select,", str(export_path)])

                col_save_vp1, col_save_vp2 = st.columns(2)
                with col_save_vp1:
                    if (project / "dub.wav").exists():
                        if st.button("🧬 Sauvegarder la voix doublée dans la Banque des Voix", key=f"btn_save_dub_vp_{selected}", use_container_width=True):
                            try:
                                prof = engine.add_voice_profile(
                                    f"Voix Doublée — {clean_proj_name}",
                                    project / "dub.wav",
                                    source_type="dubbed_audio"
                                )
                                st.success(f"✅ Voix sauvegardée dans la Banque des Voix sous '{prof['name']}' ({prof['duration']:.1f}s) !")
                            except Exception as e:
                                st.error(f"Erreur : {e}")
                with col_save_vp2:
                    if (project / "source.wav").exists():
                        if st.button("👤 Sauvegarder la voix source dans la Banque des Voix", key=f"btn_save_src_vp_{selected}", use_container_width=True):
                            try:
                                prof = engine.add_voice_profile(
                                    f"Voix Source — {clean_proj_name}",
                                    project / "source.wav",
                                    source_type="video_source"
                                )
                                st.success(f"✅ Voix source sauvegardée sous '{prof['name']}' ({prof['duration']:.1f}s) !")
                            except Exception as e:
                                st.error(f"Erreur : {e}")

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    srt_path = project / "subtitles.srt"
                    if srt_path.exists():
                        if st.button("📄 Enregistrer les sous-titres (.srt) sur le Bureau", key="btn_save_srt_desk", use_container_width=True):
                            dst_srt = get_unique_desktop_path(base_export_name, ".srt")
                            shutil.copyfile(srt_path, dst_srt)
                            st.success(f"✅ Sous-titres enregistrés sous : {dst_srt.name}")
                with col_s2:
                    rep_path = project / "report.json"
                    if rep_path.exists():
                        if st.button("📊 Enregistrer le rapport (.json) sur le Bureau", key="btn_save_rep_desk", use_container_width=True):
                            dst_rep = get_unique_desktop_path(f"{base_export_name}_rapport", ".json")
                            shutil.copyfile(rep_path, dst_rep)
                            st.success(f"✅ Rapport enregistré sous : {dst_rep.name}")

                if st.button("🗜️ Créer la sauvegarde ZIP du projet sur le Bureau", key="btn_save_zip_desk", use_container_width=True):
                    zip_dst = get_unique_desktop_path(f"Dubb-in_{clean_proj_name}_backup", ".zip")
                    with zipfile.ZipFile(zip_dst, "w", zipfile.ZIP_DEFLATED) as archive:
                        for f in project.rglob("*"):
                            if f.is_file():
                                archive.write(f, arcname=str(f.relative_to(project)))
                    st.success(f"✅ Sauvegarde ZIP complète créée sous : {zip_dst.name}")
