"""
Dubb-in AI Studio — Lip-Sync Studio UI Module
Ultra-fast Lip-Sync Studio with .dlf facial geometry acceleration.
Zero PyTorch • Numba AVX Blending • ONNX Runtime GPU micro-batching.
"""

import os
import time
import json
import shutil
from pathlib import Path
import streamlit as st
import numpy as np

import lipsync
from lipsync import CONFIG, DLFContainer, download_wav2lip_model, run_lipsync_process


def render_lipsync_studio(ROOT: Path, hw: dict, T=None):
    """Renders the autonomous Lip-Sync Studio view."""
    if T is None:
        T = lambda k, **kw: k

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(255, 51, 102, 0.12) 0%, rgba(139, 92, 246, 0.15) 100%);
                    border: 1px solid rgba(255, 51, 102, 0.35); border-radius: 14px; padding: 18px 22px; margin-bottom: 20px;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                <span style="font-size:28px;">👄</span>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:19px; font-weight:800; color:#f8fafc; letter-spacing:0.02em;">
                        STUDIO LIP-SYNC HAUTE PERFORMANCE (.DLF)
                    </div>
                    <div style="font-size:13px; color:#cbd5e1;">
                        Synchronisation labiale asymétrique 0 PyTorch • Moteur ONNX Runtime & Fusion Numba AVX LLVM
                    </div>
                </div>
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:10px;">
                <span class="hw-tag" style="background:rgba(255,51,102,0.18); border:1px solid rgba(255,51,102,0.4); color:#ff6b8b;">Wav2Lip GAN ONNX (145 Mo)</span>
                <span class="hw-tag" style="background:rgba(139,92,246,0.18); border:1px solid rgba(139,92,246,0.4); color:#c4b5fd;">Accélération .dlf 3x</span>
                <span class="hw-tag" style="background:rgba(16,185,129,0.18); border:1px solid rgba(16,185,129,0.4); color:#34d399;">Numba AVX LLVM Fusion</span>
                <span class="hw-tag" style="background:rgba(59,130,246,0.18); border:1px solid rgba(59,130,246,0.4); color:#60a5fa;">YuNet C++ Tracking (~1ms)</span>
                <span class="hw-tag" style="background:rgba(245,158,11,0.18); border:1px solid rgba(245,158,11,0.4); color:#fbbf24;">Arena VRAM : 650 Mo Stricte</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Détection Multi-Emplacements & Pérennité du Modèle ONNX
    candidate_paths = [
        Path("D:/Dubb-in/lipsync/models/wav2lip_gan.onnx"),
        Path("D:/Dubb-in/models/wav2lip_gan.onnx"),
        Path(CONFIG.model_path) if hasattr(CONFIG, "model_path") else None,
        Path(__file__).parent / "lipsync" / "models" / "wav2lip_gan.onnx",
        Path(__file__).parent / "models" / "wav2lip_gan.onnx",
    ]

    active_model_path = None
    for cp in candidate_paths:
        if cp is not None:
            try:
                if cp.exists() and cp.is_file() and cp.stat().st_size > 50 * 1024 * 1024:
                    active_model_path = cp.resolve()
                    break
            except Exception:
                pass

    model_exists = active_model_path is not None
    model_path = active_model_path or Path("D:/Dubb-in/lipsync/models/wav2lip_gan.onnx")

    if model_exists:
        sz_mb = active_model_path.stat().st_size / (1024 * 1024)
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4);
                        border-radius: 10px; padding: 12px 18px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 22px;">✅</span>
                    <div>
                        <div style="font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:700; color:#34d399;">
                            Téléchargement terminé • Synchronisation OK
                        </div>
                        <div style="font-size:12.5px; color:#cbd5e1;">
                            Modèle <b>wav2lip_gan.onnx</b> ({sz_mb:.1f} Mo) conservé définitivement sur votre disque. Zéro re-téléchargement requis.
                        </div>
                    </div>
                </div>
                <span class="hw-tag" style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #34d399;">OPÉRATIONNEL</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        col_m1, col_m2 = st.columns([3, 1])
        with col_m1:
            st.warning("⚠️ Le modèle ONNX optimisé `wav2lip_gan.onnx` (145 Mo) doit être téléchargé pour exécuter l'inférence.")
        with col_m2:
            if st.button("📥 Télécharger le modèle (145 Mo)", key="btn_dl_onnx_model", use_container_width=True):
                prog_dl = st.progress(0.0)
                status_dl = st.empty()

                def _dl_cb(pct, down, tot):
                    prog_dl.progress(pct)
                    status_dl.caption(f"Téléchargement : {down / (1024*1024):.1f} / {tot / (1024*1024):.1f} Mo ({int(pct*100)}%)")

                with st.spinner("Téléchargement du modèle Wav2Lip GAN ONNX..."):
                    download_wav2lip_model(progress_callback=_dl_cb)
                st.success("Modèle installé avec succès !")
                st.rerun()

    # 2. Source Media Selection
    st.markdown(
        """
        <div style="font-family:'Space Grotesk',sans-serif; font-size:15px; font-weight:700; color:#f1f5f9; margin-bottom:8px;">
            🎬 1. Sélection des Sources Vidéo & Audio
        </div>
        """,
        unsafe_allow_html=True
    )

    src_mode = st.radio(
        "Origine des médias",
        ["📁 Utiliser un projet Dubb-in existant", "📤 Importer des fichiers personnalisés (Vidéo + Audio)"],
        horizontal=True,
        key="ls_src_mode"
    )

    target_video = None
    target_audio = None
    target_dlf = None
    base_out_dir = ROOT / "lipsync_exports"
    base_out_dir.mkdir(exist_ok=True)

    if src_mode == "📁 Utiliser un projet Dubb-in existant":
        existing_projs = sorted(
            [p for p in ROOT.iterdir() if p.is_dir() and (p / "project.json").exists()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not existing_projs:
            st.info("ℹ️ Aucun projet de doublage existant dans votre bibliothèque. Vous pouvez créer un projet dans l'onglet **🎙️ Studio Doublage** ou importer vos propres fichiers ci-dessus.")
        else:
            proj_labels = {}
            for p in existing_projs:
                try:
                    meta = json.loads((p / "project.json").read_text(encoding="utf-8"))
                    name = meta.get("name", "Projet")
                except Exception:
                    name = "Projet"
                has_exp = (p / "export.mp4").exists()
                has_dub = (p / "dub.wav").exists()
                tag = "✅ Doublage prêt" if (has_exp and has_dub) else "⚠️ Partiel"
                proj_labels[p.name] = f"{name} ({p.name[:8]}) — [{tag}]"

            p_keys = list(proj_labels.keys())
            def_proj = st.session_state.get("active", p_keys[0])
            def_idx = p_keys.index(def_proj) if def_proj in p_keys else 0

            sel_proj_id = st.selectbox(
                "Sélectionnez le projet à synchroniser",
                p_keys,
                index=def_idx,
                format_func=lambda k: proj_labels[k],
                key="ls_sel_proj"
            )

            sel_proj_path = ROOT / sel_proj_id
            vid_options = []
            if (sel_proj_path / "export.mp4").exists():
                vid_options.append(("export.mp4 (Vidéo finale doublée)", str(sel_proj_path / "export.mp4")))
            if (sel_proj_path / "video.input").exists():
                vid_options.append(("video.input (Vidéo originale source)", str(sel_proj_path / "video.input")))

            aud_options = []
            if (sel_proj_path / "dub.wav").exists():
                aud_options.append(("dub.wav (Voix clonée cible)", str(sel_proj_path / "dub.wav")))
            if (sel_proj_path / "reference.wav").exists():
                aud_options.append(("reference.wav (Voix de référence)", str(sel_proj_path / "reference.wav")))
            if (sel_proj_path / "source.wav").exists():
                aud_options.append(("source.wav (Audio original extrait)", str(sel_proj_path / "source.wav")))

            # Intégration directe des empreintes de la Banque des Voix
            try:
                import engine
                bank_profs = engine.list_voice_profiles()
                for bp in bank_profs:
                    if bp.get("path") and bp["path"].exists():
                        aud_options.append((f"🧬 [Banque Vocale] {bp['name']} ({bp['duration']:.1f}s)", str(bp["path"])))
            except Exception:
                pass

            col_p_vid, col_p_aud = st.columns(2)
            with col_p_vid:
                if vid_options:
                    chosen_vid_label = st.selectbox("Vidéo à synchroniser :", [o[0] for o in vid_options], key="ls_vid_choice")
                    target_video = next(o[1] for o in vid_options if o[0] == chosen_vid_label)
                    st.video(target_video)
                else:
                    st.error("Aucun fichier vidéo trouvé dans ce projet.")

            with col_p_aud:
                if aud_options:
                    chosen_aud_label = st.selectbox("Audio vocal synchronisé :", [o[0] for o in aud_options], key="ls_aud_choice")
                    target_audio = next(o[1] for o in aud_options if o[0] == chosen_aud_label)
                    st.audio(target_audio)
                else:
                    st.error("Aucun fichier audio trouvé dans ce projet.")

            target_dlf = str(sel_proj_path / f"tracking_{Path(target_video).stem if target_video else 'video'}.dlf")
            out_target_path = str(sel_proj_path / "lipsync_final.mp4")

    else:
        # Custom upload mode
        col_up_v, col_up_a = st.columns(2)
        with col_up_v:
            up_v = st.file_uploader("Fichier vidéo source", type=["mp4", "mov", "mkv", "webm"], key="ls_up_video")
            if up_v:
                saved_v = base_out_dir / f"custom_{up_v.name}"
                saved_v.write_bytes(up_v.getvalue())
                target_video = str(saved_v)
                st.video(target_video)

        with col_up_a:
            up_a = st.file_uploader("Fichier vocal audio (WAV, MP3, M4A...)", type=["wav", "mp3", "m4a", "flac", "ogg"], key="ls_up_audio")
            if up_a:
                saved_a = base_out_dir / f"custom_{up_a.name}"
                saved_a.write_bytes(up_a.getvalue())
                target_audio = str(saved_a)
                st.audio(target_audio)

        if target_video:
            target_dlf = str(Path(target_video).with_suffix(".dlf"))
            out_target_path = str(Path(target_video).with_name(f"{Path(target_video).stem}_lipsync.mp4"))
        else:
            out_target_path = str(base_out_dir / "output_lipsync.mp4")

    # 3. Parameters & .dlf Caching
    st.write("")
    with st.expander("⚡ Paramètres & Optimisation Géométrie .dlf", expanded=True):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            batch_sz = st.slider("Micro-batch GPU (Frames)", min_value=1, max_value=8, value=4, help="Micro-batch optimisé pour NVIDIA GTX 960 (650 Mo VRAM arène max)")
        with col_p2:
            use_dlf_accel = st.checkbox("Activer l'accélération .dlf", value=True, help="Pré-calcule et stocke la géométrie faciale et les lèvres dans un conteneur .dlf ultra-léger. Ré-exécutions 3x plus rapides !")
            dlf_status = "Prêt (Nouveau)"
            if target_dlf and os.path.exists(target_dlf):
                sz_kb = os.path.getsize(target_dlf) / 1024.0
                dlf_status = f"✅ Cache Existant ({sz_kb:.1f} Ko)"
            st.caption(f"Statut .dlf : **{dlf_status}**")
        with col_p3:
            smooth_val = st.slider("Lissage temporel EMA", min_value=0.10, max_value=0.85, value=0.35, step=0.05, help="Filtre anti-tremblement réactif des lèvres et de la mâchoire")

    # 4. Action Button
    can_run = bool(target_video and target_audio and model_exists and os.path.exists(target_video) and os.path.exists(target_audio))

    st.write("")
    btn_run_ls = st.button(
        "🚀 LANCER LA SYNCHRONISATION LABIALE (.DLF)",
        disabled=not can_run,
        type="primary",
        use_container_width=True,
        key="btn_run_lipsync_engine"
    )

    if btn_run_ls:
        prog_bar = st.progress(0.0)
        status_box = st.empty()
        time_start = time.time()

        def _update_ls(*args):
            if len(args) == 5:
                stage, curr, total, fps, eta = args
            elif len(args) == 4:
                curr, total, fps, eta = args
                stage = "sync"
            elif len(args) == 2:
                curr, total = args
                stage = "tracking"
                fps, eta = 0.0, 0.0
            else:
                return

            ratio = min(1.0, max(0.0, curr / max(1, total))) if total > 0 else 0.0
            prog_bar.progress(ratio)
            pct = int(ratio * 100)
            if stage == "tracking":
                title = "🔍 Analyse géométrie faciale (.dlf)..."
                info = f"Frame {curr}/{total}"
            else:
                title = "👄 Synchronisation labiale ONNX GPU..."
                info = f"{curr}/{total} frames • {fps:.1f} FPS • Restant : ~{int(eta)}s"

            status_box.markdown(
                f"""
                <div class="win11-progress-container">
                    <div class="win11-progress-header">
                        <div class="win11-progress-title">
                            <span class="pulse-led"></span>
                            <span>{title}</span>
                            <span style="font-size:12px; color:#94a3b8; margin-left:8px;">({info})</span>
                        </div>
                        <span class="win11-progress-pill">{pct}%</span>
                    </div>
                    <div class="win11-track">
                        <div class="win11-fill" style="width: {pct}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        try:
            with st.spinner("Initialisation du pipeline asymétrique et synchronisation..."):
                final_out = run_lipsync_process(
                    video_path=target_video,
                    audio_path=target_audio,
                    output_path=out_target_path,
                    model_path=str(model_path),
                    batch_size=batch_sz,
                    dlf_cache_path=target_dlf if use_dlf_accel else None,
                    progress_callback=_update_ls
                )
            prog_bar.progress(1.0)
            elapsed = time.time() - time_start
            status_box.success(f"✅ Synchronisation labiale achevée en {elapsed:.1f} secondes !")
            st.session_state["last_lipsync_result"] = final_out
            st.rerun()
        except Exception as e:
            st.error(f"Erreur durant la synchronisation : {e}")

    # 5. Display Result
    result_path = st.session_state.get("last_lipsync_result")
    if not (result_path and os.path.exists(result_path)) and out_target_path and os.path.exists(out_target_path):
        result_path = out_target_path
    if result_path and os.path.exists(result_path):
        st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin:24px 0;'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; color:#34d399;">
                    ✨ RÉSULTAT : VIDÉO SYNCHRONISÉE AVEC SUCCÈS
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_res_v, col_res_actions = st.columns([3, 2])
        with col_res_v:
            st.video(result_path)

        with col_res_actions:
            st.markdown(f"**Fichier généré :** `{Path(result_path).name}`")
            sz_mb = os.path.getsize(result_path) / (1024 * 1024)
            st.caption(f"Taille : {sz_mb:.2f} Mo")

            if st.button("💾 Enregistrer la vidéo Lip-Sync sur le Bureau", type="primary", use_container_width=True, key="btn_desk_ls_ui"):
                desktop = Path.home() / "Desktop"
                dst = desktop / f"Dubb-in_LipSync_{Path(result_path).stem}_{int(time.time())}.mp4"
                shutil.copyfile(result_path, dst)
                st.success(f"✅ Vidéo Lip-Sync enregistrée avec succès sur votre Bureau : {dst.name}")

            with open(result_path, "rb") as f:
                st.download_button(
                    label="⬇️ Télécharger le fichier (.mp4)",
                    data=f.read(),
                    file_name=f"lipsync_{Path(result_path).name}",
                    mime="video/mp4",
                    use_container_width=True
                )

            if target_dlf and os.path.exists(target_dlf):
                st.info(f"⚡ Fichier d'accélération `.dlf` mis en cache : `{Path(target_dlf).name}` ({os.path.getsize(target_dlf) / 1024:.1f} Ko).")
