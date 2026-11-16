"""Dubb-in - couche de traduction d'interface a l'execution (source: francais).

Les textes francais ecrits en dur dans app.py / lipsync_ui.py sont extraits par AST,
traduits une seule fois (Google Translate, comme engine.translate) puis mis en cache
dans _ui_tr_cache.json. Les valeurs retournees par les widgets restent en francais :
seul l'affichage change (format_func), la logique de l'application n'est pas touchee.
"""
import ast
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent
CACHE = BASE / "_ui_tr_cache.json"
SOURCES = ("app.py", "lipsync_ui.py")
GCODE = {"zh": "zh-CN"}
CUR = "fr"
_lock = threading.Lock()
_S = {"segs": None, "cache": None, "rx": {}, "done": set(), "cool": {}}
_COOLDOWN = 900  # secondes sans nouvelle tentative apres un echec (evite le 429 Google)

_ACC = re.compile(r"[\u00e0\u00e2\u00e4\u00e7\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef\u00f4\u00f6\u00f9\u00fb\u00fc\u00ff\u0153\u00c0\u00c2\u00c7\u00c9\u00c8\u00ca\u00ce\u00d4\u00d9\u00db]")
_FRW = re.compile(
    r"\b(?:le|la|les|de|du|des|un|une|et|ou|en|pour|sur|avec|dans|votre|vos|cette|ces|aux|par|sans"
    r"|Voix|Video|Projet|Projets|Creer|Langue|Fichier|Duree|Debut|Aucun|Aucune|Cliquez|Enregistr\w*"
    r"|Supprim\w*|Traduction|Traduire|Transcri\w*|Selection\w*|Sauvegard\w*|Ouvrir|Generer|Telecharg\w*"
    r"|Erreur|Analyse|Empreinte|Materiel|Bibliotheque|Soutien|Dons|Studio|Doublage|termine\w*|succes"
    r"|Pret|Chargement|Initialisation|Lisez|Rapide|Lent)\b",
    re.I,
)
_BAD = re.compile(r"[{}=<>\\]|https?://")


def _pieces(s):
    if "<script" in s or "<style" in s or "@keyframes" in s:
        return []
    out = []
    if "<" in s or ">" in s:
        for m in re.finditer(r'(?:title|placeholder|alt|aria-label)="([^"{}]+)"', s):
            out.append(m.group(1))
        t = s
        if ">" in t:
            head = t[: t.index(">")]
            if "<" not in head and '="' in head:
                t = t[t.index(">") + 1:]
        parts = re.split(r"<[^>]*>?", t)
    else:
        parts = [s]
    for p in parts:
        out.extend(p.split("\n"))
    return out


def _clean(x):
    x = re.sub(r"^[^\w]+", "", x)
    x = re.sub(r"[\s*_`]+$", "", x)
    if len(x) < 3 or _BAD.search(x):
        return None
    if not (_ACC.search(x) or _FRW.search(x)):
        return None
    if not re.search(r"[A-Za-z\u00c0-\u00ff]{2}", x):
        return None
    return x


def _segments(extra=()):
    segs = set()
    for name in SOURCES:
        try:
            tree = ast.parse((BASE / name).read_text(encoding="utf-8"))
        except Exception:
            continue
        skip = {id(n.value) for n in ast.walk(tree)
                if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip:
                for piece in _pieces(n.value):
                    c = _clean(piece)
                    if c:
                        segs.add(c)
    for e in extra:
        if isinstance(e, str) and e.strip():
            segs.add(e.strip())
    return sorted(segs, key=len, reverse=True)


def _load():
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(cache):
    try:
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    except Exception:
        pass


def ensure_lang(lang, extra=(), workers=1):
    """Traduit (une fois) les segments manquants pour `lang`. Silencieux si hors-ligne."""
    if lang == "fr" or lang in _S["done"] or time.time() < _S["cool"].get(lang, 0):
        return
    with _lock:
        if lang in _S["done"]:
            return
        _S["done"].add(lang)
        if _S["segs"] is None:
            _S["segs"] = _segments(extra)
        if _S["cache"] is None:
            _S["cache"] = _load()
        d = _S["cache"].setdefault(lang, {})
        missing = [s for s in _S["segs"] if s not in d]
        if not missing:
            return
        try:
            from deep_translator import GoogleTranslator
        except Exception:
            return
        code = GCODE.get(lang, lang)

        stop = threading.Event()

        def one(s):
            for k in range(3):
                if stop.is_set():
                    return s, None
                try:
                    r = GoogleTranslator(source="fr", target=code).translate(s)
                    if r:
                        time.sleep(0.4)
                        return s, r
                except Exception as e:
                    if "TooMany" in type(e).__name__ or "too many" in str(e).lower():
                        stop.set()
                        return s, None
                    time.sleep(2 * (k + 1))
            return s, None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(one, missing))
        n = 0
        for s, r in res:
            if r:
                d[s] = r
                n += 1
        if n:
            _save(_S["cache"])
            _S["rx"].pop(lang, None)
        if n < len(missing):
            _S["done"].discard(lang)
            _S["cool"][lang] = time.time() + _COOLDOWN


def _rx(lang):
    if _S["cache"] is None:
        _S["cache"] = _load()
    cache = _S["cache"]
    d = cache.get(lang) if cache else None
    if not d:
        return None, None
    ent = _S["rx"].get(lang)
    if ent is None or ent[2] != len(d):
        keys = sorted(d, key=len, reverse=True)
        rx = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(k) for k in keys) + r")(?!\w)")
        ent = (rx, d, len(d))
        _S["rx"][lang] = ent
    return ent[0], ent[1]


def tr(s, lang=None):
    lang = lang or CUR
    if lang == "fr" or not isinstance(s, str) or not s:
        return s
    rx, d = _rx(lang)
    if rx is None:
        return s
    return rx.sub(lambda m: d.get(m.group(0), m.group(0)), s)


def _fmt(x):
    return tr(x) if isinstance(x, str) else str(x)


_NAMES = (
    "markdown", "caption", "text", "info", "success", "warning", "error", "header", "subheader",
    "title", "button", "download_button", "link_button", "checkbox", "toggle", "radio", "selectbox",
    "multiselect", "select_slider", "slider", "text_input", "text_area", "number_input", "date_input",
    "time_input", "file_uploader", "camera_input", "color_picker", "metric", "expander", "spinner",
    "status", "popover", "form_submit_button", "toast", "write", "tabs", "progress",
)
_KW = ("label", "body", "help", "placeholder", "caption", "text")
_FMT = ("radio", "selectbox", "multiselect", "select_slider")


def _mk(orig, name, method):
    off = 1 if method else 0

    def w(*args, **kw):
        if CUR == "fr":
            return orig(*args, **kw)
        a = list(args)
        if name == "tabs":
            if len(a) > off and isinstance(a[off], (list, tuple)):
                a[off] = [tr(x) if isinstance(x, str) else x for x in a[off]]
        elif name == "write":
            a = a[:off] + [tr(x) if isinstance(x, str) else x for x in a[off:]]
        elif name == "progress":
            if len(a) > off + 1 and isinstance(a[off + 1], str):
                a[off + 1] = tr(a[off + 1])
        elif len(a) > off and isinstance(a[off], str):
            a[off] = tr(a[off])
        for k in _KW:
            if isinstance(kw.get(k), str):
                kw[k] = tr(kw[k])
        if name in _FMT and "format_func" not in kw and len(a) <= off + 3:
            kw["format_func"] = _fmt
        return orig(*a, **kw)

    return w


def install(st, lang, extra=()):
    """A appeler a chaque execution du script, apres le calcul de la langue courante."""
    global CUR
    CUR = lang or "fr"
    ensure_lang(CUR, extra)
    store = getattr(st, "_autotr_orig", None)
    if store is None:
        store = {}
        try:
            setattr(st, "_autotr_orig", store)
        except Exception:
            pass
    from streamlit.delta_generator import DeltaGenerator as DG

    for name in _NAMES:
        o = store.get(("m", name))
        if o is None and hasattr(st, name):
            o = getattr(st, name)
            store[("m", name)] = o
        if o is not None:
            try:
                setattr(st, name, _mk(o, name, False))
            except Exception:
                pass
        c = store.get(("c", name))
        if c is None:
            c = getattr(DG, name, None)
            if c is not None:
                store[("c", name)] = c
        if c is not None:
            try:
                setattr(DG, name, _mk(c, name, True))
            except Exception:
                pass
