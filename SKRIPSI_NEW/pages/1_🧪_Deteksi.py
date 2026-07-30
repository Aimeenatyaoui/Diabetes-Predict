from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as npfrom __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from cbr.engine import ZERO_AS_MISSING, build_artifacts_from_case_base, load_case_base_df, predict_cbr
from cbr.weights_config import MULTISURF_WEIGHTS_ARRAY, OPTIMAL_K
from ui.theme import card_close, card_open, divider, inject_glass_theme, section_label

st.set_page_config(page_title="Deteksi — DM Tipe 2", page_icon="🧪", layout="wide")
inject_glass_theme()

# ── DESIGN TOKENS — Satu sumber kebenaran untuk seluruh halaman ─────────
# Warna dasar
BG_PAGE = "#f0f2f5"  # Abu-abu sangat terang untuk latar
BG_GLASS = "rgba(255, 255, 255, 0.72)"
BG_GLASS_HOVER = "rgba(255, 255, 255, 0.85)"
BORDER_GLASS = "rgba(255, 255, 255, 0.3)"
SHADOW_GLASS = "0 8px 32px rgba(0, 0, 0, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04)"
BLUR_GLASS = "blur(20px)"

# Warna teks - kontras tinggi untuk keterbacaan
TEXT_PRIMARY = "#1a1a2e"  # Biru tua gelap
TEXT_SECONDARY = "#4a4a6a"  # Ungu kebiruan medium
TEXT_MUTED = "#6b6b8a"  # Abu-abu kebiruan
TEXT_INVERSE = "#ffffff"  # Putih untuk di atas gelap

# Warna aksen
ACCENT = "#5b5ea6"  # Ungu kebiruan utama
ACCENT_LIGHT = "#7c7fb8"  # Ungu kebiruan terang
ACCENT_GLOW = "rgba(91, 94, 166, 0.15)"
ACCENT_SOFT = "rgba(91, 94, 166, 0.08)"

# Status warna
SUCCESS = "#34c759"  # Hijau iOS
SUCCESS_BG = "rgba(52, 199, 89, 0.12)"
DANGER = "#ff3b30"  # Merah iOS
DANGER_BG = "rgba(255, 59, 48, 0.12)"
WARNING = "#ff9500"  # Oranye iOS
WARNING_BG = "rgba(255, 149, 0, 0.12)"
INFO = "#007aff"  # Biru iOS
INFO_BG = "rgba(0, 122, 255, 0.12)"

# Sidebar
SIDEBAR_BG = "rgba(26, 26, 46, 0.92)"
SIDEBAR_BLUR = "blur(30px)"
SIDEBAR_BORDER = "rgba(255, 255, 255, 0.08)"
SIDEBAR_TEXT = "#f0f0f5"

# ── STYLE GLOBAL ──────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Reset & Base */
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}

html, body, .stApp {{
    background: {BG_PAGE} !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', sans-serif !important;
}}

/* Scrollbar styling */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {ACCENT_LIGHT}; border-radius: 10px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

/* Sidebar - Glassmorphism gelap */
section[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    backdrop-filter: {SIDEBAR_BLUR} !important;
    -webkit-backdrop-filter: {SIDEBAR_BLUR} !important;
    border-right: 1px solid {SIDEBAR_BORDER} !important;
}}

section[data-testid="stSidebar"] * {{
    color: {SIDEBAR_TEXT} !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', sans-serif !important;
}}

section[data-testid="stSidebar"] .stMarkdown {{
    color: {SIDEBAR_TEXT} !important;
}}

/* Main content area - padding lebih lega */
.main .block-container {{
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1200px !important;
}}

/* Glass Card - Komponen utama */
.glass-card {{
    background: {BG_GLASS} !important;
    backdrop-filter: {BLUR_GLASS} !important;
    -webkit-backdrop-filter: {BLUR_GLASS} !important;
    border: 1px solid {BORDER_GLASS} !important;
    border-radius: 20px !important;
    box-shadow: {SHADOW_GLASS} !important;
    padding: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

.glass-card:hover {{
    background: {BG_GLASS_HOVER} !important;
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.1), 0 4px 12px rgba(0, 0, 0, 0.06) !important;
}}

/* Override semua container Streamlit menjadi glass */
div[data-testid="stForm"],
div[data-testid="stExpander"],
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stBlock"] {{
    background: {BG_GLASS} !important;
    backdrop-filter: {BLUR_GLASS} !important;
    -webkit-backdrop-filter: {BLUR_GLASS} !important;
    border: 1px solid {BORDER_GLASS} !important;
    border-radius: 20px !important;
    box-shadow: {SHADOW_GLASS} !important;
    padding: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}

div[data-testid="stForm"]:hover,
div[data-testid="stExpander"]:hover,
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    background: {BG_GLASS_HOVER} !important;
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.1), 0 4px 12px rgba(0, 0, 0, 0.06) !important;
}}

/* Typography - iOS style */
h1, h2, h3, h4, h5, h6 {{
    color: {TEXT_PRIMARY} !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}}

h1 {{
    font-size: 2.5rem !important;
    line-height: 1.2 !important;
}}

h2 {{
    font-size: 1.8rem !important;
    line-height: 1.3 !important;
}}

h3 {{
    font-size: 1.3rem !important;
    line-height: 1.4 !important;
}}

p, span, li, label, .stMarkdown p {{
    color: {TEXT_SECONDARY} !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    font-weight: 400 !important;
}}

/* Pastikan semua teks di main terbaca */
.main *, [data-testid="stMain"] * {{
    color: {TEXT_SECONDARY} !important;
}}

/* Kecuali heading tetap gelap */
.main h1, .main h2, .main h3, .main h4,
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3 {{
    color: {TEXT_PRIMARY} !important;
}}

/* Input fields - iOS style */
input, textarea, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
    font-size: 1rem !important;
    color: {TEXT_PRIMARY} !important;
    background: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1.5px solid rgba(0, 0, 0, 0.06) !important;
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s ease !important;
}}

input:focus, textarea:focus, .stNumberInput input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 4px {ACCENT_GLOW} !important;
    background: rgba(255, 255, 255, 0.8) !important;
    outline: none !important;
}}

input::placeholder, textarea::placeholder {{
    color: {TEXT_MUTED} !important;
    opacity: 0.6 !important;
}}

/* Labels */
label, .stTextInput label, .stNumberInput label, .stSelectbox label {{
    color: {TEXT_PRIMARY} !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0.3rem !important;
}}

/* Buttons - iOS style */
button {{
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: none !important;
    letter-spacing: -0.01em !important;
}}

button[kind="primary"] {{
    background: {ACCENT} !important;
    color: {TEXT_INVERSE} !important;
    box-shadow: 0 4px 12px {ACCENT_GLOW} !important;
}}

button[kind="primary"]:hover {{
    background: {ACCENT_LIGHT} !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px {ACCENT_GLOW} !important;
}}

button[kind="primary"]:active {{
    transform: translateY(0) scale(0.98) !important;
}}

button[kind="primary"] p, button[kind="primary"] span {{
    color: {TEXT_INVERSE} !important;
    font-weight: 600 !important;
}}

/* Secondary button */
button:not([kind="primary"]) {{
    background: rgba(255, 255, 255, 0.5) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(0, 0, 0, 0.06) !important;
    color: {TEXT_SECONDARY} !important;
}}

button:not([kind="primary"]):hover {{
    background: rgba(255, 255, 255, 0.8) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
}}

/* Radio buttons - iOS style */
div[data-testid="stRadio"] {{
    background: rgba(255, 255, 255, 0.3) !important;
    border-radius: 12px !important;
    padding: 0.3rem !important;
}}

div[data-testid="stRadio"] > label {{
    padding: 0.5rem 1rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}}

div[data-testid="stRadio"] > label[data-baseweb="radio"] > div:first-child {{
    background-color: {ACCENT} !important;
}}

/* Expander */
div[data-testid="stExpander"] {{
    background: rgba(255, 255, 255, 0.5) !important;
}}

div[data-testid="stExpander"] summary {{
    color: {TEXT_PRIMARY} !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.5rem 0 !important;
}}

/* Alert/Info boxes */
div[data-testid="stAlert"] {{
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    padding: 1rem 1.25rem !important;
}}

/* Success alert */
div[data-testid="stAlert"][data-baseweb="notification"] {{
    background: {SUCCESS_BG} !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border-color: rgba(52, 199, 89, 0.2) !important;
}}

div[data-testid="stAlert"][data-baseweb="notification"] * {{
    color: {SUCCESS} !important;
}}

/* Error alert */
div[data-testid="stAlert"][data-baseweb="notification"][data-kind="error"] {{
    background: {DANGER_BG} !important;
    border-color: rgba(255, 59, 48, 0.2) !important;
}}

div[data-testid="stAlert"][data-baseweb="notification"][data-kind="error"] * {{
    color: {DANGER} !important;
}}

/* Warning alert */
div[data-testid="stAlert"][data-baseweb="notification"][data-kind="warning"] {{
    background: {WARNING_BG} !important;
    border-color: rgba(255, 149, 0, 0.2) !important;
}}

div[data-testid="stAlert"][data-baseweb="notification"][data-kind="warning"] * {{
    color: {WARNING} !important;
}}

/* Info alert */
div[data-testid="stAlert"][data-baseweb="notification"][data-kind="info"] {{
    background: {INFO_BG} !important;
    border-color: rgba(0, 122, 255, 0.2) !important;
}}

div[data-testid="stAlert"][data-baseweb="notification"][data-kind="info"] * {{
    color: {INFO} !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    background: rgba(255, 255, 255, 0.4) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    overflow: hidden !important;
}}

[data-testid="stDataFrame"] * {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.9rem !important;
}}

[data-testid="stDataFrame"] thead th {{
    background: rgba(255, 255, 255, 0.3) !important;
    color: {TEXT_PRIMARY} !important;
    font-weight: 600 !important;
}}

/* Badge styles */
.badge {{
    font-size: 0.9rem !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
    border-radius: 30px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
}}

.badge-high {{
    color: {DANGER} !important;
    background: {DANGER_BG} !important;
    border-color: rgba(255, 59, 48, 0.2) !important;
}}

.badge-low {{
    color: {SUCCESS} !important;
    background: {SUCCESS_BG} !important;
    border-color: rgba(52, 199, 89, 0.2) !important;
}}

/* Mini metric cards */
.mini-metric {{
    background: rgba(255, 255, 255, 0.4) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    transition: all 0.3s ease !important;
}}

.mini-metric:hover {{
    background: rgba(255, 255, 255, 0.6) !important;
    transform: translateY(-2px) !important;
}}

.mini-metric .val {{
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: {ACCENT} !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}}

.mini-metric .lbl {{
    font-size: 0.85rem !important;
    color: {TEXT_MUTED} !important;
    font-weight: 500 !important;
    margin-top: 0.2rem !important;
}}

/* Section label - iOS style */
.section-label, [class*="section-label"], [class*="section-title"] {{
    color: {TEXT_PRIMARY} !important;
    font-weight: 700 !important;
    font-size: 1.3rem !important;
    letter-spacing: -0.02em !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.75rem !important;
    margin-bottom: 0.75rem !important;
}}

.section-label::before {{
    content: '' !important;
    width: 4px !important;
    height: 24px !important;
    background: {ACCENT} !important;
    border-radius: 4px !important;
    display: inline-block !important;
}}

/* Custom field label */
.field-label {{
    font-weight: 600 !important;
    color: {TEXT_PRIMARY} !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.4rem !important;
    display: block !important;
    letter-spacing: -0.01em !important;
}}

/* Divider */
.divider {{
    height: 1px !important;
    background: linear-gradient(to right, rgba(0,0,0,0.05), rgba(0,0,0,0.1), rgba(0,0,0,0.05)) !important;
    margin: 1.5rem 0 !important;
    border: none !important;
}}

/* Caption */
.caption, .stCaption {{
    color: {TEXT_MUTED} !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
}}

/* Status message inline */
.status-text {{
    padding: 0.75rem 1rem !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    background: rgba(255, 255, 255, 0.3) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}}

/* Utility - text colors */
.text-primary {{ color: {TEXT_PRIMARY} !important; }}
.text-secondary {{ color: {TEXT_SECONDARY} !important; }}
.text-muted {{ color: {TEXT_MUTED} !important; }}
.text-accent {{ color: {ACCENT} !important; }}

/* Responsive tweaks */
@media (max-width: 768px) {{
    .main .block-container {{
        padding: 1rem 1rem 2rem !important;
    }}
    .glass-card {{
        padding: 1rem !important;
        border-radius: 16px !important;
    }}
    .mini-metric .val {{
        font-size: 1.6rem !important;
    }}
}}

/* Animasi halus untuk semua transisi */
* {{
    transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
}}
</style>
""", unsafe_allow_html=True)

CASE_BASE_PATH = Path("data/case_base.csv")
CASE_BASE_SEED_URLS = [
    "https://raw.githubusercontent.com/aimeenatyaoui/SKRIPSI/main/SKRIPSI/data.xlsx",
    "https://raw.githubusercontent.com/aimeenatyaoui/SKRIPSI/master/SKRIPSI/data.xlsx",
]


@dataclass(frozen=True)
class PatientInput:
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


def load_case_base() -> pd.DataFrame:
    if not CASE_BASE_PATH.exists():
        return pd.DataFrame(columns=[
            "timestamp", "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
            "Age", "predicted_outcome", "validated_outcome", "validation_note",
        ])
    return pd.read_csv(CASE_BASE_PATH)


def retain_case(row: dict[str, Any]) -> None:
    CASE_BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_case_base()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CASE_BASE_PATH, index=False)


def validate_input(pi: PatientInput) -> list[str]:
    errors: list[str] = []
    d = asdict(pi)
    for k, v in d.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            errors.append(f"**{k}** wajib diisi.")
    for k in ZERO_AS_MISSING:
        if k in d and float(d[k]) == 0.0:
            errors.append(f"**{k}** tidak boleh 0 (nilai 0 dianggap data kosong).")
    return errors


def run_predict(pi: PatientInput) -> dict[str, Any]:
    df = load_case_base_df(CASE_BASE_PATH)
    source = "data/case_base.csv"
    if df.empty:
        for url in CASE_BASE_SEED_URLS:
            df = load_case_base_df(url)
            if not df.empty:
                source = url
                break

    artifacts = build_artifacts_from_case_base(df)
    pred = predict_cbr(artifacts, asdict(pi), k=OPTIMAL_K, weights=MULTISURF_WEIGHTS_ARRAY)
    pred["case_base_source"] = source
    return pred


# ── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;">
        <span style="font-size:1.8rem;">🧪</span>
        <span style="font-size:1.3rem;font-weight:700;letter-spacing:-0.02em;">CBR Deteksi</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:rgba(255,255,255,0.06);border-radius:16px;padding:1.25rem;
         backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
         border:1px solid rgba(255,255,255,0.06);">
        <div style="display:flex;flex-direction:column;gap:0.75rem;">
            <div>
                <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;
                     color:rgba(255,255,255,0.4);font-weight:600;">Metode</span>
                <div style="font-weight:600;font-size:1.05rem;">CBR + MultiSURF</div>
            </div>
            <div>
                <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;
                     color:rgba(255,255,255,0.4);font-weight:600;">K Optimal</span>
                <div style="font-weight:600;font-size:1.05rem;">{OPTIMAL_K} tetangga</div>
            </div>
            <div>
                <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;
                     color:rgba(255,255,255,0.4);font-weight:600;">Akurasi</span>
                <div style="font-weight:600;font-size:1.05rem;">76,03%</div>
            </div>
        </div>
    </div>
    """.format(OPTIMAL_K=OPTIMAL_K), unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top:1rem;padding:0.75rem 1rem;background:rgba(255,255,255,0.04);
         border-radius:12px;border:1px solid rgba(255,255,255,0.04);">
        <span style="font-size:0.8rem;color:rgba(255,255,255,0.4);line-height:1.6;">
            ℹ️ K=9 adalah hasil terbaik dari 10-Fold Stratified Cross Validation
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── PAGE HEADER ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:2rem;">
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;">
        <span style="font-size:0.75rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;
             color:{ACCENT};background:rgba(91,94,166,0.1);padding:0.25rem 0.75rem;border-radius:20px;">
            Halaman Deteksi
        </span>
    </div>
    <h1 style="font-size:2.8rem;font-weight:800;margin:0.25rem 0 0.25rem 0;letter-spacing:-0.03em;">
        Estimasi Risiko Diabetes
    </h1>
    <p style="font-size:1.1rem;color:{TEXT_MUTED};margin:0;">
        Isi data pemeriksaan pasien di bawah, lalu klik <strong style="color:{TEXT_PRIMARY};">Proses Estimasi</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

# ── CARD A: INPUT DATA PEMERIKSAAN PASIEN ─────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
    <span style="font-size:1.3rem;font-weight:700;color:{TEXT_PRIMARY};">A</span>
    <span style="font-size:1.1rem;font-weight:600;color:{TEXT_PRIMARY};">Input Data Pemeriksaan Pasien</span>
</div>
""", unsafe_allow_html=True)

with st.form("patient_form", clear_on_submit=False):
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(f"""
        <div style="font-weight:600;color:{TEXT_PRIMARY};font-size:0.95rem;margin-bottom:0.75rem;">
            Obstetri &amp; Metabolisme
        </div>
        """, unsafe_allow_html=True)
        
        pregnancies = st.number_input(
            "Kehamilan (Pregnancies)", min_value=0, max_value=20, value=1, step=1,
            help="Jumlah riwayat kehamilan"
        )
        glucose = st.number_input(
            "Glukosa (Glucose) mg/dL", min_value=1, max_value=500, value=100, step=1,
            help="Konsentrasi glukosa plasma 2 jam setelah tes toleransi glukosa oral"
        )
        blood_pressure = st.number_input(
            "Tekanan Darah (BloodPressure) mmHg", min_value=1, max_value=300, value=72, step=1,
            help="Tekanan darah diastolik"
        )
        skin_thickness = st.number_input(
            "Ketebalan Kulit (SkinThickness) mm", min_value=1, max_value=200, value=20, step=1,
            help="Ketebalan lipatan kulit trisep"
        )

    with c2:
        st.markdown(f"""
        <div style="font-weight:600;color:{TEXT_PRIMARY};font-size:0.95rem;margin-bottom:0.75rem;">
            Hormon &amp; Antropometri
        </div>
        """, unsafe_allow_html=True)
        
        insulin = st.number_input(
            "Insulin (mu U/mL)", min_value=1, max_value=2000, value=79, step=1,
            help="Kadar insulin serum 2 jam"
        )
        bmi = st.number_input(
            "BMI (kg/m²)", min_value=0.1, max_value=80.0, value=25.0, step=0.1,
            help="Indeks massa tubuh"
        )
        dpf = st.number_input(
            "DiabetesPedigreeFunction", min_value=0.001, max_value=5.0, value=0.350, step=0.001,
            help="Skor riwayat diabetes dalam keluarga"
        )
        age = st.number_input(
            "Usia (Age) tahun", min_value=1, max_value=120, value=30, step=1,
            help="Usia pasien"
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        submitted = st.form_submit_button(
            "🔍 Proses Estimasi", type="primary", use_container_width=True
        )

with st.expander("📌 Panduan Pengisian", expanded=True):
    st.markdown(f"""
    <div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.8;">
        • <strong style="color:{TEXT_PRIMARY};">Glucose, BloodPressure, SkinThickness, Insulin, BMI</strong> tidak boleh diisi 0 —
        nilai 0 dianggap data tidak tersedia.<br>
        • Semua nilai harus sesuai hasil pemeriksaan aktual pasien.<br>
        • Hasil estimasi bersifat <strong style="color:{TEXT_PRIMARY};">pendukung klinis</strong>, bukan diagnosis final.
    </div>
    """, unsafe_allow_html=True)

# ── PROSES ESTIMASI ────────────────────────────────────────────────────
if submitted:
    pi = PatientInput(
        Pregnancies=float(pregnancies), Glucose=float(glucose),
        BloodPressure=float(blood_pressure), SkinThickness=float(skin_thickness),
        Insulin=float(insulin), BMI=float(bmi),
        DiabetesPedigreeFunction=float(dpf), Age=float(age),
    )
    errors = validate_input(pi)
    if errors:
        for e in errors:
            st.error(e)
    else:
        with st.spinner("Menghitung kemiripan kasus…"):
            try:
                pred = run_predict(pi)
                st.session_state["latest_input"] = asdict(pi)
                st.session_state["latest_pred"] = pred
            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")

latest_input = st.session_state.get("latest_input")
latest_pred = st.session_state.get("latest_pred")

# ── CARD B: HASIL ESTIMASI ─────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.75rem;margin:2rem 0 1rem 0;">
    <span style="font-size:1.3rem;font-weight:700;color:{TEXT_PRIMARY};">B</span>
    <span style="font-size:1.1rem;font-weight:600;color:{TEXT_PRIMARY};">Hasil Estimasi</span>
</div>
""", unsafe_allow_html=True)

if not latest_input or not latest_pred:
    st.markdown(f"""
    <div style="text-align:center;padding:3rem 1rem;background:rgba(255,255,255,0.3);
         backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
         border-radius:20px;border:1px solid rgba(255,255,255,0.2);">
        <div style="font-size:4rem;margin-bottom:0.75rem;">🩺</div>
        <div style="font-size:1.1rem;color:{TEXT_MUTED};">
            Belum ada hasil.<br>Isi form di atas lalu klik <strong style="color:{TEXT_PRIMARY};">Proses Estimasi</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    predicted = int(latest_pred["predicted_outcome"])
    risk_score = float(latest_pred.get("risk_score", 0.0))
    counts = latest_pred.get("neighbor_counts", {0: 0, 1: 0})

    b1, b2, b3, b4 = st.columns(4, gap="small")

    with b1:
        if predicted == 1:
            st.markdown(f"""
            <div class="mini-metric" style="text-align:left;">
                <div class="badge badge-high" style="margin-bottom:0.25rem;">⚠️ Beresiko Diabetes</div>
                <div style="font-size:0.75rem;color:{TEXT_MUTED};font-weight:500;letter-spacing:0.04em;text-transform:uppercase;">Prediksi</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="mini-metric" style="text-align:left;">
                <div class="badge badge-low" style="margin-bottom:0.25rem;">✅ Tidak Beresiko</div>
                <div style="font-size:0.75rem;color:{TEXT_MUTED};font-weight:500;letter-spacing:0.04em;text-transform:uppercase;">Prediksi</div>
            </div>
            """, unsafe_allow_html=True)

    with b2:
        st.markdown(f"""
        <div class="mini-metric">
            <div class="val">{risk_score:.0%}</div>
            <div class="lbl">Probabilitas Risiko</div>
        </div>
        """, unsafe_allow_html=True)
    
    with b3:
        outcome_label = "Tidak Diabetes" if predicted == 0 else "Diabetes"
        st.markdown(f"""
        <div class="mini-metric">
            <div class="val">{predicted}</div>
            <div class="lbl">Prediksi Outcome<br><span style="font-size:0.8rem;">{outcome_label}</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with b4:
        st.markdown(f"""
        <div class="mini-metric">
            <div class="val">{counts.get(1,0)}/{OPTIMAL_K}</div>
            <div class="lbl">K Tetangga</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    preview = latest_pred.get("nearest_cases_preview")
    if isinstance(preview, pd.DataFrame) and not preview.empty:
        with st.expander("📊 Detail Tetangga Terdekat", expanded=True):
            st.dataframe(preview, use_container_width=True, hide_index=True)

    # ── CARD C: VALIDASI KLINIS (REVISE) ───────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.75rem;margin:2rem 0 1rem 0;">
        <span style="font-size:1.3rem;font-weight:700;color:{TEXT_PRIMARY};">C</span>
        <span style="font-size:1.1rem;font-weight:600;color:{TEXT_PRIMARY};">Validasi Klinis (Revise)</span>
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns([1, 1, 1.2], gap="medium")

    with rc1:
        st.markdown('<span class="field-label">Apakah hasil estimasi sesuai pertimbangan klinis?</span>', unsafe_allow_html=True)
        validated = st.radio(
            "Apakah hasil estimasi sesuai pertimbangan klinis?",
            options=["Sesuai", "Tidak sesuai"], horizontal=True,
            key="validated_radio", label_visibility="collapsed",
        )

    with rc2:
        st.markdown('<span class="field-label">Tentukan outcome akhir:</span>', unsafe_allow_html=True)
        validated_outcome = st.radio(
            "Tentukan outcome akhir:",
            options=[0, 1],
            index=predicted,
            format_func=lambda v: "0 — Tidak Diabetes" if v == 0 else "1 — Diabetes",
            horizontal=True, key="override_outcome", label_visibility="collapsed",
        )

    with rc3:
        st.markdown('<span class="field-label">Catatan klinis (opsional)</span>', unsafe_allow_html=True)
        note = st.text_area(
            "Catatan klinis (opsional)",
            placeholder="Tulis catatan di sini…",
            height=80, max_chars=200, key="note_area", label_visibility="collapsed"
        )

    if validated == "Sesuai" and validated_outcome == predicted:
        st.markdown(f"""
        <div style="background:{SUCCESS_BG};border-radius:12px;padding:0.75rem 1rem;
             border:1px solid rgba(52,199,89,0.15);">
            <span style="color:{SUCCESS};font-weight:500;">✅ Outcome dikonfirmasi:</span>
            <span style="color:{TEXT_PRIMARY};font-weight:600;">{'Diabetes' if predicted == 1 else 'Tidak Diabetes'}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:{WARNING_BG};border-radius:12px;padding:0.75rem 1rem;
             border:1px solid rgba(255,149,0,0.15);">
            <span style="color:{WARNING};font-weight:500;">⚠️ Outcome dikoreksi oleh tenaga kesehatan.</span>
        </div>
        """, unsafe_allow_html=True)

    # ── CARD D: SIMPAN KE BASIS KASUS (RETAIN) ─────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.75rem;margin:2rem 0 1rem 0;">
        <span style="font-size:1.3rem;font-weight:700;color:{TEXT_PRIMARY};">D</span>
        <span style="font-size:1.1rem;font-weight:600;color:{TEXT_PRIMARY};">Simpan ke Basis Kasus (Retain)</span>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns([1, 3], gap="medium")
    with sc1:
        if st.button("💾 Simpan Kasus", type="primary", use_container_width=True):
            row = {
                "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                **latest_input,
                "predicted_outcome": predicted,
                "validated_outcome": int(validated_outcome),
                "validation_note": note.strip(),
            }
            retain_case(row)
            st.success("✅ Kasus berhasil disimpan ke basis kasus.")
    with sc2:
        if st.button("🔄 Reset Form", use_container_width=True):
            st.session_state.pop("latest_input", None)
            st.session_state.pop("latest_pred", None)
            st.rerun()

# ── CATATAN BAWAH HALAMAN ───────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:2.5rem;padding:1rem 1.5rem;background:rgba(255,255,255,0.3);
     backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
     border-radius:16px;border:1px solid rgba(255,255,255,0.2);
     display:flex;align-items:flex-start;gap:0.75rem;">
    <span style="font-size:1.2rem;flex-shrink:0;">ℹ️</span>
    <div style="font-size:0.9rem;color:{TEXT_SECONDARY};line-height:1.7;">
        <strong style="color:{TEXT_PRIMARY};">Catatan:</strong> Normalisasi MinMax dan pembobotan menggunakan bobot MultiSURF.
        Hasil estimasi bersifat pendukung klinis, bukan diagnosis final.
    </div>
</div>
""", unsafe_allow_html=True)
import pandas as pd
import streamlit as st

from cbr.engine import ZERO_AS_MISSING, build_artifacts_from_case_base, load_case_base_df, predict_cbr
from cbr.weights_config import MULTISURF_WEIGHTS_ARRAY, OPTIMAL_K
from ui.theme import card_close, card_open, divider, inject_glass_theme, section_label

st.set_page_config(page_title="Deteksi — DM Tipe 2", page_icon="🧪", layout="wide")
inject_glass_theme()

# ── DESIGN TOKENS — satu sumber kebenaran untuk seluruh halaman ─────────
BG_PAGE = "#f1f5f9"        # slate-100 — latar halaman, netral & lembut
BG_CARD = "#ffffff"        # kartu solid putih (bukan glass) — lebih terbaca & standar
BORDER_CARD = "#e2e8f0"    # slate-200 — garis tepi kartu
TEXT_PRIMARY = "#0f172a"   # slate-900 — teks utama
TEXT_MUTED = "#64748b"     # slate-500 — teks sekunder/caption
ACCENT = "#2563eb"         # blue-600 — satu warna aksen: tombol utama, angka penting
ACCENT_SOFT = "#dbeafe"    # blue-100 — latar chip/label kecil
SIDEBAR_BG = "#0f172a"     # slate-900 — sidebar gelap, senada TEXT_PRIMARY
SIDEBAR_TEXT = "#f1f5f9"   # teks sidebar terang, kontras dgn latar gelap

DANGER_TEXT, DANGER_BG, DANGER_BORDER = "#b91c1c", "#fef2f2", "#fecaca"   # merah — risiko
SUCCESS_TEXT, SUCCESS_BG, SUCCESS_BORDER = "#15803d", "#f0fdf4", "#bbf7d0"  # hijau — aman

# ── STYLE GLOBAL ──────────────────────────────────────────────────────────
st.markdown(f"""
<style>
:root {{ color-scheme: light; }}

.stApp {{ background: {BG_PAGE} !important; }}
html, body, [class^="css"] {{ font-size: 16px !important; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {SIDEBAR_BG} !important; }}
section[data-testid="stSidebar"] * {{ color: {SIDEBAR_TEXT} !important; font-size: 1rem !important; }}

/* Kartu — flat, solid, konsisten */
div[data-testid="stForm"],
div[data-testid="stExpander"],
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER_CARD} !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,.06), 0 1px 2px rgba(15,23,42,.04) !important;
}}
div[data-testid="stForm"] *,
div[data-testid="stExpander"] *,
div[data-testid="stVerticalBlockBorderWrapper"] * {{
    color: {TEXT_PRIMARY} !important;
}}

h1, h2, h3 {{ color: {TEXT_PRIMARY} !important; }}

/* Catch-all: pastikan tidak ada teks putih tersisa di area konten utama */
div.main *, [data-testid="stMain"] * {{ color: {TEXT_PRIMARY} !important; }}

label, .stMarkdown p, .stMarkdown li, .stCaption, p, span {{
    font-size: 1rem !important;
    line-height: 1.6 !important;
}}

input, textarea, .stNumberInput input {{
    font-size: 1.05rem !important;
    color: {TEXT_PRIMARY} !important;
    background: #ffffff !important;
    border: 1px solid {BORDER_CARD} !important;
    border-radius: 8px !important;
}}

/* Tombol sekunder (mis. Reset) — outline netral, teks gelap */
button {{ border-radius: 8px !important; }}
button p {{ font-size: 1rem !important; font-weight: 600 !important; }}

/* Tombol utama — aksen biru solid, teks putih (lebih spesifik dari catch-all) */
button[kind="primary"] {{
    background: {ACCENT} !important;
    border: 1px solid {ACCENT} !important;
}}
button[kind="primary"] p, button[kind="primary"] span {{
    color: #ffffff !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}}

/* Radio & text_area — dipaksa hitam (lapisan tambahan, jaga-jaga) */
div[data-testid="stRadio"] *, div[data-testid="stTextArea"] * {{
    color: {TEXT_PRIMARY} !important;
}}

[data-testid="stDataFrame"] * {{ font-size: .95rem !important; color: {TEXT_PRIMARY} !important; }}

div[data-testid="stAlert"] {{ border-radius: 10px !important; font-size: 1rem !important; }}

.mini-metric .val {{ font-size: 2rem !important; color: {ACCENT} !important; font-weight: 800 !important; }}
.mini-metric .lbl {{ font-size: .95rem !important; color: {TEXT_MUTED} !important; }}

.badge {{
    font-size: 1.05rem !important; padding: 8px 14px !important; font-weight: 700 !important;
    border-radius: 10px !important; display: inline-block !important; border: 1px solid transparent !important;
}}
.badge-high {{ color: {DANGER_TEXT} !important; background: {DANGER_BG} !important; border-color: {DANGER_BORDER} !important; }}
.badge-low  {{ color: {SUCCESS_TEXT} !important; background: {SUCCESS_BG} !important; border-color: {SUCCESS_BORDER} !important; }}

/* Judul kartu ("A — ...", dst) dari section_label() di ui/theme.py — timpa ke hitam */
[style*="2dd4bf"], [style*="45,212,191"], [style*="14b8a6"], [style*="0d9488"] {{
    color: {TEXT_PRIMARY} !important;
}}
[class*="section-label"], [class*="section-title"], [class*="card-title"],
[class*="section-label"] *, [class*="section-title"] *, [class*="card-title"] * {{
    color: {TEXT_PRIMARY} !important;
}}

/* Label mini kustom (dipakai utk pertanyaan Card C) */
.field-label {{
    font-weight: 700 !important; color: {TEXT_PRIMARY} !important;
    font-size: 1rem !important; margin-bottom: 6px !important; display: block !important;
}}
</style>
""", unsafe_allow_html=True)

CASE_BASE_PATH = Path("data/case_base.csv")
CASE_BASE_SEED_URLS = [
    "https://raw.githubusercontent.com/aimeenatyaoui/SKRIPSI/main/SKRIPSI/data.xlsx",
    "https://raw.githubusercontent.com/aimeenatyaoui/SKRIPSI/master/SKRIPSI/data.xlsx",
]


@dataclass(frozen=True)
class PatientInput:
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


def load_case_base() -> pd.DataFrame:
    if not CASE_BASE_PATH.exists():
        return pd.DataFrame(columns=[
            "timestamp", "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
            "Age", "predicted_outcome", "validated_outcome", "validation_note",
        ])
    return pd.read_csv(CASE_BASE_PATH)


def retain_case(row: dict[str, Any]) -> None:
    CASE_BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_case_base()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CASE_BASE_PATH, index=False)


def validate_input(pi: PatientInput) -> list[str]:
    errors: list[str] = []
    d = asdict(pi)
    for k, v in d.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            errors.append(f"**{k}** wajib diisi.")
    for k in ZERO_AS_MISSING:
        if k in d and float(d[k]) == 0.0:
            errors.append(f"**{k}** tidak boleh 0 (nilai 0 dianggap data kosong).")
    return errors


def run_predict(pi: PatientInput) -> dict[str, Any]:
    df = load_case_base_df(CASE_BASE_PATH)
    source = "data/case_base.csv"
    if df.empty:
        for url in CASE_BASE_SEED_URLS:
            df = load_case_base_df(url)
            if not df.empty:
                source = url
                break

    artifacts = build_artifacts_from_case_base(df)
    pred = predict_cbr(artifacts, asdict(pi), k=OPTIMAL_K, weights=MULTISURF_WEIGHTS_ARRAY)
    pred["case_base_source"] = source
    return pred


# ── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi CBR")
    st.markdown(f"""
    <div style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.16);
         border-radius:10px;padding:14px 16px;font-size:1rem;line-height:1.8;color:{SIDEBAR_TEXT} !important">
      <b>Metode:</b> CBR + MultiSURF<br>
      <b>K Optimal:</b> {OPTIMAL_K} tetangga<br>
      <b>Akurasi:</b> 76,03%
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.caption("Konfigurasi K=9 adalah hasil terbaik dari pengujian 10-Fold Stratified Cross Validation.")

# ── PAGE HEADER ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:1.2rem">
  <div style="font-size:.85rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
       color:{ACCENT};background:{ACCENT_SOFT};display:inline-block;padding:4px 12px;border-radius:8px">
    Halaman Deteksi
  </div>
  <div style="font-size:2rem;font-weight:800;color:{TEXT_PRIMARY};letter-spacing:-.02em;margin-top:.4rem">
    Estimasi Risiko Diabetes Melitus Tipe 2
  </div>
  <div style="color:{TEXT_MUTED};font-size:1.05rem;margin-top:.2rem">
    Isi data pemeriksaan pasien di bawah, lalu klik <b>Proses Estimasi</b>.
  </div>
</div>
""", unsafe_allow_html=True)

# ── CARD A: INPUT DATA PEMERIKSAAN PASIEN ─────────────────────────────────
section_label("A — Input Data Pemeriksaan Pasien")
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

with st.form("patient_form", clear_on_submit=False):
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Obstetri & Metabolisme**")
        pregnancies = st.number_input(
            "Kehamilan (Pregnancies)", min_value=0, max_value=20, value=1, step=1,
            help="Jumlah riwayat kehamilan"
        )
        glucose = st.number_input(
            "Glukosa (Glucose) mg/dL", min_value=1, max_value=500, value=100, step=1,
            help="Konsentrasi glukosa plasma 2 jam setelah tes toleransi glukosa oral"
        )
        blood_pressure = st.number_input(
            "Tekanan Darah (BloodPressure) mmHg", min_value=1, max_value=300, value=72, step=1,
            help="Tekanan darah diastolik"
        )
        skin_thickness = st.number_input(
            "Ketebalan Kulit (SkinThickness) mm", min_value=1, max_value=200, value=20, step=1,
            help="Ketebalan lipatan kulit trisep"
        )

    with c2:
        st.markdown("**Hormon & Antropometri**")
        insulin = st.number_input(
            "Insulin (mu U/mL)", min_value=1, max_value=2000, value=79, step=1,
            help="Kadar insulin serum 2 jam"
        )
        bmi = st.number_input(
            "BMI (kg/m²)", min_value=0.1, max_value=80.0, value=25.0, step=0.1,
            help="Indeks massa tubuh"
        )
        dpf = st.number_input(
            "DiabetesPedigreeFunction", min_value=0.001, max_value=5.0, value=0.350, step=0.001,
            help="Skor riwayat diabetes dalam keluarga"
        )
        age = st.number_input(
            "Usia (Age) tahun", min_value=1, max_value=120, value=30, step=1,
            help="Usia pasien"
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "🔍 Proses Estimasi", type="primary", use_container_width=True
    )

with st.expander("📌 Panduan Pengisian", expanded=True):
    st.markdown(f"""
    <div style="color:{TEXT_PRIMARY};font-size:1rem;line-height:1.85">
      • <b>Glucose, BloodPressure, SkinThickness, Insulin, BMI</b> tidak boleh diisi 0 —
        nilai 0 dianggap data tidak tersedia.<br>
      • Semua nilai harus sesuai hasil pemeriksaan aktual pasien.<br>
      • Hasil estimasi bersifat <b>pendukung klinis</b>, bukan diagnosis final.
    </div>
    """, unsafe_allow_html=True)

card_close()  # tutup Card A

# ── PROSES ESTIMASI ────────────────────────────────────────────────────
if submitted:
    pi = PatientInput(
        Pregnancies=float(pregnancies), Glucose=float(glucose),
        BloodPressure=float(blood_pressure), SkinThickness=float(skin_thickness),
        Insulin=float(insulin), BMI=float(bmi),
        DiabetesPedigreeFunction=float(dpf), Age=float(age),
    )
    errors = validate_input(pi)
    if errors:
        for e in errors:
            st.error(e)
    else:
        with st.spinner("Menghitung kemiripan kasus…"):
            try:
                pred = run_predict(pi)
                st.session_state["latest_input"] = asdict(pi)
                st.session_state["latest_pred"] = pred
            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")

latest_input = st.session_state.get("latest_input")
latest_pred = st.session_state.get("latest_pred")

# ── CARD B: HASIL ESTIMASI ─────────────────────────────────────────────
section_label("B — Hasil Estimasi")

if not latest_input or not latest_pred:
    st.markdown(f"""
    <div style="text-align:center;padding:32px 0;color:{TEXT_MUTED}">
      <div style="font-size:3rem;margin-bottom:10px">🩺</div>
      <div style="font-size:1.05rem">Belum ada hasil.<br>Isi form di atas lalu klik <b>Proses Estimasi</b>.</div>
    </div>
    """, unsafe_allow_html=True)
    card_close()
else:
    predicted = int(latest_pred["predicted_outcome"])
    risk_score = float(latest_pred.get("risk_score", 0.0))
    counts = latest_pred.get("neighbor_counts", {0: 0, 1: 0})

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        if predicted == 1:
            st.markdown(f"""
            <div class="mini-metric" style="text-align:left">
              <div class="badge badge-high" style="margin-bottom:4px">⚠️ Beresiko Diabetes</div>
              <div style="font-size:.85rem;color:{TEXT_MUTED};font-weight:600">(PREDIKSI)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="mini-metric" style="text-align:left">
              <div class="badge badge-low" style="margin-bottom:4px">✅ Tidak Beresiko Diabetes</div>
              <div style="font-size:.85rem;color:{TEXT_MUTED};font-weight:600">(PREDIKSI)</div>
            </div>
            """, unsafe_allow_html=True)

    with b2:
        st.markdown(f"""<div class="mini-metric"><div class="val">{risk_score:.0%}</div><div class="lbl">Probabilitas Risiko</div></div>""", unsafe_allow_html=True)
    with b3:
        outcome_label = "Tidak Diabetes" if predicted == 0 else "Diabetes"
        st.markdown(f"""<div class="mini-metric"><div class="val">{predicted}</div><div class="lbl">Prediksi Outcome<br>({outcome_label})</div></div>""", unsafe_allow_html=True)
    with b4:
        st.markdown(f"""<div class="mini-metric"><div class="val">{counts.get(1,0)}/{OPTIMAL_K}</div><div class="lbl">K Tetangga: {OPTIMAL_K}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    preview = latest_pred.get("nearest_cases_preview")
    if isinstance(preview, pd.DataFrame) and not preview.empty:
        with st.expander("Lihat detail tetangga terdekat", expanded=True):
            st.dataframe(preview, use_container_width=True, hide_index=True)

    card_close()  # tutup Card B

    # ── CARD C: VALIDASI KLINIS (REVISE) ───────────────────────────────
    section_label("C — Validasi Klinis (Revise)")

    rc1, rc2, rc3 = st.columns(3)

    # Label pertanyaan dirender manual (hitam, dijamin) + widget dgn label bawaan disembunyikan,
    # supaya warna teks pertanyaan tidak lagi bergantung pada override CSS terhadap widget internal.
    with rc1:
        st.markdown('<span class="field-label">Apakah hasil estimasi sesuai pertimbangan klinis?</span>', unsafe_allow_html=True)
        validated = st.radio(
            "Apakah hasil estimasi sesuai pertimbangan klinis?",
            options=["Sesuai", "Tidak sesuai"], horizontal=True,
            key="validated_radio", label_visibility="collapsed",
        )

    with rc2:
        st.markdown('<span class="field-label">Tentukan outcome akhir:</span>', unsafe_allow_html=True)
        validated_outcome = st.radio(
            "Tentukan outcome akhir:",
            options=[0, 1],
            index=predicted,
            format_func=lambda v: "0 — Tidak Diabetes" if v == 0 else "1 — Diabetes",
            horizontal=True, key="override_outcome", label_visibility="collapsed",
        )

    with rc3:
        note = st.text_area(
            "Catatan klinis (opsional)",
            placeholder="Tulis catatan di sini…",
            height=80, max_chars=200, key="note_area"
        )

    if validated == "Sesuai" and validated_outcome == predicted:
        st.success(f"Outcome dikonfirmasi: **{'Diabetes' if predicted == 1 else 'Tidak Diabetes'}**")
    else:
        st.warning("Outcome dikoreksi oleh tenaga kesehatan.")

    card_close()  # tutup Card C

    # ── CARD D: SIMPAN KE BASIS KASUS (RETAIN) ─────────────────────────
    section_label("D — Simpan ke Basis Kasus (Retain)")

    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("💾 Simpan Kasus", type="primary", use_container_width=True):
            row = {
                "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                **latest_input,
                "predicted_outcome": predicted,
                "validated_outcome": int(validated_outcome),
                "validation_note": note.strip(),
            }
            retain_case(row)
            st.success("✅ Kasus berhasil disimpan ke basis kasus.")
    with sc2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.pop("latest_input", None)
            st.session_state.pop("latest_pred", None)
            st.rerun()

    card_close()  # tutup Card D

# ── CATATAN BAWAH HALAMAN ───────────────────────────────────────────────
st.markdown(f"""
<div style="background:{ACCENT_SOFT};border:1px solid #bfdbfe;border-radius:12px;padding:14px 18px;
     font-size:1rem;color:{TEXT_PRIMARY};font-weight:500;
     display:flex;align-items:center;gap:10px;margin-top:.7rem">
  <span style="font-size:1.2rem">ℹ️</span>
  <span>Catatan: Normalisasi MinMax dan pembobotan menggunakan bobot MultiSURF. Hasil estimasi bersifat pendukung klinis.</span>
</div>
""", unsafe_allow_html=True)
