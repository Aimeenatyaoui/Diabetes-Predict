from __future__ import annotations

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

/* Tombol utama — dark navy solid, teks putih */
button[kind="primary"] {{
    background: {SIDEBAR_BG} !important;
    border: 1px solid {SIDEBAR_BG} !important;
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

.mini-metric .val {{ 
    font-size: 2rem !important; 
    color: {ACCENT} !important; 
    font-weight: 800 !important; 
}}
.mini-metric .lbl {{ 
    font-size: .95rem !important; 
    color: {TEXT_MUTED} !important; 
}}

.badge {{
    font-size: 1.05rem !important; padding: 8px 14px !important; font-weight: 700 !important;
    border-radius: 10px !important; display: inline-block !important; border: 1px solid transparent !important;
}}
.badge-high {{ 
    color: {DANGER_TEXT} !important; 
    background: {DANGER_BG} !important; 
    border-color: {DANGER_BORDER} !important; 
}}
.badge-low  {{ 
    color: {SUCCESS_TEXT} !important; 
    background: {SUCCESS_BG} !important; 
    border-color: {SUCCESS_BORDER} !important; 
}}

/* Perbaikan: pastikan teks di badge terbaca */
.badge-high, .badge-high * {{ color: {DANGER_TEXT} !important; }}
.badge-low, .badge-low * {{ color: {SUCCESS_TEXT} !important; }}

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

/* Perbaikan tambahan untuk hasil estimasi - pastikan semua teks hitam */
[data-testid="stVerticalBlock"] .mini-metric *,
[data-testid="stColumn"] .mini-metric *,
.mini-metric,
.mini-metric div,
.mini-metric span {{
    color: {TEXT_PRIMARY} !important;
}}
/* Kecuali value yang tetap biru aksen */
.mini-metric .val {{
    color: {ACCENT} !important;
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
