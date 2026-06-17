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
    <div style="background:rgba(45,212,191,.1);border:1px solid rgba(45,212,191,.3);
         border-radius:10px;padding:12px 14px;font-size:.85rem;color:rgba(255,255,255,.85)">
      <b>Metode:</b> CBR + MultiSURF<br>
      <b>K Optimal:</b> {OPTIMAL_K} tetangga<br>
      <b>Akurasi:</b> 76,03%
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.caption("Konfigurasi K=9 adalah hasil terbaik dari pengujian 10-Fold Stratified Cross Validation.")

# ── PAGE HEADER ──────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1rem">
  <div style="font-size:.72rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#2dd4bf">
    Halaman Deteksi
  </div>
  <div style="font-size:1.5rem;font-weight:800;color:#fff;letter-spacing:-.02em">
    Estimasi Risiko Diabetes Melitus Tipe 2
  </div>
  <div style="color:rgba(255,255,255,.6);font-size:.88rem">
    Isi data pemeriksaan pasien di bawah, lalu klik <b>Proses Estimasi</b>.
  </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.1, 1], gap="large")

# ── LEFT: INPUT FORM ──────────────────────────────────────────────────────
with left:
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

    card_close()

    # GUIDE CARD
    card_open()
    section_label("Panduan Pengisian")
    st.markdown("""
    <div style="color:rgba(255,255,255,.72);font-size:.82rem;line-height:1.75">
      • <b>Glucose, BloodPressure, SkinThickness, Insulin, BMI</b> tidak boleh diisi 0 —
        nilai 0 dianggap data tidak tersedia.<br>
      • Semua nilai harus sesuai hasil pemeriksaan aktual pasien.<br>
      • Hasil estimasi bersifat <b>pendukung klinis</b>, bukan diagnosis final.
    </div>
    """, unsafe_allow_html=True)
    card_close()

# ── RIGHT: RESULT PANEL ────────────────────────────────────────────────────
with right:
    section_label("B — Hasil Estimasi")

    if submitted:
        pi = PatientInput(
            Pregnancies=float(pregnancies), Glucose=float(glucose),
            BloodPressure=float(blood_pressure), SkinThickness=float(skin_thickness),
            Insulin=float(insulin), BMI=float(bmi),
            DiabetesPedigreeFunction=float(dpf), Age=float(age),
        )
        errors = validate_input(pi)
        if errors:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            for e in errors:
                st.error(e)
            card_close()
            st.stop()

        with st.spinner("Menghitung kemiripan kasus…"):
            try:
                pred = run_predict(pi)
                st.session_state["latest_input"] = asdict(pi)
                st.session_state["latest_pred"] = pred
            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")
                card_close()
                st.stop()

    latest_input = st.session_state.get("latest_input")
    latest_pred = st.session_state.get("latest_pred")

    if not latest_input or not latest_pred:
        st.markdown("""
        <div style="text-align:center;padding:32px 0;color:rgba(255,255,255,.4)">
          <div style="font-size:2.5rem;margin-bottom:8px">🩺</div>
          <div style="font-size:.88rem">Belum ada hasil.<br>Isi form di kiri lalu klik <b>Proses Estimasi</b>.</div>
        </div>
        """, unsafe_allow_html=True)
        card_close()
    else:
        predicted = int(latest_pred["predicted_outcome"])
        risk_score = float(latest_pred.get("risk_score", 0.0))
        counts = latest_pred.get("neighbor_counts", {0: 0, 1: 0})

        # Badge hasil
        if predicted == 1:
            st.markdown('<div class="badge badge-high">⚠️ RISIKO TINGGI</div>', unsafe_allow_html=True)
            st.markdown("<div style='color:rgba(255,255,255,.7);font-size:.88rem;margin-bottom:12px'>Sistem mengestimasi pasien <b>berisiko diabetes</b>. Diperlukan evaluasi klinis lebih lanjut.</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge badge-low">✅ RISIKO RENDAH</div>', unsafe_allow_html=True)
            st.markdown("<div style='color:rgba(255,255,255,.7);font-size:.88rem;margin-bottom:12px'>Sistem mengestimasi pasien <b>tidak berisiko diabetes</b> saat ini.</div>", unsafe_allow_html=True)

        # Metric row
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""<div class="mini-metric"><div class="val">{risk_score:.0%}</div><div class="lbl">Skor Risiko</div></div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""<div class="mini-metric"><div class="val">{counts.get(1,0)}/{OPTIMAL_K}</div><div class="lbl">Tetangga DM</div></div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""<div class="mini-metric"><div class="val">{OPTIMAL_K}</div><div class="lbl">K Tetangga</div></div>""", unsafe_allow_html=True)

        # Neighbor preview
        preview = latest_pred.get("nearest_cases_preview")
        if isinstance(preview, pd.DataFrame) and not preview.empty:
            with st.expander("Lihat detail tetangga terdekat"):
                st.dataframe(preview, use_container_width=True, hide_index=True)

        divider()

        # ── REVISE ──
        section_label("C — Validasi Klinis (Revise)")
        validated = st.radio(
            "Apakah hasil estimasi sesuai pertimbangan klinis?",
            options=["Sesuai", "Tidak sesuai"], horizontal=True,
            key="validated_radio"
        )

        if validated == "Sesuai":
            validated_outcome = predicted
            st.success(f"Outcome dikonfirmasi: **{'Diabetes' if predicted == 1 else 'Tidak Diabetes'}**")
        else:
            validated_outcome = st.radio(
                "Tentukan outcome akhir:",
                options=[0, 1],
                format_func=lambda v: "0 — Tidak Diabetes" if v == 0 else "1 — Diabetes",
                horizontal=True, key="override_outcome"
            )
            st.warning("Outcome dikoreksi oleh tenaga kesehatan.")

        note = st.text_area(
            "Catatan klinis (opsional)",
            placeholder="Contoh: hasil lab tambahan, pertimbangan rujukan, kondisi khusus pasien…",
            height=80, key="note_area"
        )

        divider()

        # ── RETAIN ──
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

        card_close()
