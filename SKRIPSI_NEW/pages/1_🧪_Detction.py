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

st.set_page_config(page_title="Detection — Type 2 DM", page_icon=":material/medical_services:", layout="wide")
inject_glass_theme()

# ── DESIGN TOKENS ──────────────────────────────────────────────────────
BG_PAGE = "#f1f5f9"        
BG_CARD = "#ffffff"        
BORDER_CARD = "#e2e8f0"    
TEXT_PRIMARY = "#0f172a"   
TEXT_MUTED = "#64748b"     
ACCENT = "#2563eb"         
ACCENT_SOFT = "#dbeafe"    
SIDEBAR_BG = "#0f172a"     
SIDEBAR_TEXT = "#f1f5f9"   

DANGER_TEXT, DANGER_BG, DANGER_BORDER = "#b91c1c", "#fef2f2", "#fecaca"
SUCCESS_TEXT, SUCCESS_BG, SUCCESS_BORDER = "#15803d", "#f0fdf4", "#bbf7d0"

# ── GLOBAL STYLE ──────────────────────────────────────────────────────────
st.markdown(f"""
<style>
:root {{ color-scheme: light; }}

.stApp {{ background: {BG_PAGE} !important; }}
html, body, [class^="css"] {{ font-size: 16px !important; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {SIDEBAR_BG} !important; }}
section[data-testid="stSidebar"] * {{ color: {SIDEBAR_TEXT} !important; font-size: 1rem !important; }}

/* Cards */
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

/* Ensure all text in the main area is black EXCEPT where specifically defined */
div.main *, [data-testid="stMain"] * {{ color: {TEXT_PRIMARY} !important; }}

label, .stMarkdown p, .stMarkdown li, .stCaption, p, span {{
    font-size: 1rem !important;
    line-height: 1.6 !important;
    color: {TEXT_PRIMARY} !important;
}}

input, textarea, .stNumberInput input {{
    font-size: 1.05rem !important;
    color: {TEXT_PRIMARY} !important;
    background: #ffffff !important;
    border: 1px solid {BORDER_CARD} !important;
    border-radius: 8px !important;
}}

/* ALL BUTTONS */
button {{ 
    border-radius: 8px !important; 
}}
button p {{ 
    font-size: 1rem !important; 
    font-weight: 600 !important; 
}}

/* PRIMARY BUTTON - DARK NAVY */
button[kind="primary"] {{
    background: #0f172a !important;
    border: 1px solid #0f172a !important;
    color: #ffffff !important;
}}
button[kind="primary"] * {{
    color: #ffffff !important;
}}
button[kind="primary"] p {{
    color: #ffffff !important;
}}

/* SECONDARY BUTTON - RESET */
button:not([kind="primary"]) {{
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    color: #0f172a !important;
}}
button:not([kind="primary"]) * {{
    color: #0f172a !important;
}}

/* RADIO BUTTON - SPECIAL FIX */
div[data-testid="stRadio"] {{
    color: {TEXT_PRIMARY} !important;
}}
div[data-testid="stRadio"] * {{
    color: {TEXT_PRIMARY} !important;
}}
div[data-testid="stRadio"] label {{
    color: {TEXT_PRIMARY} !important;
}}
div[data-testid="stRadio"] span {{
    color: {TEXT_PRIMARY} !important;
}}
div[data-testid="stRadio"] p {{
    color: {TEXT_PRIMARY} !important;
}}
/* Radio button horizontal */
div[role="radiogroup"] {{
    color: {TEXT_PRIMARY} !important;
}}
div[role="radiogroup"] * {{
    color: {TEXT_PRIMARY} !important;
}}
div[role="radiogroup"] label {{
    color: {TEXT_PRIMARY} !important;
}}
div[role="radiogroup"] span {{
    color: {TEXT_PRIMARY} !important;
}}
/* Styling for selected radio options */
div[role="radiogroup"] div[data-testid="stMarkdownContainer"] {{
    color: {TEXT_PRIMARY} !important;
}}
div[role="radiogroup"] div[data-testid="stMarkdownContainer"] * {{
    color: {TEXT_PRIMARY} !important;
}}
/* Force all text inside radio to be black */
.stRadio div[role="radiogroup"] label span {{
    color: #0f172a !important;
}}
.stRadio label span:last-child {{
    color: #0f172a !important;
}}

/* Text area */
div[data-testid="stTextArea"] * {{
    color: {TEXT_PRIMARY} !important;
}}

[data-testid="stDataFrame"] * {{ font-size: .95rem !important; color: {TEXT_PRIMARY} !important; }}

div[data-testid="stAlert"] {{ border-radius: 10px !important; font-size: 1rem !important; }}

/* ESTIMATION RESULTS */
.mini-metric {{
    color: {TEXT_PRIMARY} !important;
}}
.mini-metric * {{
    color: {TEXT_PRIMARY} !important;
}}
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
    font-size: 1.05rem !important; 
    padding: 8px 14px !important; 
    font-weight: 700 !important;
    border-radius: 10px !important; 
    display: inline-block !important; 
    border: 1px solid transparent !important;
}}
.badge-high {{ 
    color: {DANGER_TEXT} !important; 
    background: {DANGER_BG} !important; 
    border-color: {DANGER_BORDER} !important; 
}}
.badge-low {{ 
    color: {SUCCESS_TEXT} !important; 
    background: {SUCCESS_BG} !important; 
    border-color: {SUCCESS_BORDER} !important; 
}}
.badge-high *,
.badge-low * {{
    color: inherit !important;
}}

/* Card titles */
[style*="2dd4bf"], [style*="45,212,191"], [style*="14b8a6"], [style*="0d9488"] {{
    color: {TEXT_PRIMARY} !important;
}}
[class*="section-label"], [class*="section-title"], [class*="card-title"],
[class*="section-label"] *, [class*="section-title"] *, [class*="card-title"] * {{
    color: {TEXT_PRIMARY} !important;
}}

/* Mini label */
.field-label {{
    font-weight: 700 !important; 
    color: {TEXT_PRIMARY} !important;
    font-size: 1rem !important; 
    margin-bottom: 6px !important; 
    display: block !important;
}}

/* Alert messages */
.stAlert {{
    color: {TEXT_PRIMARY} !important;
}}
.stAlert * {{
    color: {TEXT_PRIMARY} !important;
}}
.stSuccess {{
    color: {TEXT_PRIMARY} !important;
}}
.stWarning {{
    color: {TEXT_PRIMARY} !important;
}}

/* EXPANDER */
.streamlit-expanderHeader {{
    color: {TEXT_PRIMARY} !important;
}}
.streamlit-expanderHeader * {{
    color: {TEXT_PRIMARY} !important;
}}

/* COLUMN */
div[data-testid="column"] * {{
    color: {TEXT_PRIMARY} !important;
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
            errors.append(f"**{k}** is required.")
    for k in ZERO_AS_MISSING:
        if k in d and float(d[k]) == 0.0:
            errors.append(f"**{k}** cannot be 0 (a value of 0 is considered missing data).")
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
    st.markdown("### :material/settings: CBR Configuration")
    st.markdown(f"""
    <div style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.16);
         border-radius:10px;padding:14px 16px;font-size:1rem;line-height:1.8;color:{SIDEBAR_TEXT} !important">
      <b>Method:</b> CBR + MultiSURF<br>
      <b>Optimal K:</b> {OPTIMAL_K} neighbors<br>
      <b>Accuracy:</b> 76,03%
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.caption("The K=9 configuration is the best result from 10-Fold Stratified Cross Validation.")

# ── PAGE HEADER ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:1.2rem">
  <div style="font-size:.85rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
       color:{ACCENT};background:{ACCENT_SOFT};display:inline-block;padding:4px 12px;border-radius:8px">
    Detection Page
  </div>
  <div style="font-size:2rem;font-weight:800;color:{TEXT_PRIMARY};letter-spacing:-.02em;margin-top:.4rem">
    Type 2 Diabetes Mellitus Risk Estimation
  </div>
  <div style="color:{TEXT_MUTED};font-size:1.05rem;margin-top:.2rem">
    Enter the patient examination data below, then click <b>Process Estimation</b>.
  </div>
</div>
""", unsafe_allow_html=True)

# ── CARD A: PATIENT EXAMINATION INPUT ─────────────────────────────────────
section_label("A — Patient Examination Input")
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

with st.form("patient_form", clear_on_submit=False):
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Obstetrics & Metabolism**")
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=1, step=1,
            help="Number of pregnancies"
        )
        glucose = st.number_input(
            "Glucose (Glucose) mg/dL", min_value=1, max_value=500, value=100, step=1,
            help="Plasma glucose concentration 2 hours after an oral glucose tolerance test"
        )
        blood_pressure = st.number_input(
            "Blood Pressure (BloodPressure) mmHg", min_value=1, max_value=300, value=72, step=1,
            help="Diastolic blood pressure"
        )
        skin_thickness = st.number_input(
            "Skin Thickness (SkinThickness) mm", min_value=1, max_value=200, value=20, step=1,
            help="Triceps skin fold thickness"
        )

    with c2:
        st.markdown("**Hormones & Anthropometry**")
        insulin = st.number_input(
            "Insulin (mu U/mL)", min_value=1, max_value=2000, value=79, step=1,
            help="2-hour serum insulin level"
        )
        bmi = st.number_input(
            "BMI (kg/m²)", min_value=0.1, max_value=80.0, value=25.0, step=0.1,
            help="Body mass index"
        )
        dpf = st.number_input(
            "DiabetesPedigreeFunction", min_value=0.001, max_value=5.0, value=0.350, step=0.001,
            help="Family diabetes history score"
        )
        age = st.number_input(
            "Age (Age) years", min_value=1, max_value=120, value=30, step=1,
            help="Patient age"
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        ":material/search: Process Estimation", type="primary", use_container_width=True
    )

with st.expander(":material/push_pin: Input Guidelines", expanded=True):
    st.markdown(f"""
    <div style="color:{TEXT_PRIMARY};font-size:1rem;line-height:1.85">
      • <b>Glucose, BloodPressure, SkinThickness, Insulin, BMI</b> cannot be filled with 0 —
        a value of 0 indicates that the data is unavailable.<br>
      • All values must match the patient's actual examination results.<br>
      • The estimation result is intended as <b>clinical support</b>, not a final diagnosis.
    </div>
    """, unsafe_allow_html=True)

card_close()

# ── ESTIMATION PROCESS ───────────────────────────────────────────────────
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
        with st.spinner("Calculating case similarity…"):
            try:
                pred = run_predict(pi)
                st.session_state["latest_input"] = asdict(pi)
                st.session_state["latest_pred"] = pred
            except Exception as e:
                st.error(f"Prediction failed: {e}")

latest_input = st.session_state.get("latest_input")
latest_pred = st.session_state.get("latest_pred")

# ── CARD B: ESTIMATION RESULTS ───────────────────────────────────────────
section_label("B — Estimation Results & Clinical Recommendations")

if not latest_input or not latest_pred:
    st.markdown(f"""
    <div style="text-align:center;padding:32px 0;color:{TEXT_MUTED}">
      <div style="font-size:3rem;margin-bottom:10px">
          <span class="material-symbols-outlined">medical_services</span>
      </div>
      <div style="font-size:1.05rem">No result yet.<br>Isi form di atas lalu klik <b>Process Estimation</b>.</div>
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
              <div class="badge badge-high" style="margin-bottom:4px">⚠️ At Risk of Diabetes</div>
              <div style="font-size:.85rem;color:{TEXT_MUTED};font-weight:600">(PREDICTION)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="mini-metric" style="text-align:left">
              <div class="badge badge-low" style="margin-bottom:4px">✅ Tidak At Risk of Diabetes</div>
              <div style="font-size:.85rem;color:{TEXT_MUTED};font-weight:600">(PREDICTION)</div>
            </div>
            """, unsafe_allow_html=True)

    with b2:
        st.markdown(f"""<div class="mini-metric"><div class="val">{risk_score:.0%}</div><div class="lbl">Risk Probability</div></div>""", unsafe_allow_html=True)
    with b3:
        outcome_label = "No Diabetes" if predicted == 0 else "Diabetes"
        st.markdown(f"""<div class="mini-metric"><div class="val">{predicted}</div><div class="lbl">Predicted Outcome<br>({outcome_label})</div></div>""", unsafe_allow_html=True)
    with b4:
        st.markdown(f"""<div class="mini-metric"><div class="val">{counts.get(1,0)}/{OPTIMAL_K}</div><div class="lbl">K Neighbors: {OPTIMAL_K}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── CLINICAL RECOMMENDATIONS FOR HEALTHCARE PROFESSIONALS (NEUTRAL & WITHOUT OVERCLAIMING) ────────────────
    if predicted == 1:
        st.markdown(f"""
        <div style="background:{DANGER_BG}; border:1px solid {DANGER_BORDER}; border-radius:12px; padding:18px; margin-bottom:16px;">
            <div style="color:{DANGER_TEXT} !important; font-size:1.1rem; font-weight:700; margin-bottom:8px; display:flex; align-items:center; gap:8px;">
                <span>
                    <span class="material-symbols-outlined">description</span> Recommended Next Actions (Risk Status: At Risk)
                </span>
            </div>
            <ul style="margin:0; padding-left:22px; color:{TEXT_PRIMARY} !important; line-height:1.7;">
                <li><b>Further Examination:</b> Schedule a confirmatory blood glucose test (such as fasting blood glucose, OGTT, or HbA1c) according to standard operating procedures.</li>
                <li><b>Patient Education:</b> Provide healthy lifestyle education covering dietary management and regular physical activity.</li>
                <li><b>Periodic Evaluation:</b> Schedule a reassessment of the patient's condition and consider referral to a specialist or attending physician if necessary.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:{SUCCESS_BG}; border:1px solid {SUCCESS_BORDER}; border-radius:12px; padding:18px; margin-bottom:16px;">
            <div style="color:{SUCCESS_TEXT} !important; font-size:1.1rem; font-weight:700; margin-bottom:8px; display:flex; align-items:center; gap:8px;">
                <span>
                    <span class="material-symbols-outlined">description</span> Recommended Next Actions (Risk Status: Not at Risk)
                </span>
            </div>
            <ul style="margin:0; padding-left:22px; color:{TEXT_PRIMARY} !important; line-height:1.7;">
                <li><b>Preventive Education:</b> Advise the patient to maintain a healthy lifestyle and ideal body weight.</li>
                <li><b>Routine Monitoring:</b> Recommend blood glucose monitoring and regular health examinations according to primary care service standards.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    preview = latest_pred.get("nearest_cases_preview")
    if isinstance(preview, pd.DataFrame) and not preview.empty:
        with st.expander("Lihat detail neighbors terdekat", expanded=True):
            st.dataframe(preview, use_container_width=True, hide_index=True)

    card_close()

    # ── CARD C: CLINICAL VALIDATION (REVISE) ────────────────────────────────
    section_label("C — Clinical Validation (Revise)")

    rc1, rc2, rc3 = st.columns(3)

    with rc1:
        st.markdown('<span class="field-label">Is the estimation result consistent with clinical judgment?</span>', unsafe_allow_html=True)
        validated = st.radio(
            "Is the estimation result consistent with clinical judgment?",
            options=["Consistent", "Not consistent"], 
            horizontal=True,
            key="validated_radio", 
            label_visibility="collapsed",
        )

    with rc2:
        st.markdown('<span class="field-label">Determine the final outcome:</span>', unsafe_allow_html=True)
        validated_outcome = st.radio(
            "Determine the final outcome:",
            options=[0, 1],
            index=predicted,
            format_func=lambda v: "0 — No Diabetes" if v == 0 else "1 — Diabetes",
            horizontal=True, 
            key="override_outcome", 
            label_visibility="collapsed",
        )

    with rc3:
        note = st.text_area(
            "Clinical notes (optional)",
            placeholder="Write notes here…",
            height=80, 
            max_chars=200, 
            key="note_area"
        )

    if validated == "Consistent" and validated_outcome == predicted:
        st.success(f"Outcome confirmed: **{'Diabetes' if predicted == 1 else 'No Diabetes'}**")
    else:
        st.warning("Outcome corrected by the healthcare professional.")

    card_close()

    # ── CARD D: SAVE TO CASE BASE (RETAIN) ─────────────────────────────────
    section_label("D — Save to Case Base (Retain)")

    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button(
            ":material/save: Save Case",
            type="primary",
            use_container_width=True
        ):
            row = {
                "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                **latest_input,
                "predicted_outcome": predicted,
                "validated_outcome": int(validated_outcome),
                "validation_note": note.strip(),
            }
            retain_case(row)
            st.success("✅ Case successfully saved to the case base.")
    with sc2:
        if st.button(
            ":material/refresh: Reset",
            use_container_width=True
        ):
            st.session_state.pop("latest_input", None)
            st.session_state.pop("latest_pred", None)
            st.rerun()

    card_close()

# ── PAGE FOOTER NOTE ─────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{ACCENT_SOFT};border:1px solid #bfdbfe;border-radius:12px;padding:14px 18px;
     font-size:1rem;color:{TEXT_PRIMARY};font-weight:500;
     display:flex;align-items:center;gap:10px;margin-top:.7rem">
  <span class="material-symbols-outlined"
        style="font-size:20px;">
      info
  </span>
  <span>Note: Min-Max normalization and weighting use MultiSURF weights. The estimation result is intended as clinical support.</span>
</div>
""", unsafe_allow_html=True)
