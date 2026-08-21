from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.theme import card_close, card_open, divider, inject_glass_theme, section_label

st.set_page_config(page_title="Case Base — Type 2 DM", page_icon=":material/menu_book:", layout="wide")
inject_glass_theme()

# ── DESIGN TOKENS — Same as Detection page ─────────────────────────
BG_PAGE = "#f1f5f9"        # slate-100 — page background
BG_CARD = "#ffffff"        # kartu solid putih
BORDER_CARD = "#e2e8f0"    # slate-200 — garis tepi kartu
TEXT_PRIMARY = "#0f172a"   # slate-900 — teks utama
TEXT_MUTED = "#64748b"     # slate-500 — teks sekunder/caption
ACCENT = "#2563eb"         # blue-600 — warna aksen
ACCENT_SOFT = "#dbeafe"    # blue-100 — latar chip/label kecil
SIDEBAR_BG = "#0f172a"     # slate-900 — sidebar gelap
SIDEBAR_TEXT = "#f1f5f9"   # light sidebar text

DANGER_TEXT, DANGER_BG, DANGER_BORDER = "#b91c1c", "#fef2f2", "#fecaca"
SUCCESS_TEXT, SUCCESS_BG, SUCCESS_BORDER = "#15803d", "#f0fdf4", "#bbf7d0"

# ── STYLE GLOBAL ──────────────────────────────────────────────────────────
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

/* Ensure all text in the main area is black */
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

/* BUTTON */
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

/* SECONDARY BUTTON */
button:not([kind="primary"]) {{
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    color: #0f172a !important;
}}
button:not([kind="primary"]) * {{
    color: #0f172a !important;
}}

/* SELECTBOX - for sidebar filter */
div[data-testid="stSelectbox"] * {{
    color: {SIDEBAR_TEXT} !important;
}}
div[data-testid="stSelectbox"] label {{
    color: {SIDEBAR_TEXT} !important;
}}
div[data-testid="stSelectbox"] div {{
    color: {SIDEBAR_TEXT} !important;
}}

/* Text input in sidebar */
section[data-testid="stSidebar"] input {{
    color: {SIDEBAR_TEXT} !important;
    background: rgba(255,255,255,.1) !important;
    border: 1px solid rgba(255,255,255,.2) !important;
}}

[data-testid="stDataFrame"] * {{ 
    font-size: .95rem !important; 
    color: {TEXT_PRIMARY} !important; 
}}

div[data-testid="stAlert"] {{ 
    border-radius: 10px !important; 
    font-size: 1rem !important; 
}}

/* METRICS */
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

/* Card titles */
[style*="2dd4bf"], [style*="45,212,191"], [style*="14b8a6"], [style*="0d9488"] {{
    color: {TEXT_PRIMARY} !important;
}}
[class*="section-label"], [class*="section-title"], [class*="card-title"],
[class*="section-label"] *, [class*="section-title"] *, [class*="card-title"] * {{
    color: {TEXT_PRIMARY} !important;
}}

/* Info message */
.stInfo {{
    color: {TEXT_PRIMARY} !important;
}}
.stInfo * {{
    color: {TEXT_PRIMARY} !important;
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

/* Muted text */
.muted {{
    color: {TEXT_MUTED} !important;
}}
</style>
""", unsafe_allow_html=True)

CASE_BASE_PATH = Path("data/case_base.csv")

# ── PAGE HEADER ───────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:1.2rem">
  <div style="font-size:.85rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
       color:{ACCENT};background:{ACCENT_SOFT};display:inline-block;padding:4px 12px;border-radius:8px">
    Data Management
  </div>
  <div style="font-size:2rem;font-weight:800;color:{TEXT_PRIMARY};letter-spacing:-.02em;margin-top:.4rem">
    Case Base
  </div>
  <div style="color:{TEXT_MUTED};font-size:1.05rem;margin-top:.2rem">
    All validated and saved cases (<b>Retain</b> stage of the CBR cycle).
  </div>
</div>
""", unsafe_allow_html=True)

def load_cases() -> pd.DataFrame:
    if not CASE_BASE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CASE_BASE_PATH)

df = load_cases()

# ── STATS ─────────────────────────────────────────────────────────────────
section_label("Statistik Case Base")
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

if df.empty:
    st.info("ℹ️ No cases have been saved yet. Use the **Detection** page to add a new case.")
    card_close()
    st.stop()

c1, c2, c3, c4 = st.columns(4, gap="small")
total = len(df)
dm = int((df["validated_outcome"] == 1).sum()) if "validated_outcome" in df.columns else 0
non_dm = total - dm
match_pct = 0
if "predicted_outcome" in df.columns and "validated_outcome" in df.columns:
    match = (df["predicted_outcome"] == df["validated_outcome"]).sum()
    match_pct = round(match / total * 100, 1)

with c1:
    st.markdown(f"""
    <div class="mini-metric">
      <div class="val">{total}</div>
      <div class="lbl">Total Cases</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="mini-metric">
      <div class="val">{dm}</div>
      <div class="lbl">Diabetes Cases</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="mini-metric">
      <div class="val">{non_dm}</div>
      <div class="lbl">Non-Diabetes Cases</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="mini-metric">
      <div class="val">{match_pct}%</div>
      <div class="lbl">Validation Consistency</div>
    </div>""", unsafe_allow_html=True)

card_close()

# ── TABLE ──────────────────────────────────────────────────────────────────
section_label("Saved Cases")
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# Sidebar filter
with st.sidebar:
    st.markdown("### :material/search: Filter")
    outcome_filter = st.selectbox(
        "Validation Outcome",
        options=["All", "Diabetes (1)", "Non-Diabetes (0)"],
    )
    search_ts = st.text_input("Search timestamp…", "")

filtered = df.copy()
if outcome_filter == "Diabetes (1)":
    filtered = filtered[filtered["validated_outcome"] == 1]
elif outcome_filter == "Non-Diabetes (0)":
    filtered = filtered[filtered["validated_outcome"] == 0]
if search_ts and "timestamp" in filtered.columns:
    filtered = filtered[filtered["timestamp"].astype(str).str.contains(search_ts)]

st.markdown(f"<div style='color:{TEXT_MUTED};margin-bottom:8px'>Showing {len(filtered)} of {total} cases.</div>", unsafe_allow_html=True)

# Style outcome column - using softer colors
def style_row(row):
    color = "rgba(248,113,113,.18)" if row.get("validated_outcome") == 1 else "rgba(34,197,94,.10)"
    return [f"background: {color}"] * len(row)

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
    column_config={
        "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
        "Pregnancies": st.column_config.NumberColumn("Pregnancies", format="%d", width="small"),
        "Glucose": st.column_config.NumberColumn("Glucose", format="%d", width="small"),
        "BloodPressure": st.column_config.NumberColumn("Blood Pressure", format="%d", width="small"),
        "SkinThickness": st.column_config.NumberColumn("Skin Thickness", format="%d", width="small"),
        "Insulin": st.column_config.NumberColumn("Insulin", format="%d", width="small"),
        "BMI": st.column_config.NumberColumn("BMI", format="%.1f", width="small"),
        "DiabetesPedigreeFunction": st.column_config.NumberColumn("DPF", format="%.3f", width="small"),
        "Age": st.column_config.NumberColumn("Age", format="%d", width="small"),
        "predicted_outcome": st.column_config.NumberColumn("Pred.", format="%d", width="small"),
        "validated_outcome": st.column_config.NumberColumn("Valid.", format="%d", width="small"),
        "validation_note": st.column_config.TextColumn("Clinical Notes", width="medium"),
    }
)
card_close()

# ── EXPORT ────────────────────────────────────────────────────────────────
section_label("Export Data")
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
ec1, ec2 = st.columns(2, gap="small")
with ec1:
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        ":material/download: Download as CSV",
        data=csv_bytes,
        file_name="case_base.csv",
        mime="text/csv",
        use_container_width=True,
    )
with ec2:
    if st.button(":material/delete: Delete all cases (reset)", use_container_width=True):
        if st.session_state.get("confirm_delete"):
            CASE_BASE_PATH.unlink(missing_ok=True)
            st.success(":material/check_circle: Case base successfully deleted.")
            st.session_state.pop("confirm_delete")
            st.rerun()
        else:
            st.session_state["confirm_delete"] = True
            st.warning(":material/warning: Click once more to confirm deletion of all cases.")
card_close()
