import streamlit as st
from ui.theme import card_close, card_open, divider, inject_glass_theme, section_label

st.set_page_config(page_title="About — Type 2 DM", page_icon=":material/info:", layout="wide")
inject_glass_theme()

st.markdown("""
<div style="margin-bottom:1.5rem">
  <div style="font-size:.72rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#2dd4bf">
    System Information
  </div>
  <div style="font-size:1.5rem;font-weight:800;color:#fff;letter-spacing:-.02em">
    About the System
  </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    card_open()
    section_label("System Description")
    st.markdown("""
    <div style="color:rgba(255,255,255,.8);font-size:.9rem;line-height:1.8">
      This system is a clinical support application for estimating the risk of
      <b>Type 2 Diabetes Mellitus</b>, developed using the
      <b>Case-Based Reasoning (CBR)</b> method with
      <b>MultiSURF</b> feature weighting.
      <br><br>
      The system is designed for use by <b>healthcare professionals</b> in
      primary healthcare facilities to support the initial screening process for patients.
    </div>
    """, unsafe_allow_html=True)
    divider()
    section_label("Implemented CBR Cycle")
    for step, desc in [
        ("Retrieve", "The system searches for historical cases that are most similar to the patient data using Weighted Euclidean Distance."),
        ("Reuse", "The system adapts the solution from the selected cases through majority voting of the classes of the K nearest neighbors."),
        ("Revise", "Healthcare professionals evaluate the estimation result based on clinical considerations and can correct the outcome."),
        ("Retain", "Validated cases are stored in the case base to improve system performance in the future."),
    ]:
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-bottom:10px;align-items:flex-start">
          <div style="background:rgba(45,212,191,.15);border:1px solid rgba(45,212,191,.3);
               border-radius:8px;padding:4px 10px;font-size:.78rem;font-weight:700;
               color:#2dd4bf;white-space:nowrap;margin-top:1px">{step}</div>
          <div style="color:rgba(255,255,255,.72);font-size:.87rem;line-height:1.6">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    card_close()

with col2:
    card_open()
    section_label("Model Performance (K=9)")
    for metric, val, color in [
        ("Accuracy", "76,03%", "#2dd4bf"),
        ("Precision", "65,87%", "#22c55e"),
        ("Recall", "64,54%", "#fbbf24"),
        ("F1-Score", "64,95%", "#a78bfa"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
             padding:10px 0;border-bottom:1px solid rgba(255,255,255,.07)">
          <div style="font-size:.88rem;color:rgba(255,255,255,.75)">{metric}</div>
          <div style="font-size:1.1rem;font-weight:700;color:{color}">{val}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div style="color:rgba(255,255,255,.45);font-size:.76rem;margin-top:10px">
      Results from Stratified 10-Fold Cross Validation.
    </div>
    """, unsafe_allow_html=True)
    divider()
    section_label("Input Features")
    feats = [
        ("Pregnancies", "Number of pregnancies"),
        ("Glucose", "Plasma glucose level (mg/dL)"),
        ("BloodPressure", "Diastolic blood pressure (mmHg)"),
        ("SkinThickness", "Triceps skin fold thickness (mm)"),
        ("Insulin", "2-hour serum insulin level (mu U/mL)"),
        ("BMI", "Body mass index (kg/m²)"),
        ("DiabetesPedigreeFunction", "Family history of diabetes score"),
        ("Age", "Patient age (years)"),
    ]
    for name, desc in feats:
        st.markdown(f"""
        <div style="margin-bottom:5px;font-size:.83rem">
          <span style="color:#2dd4bf;font-weight:600">{name}</span>
          <span style="color:rgba(255,255,255,.55)"> — {desc}</span>
        </div>
        """, unsafe_allow_html=True)
    card_close()
