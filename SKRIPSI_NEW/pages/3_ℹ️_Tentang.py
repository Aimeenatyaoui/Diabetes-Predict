import streamlit as st
from ui.theme import card_close, card_open, divider, inject_glass_theme, section_label

st.set_page_config(page_title="Tentang — DM Tipe 2", page_icon="ℹ️", layout="wide")
inject_glass_theme()

st.markdown("""
<div style="margin-bottom:1.5rem">
  <div style="font-size:.72rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#2dd4bf">
    Informasi Sistem
  </div>
  <div style="font-size:1.5rem;font-weight:800;color:#fff;letter-spacing:-.02em">
    Tentang Sistem
  </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    card_open()
    section_label("Deskripsi Sistem")
    st.markdown("""
    <div style="color:rgba(255,255,255,.8);font-size:.9rem;line-height:1.8">
      Sistem ini merupakan aplikasi pendukung klinis untuk estimasi risiko
      <b>Diabetes Melitus Tipe 2</b> yang dikembangkan menggunakan metode
      <b>Case-Based Reasoning (CBR)</b> dengan pembobotan fitur <b>MultiSURF</b>.
      <br><br>
      Sistem dirancang untuk digunakan oleh <b>tenaga kesehatan</b> di fasilitas
      kesehatan tingkat 1 dalam membantu proses skrining awal pasien.
    </div>
    """, unsafe_allow_html=True)
    divider()
    section_label("Siklus CBR yang Diimplementasikan")
    for step, desc in [
        ("Retrieve", "Sistem mencari kasus-kasus historis yang paling mirip dengan data pasien menggunakan Weighted Euclidean Distance."),
        ("Reuse", "Sistem mengadaptasi solusi dari kasus terpilih melalui voting mayoritas kelas dari K tetangga terdekat."),
        ("Revise", "Tenaga kesehatan mengevaluasi hasil estimasi berdasarkan pertimbangan klinis dan dapat mengoreksi outcome."),
        ("Retain", "Kasus yang telah divalidasi disimpan ke basis kasus untuk meningkatkan kinerja sistem di masa mendatang."),
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
    section_label("Performa Model (K=9)")
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
      Hasil pengujian Stratified 10-Fold Cross Validation.
    </div>
    """, unsafe_allow_html=True)
    divider()
    section_label("Fitur Input")
    feats = [
        ("Pregnancies", "Jumlah riwayat kehamilan"),
        ("Glucose", "Kadar glukosa plasma (mg/dL)"),
        ("BloodPressure", "Tekanan darah diastolik (mmHg)"),
        ("SkinThickness", "Ketebalan lipatan kulit trisep (mm)"),
        ("Insulin", "Kadar insulin serum 2 jam (mu U/mL)"),
        ("BMI", "Indeks massa tubuh (kg/m²)"),
        ("DiabetesPedigreeFunction", "Skor riwayat diabetes keluarga"),
        ("Age", "Usia pasien (tahun)"),
    ]
    for name, desc in feats:
        st.markdown(f"""
        <div style="margin-bottom:5px;font-size:.83rem">
          <span style="color:#2dd4bf;font-weight:600">{name}</span>
          <span style="color:rgba(255,255,255,.55)"> — {desc}</span>
        </div>
        """, unsafe_allow_html=True)
    card_close()
