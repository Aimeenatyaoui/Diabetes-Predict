import streamlit as st
from ui.theme import inject_glass_theme, mini_metric_html

st.set_page_config(
    page_title="Estimasi Risiko DM Tipe 2 | CBR",
    page_icon="🩺",
    layout="wide",
)

inject_glass_theme()

# ── HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem">
  <div style="font-size:.75rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#2dd4bf;margin-bottom:6px">
    Sistem Pendukung Klinis
  </div>
  <div style="font-size:1.85rem;font-weight:800;letter-spacing:-.025em;line-height:1.2;color:#fff">
    Estimasi Risiko Diabetes Melitus Tipe 2
  </div>
  <div style="color:rgba(255,255,255,.65);margin-top:.4rem;font-size:.95rem">
    Berbasis <strong>Case-Based Reasoning</strong> dengan pembobotan fitur <strong>MultiSURF</strong> —
    dirancang untuk tenaga kesehatan di Fasilitas Kesehatan Tingkat 1.
  </div>
</div>
""", unsafe_allow_html=True)

# ── METRIC ROW ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4, gap="small")
with c1:
    st.markdown(mini_metric_html("76,03%", "Akurasi (K=9)"), unsafe_allow_html=True)
with c2:
    st.markdown(mini_metric_html("65,87%", "Precision"), unsafe_allow_html=True)
with c3:
    st.markdown(mini_metric_html("64,54%", "Recall"), unsafe_allow_html=True)
with c4:
    st.markdown(mini_metric_html("64,95%", "F1-Score"), unsafe_allow_html=True)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ── CONTENT ──────────────────────────────────────────────────────────────
col_a, col_b = st.columns([1.3, 1], gap="large")

with col_a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Cara Penggunaan</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:rgba(255,255,255,.85);font-size:.9rem;line-height:1.8">
      <b>1. Input Data</b> — Buka halaman <b>Deteksi</b>, isi 8 atribut pemeriksaan pasien.<br>
      <b>2. Proses Estimasi</b> — Sistem menghitung kemiripan kasus (Retrieve &amp; Reuse).<br>
      <b>3. Validasi Klinis</b> — Periksa hasil estimasi sesuai pertimbangan klinis (Revise).<br>
      <b>4. Simpan Kasus</b> — Kasus baru tersimpan ke basis kasus untuk meningkatkan akurasi (Retain).
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Metode</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:rgba(255,255,255,.7);font-size:.85rem;line-height:1.7">
      • Algoritma: <b>Case-Based Reasoning (CBR)</b><br>
      • Pembobotan fitur: <b>MultiSURF</b> (rata-rata 10-Fold CV)<br>
      • Pengukuran kemiripan: <b>Weighted Euclidean Distance</b><br>
      • Konfigurasi terbaik: <b>K = 9 tetangga terdekat</b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Bobot Fitur MultiSURF</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    weights = {
        "Glucose": 1.0000,
        "Age": 0.2594,
        "BMI": 0.2342,
        "DiabetesPedigreeFunction": 0.1447,
        "Insulin": 0.0983,
        "Pregnancies": 0.0417,
        "SkinThickness": 0.0354,
        "BloodPressure": 0.0155,
    }
    for feat, w in weights.items():
        pct = w * 100
        bar_w = int(pct)
        color = "#2dd4bf" if w >= 0.15 else "rgba(45,212,191,.4)"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
          <div style="font-size:.8rem;color:rgba(255,255,255,.8);width:160px;white-space:nowrap">{feat}</div>
          <div style="flex:1;background:rgba(255,255,255,.07);border-radius:4px;height:8px">
            <div style="width:{bar_w}%;background:{color};height:8px;border-radius:4px"></div>
          </div>
          <div style="font-size:.75rem;color:rgba(255,255,255,.6);width:36px;text-align:right">{w:.3f}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
st.info("👉 Gunakan menu **Deteksi** di sidebar untuk memulai estimasi risiko pasien.")
