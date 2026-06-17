from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.theme import card_close, card_open, divider, inject_glass_theme, section_label

st.set_page_config(page_title="Basis Kasus — DM Tipe 2", page_icon="📚", layout="wide")
inject_glass_theme()

CASE_BASE_PATH = Path("data/case_base.csv")

st.markdown("""
<div style="margin-bottom:1rem">
  <div style="font-size:.72rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#2dd4bf">
    Manajemen Data
  </div>
  <div style="font-size:1.5rem;font-weight:800;color:#fff;letter-spacing:-.02em">
    Basis Kasus
  </div>
  <div style="color:rgba(255,255,255,.6);font-size:.88rem">
    Seluruh kasus yang telah divalidasi dan disimpan (tahap <b>Retain</b> siklus CBR).
  </div>
</div>
""", unsafe_allow_html=True)

def load_cases() -> pd.DataFrame:
    if not CASE_BASE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CASE_BASE_PATH)

df = load_cases()

# ── STATS ─────────────────────────────────────────────────────────────────
section_label("Statistik Basis Kasus")
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

if df.empty:
    st.info("Belum ada kasus tersimpan. Gunakan halaman **Deteksi** untuk menambahkan kasus baru.")
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

for col, val, lbl in zip(
    [c1, c2, c3, c4],
    [total, dm, non_dm, f"{match_pct}%"],
    ["Total Kasus", "Kasus Diabetes", "Kasus Non-Diabetes", "Konsistensi Validasi"],
):
    with col:
        st.markdown(f"""
        <div class="mini-metric">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

card_close()

# ── TABLE ──────────────────────────────────────────────────────────────────
section_label("Daftar Kasus Tersimpan")
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# Filter sidebar
with st.sidebar:
    st.markdown("### 🔍 Filter")
    outcome_filter = st.selectbox(
        "Outcome Validasi",
        options=["Semua", "Diabetes (1)", "Non-Diabetes (0)"],
    )
    search_ts = st.text_input("Cari timestamp…", "")

filtered = df.copy()
if outcome_filter == "Diabetes (1)":
    filtered = filtered[filtered["validated_outcome"] == 1]
elif outcome_filter == "Non-Diabetes (0)":
    filtered = filtered[filtered["validated_outcome"] == 0]
if search_ts and "timestamp" in filtered.columns:
    filtered = filtered[filtered["timestamp"].astype(str).str.contains(search_ts)]

st.markdown(f"<div class='muted' style='margin-bottom:8px'>Menampilkan {len(filtered)} dari {total} kasus.</div>", unsafe_allow_html=True)

# Style outcome column
def style_row(row):
    color = "rgba(248,113,113,.18)" if row.get("validated_outcome") == 1 else "rgba(34,197,94,.10)"
    return [f"background: {color}"] * len(row)

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
    column_config={
        "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
        "predicted_outcome": st.column_config.NumberColumn("Pred.", format="%d", width="small"),
        "validated_outcome": st.column_config.NumberColumn("Valid.", format="%d", width="small"),
        "validation_note": st.column_config.TextColumn("Catatan Klinis"),
    }
)
card_close()

# ── EXPORT ────────────────────────────────────────────────────────────────
section_label("Ekspor Data")
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
ec1, ec2 = st.columns(2, gap="small")
with ec1:
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Unduh sebagai CSV",
        data=csv_bytes,
        file_name="basis_kasus.csv",
        mime="text/csv",
        use_container_width=True,
    )
with ec2:
    if st.button("🗑️ Hapus semua kasus (reset)", use_container_width=True):
        if st.session_state.get("confirm_delete"):
            CASE_BASE_PATH.unlink(missing_ok=True)
            st.success("Basis kasus berhasil dihapus.")
            st.session_state.pop("confirm_delete")
            st.rerun()
        else:
            st.session_state["confirm_delete"] = True
            st.warning("Klik sekali lagi untuk konfirmasi penghapusan semua kasus.")
card_close()
