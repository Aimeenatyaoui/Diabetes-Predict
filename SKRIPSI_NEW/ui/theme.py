from __future__ import annotations
import streamlit as st


def inject_glass_theme() -> None:
    st.markdown(
        """
        <style>
          :root {
            --bg0: #050b1f;
            --bg1: #0b1b3a;
            --bg2: #0a3a44;
            --glass: rgba(255,255,255,.10);
            --glass2: rgba(255,255,255,.06);
            --stroke: rgba(255,255,255,.15);
            --text: #ffffff;
            --muted: rgba(255,255,255,0.65);
            --accent: #2dd4bf;
            --accent2: #22c55e;
            --danger: #f87171;
            --warn: #fbbf24;
          }

          /* ── HEADER & DECORATION ── */
          header[data-testid="stHeader"] { background: transparent !important; }
          [data-testid="stDecoration"] { display: none; }

          /* ── BACKGROUND ── */
          .stApp {
            background:
              radial-gradient(1100px 600px at 10% 0%, rgba(45,212,191,.18), transparent 55%),
              radial-gradient(900px 600px at 88% 8%, rgba(34,197,94,.13), transparent 50%),
              linear-gradient(135deg, var(--bg0), var(--bg1) 50%, var(--bg2));
            color: var(--text);
            font-family: 'Inter', -apple-system, sans-serif;
          }

          /* ── SIDEBAR ── */
          section[data-testid="stSidebar"] { width: 270px !important; }
          section[data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(160deg, rgba(5,11,31,.85), rgba(11,27,58,.75)) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-right: 1px solid var(--stroke);
          }
          section[data-testid="stSidebar"] label,
          section[data-testid="stSidebar"] span,
          section[data-testid="stSidebar"] p {
            color: rgba(255,255,255,.9) !important;
          }

          /* ── LAYOUT ── */
          .block-container {
            padding-top: 2.5rem !important;
            padding-bottom: 2rem;
            max-width: 1200px;
          }

          h1, h2, h3, h4 {
            color: white !important;
            font-weight: 700;
            letter-spacing: -0.02em;
          }

          /* ── GLASS CARD ── */
          .card {
            background: linear-gradient(160deg, var(--glass), var(--glass2));
            border: 1px solid var(--stroke);
            border-radius: 16px;
            box-shadow: 0 8px 28px rgba(0,0,0,.30);
            backdrop-filter: blur(22px);
            -webkit-backdrop-filter: blur(22px);
            padding: 24px;
            margin-bottom: 12px;
          }

          /* ── SECTION LABEL ── */
          .section-label {
            font-size: .7rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 4px;
          }

          /* ── INPUTS ── */
          div[data-baseweb="input"] > div {
            background: rgba(255,255,255,.07) !important;
            border: 1px solid var(--stroke) !important;
            border-radius: 10px !important;
            transition: border-color .2s;
          }
          div[data-baseweb="input"]:focus-within > div {
            border-color: var(--accent) !important;
          }
          input { color: black !important; }
          label { color: rgba(255,255,255,.85) !important; font-size: .85rem !important; }

          /* ── BUTTONS ── */
          .stButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: all 0.25s ease;
            letter-spacing: 0.01em;
          }
          .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2dd4bf, #22c55e) !important;
            color: #050b1f !important;
            border: none !important;
            box-shadow: 0 6px 18px rgba(45,212,191,.25);
          }
          .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(0,0,0,.28);
          }

          /* ── RESULT BADGE ── */
          .badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1rem;
            letter-spacing: .04em;
            margin-bottom: 16px;
          }
          .badge-high {
            background: rgba(248,113,113,.15);
            border: 1.5px solid var(--danger);
            color: var(--danger);
          }
          .badge-low {
            background: rgba(34,197,94,.12);
            border: 1.5px solid var(--accent2);
            color: var(--accent2);
          }

          /* ── DIVIDER ── */
          .hdivider {
            border: none;
            border-top: 1px solid var(--stroke);
            margin: 16px 0;
          }

          /* ── METRIC MINI ── */
          .mini-metric {
            background: rgba(255,255,255,.05);
            border: 1px solid var(--stroke);
            border-radius: 10px;
            padding: 10px 14px;
            text-align: center;
          }
          .mini-metric .val {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--accent);
          }
          .mini-metric .lbl {
            font-size: .72rem;
            color: var(--muted);
            margin-top: 2px;
          }

          /* ── INFO / WARNING OVERRIDES ── */
          [data-testid="stAlert"] {
            border-radius: 10px !important;
          }

          /* ── DATAFRAME ── */
          [data-testid="stDataFrame"] {
            border-radius: 10px !important;
            overflow: hidden;
          }

          /* ── RADIO ── */
          .stRadio label { font-size: .88rem !important; }

          /* ── CAPTION ── */
          .muted { color: var(--muted); font-size: .85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card_open(extra_style: str = "") -> None:
    st.markdown(f'<div class="card" style="{extra_style}">', unsafe_allow_html=True)


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<hr class="hdivider">', unsafe_allow_html=True)


def mini_metric_html(value: str, label: str) -> str:
    return f"""
    <div class="mini-metric">
      <div class="val">{value}</div>
      <div class="lbl">{label}</div>
    </div>"""
