"""
NIDS-XAI  ·  Dark Command Center Dashboard
Run: streamlit run app/dashboard.py
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap, joblib
import streamlit as st

warnings.filterwarnings("ignore")

MODEL_PATH  = "outputs/models/xgb_model.pkl"
SCALER_PATH = "outputs/models/scaler.pkl"
SAMPLE_PATH = "data/processed/X_test.parquet"
SHAP_PATH   = "outputs/reports/shap_feature_ranking_xgb.csv"

st.set_page_config(
    page_title="NIDS-XAI · Command Center",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path fill='%2300e5b0' d='M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
BG      = "#07090f"
CARD    = "#0c1120"
CARD2   = "#101827"
BORDER  = "#161f30"
BORDER2 = "#1e2e44"

TEXT1   = "#edf2f7"   # primary — near white
TEXT2   = "#8ba8c4"   # secondary — light silver-blue (was too dark)
TEXT3   = "#4a6580"   # dim — now visible on projectors (was #2d3f55)

MINT    = "#00e5b0"
CRIMSON = "#ff2d54"
BLUE    = "#3d8bff"
AMBER   = "#ffb020"
VIOLET  = "#9c6ffa"
TEAL2   = "#00c8e0"
ROSE    = "#ff4d7a"
PURPLE  = "#cc55ff"

PAL     = [BLUE, VIOLET, AMBER, MINT, CRIMSON, TEAL2, ROSE, PURPLE, "#40d9a0", "#ff7744"]

CH_BG   = "#0c1120"
CH_SURF = "#080d16"
CH_GRID = "#101827"

# ── Inline SVG icons ───────────────────────────────────────────────────────────
SVG_SHIELD = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00e5b0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>"""

SVG_ACTIVITY = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>"""

SVG_DATABASE = """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>"""

SVG_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>"""

SVG_UPLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3d8bff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>"""

SVG_ALERT = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ff2d54" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""

SVG_CHECK = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00e5b0" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>"""

SVG_CPU = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3d8bff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>"""

SVG_BULB = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ffb020" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="9" y1="18" x2="15" y2="18"/><line x1="10" y1="22" x2="14" y2="22"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/></svg>"""

SVG_DOWNLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>"""

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

/* Inter for body — highly legible at all sizes */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}}

.stApp {{ background: {BG} !important; }}
.block-container {{ padding: 1.6rem 2rem 3rem !important; max-width: 1280px !important; }}

/* Blueprint grid — very subtle */
.stApp::before {{
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient({BORDER}90 1px, transparent 1px),
        linear-gradient(90deg, {BORDER}90 1px, transparent 1px);
    background-size: 48px 48px;
    opacity: 0.4;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {CARD} !important;
    border-right: 1px solid #1a2540 !important;
    box-shadow: 4px 0 24px rgba(0,0,0,.55), 1px 0 0 #0d1828 !important;
}}
[data-testid="stSidebar"] .block-container {{ padding: 1.4rem 1.1rem !important; }}
[data-testid="stSidebar"] hr {{ border-color: {BORDER} !important; }}

[data-testid="stSidebar"] [data-testid="metric-container"] {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    padding: .65rem .85rem !important;
    transition: border-color .2s, box-shadow .2s !important;
}}
[data-testid="stSidebar"] [data-testid="metric-container"]:hover {{
    border-color: {BLUE}66 !important;
    box-shadow: 0 0 14px {BLUE}1a !important;
}}
[data-testid="stSidebar"] [data-testid="metric-container"] label {{
    font-family: 'Roboto Mono', monospace !important;
    font-size: .6rem !important; letter-spacing: .07em !important;
    text-transform: uppercase !important; color: {TEXT3} !important;
}}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', 'Roboto Mono', monospace !important;
    color: {TEXT1} !important; font-size: 1.25rem !important; font-weight: 600 !important;
}}

/* ── Main metric cards ── */
[data-testid="metric-container"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 1rem 1.15rem !important;
    transition: border-color .2s, box-shadow .22s, transform .18s !important;
    cursor: default !important;
}}
[data-testid="metric-container"]:hover {{
    border-color: {BLUE}55 !important;
    box-shadow: 0 0 22px {BLUE}18, 0 4px 24px rgba(0,0,0,.5) !important;
    transform: translateY(-2px) !important;
}}
[data-testid="metric-container"] label {{
    font-family: 'Inter', sans-serif !important;
    font-size: .7rem !important; font-weight: 600 !important;
    letter-spacing: .04em !important; text-transform: uppercase !important;
    color: {TEXT2} !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', 'Roboto Mono', monospace !important;
    font-size: 1.7rem !important; font-weight: 700 !important;
    color: {TEXT1} !important; letter-spacing: -.02em !important; line-height: 1.1 !important;
}}
[data-testid="stMetricDelta"] {{
    font-family: 'Roboto Mono', monospace !important;
    font-size: .68rem !important; color: {TEXT2} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    font-family: 'Inter', sans-serif !important;
    font-size: .78rem !important; font-weight: 500 !important;
    background: {CARD2} !important; border: 1px solid {BORDER2} !important;
    color: {TEXT2} !important; border-radius: 7px !important;
    padding: .43rem 1.1rem !important; transition: all .18s ease !important;
}}
.stButton > button:hover {{
    background: {BLUE}1a !important; border-color: {BLUE} !important;
    color: {BLUE} !important; box-shadow: 0 0 16px {BLUE}33 !important;
    transform: translateY(-1px) !important;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 10px !important; margin-bottom: 8px !important;
    transition: border-color .18s, box-shadow .18s !important;
}}
[data-testid="stExpander"]:hover {{
    border-color: {BORDER2} !important;
    box-shadow: 0 4px 20px rgba(0,0,0,.4) !important;
}}
.streamlit-expanderHeader {{
    font-family: 'Inter', sans-serif !important;
    font-size: .82rem !important; font-weight: 500 !important;
    color: {TEXT2} !important; padding: .7rem 1rem !important;
}}

/* ── File uploader — cyan dashed border ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"] {{
    background: {CARD2} !important;
    border: 1.5px dashed {MINT}55 !important;
    border-radius: 8px !important; transition: border-color .2s, box-shadow .2s !important;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {MINT}cc !important;
    box-shadow: 0 0 18px {MINT}18 !important;
}}

/* ── SHAP waterfall glassmorphism container ── */
[data-testid="stExpander"] > div:last-child {{
    background: rgba(13, 24, 44, 0.72) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-top: 1px solid {BORDER2} !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important; overflow: hidden !important;
}}

/* ── Alerts ── */
.stInfo, .stSuccess, .stWarning, .stError {{
    border-radius: 8px !important; font-size: .8rem !important;
    background: {CARD2} !important; font-family: 'Inter', sans-serif !important;
}}

/* ── Checkbox / slider ── */
.stCheckbox label {{
    color: {TEXT2} !important; font-size: .8rem !important;
    font-family: 'Inter', sans-serif !important;
}}
.stSlider label {{
    color: {TEXT2} !important; font-size: .78rem !important;
    font-family: 'Inter', sans-serif !important;
}}

/* ── Caption / small text ── */
.stCaption, [data-testid="stCaption"] {{
    color: {TEXT2} !important; font-size: .78rem !important;
    font-family: 'Inter', sans-serif !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 4px; }}
hr {{ border-color: {BORDER} !important; margin: .85rem 0 !important; }}
</style>
""", unsafe_allow_html=True)

# ── Chart helpers ──────────────────────────────────────────────────────────────
def dark_chart(fig, ax, hide_ygrid=False):
    fig.patch.set_facecolor(CH_BG)
    ax.set_facecolor(CH_SURF)
    ax.tick_params(colors=TEXT2, labelsize=9, length=0, pad=4)
    ax.xaxis.label.set_color(TEXT2)
    ax.yaxis.label.set_color(TEXT2)
    ax.yaxis.set_tick_params(labelcolor=TEXT1)
    ax.title.set_color(TEXT1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.set_axisbelow(True)
    if hide_ygrid:
        ax.grid(axis="x", color=CH_GRID, linewidth=.7)
        ax.yaxis.grid(False)
    else:
        ax.grid(color=CH_GRID, linewidth=.7)

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

@st.cache_resource(show_spinner=False)
def load_explainer(_m):
    return shap.TreeExplainer(_m)

def preprocess(df, sc):
    df = df.copy(); df.columns = df.columns.str.strip()
    for c in ["Label", " Label"]:
        if c in df.columns: df.drop(columns=[c], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.select_dtypes(include=[np.number])
    return pd.DataFrame(sc.transform(df), columns=df.columns, index=df.index)

try:
    model, scaler = load_model()
    explainer     = load_explainer(model)
    MODEL_OK = True
except FileNotFoundError:
    MODEL_OK = False

# ── Helper: section label ──────────────────────────────────────────────────────
def section(icon_svg, txt):
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:7px;"
        f"margin-bottom:.55rem;margin-top:.1rem;'>"
        f"<span style='color:{TEXT2};display:flex;align-items:center;'>{icon_svg}</span>"
        f"<span style='font-size:.7rem;font-weight:600;letter-spacing:.07em;"
        f"text-transform:uppercase;color:#a8c0d8;'>{txt}</span>"
        f"<div style='flex:1;height:1px;background:{BORDER};margin-left:6px;'></div>"
        f"</div>",
        unsafe_allow_html=True
    )

def chart_label(txt):
    st.markdown(
        f"<p style='font-size:.8rem;font-weight:600;color:{TEXT1};"
        f"margin-bottom:.3rem;'>{txt}</p>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand: SVG shield + text
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:.9rem;'>"
        f"<div style='width:34px;height:34px;background:{CARD2};border:1px solid {BORDER2};"
        f"border-radius:8px;display:flex;align-items:center;justify-content:center;'>"
        f"{SVG_SHIELD}</div>"
        f"<div>"
        f"<div style='font-size:.95rem;font-weight:700;color:{TEXT1};"
        f"letter-spacing:-.02em;line-height:1.1;'>NIDS-XAI</div>"
        f"<div style='font-size:.68rem;color:{TEXT3};margin-top:1px;'>Command Center</div>"
        f"</div></div>",
        unsafe_allow_html=True
    )
    st.divider()

    # Model metrics
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:.5rem;'>"
        f"<span style='color:{TEXT3};display:flex;'>{SVG_CPU}</span>"
        f"<span style='font-size:.68rem;font-weight:600;letter-spacing:.06em;"
        f"text-transform:uppercase;color:{TEXT3};'>Model Metrics</span></div>",
        unsafe_allow_html=True
    )
    sa, sb = st.columns(2)
    sa.metric("F1",      "0.9999")
    sb.metric("PR-AUC",  "1.0000")
    sa.metric("ROC-AUC", "1.0000")
    sb.metric("Feats",   "68")

    st.divider()

    # Dataset
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:.5rem;'>"
        f"<span style='color:{TEXT3};display:flex;'>{SVG_DATABASE}</span>"
        f"<span style='font-size:.68rem;font-weight:600;letter-spacing:.06em;"
        f"text-transform:uppercase;color:{TEXT3};'>Dataset</span></div>",
        unsafe_allow_html=True
    )
    for k, v in [("Source","CICIDS2017"),("Algorithm","XGBoost"),
                 ("Train","178,465 rows"),("Test","44,617 rows"),
                 ("Labels","BENIGN / DDoS")]:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;padding:5px 0;border-bottom:1px solid {BORDER};'>"
            f"<span style='font-size:.72rem;color:{TEXT3};'>{k}</span>"
            f"<span style='font-family:Roboto Mono,monospace;font-size:.72rem;"
            f"font-weight:500;color:{TEXT2};'>{v}</span></div>",
            unsafe_allow_html=True
        )

    st.divider()

    # Controls
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:.5rem;'>"
        f"<span style='color:{TEXT3};display:flex;'>{SVG_SETTINGS}</span>"
        f"<span style='font-size:.68rem;font-weight:600;letter-spacing:.06em;"
        f"text-transform:uppercase;color:{TEXT3};'>Controls</span></div>",
        unsafe_allow_html=True
    )
    show_shap = st.checkbox("SHAP Waterfall", value=True)
    n_shap    = st.slider("Flows to explain", 1, 8, 3)

    st.divider()
    st.markdown(
        f"<div style='padding:10px 4px 4px 4px;border-top:1px solid {BORDER};margin-top:.4rem;'>"
        f"<div style='font-size:.68rem;color:{TEXT3};line-height:2;'>"
        f"NIDS-XAI &nbsp;·&nbsp; v1.0<br>"
        f"XGBoost + SHAP &nbsp;·&nbsp; CICIDS2017<br>"
        f"</div></div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
hl, hr = st.columns([3, 1])
with hl:
    st.markdown(
        f"<h1 style='font-size:1.55rem;font-weight:700;color:{TEXT1};"
        f"letter-spacing:-.03em;margin:0;line-height:1.2;'>"
        f"Network Intrusion Detection</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-family:Roboto Mono,monospace;font-size:.7rem;"
        f"color:{TEXT3};margin-top:6px;'>"
        f"XGBoost + SHAP &nbsp;·&nbsp; CICIDS2017 &nbsp;·&nbsp; "
        f"68 network flow features &nbsp;·&nbsp; Binary classification</p>",
        unsafe_allow_html=True
    )
with hr:
    st.write(""); st.write("")
    if MODEL_OK:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:7px;"
            f"background:{CARD};border:1px solid #00e5b022;"
            f"border-radius:8px;padding:8px 12px;'>"
            f"{SVG_CHECK}"
            f"<span style='font-size:.78rem;font-weight:500;color:{MINT};'>Model active</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:7px;"
            f"background:{CARD};border:1px solid #ff2d5422;"
            f"border-radius:8px;padding:8px 12px;'>"
            f"{SVG_ALERT}"
            f"<span style='font-size:.78rem;font-weight:500;color:{CRIMSON};'>Model not found</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.stop()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODEL KPIS
# ══════════════════════════════════════════════════════════════════════════════
section(SVG_CPU, "Model Performance")
p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Algorithm",  "XGBoost",  help="Gradient Boosted Trees")
p2.metric("F1-Score",   "0.9999",   help="Macro average · test set")
p3.metric("PR-AUC",     "1.0000",   help="Precision-Recall AUC")
p4.metric("Train Rows", "178,465",  help="80% stratified split")
p5.metric("Test Rows",  "44,617",   help="20% stratified split")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════════════════
section(SVG_UPLOAD, "Traffic Classification")
st.caption("Upload a CICIDS2017-format network flow CSV, or load 100 sample flows to test live inference.")

uc, bc = st.columns([4.5, 1])
with uc:
    uploaded = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
with bc:
    st.write(""); st.write("")
    use_sample = st.button("Load sample data", use_container_width=True)

df_in = None
if uploaded:
    try:
        df_in = pd.read_csv(uploaded, low_memory=False)
        st.success(f"Loaded **{uploaded.name}** — {df_in.shape[0]:,} rows × {df_in.shape[1]} cols")
    except Exception as e:
        st.error(str(e))
elif use_sample or st.session_state.get("_smpl"):
    if os.path.exists(SAMPLE_PATH):
        df_in = pd.read_parquet(SAMPLE_PATH).sample(100, random_state=42)
        st.session_state["_pre"]  = True
        st.session_state["_smpl"] = False
        st.info("Loaded 100 sample flows from the pre-processed test set.")
    else:
        st.warning("Sample data not found — run `preprocess.py` first.")

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE + RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if df_in is not None:
    try:
        if st.session_state.get("_pre"):
            df = df_in.copy(); st.session_state["_pre"] = False
        else:
            with st.spinner("Preprocessing…"):
                df = preprocess(df_in, scaler)

        with st.spinner("Running inference…"):
            t0    = time.time()
            pred  = model.predict(df)
            proba = model.predict_proba(df)[:, 1]
            ms    = (time.time()-t0)*1000

        n   = len(pred)
        nd  = int((pred==1).sum()); nb = n - nd
        thr = nd/n*100
        avc = float((np.maximum(proba, 1-proba)*100).mean())
        lvl = "HIGH" if thr>50 else "MEDIUM" if thr>20 else "LOW"

        st.divider()
        section(SVG_ALERT, "Detection Results")

        d1,d2,d3,d4,d5 = st.columns(5)
        d1.metric("Total Flows",    f"{n:,}",       f"in {ms:.0f} ms")
        d2.metric("DDoS Detected",  f"{nd:,}",      f"{thr:.1f}% of traffic")
        d3.metric("Benign Traffic", f"{nb:,}",      f"{100-thr:.1f}% of traffic")
        d4.metric("Avg Confidence", f"{avc:.1f}%",  "model posterior")
        d5.metric("Threat Level",   lvl,            f"{thr:.1f}% threat rate")

        # SHAP values (computed once)
        sv_a = explainer.shap_values(df.values)
        sv_p = sv_a[1] if isinstance(sv_a, list) else sv_a
        msv  = np.abs(sv_p).mean(axis=0)
        ti   = np.argsort(msv)[::-1]

        st.divider()
        section(SVG_ACTIVITY, "Visual Analysis")

        # ── Row A: Donut + Confidence ─────────────────────────────────────────
        ch1, ch2 = st.columns([1, 2])

        with ch1:
            chart_label("Traffic Breakdown")
            fig, ax = plt.subplots(figsize=(3.5, 3.4), subplot_kw={"aspect":"equal"})
            ax.pie([nb, nd], colors=[MINT, CRIMSON], startangle=90,
                   wedgeprops={"width":0.22, "edgecolor":CH_BG, "linewidth":2})
            ax.text(0,  0.1,  f"{thr:.0f}%",  ha="center", va="center",
                    fontsize=26, fontweight="700", color=CRIMSON)
            ax.text(0, -0.2, "threat rate",   ha="center", va="center",
                    fontsize=8.5, color=TEXT2)
            ax.legend(
                handles=[
                    mpatches.Patch(color=MINT,    label=f"Benign  {nb:,}"),
                    mpatches.Patch(color=CRIMSON, label=f"DDoS  {nd:,}")
                ],
                loc="lower center", bbox_to_anchor=(0.5, -0.07),
                ncol=2, fontsize=9, framealpha=0, labelcolor=TEXT1
            )
            fig.patch.set_facecolor(CH_BG); ax.set_facecolor(CH_BG)
            st.pyplot(fig, use_container_width=True); plt.close()

        with ch2:
            chart_label("Prediction Confidence Distribution")
            fig, ax = plt.subplots(figsize=(7.5, 3.4))
            bins = np.linspace(50, 100, 26)
            cb = np.maximum(proba[pred==0], 1-proba[pred==0])*100
            cd = np.maximum(proba[pred==1], 1-proba[pred==1])*100
            if len(cb): ax.hist(cb, bins=bins, color=MINT,    alpha=.5,
                                label="Benign", edgecolor=CH_BG, linewidth=.4, rwidth=.86)
            if len(cd): ax.hist(cd, bins=bins, color=CRIMSON, alpha=.5,
                                label="DDoS",   edgecolor=CH_BG, linewidth=.4, rwidth=.86)
            ax.set_xlabel("Model confidence (%)", fontsize=9)
            ax.set_ylabel("Flow count", fontsize=9)
            ax.legend(prop={"size":9.5}, framealpha=0, labelcolor=TEXT1)
            dark_chart(fig, ax)
            ax.yaxis.grid(True, color=CH_GRID, linewidth=.7); ax.xaxis.grid(False)
            st.pyplot(fig, use_container_width=True); plt.close()

        # ── Row B: Top 5 SHAP full-width ──────────────────────────────────────
        chart_label("Top 5 SHAP Features  —  Most influential for DDoS classification")
        top5_v = [msv[ti[i]] for i in range(5)]
        top5_f = [df.columns[ti[i]] for i in range(5)]
        cols5  = [BLUE, VIOLET, AMBER, MINT, CRIMSON]

        fig, ax = plt.subplots(figsize=(13, 3.0))
        # Glow layer
        for i, (val, col) in enumerate(zip(top5_v[::-1], cols5[::-1])):
            ax.barh(i, val, height=0.72, color=col, alpha=.12, edgecolor="none", zorder=0)
        # Main bars
        bars = ax.barh(range(5), top5_v[::-1], color=cols5[::-1],
                       edgecolor=CH_BG, linewidth=.3, height=0.5, zorder=2)
        ax.set_yticks(range(5))
        ax.set_yticklabels(top5_f[::-1], fontsize=10.5, color=TEXT1)
        ax.set_xlabel("Mean |SHAP| value", fontsize=9.5)
        for bar, val in zip(bars, top5_v[::-1]):
            ax.text(val + max(top5_v)*.01, bar.get_y()+bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=9, color=TEXT2)
        dark_chart(fig, ax, hide_ygrid=True); ax.yaxis.grid(False)
        st.pyplot(fig, use_container_width=True); plt.close()

        # ── Prediction table ───────────────────────────────────────────────────
        st.divider()
        tc, dc = st.columns([5.5, 1])
        with tc:
            section(SVG_ACTIVITY, f"Flow-Level Predictions — top 50 of {n:,} flows")

        res = df.copy()
        res.insert(0, "Label",      ["DDoS" if p==1 else "BENIGN" for p in pred])
        res.insert(1, "Confidence", (np.maximum(proba, 1-proba)*100).round(1))
        disp = res[["Label","Confidence"] + list(df.columns[:5])].head(50)

        def _style(v):
            if v=="DDoS":   return f"background:#280a12;color:{CRIMSON};font-weight:600"
            if v=="BENIGN": return f"background:#041a12;color:{MINT};font-weight:600"
            return ""

        st.dataframe(
            disp.style.applymap(_style, subset=["Label"])
                      .format({"Confidence":"{:.1f}%"}),
            use_container_width=True, height=260
        )
        with dc:
            st.write("")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:5px;"
                f"justify-content:center;color:{TEXT2};font-size:.72rem;margin-top:4px;'>"
                f"{SVG_DOWNLOAD}</div>",
                unsafe_allow_html=True
            )
            st.download_button("Export CSV",
                res[["Label","Confidence"]].to_csv(index=False),
                "predictions.csv","text/csv", use_container_width=True)

        # ── SHAP Waterfalls ────────────────────────────────────────────────────
        if show_shap:
            st.divider()
            section(SVG_ACTIVITY, "SHAP Explanations")
            st.caption("Per-flow waterfall — red bars increase DDoS probability, blue bars decrease it.")

            for i in range(min(n_shap, len(df))):
                row  = df.iloc[i]
                lbl  = "DDoS" if pred[i]==1 else "BENIGN"
                conf = np.maximum(proba[i], 1-proba[i])*100
                col  = CRIMSON if pred[i]==1 else MINT
                dot  = f"<svg xmlns='http://www.w3.org/2000/svg' width='8' height='8'><circle cx='4' cy='4' r='4' fill='{col}'/></svg>"

                with st.expander(
                    f"Flow {i+1:02d}   ·   {lbl}   ·   {conf:.1f}% confidence",
                    expanded=(i==0)
                ):
                    sv_r = explainer.shap_values(row.values.reshape(1,-1))
                    svp  = sv_r[1][0] if isinstance(sv_r, list) else sv_r[0]
                    base = (explainer.expected_value[1]
                            if isinstance(explainer.expected_value, list)
                            else explainer.expected_value)
                    exp  = shap.Explanation(
                        values=svp, base_values=base,
                        data=row.values, feature_names=list(row.index)
                    )
                    plt.rcParams.update({
                        "figure.facecolor": CH_BG, "axes.facecolor": CH_SURF,
                        "axes.edgecolor": BORDER, "text.color": TEXT1,
                        "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                        "ytick.color": TEXT1,
                    })
                    # 65% width via columns — golden ratio centering
                    _gap, _mid, _gap2 = st.columns([0.175, 0.65, 0.175])
                    with _mid:
                        fig, ax_w = plt.subplots(figsize=(8, 4.2))
                        shap.waterfall_plot(exp, max_display=10, show=False)
                        ax_w = plt.gca()
                        fig.patch.set_facecolor(CH_BG)
                        ax_w.set_facecolor(CH_SURF)
                        for sp in ax_w.spines.values(): sp.set_color(BORDER)
                        # Thinner bars + smaller feature font
                        ax_w.tick_params(labelsize=8, axis="y")
                        ax_w.tick_params(labelsize=8, axis="x")
                        ax_w.set_yticklabels(
                            [t.get_text() for t in ax_w.get_yticklabels()],
                            fontsize=8, color=TEXT1
                        )
                        # Annotate f(x) meaning
                        fig.text(0.98, 0.97, "f(x) = Final Prediction Logit",
                                 ha="right", va="top", fontsize=7.5,
                                 color=TEXT2, fontstyle="italic",
                                 fontfamily="monospace")
                        plt.tight_layout(pad=1.2)
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    plt.rcParams.update(plt.rcParamsDefault)

                    ti2 = np.argsort(np.abs(svp))[::-1][:6]
                    st.dataframe(pd.DataFrame({
                        "Feature":   [df.columns[j] for j in ti2],
                        "SHAP":      [f"{svp[j]:+.4f}" for j in ti2],
                        "Direction": ["→ DDoS" if svp[j]>0 else "→ BENIGN" for j in ti2],
                    }), use_container_width=True, hide_index=True, height=232)

    except Exception as e:
        st.error(str(e)); st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL SHAP
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
section(SVG_BULB, "Global Feature Intelligence")
st.caption("Mean absolute SHAP values across the full test set — which features drive DDoS detection.")

if os.path.exists(SHAP_PATH):
    rank  = pd.read_csv(SHAP_PATH, index_col=0)
    top10 = rank.head(10)
    gc, rc = st.columns([1.65, 1])

    with gc:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        # Glow layer
        for i, (val, col) in enumerate(zip(top10["Mean_SHAP"].values[::-1], PAL[::-1])):
            ax.barh(i, val, height=.85, color=col, alpha=.1, edgecolor="none", zorder=0)
        ax.barh(range(10), top10["Mean_SHAP"].values[::-1],
                color=PAL[::-1], edgecolor=CH_BG, linewidth=.3, height=.6, zorder=2)
        ax.set_yticks(range(10))
        ax.set_yticklabels(top10["Feature"].values[::-1], fontsize=10, color=TEXT1)
        ax.set_xlabel("Mean |SHAP| Value", fontsize=9.5)
        ax.set_title("Top 10 Feature Importances — XGBoost + SHAP",
                     fontsize=11, pad=14, fontweight="600", loc="left", color=TEXT1)
        for bar in ax.patches[10:]:
            val = bar.get_width()
            ax.text(val+.013, bar.get_y()+bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9, color=TEXT2)
        dark_chart(fig, ax, hide_ygrid=True); ax.yaxis.grid(False)
        st.pyplot(fig, use_container_width=True); plt.close()

    with rc:
        chart_label("Feature Rankings")
        st.dataframe(
            top10[["Feature","Mean_SHAP","Mean_SHAP_pct"]]
                .rename(columns={"Mean_SHAP":"SHAP","Mean_SHAP_pct":"%"})
                .style.format({"SHAP":"{:.4f}","%":"{:.1f}"})
                .bar(subset=["SHAP"], color="#1a3a6a", vmin=0),
            use_container_width=True, height=396, hide_index=False
        )

    st.info(
        "**Key finding** — `Fwd Packet Length Max` contributes **23.18%** of total SHAP "
        "attribution. DDoS flows produce abnormally large forward packets vs. benign traffic. "
        "The top 4 features account for **54.7%** of all prediction decisions.",
        icon="💡"
    )
else:
    st.info("Run `python src/explainability/shap_analysis.py` to populate this section.", icon="📊")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"<p style='font-family:Roboto Mono,monospace;font-size:.65rem;"
    f"color:{TEXT3};text-align:center;'>"
    f"NIDS-XAI &nbsp;·&nbsp; XGBoost + SHAP &nbsp;·&nbsp; CICIDS2017 &nbsp;·&nbsp; "
    f"DDoS Detection &nbsp;·&nbsp; FYP Research Project &nbsp;·&nbsp; 2026"
    f"</p>",
    unsafe_allow_html=True
)
