"""
Golf Performance Analytics
Built for varsity golf team season prep — track fairways, approach quality,
putting patterns, and get a personalized data-driven practice plan.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import shap
import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Golf Analytics",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "green":      "#16a34a",
    "green_dark": "#14532d",
    "green_mid":  "#166534",
    "green_light":"#dcfce7",
    "green_pale": "#f0fdf4",
    "gold":       "#ca8a04",
    "gold_light": "#fef9c3",
    "red":        "#dc2626",
    "red_light":  "#fee2e2",
    "blue":       "#2563eb",
    "blue_light": "#dbeafe",
    "purple":     "#7c3aed",
    "gray":       "#6b7280",
    "gray_light": "#f3f4f6",
    "white":      "#ffffff",
    "text":       "#111827",
    "text_mid":   "#374151",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}
.main {{ padding: 0; }}
.block-container {{ padding: 1rem 2rem 3rem; max-width: 1240px; }}

/* ─ Hero ─ */
.hero {{
  background: linear-gradient(135deg, {C['green_dark']} 0%, {C['green_mid']} 45%, #15803d 100%);
  border-radius: 20px;
  padding: 36px 44px;
  margin-bottom: 28px;
  color: white;
  position: relative;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(20,83,45,.3);
}}
.hero::before {{
  content: '';
  position: absolute; inset: 0;
  background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  pointer-events: none;
}}
.hero::after {{
  content: '⛳';
  position: absolute; right: 44px; top: 50%;
  transform: translateY(-50%);
  font-size: 8rem; opacity: 0.06; pointer-events: none;
}}
.hero-badge {{
  display: inline-block;
  background: rgba(255,255,255,.15);
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255,255,255,.2);
  border-radius: 20px;
  padding: 4px 14px;
  font-size: .72rem; font-weight: 700;
  letter-spacing: 1.5px; text-transform: uppercase;
  margin-bottom: 14px;
}}
.hero h1 {{
  margin: 0 0 12px; font-size: 2.1rem; font-weight: 900;
  line-height: 1.15; letter-spacing: -0.5px;
}}
.hero p {{
  margin: 0; opacity: .82; font-size: .95rem;
  max-width: 560px; line-height: 1.65;
}}

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"] {{
  gap: 2px; background: {C['green_pale']};
  border-radius: 14px; padding: 5px; border-bottom: none;
  margin-bottom: 8px;
}}
.stTabs [data-baseweb="tab"] {{
  height: 40px; padding: 0 18px; border-radius: 10px;
  font-weight: 600; font-size: .84rem;
  background: transparent; color: {C['gray']};
  transition: all .18s ease; border: none;
}}
.stTabs [aria-selected="true"] {{
  background: {C['green']} !important; color: white !important;
  box-shadow: 0 2px 10px rgba(22,163,74,.28) !important;
}}

/* ─ KPI cards ─ */
.kpi {{
  background: white;
  border: 1.5px solid #f0f0f0;
  border-radius: 18px;
  padding: 22px 18px 18px;
  box-shadow: 0 2px 20px rgba(0,0,0,.05);
  position: relative; overflow: hidden;
  transition: box-shadow .22s, transform .22s;
}}
.kpi:hover {{ box-shadow: 0 8px 32px rgba(0,0,0,.1); transform: translateY(-2px); }}
.kpi::before {{
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 3px; border-radius: 18px 18px 0 0;
  background: {C['green']};
}}
.kpi.gold::before  {{ background: {C['gold']}; }}
.kpi.red::before   {{ background: {C['red']}; }}
.kpi.blue::before  {{ background: {C['blue']}; }}
.kpi.purple::before{{ background: {C['purple']}; }}
.kpi-icon  {{ font-size: 1.6rem; margin-bottom: 10px; }}
.kpi-value {{
  font-size: 2.1rem; font-weight: 800;
  color: {C['text']}; line-height: 1; margin-bottom: 6px;
}}
.kpi-label {{
  font-size: .64rem; font-weight: 700;
  letter-spacing: 1.5px; text-transform: uppercase; color: {C['gray']};
}}
.kpi-sub  {{ font-size: .78rem; color: {C['gray']}; margin-top: 5px; }}
.kpi-trend {{
  display: inline-flex; align-items: center; gap: 4px;
  margin-top: 8px; font-size: .75rem; font-weight: 700;
  padding: 4px 12px; border-radius: 20px;
}}
.trend-up   {{ background: {C['green_light']}; color: {C['green']}; }}
.trend-down {{ background: {C['red_light']};   color: {C['red']};   }}
.trend-flat {{ background: {C['gray_light']};  color: {C['gray']};  }}

/* ─ Feature cards (landing) ─ */
.feature-card {{
  background: white;
  border: 1.5px solid #f0f0f0;
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 2px 14px rgba(0,0,0,.05);
  transition: transform .22s, box-shadow .22s, border-color .22s;
  height: 100%;
}}
.feature-card:hover {{
  transform: translateY(-5px);
  box-shadow: 0 14px 36px rgba(22,163,74,.13);
  border-color: {C['green_light']};
}}
.feature-top {{
  background: linear-gradient(135deg, {C['green_pale']} 0%, {C['green_light']} 100%);
  padding: 26px 22px 20px; text-align: center;
  border-bottom: 1px solid {C['green_light']};
}}
.feature-icon {{ font-size: 2.2rem; display: block; margin-bottom: 12px; }}
.feature-title {{ font-size: .94rem; font-weight: 800; color: {C['text']}; }}
.feature-body  {{ padding: 18px 20px 22px; }}
.feature-desc  {{ font-size: .82rem; color: {C['gray']}; line-height: 1.65; }}

/* ─ Steps (landing) ─ */
.step-card {{
  background: white;
  border: 2px solid {C['green_light']};
  border-radius: 18px;
  padding: 28px 20px; text-align: center;
  box-shadow: 0 4px 20px rgba(22,163,74,.08);
  transition: transform .2s, box-shadow .2s;
}}
.step-card:hover {{ transform: translateY(-3px); box-shadow: 0 10px 30px rgba(22,163,74,.15); }}
.step-num {{
  font-size: 3.2rem; font-weight: 900;
  color: {C['green_pale']};
  -webkit-text-stroke: 2.5px {C['green']};
  line-height: 1; margin-bottom: 12px;
}}
.step-text {{ font-size: .88rem; line-height: 1.6; font-weight: 500; color: {C['text_mid']}; }}

/* ─ Achievement badges ─ */
.achievement {{
  display: inline-flex; align-items: center; gap: 8px;
  background: linear-gradient(135deg, {C['gold_light']}, #fde68a);
  border: 1.5px solid {C['gold']};
  border-radius: 10px; padding: 10px 16px;
  font-size: .84rem; font-weight: 600; color: #92400e;
  margin: 4px 6px 4px 0;
}}

/* ─ Section headers ─ */
.sec-hdr {{
  display: flex; align-items: center; gap: 12px;
  margin: 34px 0 18px; padding-bottom: 12px;
  border-bottom: 2px solid {C['green_light']};
}}
.sec-hdr h3 {{
  margin: 0; font-size: 1.15rem; font-weight: 800;
  color: {C['text']}; letter-spacing: -.3px;
}}
.sec-dot {{
  width: 10px; height: 10px; border-radius: 50%;
  background: {C['green']}; flex-shrink: 0;
  box-shadow: 0 0 0 3px {C['green_pale']};
}}

/* ─ Drill cards ─ */
.drill {{
  background: white; border: 1px solid #e9ecef;
  border-left: 4px solid {C['green']};
  border-radius: 0 12px 12px 0;
  padding: 16px 20px; margin: 10px 0;
  box-shadow: 0 1px 5px rgba(0,0,0,.04);
}}
.drill.p1 {{ border-left-color: {C['red']};   }}
.drill.p2 {{ border-left-color: {C['gold']};  }}
.drill.p3 {{ border-left-color: {C['blue']};  }}

/* ─ Progress bar ─ */
.prog-wrap {{
  background: {C['gray_light']}; border-radius: 8px;
  overflow: hidden; height: 10px; margin: 6px 0 12px;
}}
.prog-fill {{
  height: 100%; border-radius: 8px;
  transition: width .4s ease;
}}

/* ─ Sidebar ─ */
[data-testid="stSidebar"] {{
  background: {C['green_pale']};
  border-right: 1.5px solid {C['green_light']};
  box-shadow: 4px 0 20px rgba(22,163,74,.07);
}}
[data-testid="stSidebar"] > div {{ padding-top: 0 !important; }}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {{ color: {C['text_mid']} !important; }}
[data-testid="stSidebar"] hr {{ border-color: {C['green_light']}; }}
[data-testid="stSidebar"] .stRadio label {{
  color: {C['gray']} !important; font-size: .84rem !important;
}}
[data-testid="stSidebar"] [role="slider"] {{
  background: {C['green']} !important;
  border: 2px solid #22c55e !important;
  box-shadow: 0 0 8px rgba(22,163,74,.4) !important;
}}
[data-testid="stSidebar"] .stDownloadButton > button {{
  background: {C['green']} !important;
  border: none !important;
  color: white !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 10px rgba(22,163,74,.25) !important;
  transition: all .18s !important;
  width: 100% !important;
}}
[data-testid="stSidebar"] .stDownloadButton > button:hover {{
  background: {C['green_mid']} !important;
  box-shadow: 0 4px 16px rgba(22,163,74,.35) !important;
  transform: translateY(-1px) !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {{
  background: white !important;
  border-color: {C['green_light']} !important;
  border-radius: 12px !important;
}}
[data-testid="stSidebar"] ::-webkit-scrollbar {{ width: 3px; }}
[data-testid="stSidebar"] ::-webkit-scrollbar-track {{ background: transparent; }}
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {{
  background: {C['green_light']}; border-radius: 2px;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] {{
  background: white !important;
  border: 1.5px solid {C['green_light']} !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}}
[data-testid="stSidebar"] [data-testid="stForm"] {{
  border: none !important; padding: 0 !important;
}}
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox select,
[data-testid="stSidebar"] .stDateInput input {{
  background: {C['green_pale']} !important;
  border-color: {C['green_light']} !important;
  border-radius: 8px !important;
}}

/* ─ Misc ─ */
[data-testid="stDataFrame"]  {{ border-radius: 14px; overflow: hidden; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}

/* Main download buttons (outside sidebar) */
.stDownloadButton > button {{
  background: {C['green']}; color: white; border: none;
  border-radius: 12px; font-weight: 700; padding: 10px 24px;
  font-size: .9rem; letter-spacing: .2px;
  box-shadow: 0 2px 10px rgba(22,163,74,.25);
  transition: all .18s;
}}
.stDownloadButton > button:hover {{
  background: {C['green_mid']};
  box-shadow: 0 6px 20px rgba(22,163,74,.35);
  transform: translateY(-1px);
}}

/* Main action buttons */
.stButton > button {{
  border-radius: 12px; font-weight: 600;
  transition: all .18s;
}}
.stButton > button:hover {{ transform: translateY(-1px); }}

/* Page background */
.stApp {{ background: #f8faf8; }}
.block-container {{ background: transparent; }}

/* Subtle card for success/info */
.stAlert {{ border-radius: 14px !important; border: none !important; }}

/* Caption text */
.stCaption p {{ color: {C['gray']} !important; font-size: .78rem !important; }}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEAT_COLS  = ["FIR%", "GIR%", "Putts", "Up&Down%"]
REQUIRED   = ["Round", "Score"] + FEAT_COLS
FIELD_AVG  = {"FIR%": 52.0, "GIR%": 55.0, "Putts": 32.5, "Up&Down%": 40.0}
STAT_ICONS = {"FIR%": "🏌️", "GIR%": "🎯", "Putts": "⛳", "Up&Down%": "🏹"}
STAT_LABELS= {"FIR%": "Fairways in Regulation", "GIR%": "Greens / Approach Quality",
              "Putts": "Total Putts", "Up&Down%": "Scrambling / Up & Down"}

# ── Chart base layout ─────────────────────────────────────────────────────────
def CL(**kw):
    """Base Plotly layout. Dict kwargs (legend, margin, xaxis, yaxis) are merged
    so callers can pass partial overrides without triggering duplicate-key errors."""
    DEFAULTS = {
        "legend":   dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=12)),
        "margin":   dict(l=14, r=14, t=48, b=14),
        "xaxis":    dict(gridcolor="#f3f4f6", linecolor="#e5e7eb", tickfont=dict(size=11), showgrid=True),
        "yaxis":    dict(gridcolor="#f3f4f6", linecolor="#e5e7eb", tickfont=dict(size=11), showgrid=True),
        "hoverlabel": dict(bgcolor="white", bordercolor="#e5e7eb",
                           font_size=13, font_family="Inter, sans-serif"),
    }
    base = dict(
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color=C["text_mid"]),
        plot_bgcolor="white", paper_bgcolor="white",
        colorway=[C["green"], C["gold"], C["blue"], C["red"], C["purple"]],
    )
    for key, default in DEFAULTS.items():
        merged = dict(default)
        if key in kw:
            merged.update(kw.pop(key))
        base[key] = merged
    base.update(kw)
    return base

# ── HTML helpers ──────────────────────────────────────────────────────────────
def kpi_card(icon, value, label, sub="", accent=""):
    trend_html = ""
    st.markdown(f"""
    <div class="kpi {accent}">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def kpi_trend_card(icon, value, label, sub, delta, lower_is_better=False):
    if abs(delta) < 0.1:
        cls, arrow = "trend-flat", f"→ stable"
    elif (delta < 0) == lower_is_better:
        cls, arrow = "trend-up",   f"↑ improving"
    else:
        cls, arrow = "trend-down", f"↓ declining"
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-sub">{sub}</div>
      <span class="kpi-trend {cls}">{arrow} {abs(delta):.1f}</span>
    </div>""", unsafe_allow_html=True)

def sec(title, emoji=""):
    st.markdown(f"""
    <div class="sec-hdr">
      <div class="sec-dot"></div>
      <h3>{emoji + '  ' if emoji else ''}{title}</h3>
    </div>""", unsafe_allow_html=True)

def prog_bar(pct, color):
    pct_safe = max(0, min(100, pct))
    return f"""<div class="prog-wrap">
      <div class="prog-fill" style="width:{pct_safe}%;background:{color};"></div>
    </div>"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def estimate_handicap(scores, rating=71.5, slope=130):
    diffs = sorted([(s - rating) * 113 / slope for s in scores])
    use   = diffs[: max(1, len(diffs) // 3 if len(diffs) < 20 else 8)]
    return round(np.mean(use) * 0.96, 1)

def norm_putts(v):
    return float(np.clip((36 - v) / (36 - 27) * 100, 0, 100))

# ── Visual components ─────────────────────────────────────────────────────────
def performance_radar(df, good_df, threshold):
    cats = ["FIR%", "GIR%", "Putting", "Up & Down%"]

    def row_vals(d):
        return [d["FIR%"].mean(), d["GIR%"].mean(),
                norm_putts(d["Putts"].mean()), d["Up&Down%"].mean()]

    field_v = [FIELD_AVG["FIR%"], FIELD_AVG["GIR%"],
               norm_putts(FIELD_AVG["Putts"]), FIELD_AVG["Up&Down%"]]
    your_v  = row_vals(df)

    fig = go.Figure()

    # Field avg (background reference)
    fig.add_trace(go.Scatterpolar(
        r=field_v + [field_v[0]], theta=cats + [cats[0]],
        fill="toself", name="Field Average",
        line=dict(color=C["blue"], width=1.5, dash="dot"),
        fillcolor="rgba(37,99,235,.05)", opacity=0.9,
    ))

    # Good round target
    if not good_df.empty:
        good_v = row_vals(good_df)
        fig.add_trace(go.Scatterpolar(
            r=good_v + [good_v[0]], theta=cats + [cats[0]],
            fill="toself", name=f"Sub-{threshold} Target",
            line=dict(color=C["gold"], width=2),
            fillcolor="rgba(202,138,4,.09)",
        ))

    # Your stats (on top)
    fig.add_trace(go.Scatterpolar(
        r=your_v + [your_v[0]], theta=cats + [cats[0]],
        fill="toself", name="Your Average",
        line=dict(color=C["green"], width=2.5),
        fillcolor="rgba(22,163,74,.22)",
        marker=dict(size=7, color=C["green"]),
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e5e7eb",
                            tickfont=dict(size=9), tickvals=[25, 50, 75, 100]),
            angularaxis=dict(tickfont=dict(size=11, color=C["text_mid"])),
            bgcolor="white",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.16, font=dict(size=11), x=0.15),
        height=390, margin=dict(l=30, r=30, t=30, b=70),
        paper_bgcolor="white",
        title=dict(text="Performance Profile", font=dict(size=14, weight="bold"),
                   x=0.5, xanchor="center"),
    )
    return fig

def stat_ring(current, target, color, invert=False):
    pct = min(100, (target / current * 100) if (invert and current > 0)
               else (current / target * 100 if target > 0 else 50))
    remaining = max(0, 100 - pct)
    inner_color = color if pct >= 80 else (C["gold"] if pct >= 55 else C["red"])

    fig = go.Figure(go.Pie(
        values=[pct, remaining],
        hole=0.74,
        marker_colors=[inner_color, "#f3f4f6"],
        textinfo="none", hoverinfo="skip", showlegend=False,
        direction="clockwise", rotation=90,
    ))
    label = f"{pct:.0f}%" if pct < 100 else "✓"
    fig.add_annotation(
        text=f"<b style='font-size:15px'>{current:.0f}</b><br>"
             f"<span style='font-size:10px;color:{C['gray']}'>{label}</span>",
        x=0.5, y=0.5, showarrow=False, xanchor="center",
        font=dict(size=13, color=C["text"]),
    )
    fig.update_layout(
        margin=dict(l=4, r=4, t=4, b=4), height=130,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def shap_dotplot(shap_values, X_scaled):
    rng, traces = np.random.default_rng(0), []
    palette = [C["green"], C["blue"], C["purple"], C["gold"]]
    for i, feat in enumerate(FEAT_COLS):
        sv  = np.array(shap_values)[:, i]
        fv  = X_scaled[:, i]
        fvn = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
        yj  = i + rng.uniform(-0.3, 0.3, size=len(sv))
        traces.append(go.Scatter(
            x=sv, y=yj, mode="markers", name=feat,
            marker=dict(size=10, color=fvn, colorscale="RdYlGn_r",
                        showscale=(i == 0),
                        colorbar=dict(title="Feature<br>Value",
                                      tickvals=[0, 1], ticktext=["Low","High"],
                                      len=0.5, thickness=12) if i == 0 else None,
                        opacity=0.88, line=dict(width=0.5, color="white")),
            hovertemplate=(f"<b>{feat}</b><br>SHAP: %{{x:.3f}}<br>"
                           "Value: %{customdata:.2f}<extra></extra>"),
            customdata=fv,
        ))
    fig = go.Figure(traces)
    fig.add_vline(x=0, line_dash="dash", line_color="#9ca3af", line_width=1.5)
    fig.update_layout(
        **CL(height=400, showlegend=False,
             title=dict(text="SHAP Beeswarm — how each stat drives your score",
                        font=dict(size=14)),
             yaxis=dict(tickvals=list(range(len(FEAT_COLS))), ticktext=FEAT_COLS,
                        gridcolor="#f3f4f6")),
        xaxis_title="SHAP value  (→ raises score · ← lowers score)",
    )
    return fig

def detect_milestones(df, score_threshold):
    badges = []
    best = df["Score"].min()
    best_rd = df.loc[df["Score"].idxmin(), "Round"]
    badges.append(("🏆", f"Personal Best: {int(best)} (Round {int(best_rd)})"))
    # Improvement trend
    if len(df) >= 6:
        first3 = df.head(3)["Score"].mean()
        last3  = df.tail(3)["Score"].mean()
        if last3 < first3 - 1.5:
            badges.append(("📈", f"Season Improvement: ↓{first3 - last3:.1f} strokes since Round 1"))
    # Best GIR round
    best_gir_rd = df.loc[df["GIR%"].idxmax(), "Round"]
    badges.append(("🎯", f"Best Greens Round: {df['GIR%'].max():.0f}% GIR (Round {int(best_gir_rd)})"))
    # Fewest putts
    best_putts_rd = df.loc[df["Putts"].idxmin(), "Round"]
    badges.append(("⛳", f"Fewest Putts: {int(df['Putts'].min())} (Round {int(best_putts_rd)})"))
    # Consecutive good rounds
    perf = df["Performance"].tolist()
    max_streak = cur_streak = 0
    for p in perf:
        if p == "Good":
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
    if max_streak >= 2:
        badges.append(("🔥", f"Best Streak: {max_streak} consecutive good rounds"))
    return badges

# ── Cached ML ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_ml(X_tup, y_score_tup, y_class_tup):
    Xs   = np.array(X_tup)
    y_sc = np.array(y_score_tup)
    y_cl = np.array(y_class_tup)

    reg_mdls = {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
        "XGBoost":       xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    }
    reg_res, loo_preds = {}, {}
    for nm, mdl in reg_mdls.items():
        maes, preds, acts = [], [], []
        for tr, te in LeaveOneOut().split(Xs):
            mdl.fit(Xs[tr], y_sc[tr])
            p = float(mdl.predict(Xs[te])[0])
            preds.append(p); acts.append(float(y_sc[te][0])); maes.append(abs(y_sc[te][0] - p))
        reg_res[nm]   = {"MAE": float(np.mean(maes)), "STD": float(np.std(maes))}
        loo_preds[nm] = {"preds": preds, "actuals": acts}
        mdl.fit(Xs, y_sc)

    clf_mdls = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        "XGBoost":       xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                            random_state=42, eval_metric="logloss"),
    }
    clf_res = {}
    for nm, mdl in clf_mdls.items():
        ps, as_ = [], []
        for tr, te in LeaveOneOut().split(Xs):
            mdl.fit(Xs[tr], y_cl[tr])
            ps.append(int(mdl.predict(Xs[te])[0])); as_.append(int(y_cl[te][0]))
        clf_res[nm] = float(accuracy_score(as_, ps))
        mdl.fit(Xs, y_cl)

    xm       = reg_mdls["XGBoost"]
    sv       = shap.TreeExplainer(xm).shap_values(Xs)
    rf_imp   = reg_mdls["Random Forest"].feature_importances_
    xgb_imp  = xm.feature_importances_
    shap_imp = np.abs(sv).mean(0)

    imp_df = pd.DataFrame({
        "Feature":       FEAT_COLS,
        "Random Forest": rf_imp  / rf_imp.sum()  * 100,
        "XGBoost":       xgb_imp / xgb_imp.sum() * 100,
        "SHAP":          shap_imp / shap_imp.sum() * 100,
    }).sort_values("SHAP", ascending=False).reset_index(drop=True)

    final_mdl = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    final_mdl.fit(Xs, y_sc)

    return dict(reg_res=reg_res, clf_res=clf_res, loo_preds=loo_preds,
                importance_df=imp_df, shap_values=sv.tolist(), final_model=final_mdl)

# ── Sample data ───────────────────────────────────────────────────────────────
def generate_sample_data():
    return pd.DataFrame({
        "Round": list(range(1, 21)),
        "Date":  pd.date_range("2024-02-06", periods=20, freq="7D").strftime("%Y-%m-%d").tolist(),
        "Score": [78,75,80,74,76,79,73,77,75,74,76,71,73,78,74,72,75,70,73,72],
        "FIR%":  [50,57.1,42.9,64.3,57.1,35.7,64.3,50,57.1,57.1,50,71.4,57.1,42.9,64.3,64.3,57.1,71.4,64.3,71.4],
        "GIR%":  [44.4,50,38.9,61.1,50,33.3,55.6,44.4,50,55.6,44.4,66.7,55.6,38.9,55.6,61.1,50,72.2,61.1,66.7],
        "Putts": [33,32,35,30,31,35,31,33,32,31,32,29,31,33,31,30,32,28,30,29],
        "Up&Down%":[33.3,40,20,55.6,44.4,16.7,50,28.6,37.5,48.7,30,60,50,25,45,55,42.9,66.7,57.1,62.5],
        "Type":  ["Practice","Practice","Tournament","Practice","Practice",
                  "Tournament","Practice","Practice","Tournament","Practice",
                  "Practice","Tournament","Practice","Practice","Tournament",
                  "Practice","Practice","Tournament","Practice","Tournament"],
    })

def generate_team_sample_data():
    rng, rows = np.random.default_rng(7), []
    for pl, base in zip([f"Player {c}" for c in "ABCDE"], [74,76,78,73,75]):
        for rnd in range(1, 11):
            s = int(base + rng.integers(-3, 4)); d = s - 75
            rows.append({"Player": pl, "Round": rnd, "Score": s,
                         "FIR%":     round(float(np.clip(52+rng.uniform(-12,12)-d*1.5,20,95)),1),
                         "GIR%":     round(float(np.clip(52+rng.uniform(-12,12)-d*2,  20,95)),1),
                         "Putts":    int(32+rng.integers(-3,4)+d*0.4),
                         "Up&Down%": round(float(np.clip(40+rng.uniform(-18,18)-d*3,  0,100)),1),
                         "Type":     "Tournament" if rnd % 3 == 0 else "Practice"})
    return pd.DataFrame(rows)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Brand header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
      background:linear-gradient(135deg,#14532d 0%,#166534 60%,#15803d 100%);
      padding:28px 16px 24px;text-align:center;
      position:relative;overflow:hidden;
    ">
      <div style="position:absolute;inset:0;pointer-events:none;
        background:url('data:image/svg+xml,%3Csvg width=\'40\' height=\'40\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Ccircle cx=\'20\' cy=\'20\' r=\'1\' fill=\'white\' fill-opacity=\'.07\'/%3E%3C/svg%3E')">
      </div>
      <div style="font-size:2.4rem;line-height:1;margin-bottom:10px;
        filter:drop-shadow(0 2px 6px rgba(0,0,0,.4));position:relative">⛳</div>
      <div style="font-size:1rem;font-weight:900;letter-spacing:2.5px;
        text-transform:uppercase;color:white;margin-bottom:5px;position:relative">
        Golf Analytics
      </div>
      <div style="font-size:.58rem;font-weight:600;letter-spacing:2px;
        text-transform:uppercase;color:rgba(255,255,255,.5);position:relative">
        Varsity · ML-Powered
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Mode ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:.56rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;
      color:#16a34a;margin:18px 0 6px;
      display:flex;align-items:center;gap:8px">
      <span style="flex:1;height:1px;background:#dcfce7"></span>
      MODE
      <span style="flex:1;height:1px;background:#dcfce7"></span>
    </div>""", unsafe_allow_html=True)
    analysis_mode = st.radio("", ["Individual", "Team"],
                             horizontal=True, label_visibility="collapsed")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:.56rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;
      color:#16a34a;margin:18px 0 6px;
      display:flex;align-items:center;gap:8px">
      <span style="flex:1;height:1px;background:#dcfce7"></span>
      UPLOAD DATA
      <span style="flex:1;height:1px;background:#dcfce7"></span>
    </div>""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    # ── Data stats card (shown when data is loaded) ───────────────────────────
    _df_sb = st.session_state.get("df")
    if _df_sb is not None:
        _n   = len(_df_sb)
        _avg = _df_sb["Score"].mean()
        _hcp = estimate_handicap(_df_sb["Score"].tolist())
        st.markdown(f"""
        <div style="
          background:white;border:1.5px solid #dcfce7;
          border-radius:14px;padding:14px 12px;margin:10px 0;
          box-shadow:0 2px 10px rgba(22,163,74,.06);
        ">
          <div style="font-size:.54rem;font-weight:700;letter-spacing:2px;
            text-transform:uppercase;color:#16a34a;
            margin-bottom:12px;display:flex;align-items:center;gap:6px">
            <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
              background:#16a34a"></span>
            Data Loaded
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;text-align:center">
            <div style="background:#f0fdf4;border:1px solid #dcfce7;border-radius:10px;padding:10px 4px">
              <div style="font-size:1.25rem;font-weight:800;color:#14532d;line-height:1">{_n}</div>
              <div style="font-size:.52rem;color:#9ca3af;text-transform:uppercase;
                letter-spacing:1px;margin-top:3px">Rounds</div>
            </div>
            <div style="background:#f0fdf4;border:1px solid #dcfce7;border-radius:10px;padding:10px 4px">
              <div style="font-size:1.25rem;font-weight:800;color:#14532d;line-height:1">{_avg:.1f}</div>
              <div style="font-size:.52rem;color:#9ca3af;text-transform:uppercase;
                letter-spacing:1px;margin-top:3px">Avg</div>
            </div>
            <div style="background:#f0fdf4;border:1px solid #dcfce7;border-radius:10px;padding:10px 4px">
              <div style="font-size:1.25rem;font-weight:800;color:#14532d;line-height:1">{_hcp}</div>
              <div style="font-size:.52rem;color:#9ca3af;text-transform:uppercase;
                letter-spacing:1px;margin-top:3px">HCP</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Round type filter ─────────────────────────────────────────────────────
    if _df_sb is not None and "Type" in _df_sb.columns:
        st.markdown("""
        <div style="font-size:.56rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;
          color:#16a34a;margin:18px 0 6px;
          display:flex;align-items:center;gap:8px">
          <span style="flex:1;height:1px;background:#dcfce7"></span>
          ROUND TYPE
          <span style="flex:1;height:1px;background:#dcfce7"></span>
        </div>""", unsafe_allow_html=True)
        _type_opts = ["All"] + sorted(_df_sb["Type"].dropna().unique().tolist())
        _tf = st.radio("", _type_opts, horizontal=True, label_visibility="collapsed")
        st.session_state["type_filter"] = _tf
    else:
        st.session_state["type_filter"] = "All"

    # ── Log a Round ───────────────────────────────────────────────────────────
    if _df_sb is not None:
        st.markdown("""
        <div style="font-size:.56rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;
          color:#16a34a;margin:18px 0 6px;
          display:flex;align-items:center;gap:8px">
          <span style="flex:1;height:1px;background:#dcfce7"></span>
          LOG ROUND
          <span style="flex:1;height:1px;background:#dcfce7"></span>
        </div>""", unsafe_allow_html=True)
        with st.expander("➕ Add a new round"):
            with st.form("log_round_form", clear_on_submit=True):
                if "Player" in _df_sb.columns:
                    _l_player = st.selectbox("Player", sorted(_df_sb["Player"].unique()))
                _l_date  = st.date_input("Date", value=datetime.date.today())
                _lc1, _lc2 = st.columns(2)
                with _lc1:
                    _l_score = st.number_input("Score", 60, 110, 76)
                    _l_fir   = st.number_input("FIR%",  0.0, 100.0, 50.0, step=1.0)
                    _l_putts = st.number_input("Putts", 20,  50,   32)
                with _lc2:
                    _l_gir  = st.number_input("GIR%",      0.0, 100.0, 50.0, step=1.0)
                    _l_ud   = st.number_input("Up&Down%",  0.0, 100.0, 40.0, step=1.0)
                    _l_type = st.selectbox("Type", ["Practice", "Tournament"])
                if st.form_submit_button("✚ Add Round"):
                    _df_new = _df_sb.copy()
                    if "Type" not in _df_new.columns:
                        _df_new["Type"] = "Practice"
                    _new_rnd = int(_df_new["Round"].max()) + 1
                    _new_row = {
                        "Round": _new_rnd, "Date": str(_l_date),
                        "Score": int(_l_score), "FIR%": float(_l_fir),
                        "GIR%": float(_l_gir),  "Putts": int(_l_putts),
                        "Up&Down%": float(_l_ud), "Type": _l_type,
                    }
                    if "Player" in _df_new.columns:
                        _new_row["Player"] = _l_player
                    st.session_state["df"] = pd.concat(
                        [_df_new, pd.DataFrame([_new_row])], ignore_index=True)
                    st.rerun()

    # ── Settings ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:.56rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;
      color:#16a34a;margin:18px 0 6px;
      display:flex;align-items:center;gap:8px">
      <span style="flex:1;height:1px;background:#dcfce7"></span>
      SETTINGS
      <span style="flex:1;height:1px;background:#dcfce7"></span>
    </div>""", unsafe_allow_html=True)
    score_threshold = st.slider("Good Round Threshold", 65, 90, 76,
                                help="Rounds below this = Good")
    show_advanced = st.checkbox("Show Advanced Analytics", value=True)
    st.markdown(f"""<div style="font-size:.72rem;font-weight:600;
      color:{C['text_mid']};margin:14px 0 3px">Target Handicap</div>""",
      unsafe_allow_html=True)
    target_hcp_input = st.number_input("Target HCP", min_value=0.0, max_value=36.0,
                                        value=4.0, step=0.5,
                                        label_visibility="collapsed",
                                        help="Your HCP goal for this season")
    st.session_state["target_hcp"] = target_hcp_input

    # ── Resources ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:.56rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;
      color:#16a34a;margin:18px 0 6px;
      display:flex;align-items:center;gap:8px">
      <span style="flex:1;height:1px;background:#dcfce7"></span>
      RESOURCES
      <span style="flex:1;height:1px;background:#dcfce7"></span>
    </div>""", unsafe_allow_html=True)
    template = "Round,Date,Score,FIR%,GIR%,Putts,Up&Down%\n1,2024-01-15,75,57.1,50.0,32,40.0\n"
    st.download_button("📥 Download CSV Template", data=template,
                       file_name="golf_template.csv", mime="text/csv")
    st.markdown("""<div style="font-size:.7rem;color:#9ca3af;
      margin-top:8px;line-height:1.75">
      Round · Score · FIR% · GIR%<br>Putts · Up&amp;Down%
      <span style="opacity:.7"> · Date · Player</span>
    </div>""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:32px;padding:14px 0 4px;
      border-top:1.5px solid #dcfce7;text-align:center">
      <div style="font-size:.52rem;color:#9ca3af;
        letter-spacing:1.5px;text-transform:uppercase">
        Golf Analytics · Open Source
      </div>
    </div>""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">⛳ Built for Varsity Golf</div>
  <h1>Know exactly what to<br>practice next.</h1>
  <p>Upload your round stats and find out which part of your game —
  fairways, greens, putting, or scrambling — is actually costing you
  the most strokes. Get a personalized, data-backed practice plan in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
if uploaded_file is None and "df" not in st.session_state:
    # ── CTA banner ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
      background:linear-gradient(135deg,{C['green_pale']} 0%,white 100%);
      border:1.5px solid {C['green_light']};
      border-radius:18px; padding:24px 28px 18px;
      margin-bottom:8px;
      display:flex; align-items:center; justify-content:space-between; gap:16px;
    ">
      <div>
        <div style="font-size:1.05rem;font-weight:800;color:{C['text']};margin-bottom:4px">
          Try it now — no signup needed
        </div>
        <div style="font-size:.85rem;color:{C['gray']}">
          Load a sample season to explore every feature, or upload your own CSV via the sidebar.
        </div>
      </div>
      <div style="font-size:.82rem;color:{C['green']};font-weight:700;white-space:nowrap;opacity:.8">
        ↓ Choose below
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("📊 Individual Demo", width="stretch"):
            st.session_state["df"]   = generate_sample_data()
            st.session_state["mode"] = "Individual"
            st.rerun()
    with c2:
        if st.button("👥 Team Demo", width="stretch"):
            st.session_state["df"]   = generate_team_sample_data()
            st.session_state["mode"] = "Team"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    sec("What You Get")
    f1, f2, f3 = st.columns(3)
    features = [
        ("📊", "Score Predictions",
         "Random Forest and XGBoost models predict your score and classify rounds as "
         "good or bad — trained with Leave-One-Out CV so small datasets actually work."),
        ("🔍", "What's Costing You Strokes",
         "SHAP analysis breaks down which stat — fairways, greens, putts, or scrambling — "
         "moved your score the most in every single round. No guessing."),
        ("💡", "Your Practice Plan",
         "Drills and a weekly schedule ranked by real impact on your score, not "
         "what feels off. Built automatically from your data gaps."),
        ("📈", "Season Trends",
         "Handicap index that updates round by round, rolling averages, consistency "
         "tracker, and your stats vs. high school varsity field averages."),
        ("🎯", "Performance Radar",
         "Radar chart showing your current stats vs. your own good-round averages "
         "and the varsity field — see your whole game in one view."),
        ("👥", "Team Mode",
         "Add a Player column and instantly get a team leaderboard, skill radar, "
         "stat comparison, and score trends for every player."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
              <div class="feature-top">
                <span class="feature-icon">{icon}</span>
                <div class="feature-title">{title}</div>
              </div>
              <div class="feature-body">
                <div class="feature-desc">{desc}</div>
              </div>
            </div><br>""", unsafe_allow_html=True)

    # How it works
    sec("How It Works")
    s1, arr1, s2, arr2, s3 = st.columns([3, 0.6, 3, 0.6, 3])
    with s1:
        st.markdown("""<div class="step-card">
          <div class="step-num">1</div>
          <div class="step-text">Upload your round data CSV — or use the demo</div>
        </div>""", unsafe_allow_html=True)
    with arr1:
        st.markdown("<br><br><div style='text-align:center;font-size:1.4rem;color:#bbf7d0;'>→</div>",
                    unsafe_allow_html=True)
    with s2:
        st.markdown("""<div class="step-card">
          <div class="step-num">2</div>
          <div class="step-text">Models run on your stats and rank what's driving your score</div>
        </div>""", unsafe_allow_html=True)
    with arr2:
        st.markdown("<br><br><div style='text-align:center;font-size:1.4rem;color:#bbf7d0;'>→</div>",
                    unsafe_allow_html=True)
    with s3:
        st.markdown("""<div class="step-card">
          <div class="step-num">3</div>
          <div class="step-text">Get a ranked practice plan built around your biggest gaps</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Sample Data Preview")
    st.dataframe(generate_sample_data().head(6), width="stretch", hide_index=True)
    st.stop()

elif uploaded_file is not None:
    try:
        df_raw  = pd.read_csv(uploaded_file)
        missing = [c for c in REQUIRED if c not in df_raw.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            st.stop()
        st.session_state["df"]   = df_raw
        st.session_state["mode"] = "Team" if "Player" in df_raw.columns else "Individual"
        st.success(f"✅ Loaded {len(df_raw)} rounds.")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

if "df" not in st.session_state:
    st.stop()

df   = st.session_state["df"].copy()
mode = st.session_state.get("mode", analysis_mode)

# Apply round type filter
_type_filter = st.session_state.get("type_filter", "All")
if _type_filter != "All" and "Type" in df.columns:
    df = df[df["Type"] == _type_filter].copy()
    if df.empty:
        st.warning(f"No {_type_filter} rounds found. Change the Round Type filter in the sidebar.")
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# TEAM MODE
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "Team" and "Player" in df.columns:
    st.markdown("## 👥 Team Performance Dashboard")
    players = sorted(df["Player"].unique())

    summary = (df.groupby("Player")
               .agg(Rounds=("Score","count"), Avg_Score=("Score","mean"),
                    Best=("Score","min"), GIR=("GIR%","mean"),
                    FIR=("FIR%","mean"), Putts=("Putts","mean"),
                    UpDown=("Up&Down%","mean"))
               .round(1).reset_index()
               .sort_values("Avg_Score")
               .rename(columns={"Avg_Score":"Avg Score","GIR":"GIR%",
                                 "FIR":"FIR%","UpDown":"Up&Down%"}))

    tA, tB, tC = st.tabs(["🏆 Leaderboard", "📊 Stat Comparison", "📈 Score Trends"])

    with tA:
        k1, k2, k3 = st.columns(3)
        low_idx = df["Score"].idxmin()
        with k1: kpi_card("🏆", str(int(df["Score"].min())), "Team Low",
                           f"by {df.loc[low_idx,'Player']}")
        with k2: kpi_card("📊", f"{df['Score'].mean():.1f}", "Team Avg",
                           f"{len(df)} total rounds", "gold")
        with k3: kpi_card("👥", str(len(players)), "Players",
                           f"~{len(df)//len(players)} rounds each", "blue")

        sec("Leaderboard")
        st.dataframe(summary.style.background_gradient(subset=["Avg Score"], cmap="RdYlGn_r"),
                     width="stretch", hide_index=True)

        fig = go.Figure(go.Bar(
            x=summary["Player"], y=summary["Avg Score"],
            marker=dict(color=summary["Avg Score"].tolist(),
                        colorscale="RdYlGn_r", showscale=True,
                        colorbar=dict(title="Avg Score", len=0.6),
                        line=dict(color="white", width=1.5)),
            text=summary["Avg Score"].round(1), textposition="outside",
        ))
        fig.add_hline(y=summary["Avg Score"].mean(), line_dash="dash", line_color=C["gold"],
                      annotation_text=f"Team avg: {summary['Avg Score'].mean():.1f}",
                      annotation_position="right")
        fig.update_layout(**CL(height=360, title="Average Score by Player"))
        st.plotly_chart(fig, width="stretch")

    with tB:
        stat = st.selectbox("Stat to compare", FEAT_COLS, format_func=lambda x: STAT_LABELS[x])
        fig  = go.Figure()
        pal  = [C["green"], C["gold"], C["blue"], C["red"], C["purple"]]
        for i, pl in enumerate(players):
            vals = df[df["Player"] == pl][stat]
            fig.add_trace(go.Box(y=vals, name=pl, boxmean=True, opacity=0.85,
                                  marker_color=pal[i % 5]))
        fig.update_layout(**CL(height=420, title=f"{STAT_LABELS[stat]} — by Player"),
                          yaxis_title=stat)
        st.plotly_chart(fig, width="stretch")

        sec("Skill Radar")
        fig2 = go.Figure()
        cats = ["FIR%", "GIR%", "Putting", "Up & Down%"]
        for i, pl in enumerate(players):
            r    = summary[summary["Player"] == pl].iloc[0]
            vals = [r["FIR%"], r["GIR%"],
                    float(np.clip(100 - (r["Putts"] - 27) * 6, 0, 100)),
                    r["Up&Down%"]]
            fig2.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name=pl, opacity=0.55,
                line=dict(color=pal[i % len(pal)], width=2)))
        fig2.update_layout(height=440,
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            title="Team Skill Radar", showlegend=True)
        st.plotly_chart(fig2, width="stretch")

    with tC:
        fig = px.line(df.sort_values("Round"), x="Round", y="Score",
                      color="Player", markers=True, title="Score Trends by Player")
        fig.update_layout(**CL(height=420), yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL MODE
# ═══════════════════════════════════════════════════════════════════════════════
df["Performance"] = df["Score"].apply(lambda x: "Good" if x < score_threshold else "Bad")
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

X        = df[FEAT_COLS].values
y_score  = df["Score"].values.astype(float)
y_class  = (df["Score"] < score_threshold).astype(int).values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

with st.spinner("Training models…"):
    ml = run_ml(tuple(tuple(r) for r in X_scaled),
                tuple(float(v) for v in y_score),
                tuple(int(v) for v in y_class))

importance_df = ml["importance_df"]
shap_values   = np.array(ml["shap_values"])

n         = len(df)
good_df   = df[df["Performance"] == "Good"]
bad_df    = df[df["Performance"] == "Bad"]
handicap  = estimate_handicap(df["Score"].tolist())
good_pct  = len(good_df) / n * 100
recent5   = df.tail(5)["Score"].mean()
prev5     = df.head(max(1, n - 5)).tail(5)["Score"].mean()

has_date = "Date" in df.columns and df["Date"].notna().any()
x_axis   = df["Date"].dt.strftime("%b %d") if has_date else df["Round"].astype(str)
x_title  = "Date" if has_date else "Round"

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "📈 Season Trends", "🎯 Statistics",
    "🤖 ML Models", "🔍 SHAP", "💡 Recommendations",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi_card("📊", f"{df['Score'].mean():.1f}", "Avg Score",
                       f"Best: {int(df['Score'].min())}  ·  Worst: {int(df['Score'].max())}")
    with k2: kpi_trend_card("📉", f"{recent5:.1f}", "Last 5 Avg",
                              f"vs {prev5:.1f} before", recent5 - prev5, lower_is_better=True)
    with k3: kpi_card("🏷️", str(handicap), "Handicap Est.",
                       f"from {n} rounds", "blue")
    with k4: kpi_card("✅", f"{good_pct:.0f}%", f"Sub-{score_threshold} Rate",
                       f"{len(good_df)} of {n} rounds")
    with k5: kpi_card("📐", f"±{df['Score'].std():.1f}", "Consistency",
                       "std dev of scores", "gold")

    st.markdown("<br>", unsafe_allow_html=True)

    # Score trend + Radar side by side
    c1, c2 = st.columns([3, 2])
    with c1:
        sec("Score History")
        fig = go.Figure()
        fig.add_hrect(y0=df["Score"].min() - 1, y1=score_threshold,
                      fillcolor=C["green_light"], opacity=0.25, line_width=0,
                      annotation_text="Good zone ✓", annotation_position="right")
        if "Type" in df.columns:
            _mkr_colors  = [C["blue"] if t == "Tournament" else C["green"]
                            for t in df["Type"]]
            _mkr_symbols = ["diamond" if t == "Tournament" else "circle"
                            for t in df["Type"]]
            _mkr = dict(size=12, color=_mkr_colors, symbol=_mkr_symbols,
                        line=dict(width=1.5, color="white"))
        else:
            _mkr = dict(size=11, color=df["Score"].tolist(),
                        colorscale="RdYlGn_r", showscale=True,
                        colorbar=dict(title="Score", len=0.5, thickness=12))
        fig.add_trace(go.Scatter(
            x=x_axis, y=df["Score"], mode="lines+markers",
            line=dict(color="#d1d5db", width=1.5, dash="dot"),
            marker=_mkr, name="Score",
            hovertemplate="Round %{x}<br><b>Score: %{y}</b><extra></extra>",
        ))
        if "Type" in df.columns:
            for _tn, _tc, _ts in [("Tournament", C["blue"], "diamond"),
                                   ("Practice",   C["green"], "circle")]:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=_tc, symbol=_ts,
                                line=dict(width=1.5, color="white")),
                    name=_tn, showlegend=True,
                ))
        if n >= 3:
            fig.add_trace(go.Scatter(
                x=x_axis, y=df["Score"].rolling(3, min_periods=2).mean(),
                mode="lines", name="3-Rnd Avg",
                line=dict(color=C["green"], width=2.5),
            ))
        if n >= 5:
            fig.add_trace(go.Scatter(
                x=x_axis, y=df["Score"].rolling(5, min_periods=3).mean(),
                mode="lines", name="5-Rnd Avg",
                line=dict(color=C["gold"], width=2.5, dash="dash"),
            ))
        fig.add_hline(y=score_threshold, line_dash="dash",
                      line_color=C["red"], opacity=0.6,
                      annotation_text=f"Target {score_threshold}",
                      annotation_position="right")
        # Annotate personal best
        best_idx = df["Score"].idxmin()
        fig.add_annotation(
            x=x_axis.iloc[best_idx], y=df["Score"].min(),
            text="🏆 PB", showarrow=True, arrowhead=2, arrowcolor=C["gold"],
            font=dict(color=C["gold"], weight="bold", size=12),
            bgcolor="white", bordercolor=C["gold"], borderwidth=1.5,
            ay=-36,
        )
        fig.update_layout(
            **CL(height=370, title="Score Trend with Rolling Averages",
                 legend=dict(orientation="h", y=1.08)),
            xaxis_title=x_title, yaxis_title="Score",
            yaxis_autorange="reversed",
        )
        st.plotly_chart(fig, width="stretch")

    with c2:
        sec("Performance Profile")
        fig_radar = performance_radar(df, good_df, score_threshold)
        st.plotly_chart(fig_radar, width="stretch")

    # Stat rings row
    st.markdown("<br>", unsafe_allow_html=True)
    sec("Stat Performance vs Good-Round Target")
    st.caption("Ring fill = % of the way to your good-round average. Outer number = your current average.")
    if not good_df.empty:
        ring_cols = st.columns(4)
        ring_colors = [C["green"], C["blue"], C["purple"], C["gold"]]
        for col, feat, color in zip(ring_cols, FEAT_COLS, ring_colors):
            with col:
                curr   = df[feat].mean()
                target = good_df[feat].mean()
                invert = (feat == "Putts")
                st.markdown(
                    f"<div style='text-align:center;font-size:.78rem;font-weight:700;"
                    f"color:{C['gray']};text-transform:uppercase;letter-spacing:1px;"
                    f"margin-bottom:2px'>{STAT_ICONS[feat]} {feat}</div>",
                    unsafe_allow_html=True)
                st.plotly_chart(stat_ring(curr, target, color, invert),
                                width="stretch")
                st.markdown(
                    f"<div style='text-align:center;font-size:.75rem;color:{C['gray']}'>"
                    f"Target: <b>{target:.1f}</b></div>",
                    unsafe_allow_html=True)

    # Achievement badges
    st.markdown("<br>", unsafe_allow_html=True)
    sec("Season Highlights")
    badges = detect_milestones(df, score_threshold)
    badge_html = "".join(
        f'<span class="achievement"><span>{icon}</span>{text}</span>'
        for icon, text in badges
    )
    st.markdown(f"<div>{badge_html}</div>", unsafe_allow_html=True)

    # ── Goal Progress ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    sec("🎯 Season Goal")

    _target_hcp = st.session_state.get("target_hcp", 4.0)
    _start_hcp  = estimate_handicap(df.head(min(3, n))["Score"].tolist())
    _total_drop = max(0.1, _start_hcp - _target_hcp)
    _pct_done   = max(0, min(100, (_start_hcp - handicap) / _total_drop * 100))

    _improve_rate = 0.0
    if n >= 4:
        _half = n // 2
        _improve_rate = (estimate_handicap(df.head(_half)["Score"].tolist()) -
                         estimate_handicap(df.tail(_half)["Score"].tolist())) / _half

    if handicap <= _target_hcp:
        _proj_msg, _proj_color = "🎉 Goal reached! Set a new target in the sidebar.", C["green"]
    elif _improve_rate > 0.01:
        _rds = max(1, int(np.ceil((handicap - _target_hcp) / _improve_rate)))
        _proj_msg  = f"At your current pace: ~{_rds} more rounds to reach HCP {_target_hcp}"
        _proj_color = C["green"]
    elif _improve_rate < -0.01:
        _proj_msg, _proj_color = "HCP trending up — review your practice focus.", C["red"]
    else:
        _proj_msg, _proj_color = "Holding steady — keep logging rounds for a projection.", C["gold"]

    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        kpi_card("🏷️", str(handicap), "Current HCP",
                  f"Target: {_target_hcp}  ·  Gap: {max(0, handicap - _target_hcp):.1f}", "blue")
    with gc2:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-icon">📈</div>
          <div class="kpi-label">Progress to Goal</div>
          <div style="margin:10px 0 4px;font-size:.75rem;
            display:flex;justify-content:space-between;color:{C['gray']}">
            <span>Start {_start_hcp:.1f}</span><span>Goal {_target_hcp}</span>
          </div>
          <div style="background:#f3f4f6;border-radius:20px;height:10px;overflow:hidden;margin-bottom:10px">
            <div style="width:{_pct_done:.0f}%;height:100%;border-radius:20px;
              background:linear-gradient(90deg,{C['green']},{C['green_light']})"></div>
          </div>
          <div style="font-size:.78rem;font-weight:600;color:{_proj_color}">{_proj_msg}</div>
        </div>""", unsafe_allow_html=True)
    with gc3:
        if "Type" in df.columns:
            _tdf = df[df["Type"] == "Tournament"]
            _pdf = df[df["Type"] == "Practice"]
            _t_avg = f"{_tdf['Score'].mean():.1f}" if not _tdf.empty else "—"
            _p_avg = f"{_pdf['Score'].mean():.1f}" if not _pdf.empty else "—"
            st.markdown(f"""
            <div class="kpi">
              <div class="kpi-icon">📋</div>
              <div class="kpi-label">By Round Type</div>
              <div style="margin-top:14px">
                <div style="display:flex;align-items:center;justify-content:space-between;
                  margin-bottom:8px;padding:9px 12px;background:{C['blue_light']};
                  border-radius:10px">
                  <span style="font-size:.82rem;font-weight:700;color:{C['blue']}">
                    🏆 Tournament</span>
                  <span style="font-size:.78rem;color:{C['text_mid']}">
                    {len(_tdf)} rds · avg {_t_avg}</span>
                </div>
                <div style="display:flex;align-items:center;justify-content:space-between;
                  padding:9px 12px;background:{C['green_light']};border-radius:10px">
                  <span style="font-size:.82rem;font-weight:700;color:{C['green']}">
                    ⛳ Practice</span>
                  <span style="font-size:.78rem;color:{C['text_mid']}">
                    {len(_pdf)} rds · avg {_p_avg}</span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            kpi_card("📋", str(n), "Rounds Logged",
                      "Add Type col (Tournament/Practice) for breakdown", "purple")

    st.markdown("<br>", unsafe_allow_html=True)

    # Data table + distribution
    c1, c2 = st.columns([3, 2])
    with c1:
        sec("Round-by-Round Data")
        display_cols = [c for c in ["Round","Date","Score"] + FEAT_COLS + ["Type"] if c in df.columns]
        st.dataframe(
            df[display_cols].style.background_gradient(subset=["Score"], cmap="RdYlGn_r").format(precision=1),
            width="stretch", height=340, hide_index=True,
        )
    with c2:
        sec("Summary Stats")
        st.dataframe(df[["Score"]+FEAT_COLS].describe().round(2), width="stretch")
        fig2 = px.histogram(df, x="Score", nbins=10, color_discrete_sequence=[C["green"]])
        fig2.add_vline(x=score_threshold, line_dash="dash", line_color=C["red"],
                       annotation_text="Target")
        fig2.add_vline(x=df["Score"].mean(), line_dash="dot", line_color=C["gold"],
                       annotation_text="Avg")
        fig2.update_traces(marker_line_color="white", marker_line_width=1.5)
        fig2.update_layout(**CL(height=240, title="Score Distribution",
                               margin=dict(l=5,r=5,t=42,b=5)))
        st.plotly_chart(fig2, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · SEASON TRENDS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if n < 4:
        st.warning("Need at least 4 rounds for trend analysis.")
    else:
        hdcp_trend  = [estimate_handicap(df["Score"].iloc[:i+1].tolist()) for i in range(n)]
        rolling_std = df["Score"].rolling(5, min_periods=3).std()

        c1, c2 = st.columns(2)
        with c1:
            sec("Handicap Index Over Season")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_axis, y=hdcp_trend, mode="lines+markers",
                fill="tozeroy", fillcolor="rgba(37,99,235,.07)",
                line=dict(color=C["blue"], width=2.5),
                marker=dict(size=8, color=C["blue"], line=dict(width=2, color="white")),
                hovertemplate="Round %{x}<br><b>Handicap: %{y:.1f}</b><extra></extra>",
            ))
            fig.update_layout(**CL(height=290, title="Estimated Handicap Index"),
                              yaxis_autorange="reversed",
                              xaxis_title=x_title, yaxis_title="Handicap Index")
            st.plotly_chart(fig, width="stretch")

        with c2:
            sec("Scoring Consistency")
            bar_c = [C["green"] if (v is not None and not np.isnan(float(v)) and v < 3)
                     else C["gold"] if (v is not None and not np.isnan(float(v)) and v < 5)
                     else C["red"] for v in rolling_std]
            fig = go.Figure(go.Bar(
                x=x_axis, y=rolling_std, marker_color=bar_c,
                hovertemplate="Round %{x}<br>Std Dev: %{y:.2f}<extra></extra>",
            ))
            fig.update_layout(**CL(height=290, title="5-Round Std Dev (lower = more consistent)"),
                              xaxis_title=x_title, yaxis_title="Std Dev (strokes)")
            st.plotly_chart(fig, width="stretch")

        sec("Stat Trends Over Season")
        stat_choice = st.multiselect("Select stats", FEAT_COLS, default=["GIR%","Putts"],
                                     format_func=lambda x: STAT_LABELS[x])
        if stat_choice:
            palette = [C["green"], C["gold"], C["blue"], C["red"]]
            fig = go.Figure()
            for i, feat in enumerate(stat_choice):
                col = palette[i % len(palette)]
                fig.add_trace(go.Scatter(
                    x=x_axis, y=df[feat], mode="markers",
                    marker=dict(color=col, size=7, opacity=0.4), showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=x_axis, y=df[feat].rolling(3, min_periods=2).mean(),
                    mode="lines", name=f"{feat} (3-rnd avg)",
                    line=dict(color=col, width=2.5),
                    fill="tozeroy" if i == 0 else None,
                    fillcolor=f"rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},.06)" if i == 0 else None,
                ))
                fig.add_hline(y=FIELD_AVG[feat], line_dash="dot", line_color=col, opacity=0.3,
                              annotation_text=f"Field avg", annotation_position="right")
            fig.update_layout(**CL(height=390, title="Stat Trends vs Field Average",
                                   legend=dict(orientation="h", y=1.08)),
                              xaxis_title=x_title, yaxis_title="Value")
            st.plotly_chart(fig, width="stretch")

        sec("Strokes vs Field Average")
        st.caption("Field averages for high school varsity: FIR% 52 · GIR% 55 · Putts 32.5 · U&D% 40")
        rows = []
        for feat in FEAT_COLS:
            p = df[feat].mean(); f = FIELD_AVG[feat]; diff = p - f
            est = -diff * 0.06 if feat != "Putts" else diff * 0.18
            rows.append({"Stat": feat, "Your Avg": round(p,1), "Field Avg": f,
                         "Est. Strokes Gained": round(est, 2)})
        sv_df = pd.DataFrame(rows)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(sv_df.style.background_gradient(
                subset=["Est. Strokes Gained"], cmap="RdYlGn"),
                width="stretch", hide_index=True)
        with c2:
            colors = [C["green"] if v >= 0 else C["red"] for v in sv_df["Est. Strokes Gained"]]
            fig = go.Figure(go.Bar(
                x=sv_df["Stat"], y=sv_df["Est. Strokes Gained"],
                marker_color=colors,
                text=sv_df["Est. Strokes Gained"].apply(lambda v: f"{v:+.2f}"),
                textposition="outside",
                marker_line_color="white", marker_line_width=1.5,
            ))
            fig.add_hline(y=0, line_color="#9ca3af", line_width=1.5)
            fig.update_layout(**CL(height=310, title="Estimated Strokes Gained vs Field"),
                              yaxis_title="Strokes (+ = gained)")
            st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    corrs = df[FEAT_COLS].corrwith(df["Score"]).sort_values()

    c1, c2 = st.columns(2)
    with c1:
        sec("Correlation with Score")
        bar_c = [C["green"] if v < 0 else C["red"] for v in corrs.values]
        fig = go.Figure(go.Bar(
            x=corrs.values, y=corrs.index, orientation="h",
            marker_color=bar_c,
            text=[f"r = {v:+.3f}" for v in corrs.values], textposition="outside",
            marker_line_color="white", marker_line_width=1,
        ))
        fig.add_vline(x=0, line_color="#9ca3af", line_width=1.5)
        fig.update_layout(**CL(height=300, title="Pearson Correlation with Score"),
                          xaxis_title="Correlation (negative = better stat → lower score)",
                          xaxis_range=[-1.15, 1.15])
        st.plotly_chart(fig, width="stretch")
        for feat, r in corrs.items():
            direction = "lower scores" if r < 0 else "higher scores"
            strength  = "Strong" if abs(r) > 0.6 else "Moderate" if abs(r) > 0.35 else "Weak"
            dot = "🟢" if (abs(r) > 0.6 and r < 0) else "🔴" if (abs(r) > 0.6 and r > 0) else "🟡"
            st.markdown(f"{dot} **{feat}** — {strength}, {direction} (r = {r:+.3f})")

    with c2:
        sec("Correlation Heatmap")
        corr_mat = df[["Score"]+FEAT_COLS].corr()
        fig = px.imshow(corr_mat, text_auto=".2f",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(**CL(height=320, margin=dict(l=5,r=5,t=10,b=5)))
        fig.update_traces(textfont=dict(size=12))
        st.plotly_chart(fig, width="stretch")

    sec("Stat vs Score — Interactive Scatter")
    sel = st.selectbox("Select statistic", FEAT_COLS,
                       format_func=lambda x: f"{STAT_ICONS[x]}  {STAT_LABELS[x]}")
    fig = px.scatter(df, x=sel, y="Score", trendline="ols",
                     color="Score", color_continuous_scale="RdYlGn_r", size_max=14,
                     hover_data={"Round": True})
    r_val = df[sel].corr(df["Score"])
    fig.add_annotation(text=f"r = {r_val:+.3f}", xref="paper", yref="paper",
                       x=0.04, y=0.96, showarrow=False, bgcolor="white",
                       bordercolor="#d1d5db", borderwidth=1.5,
                       font=dict(size=13, weight="bold"))
    fig.update_layout(**CL(height=380, title=f"{STAT_LABELS[sel]} vs Score"),
                      yaxis_autorange="reversed",
                      xaxis_title=STAT_LABELS[sel], yaxis_title="Score")
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")),
                      selector=dict(mode="markers"))
    st.plotly_chart(fig, width="stretch")

    sec("Good vs Bad Rounds")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.markdown(f"**Good Rounds** (< {score_threshold})  ·  {len(good_df)} rounds")
        if not good_df.empty:
            st.dataframe(good_df[FEAT_COLS].mean().round(1).to_frame("Avg"),
                         width="stretch")
    with c2:
        st.markdown(f"**Bad Rounds** (≥ {score_threshold})  ·  {len(bad_df)} rounds")
        if not bad_df.empty:
            st.dataframe(bad_df[FEAT_COLS].mean().round(1).to_frame("Avg"),
                         width="stretch")
    with c3:
        if not good_df.empty and not bad_df.empty:
            cmp = pd.DataFrame({"Good": good_df[FEAT_COLS].mean(),
                                 "Bad":  bad_df[FEAT_COLS].mean()}).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(name=f"Good (<{score_threshold})", x=cmp["index"],
                                  y=cmp["Good"], marker_color=C["green"],
                                  text=cmp["Good"].round(1), textposition="outside",
                                  marker_line_color="white", marker_line_width=1.5))
            fig.add_trace(go.Bar(name=f"Bad (≥{score_threshold})", x=cmp["index"],
                                  y=cmp["Bad"], marker_color=C["red"],
                                  text=cmp["Bad"].round(1), textposition="outside",
                                  marker_line_color="white", marker_line_width=1.5))
            fig.update_layout(**CL(height=320, title="Good vs Bad Rounds"),
                              barmode="group", yaxis_title="Average")
            st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · ML MODELS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    sec("Score Prediction — Leave-One-Out CV")
    st.caption("LOO-CV: each round is held out once and predicted by a model trained on all others.")

    c1, c2, c3 = st.columns([1, 1, 2])
    for col, (nm, res) in zip([c1, c2], ml["reg_res"].items()):
        with col:
            st.metric(nm, f"±{res['MAE']:.2f} strokes", f"σ = {res['STD']:.2f}")
    with c3:
        st.info("**MAE** = average prediction error. ±2.5 strokes means a predicted 75 "
                "is typically between 72.5–77.5.")

    best_nm = min(ml["reg_res"], key=lambda k: ml["reg_res"][k]["MAE"])
    lp = ml["loo_preds"][best_nm]
    lo = min(min(lp["actuals"]), min(lp["preds"])) - 1
    hi = max(max(lp["actuals"]), max(lp["preds"])) + 1

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lp["actuals"], y=lp["preds"], mode="markers",
            marker=dict(size=11, color=lp["actuals"], colorscale="RdYlGn_r",
                        showscale=True, colorbar=dict(title="Actual", len=0.6, thickness=12),
                        line=dict(width=1, color="white")),
            hovertemplate="Actual: %{x}<br>Predicted: %{y:.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
                                  line=dict(dash="dash", color="#9ca3af", width=1.5),
                                  name="Perfect", showlegend=True))
        fig.update_layout(**CL(height=360, title=f"Actual vs Predicted ({best_nm})"),
                          xaxis_title="Actual Score", yaxis_title="Predicted Score")
        st.plotly_chart(fig, width="stretch")

    with c2:
        sec("Classification Accuracy")
        for nm, acc in ml["clf_res"].items():
            st.metric(nm, f"{acc:.1%}", "good/bad classification")

        # Error distribution
        errs = [abs(a - p) for a, p in zip(lp["actuals"], lp["preds"])]
        fig2 = px.histogram(x=errs, nbins=10, color_discrete_sequence=[C["green"]],
                            title="Prediction Error Distribution")
        fig2.add_vline(x=np.mean(errs), line_dash="dash", line_color=C["gold"],
                       annotation_text=f"Avg: {np.mean(errs):.1f}")
        fig2.update_traces(marker_line_color="white", marker_line_width=1.5)
        fig2.update_layout(**CL(height=250, title="Error Distribution",
                               margin=dict(l=5,r=5,t=42,b=5)),
                           xaxis_title="Error (strokes)", yaxis_title="Rounds")
        st.plotly_chart(fig2, width="stretch")

    if show_advanced:
        sec("Predict Your Next Round")
        c1, c2, c3, c4 = st.columns(4)
        with c1: p_fir   = st.slider("FIR%",     0.0,100.0, float(df["FIR%"].mean()),    5.0)
        with c2: p_gir   = st.slider("GIR%",     0.0,100.0, float(df["GIR%"].mean()),    5.0)
        with c3: p_putts = st.slider("Putts",    20, 45,    int(df["Putts"].mean()))
        with c4: p_ud    = st.slider("Up&Down%", 0.0,100.0, float(df["Up&Down%"].mean()),5.0)

        inp    = scaler.transform([[p_fir, p_gir, p_putts, p_ud]])
        pred_s = float(ml["final_model"].predict(inp)[0])

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if pred_s < score_threshold:
                st.success(f"### Predicted Score: {pred_s:.1f}  ✅  Good round!")
            else:
                st.warning(f"### Predicted Score: {pred_s:.1f}  — Above target")

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred_s,
                delta={"reference": score_threshold, "valueformat": ".1f"},
                gauge=dict(
                    axis=dict(range=[65, 88], tickwidth=1),
                    bar=dict(color=C["green"] if pred_s < score_threshold else C["red"],
                             thickness=0.3),
                    bgcolor="white",
                    steps=[dict(range=[65, score_threshold], color=C["green_light"]),
                           dict(range=[score_threshold, 88], color=C["red_light"])],
                    threshold=dict(line=dict(color=C["red"], width=3),
                                   thickness=0.75, value=score_threshold),
                ),
                title={"text": "Predicted Score"},
            ))
            fig.update_layout(height=290, margin=dict(l=20,r=20,t=50,b=10))
            st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · SHAP
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown(f"""
    <div style='background:{C["green_pale"]};border:1.5px solid {C["green_light"]};
    border-radius:12px;padding:16px 22px;margin-bottom:24px;'>
    <b>🔍 What is SHAP?</b>  SHapley Additive exPlanations assigns each stat a score
    showing exactly how much it pushed your score up or down in <em>each specific round</em> —
    going beyond correlation to show real contribution.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        sec("Importance Rankings (%)")
        st.dataframe(
            importance_df.style
            .background_gradient(subset=["SHAP"], cmap="YlGn")
            .format({"Random Forest":"{:.1f}","XGBoost":"{:.1f}","SHAP":"{:.1f}"}),
            width="stretch", hide_index=True, height=210,
        )
    with c2:
        sec("Method Comparison")
        fig = go.Figure()
        for method, color in zip(["Random Forest","XGBoost","SHAP"],
                                  [C["green"], C["blue"], C["gold"]]):
            fig.add_trace(go.Bar(
                name=method, x=importance_df["Feature"], y=importance_df[method],
                marker_color=color, opacity=0.88,
                text=importance_df[method].round(1), textposition="inside",
                textfont=dict(color="white", size=11),
                marker_line_color="white", marker_line_width=1.5,
            ))
        fig.update_layout(**CL(height=270, title="Feature Importance (%) — 3 Methods"),
                          barmode="group", yaxis_title="Importance (%)")
        st.plotly_chart(fig, width="stretch")

    sec("SHAP Beeswarm Plot")
    st.caption("Each dot = one round. Red = high stat value. Blue = low stat value. "
               "Right of zero raises your score; left lowers it.")
    st.plotly_chart(shap_dotplot(shap_values, X_scaled), width="stretch")

    with st.expander("📖 How to read this chart"):
        st.markdown("""
        - **Feature at the top** = biggest impact on your scores
        - **Red dot, right of zero** → high stat value in that round *and* it hurt you
        - **Blue dot, left of zero** → low stat value *and* it helped you
        - **Tight cluster near zero** → that stat barely moved your score
        - **Wide spread** → that stat is highly variable and influential
        """)

    if show_advanced:
        sec("Per-Round SHAP Values")
        sv_df = pd.DataFrame(shap_values, columns=FEAT_COLS)
        sv_df.insert(0, "Round", df["Round"].values)
        sv_df.insert(1, "Score", df["Score"].values)
        st.dataframe(sv_df.style.background_gradient(cmap="RdBu_r", subset=FEAT_COLS),
                     width="stretch", height=320, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 · RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    DRILLS = {
        "GIR%": {
            "why": "GIR% is the strongest predictor of your score. "
                   "Every green hit = a birdie chance + one less scramble attempt. "
                   "A 10-point GIR% gain is worth ~1.5 strokes per round.",
            "drills": [
                "**Distance ladder** — hit 10 balls each to 100 / 125 / 150 / 175 yd targets; log carries",
                "**Dispersion chart** — mark every iron's landing zone for 2 weeks to find your miss pattern",
                "**Miss-club audit** — which club do you miss most GIRs with? Target that club specifically",
                "**Gap wedge mastery** — most GIR misses are 80–130 yd; dial in 3 exact distances",
                "**Simulator approach session** — focus on carry distance and approach angle variety",
            ],
        },
        "Putts": {
            "why": "Every extra putt goes straight on the scorecard. "
                   "Lag putting and eliminating 3-putts are the fastest improvements you can make.",
            "drills": [
                "**Lag gate drill** — putt from 30–40 ft; ball must stop within 2 ft past the hole",
                "**6-foot circle** — make 10 in a row walking around the cup; do this daily",
                "**Backstroke length control** — mark tape on putter grip to control distance",
                "**3-putt log** — record every 3-putt and its distance; review trends weekly",
                "**Green reading reps** — read 20 putts per session using AimPoint or plumb-bob",
            ],
        },
        "Up&Down%": {
            "why": "Scrambling turns potential bogeys into pars when you miss a green. "
                   "A 10-point improvement in Up&Down% saves roughly 0.6 strokes per round.",
            "drills": [
                "**Multi-target chipping** — chip to 5 different holes from the same lie; 2 makes each",
                "**Bunker variety** — set up 5 different lies (buried, upslope, downslope, plugged, etc.)",
                "**Up-and-down challenge** — play 9-hole short game vs. par from off the green",
                "**Landing zone pitching** — mark 5/10/15 yd carry spots on green; hit each repeatedly",
                "**Flop vs. bump-and-run** — practice both shots to the same hole from 20 yd",
            ],
        },
        "FIR%": {
            "why": "Fairways give better lies, better angles, and easier approaches. "
                   "FIR% has a cascading effect — missing one tee shot often costs two shots total.",
            "drills": [
                "**Gate drill** — alignment sticks 20 yd apart as a fairway; hit driver through",
                "**Course management audit** — find the 3 holes you miss most and consider clubbing down",
                "**3-wood accuracy day** — sometimes 3-wood gains more net strokes than driver on tight holes",
                "**Shot shape control** — 5 draws + 5 fades per session until you own both shapes",
                "**Tee height experiment** — 3 heights per session; track which produces the tightest pattern",
            ],
        },
    }

    priority_classes = ["drill p1", "drill p2", "drill p3", "drill"]
    badge_styles = [
        f"background:{C['red_light']};color:{C['red']};",
        f"background:{C['gold_light']};color:{C['gold']};",
        f"background:{C['blue_light']};color:{C['blue']};",
        f"background:{C['green_light']};color:{C['green']};",
    ]
    badge_labels = ["#1 Priority", "#2 Priority", "#3 Priority", "#4 Priority"]

    # Gap analysis
    sec("Gap Analysis — Current vs Good-Round Target")
    if good_df.empty:
        st.warning(f"No good rounds yet (sub-{score_threshold}). Lower the threshold in the sidebar.")
    else:
        cols = st.columns(4)
        for i, feat in enumerate(importance_df["Feature"].tolist()):
            curr   = df[feat].mean()
            target = good_df[feat].mean()
            gap    = target - curr
            with cols[i]:
                st.metric(feat, f"{curr:.1f}",
                          delta=f"{gap:+.1f} to target" if abs(gap) > 0.5 else "On target ✓")

        # Progress bars
        sec("Progress to Target")
        for feat in importance_df["Feature"].tolist():
            curr   = df[feat].mean()
            target = good_df[feat].mean()
            low    = bad_df[feat].mean() if not bad_df.empty else (curr * 0.85 if feat != "Putts" else curr * 1.1)
            prog   = float(np.clip((target/curr * 100 if (feat == "Putts" and curr > 0) else
                                    (curr - low)/(target - low + .01) * 100), 0, 100))
            color  = C["green"] if prog >= 70 else C["gold"] if prog >= 40 else C["red"]
            st.markdown(
                f"**{STAT_ICONS[feat]} {feat}** — Current: **{curr:.1f}** → Target: **{target:.1f}**  "
                f"<span style='color:{color};font-weight:700;font-size:.85rem'>{prog:.0f}% there</span>",
                unsafe_allow_html=True)
            st.markdown(prog_bar(prog, color), unsafe_allow_html=True)

    sec("Practice Priorities — Ranked by SHAP Impact")
    for rank, (_, row) in enumerate(importance_df.iterrows()):
        feat = row["Feature"]
        info = DRILLS.get(feat, {})
        with st.expander(
            f"{badge_labels[rank]}: {feat}  ·  {row['SHAP']:.1f}% of score impact",
            expanded=(rank == 0),
        ):
            c1, c2 = st.columns([1, 1])
            with c1:
                bs = badge_styles[rank]
                st.markdown(f"""
                <div class="{priority_classes[rank]}">
                  <span style='display:inline-block;{bs}padding:3px 10px;border-radius:20px;
                  font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;
                  margin-bottom:12px'>{badge_labels[rank]}</span>
                  <p style='margin:0;font-size:.88rem;line-height:1.6;color:{C["text_mid"]}'>{info.get("why","")}</p>
                </div>""", unsafe_allow_html=True)

                if not good_df.empty:
                    st.markdown("")
                    m1, m2, m3 = st.columns(3)
                    with m1: st.metric("Your Avg",    f"{df[feat].mean():.1f}")
                    with m2: st.metric("Good Rounds", f"{good_df[feat].mean():.1f}")
                    with m3: st.metric("Bad Rounds",
                                       f"{bad_df[feat].mean():.1f}" if not bad_df.empty else "—")
            with c2:
                st.markdown("**Practice drills:**")
                for drill in info.get("drills", []):
                    st.markdown(f"- {drill}")

    sec("Weekly Practice Schedule")
    top1 = importance_df.iloc[0]["Feature"]
    top2 = importance_df.iloc[1]["Feature"]
    top3 = importance_df.iloc[2]["Feature"] if len(importance_df) > 2 else top1
    sched = pd.DataFrame([
        ["Monday",    f"60 min — {STAT_LABELS[top1]} drills (biggest lever)"],
        ["Tuesday",   "60 min — Full swing range session + driver accuracy"],
        ["Wednesday", f"45 min — {STAT_LABELS[top2]} drills + 9 holes on-course"],
        ["Thursday",  "Rest or easy putting green (no pressure session)"],
        ["Friday",    f"60 min — {STAT_LABELS[top3]} drills + chipping game"],
        ["Saturday",  "18-hole round — track all 4 stats carefully"],
        ["Sunday",    "10 min — upload round data, review trends, update plan"],
    ], columns=["Day", "Focus"])
    st.dataframe(sched, width="stretch", hide_index=True)

    sec("Download Your Report")
    good_targets = {f: good_df[f].mean() for f in FEAT_COLS} if not good_df.empty else {}
    lines = [
        "GOLF PERFORMANCE ANALYTICS — REPORT", "=" * 44, "",
        f"Rounds:          {n}",
        f"Avg Score:       {df['Score'].mean():.1f}",
        f"Best:            {int(df['Score'].min())}",
        f"Handicap est.:   {handicap}",
        f"Good round rate: {good_pct:.0f}% (sub-{score_threshold})",
        "", "STAT AVERAGES", "-" * 28,
    ] + [f"{f:12s}: {df[f].mean():.1f}" for f in FEAT_COLS] + [
        "", "FEATURE IMPORTANCE (SHAP)", "-" * 28,
    ] + [f"#{i+1} {r['Feature']:12s}: {r['SHAP']:.1f}% impact"
         for i, (_, r) in enumerate(importance_df.iterrows())] + [
        "", "GOOD-ROUND TARGETS", "-" * 28,
    ] + ([f"{f:12s}: {good_targets[f]:.1f}" for f in FEAT_COLS]
         if good_targets else ["Not enough good rounds yet."]) + [
        "", "TOP 3 PRIORITIES", "-" * 28,
    ] + [f"#{i+1} {r['Feature']}: {STAT_LABELS[r['Feature']]}"
         for i, (_, r) in enumerate(importance_df.head(3).iterrows())]

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📄 Download Report (.txt)", data="\n".join(lines),
                           file_name="golf_analysis_report.txt",
                           mime="text/plain", width="stretch")
    with c2:
        st.download_button("📊 Export Data (.csv)", data=df.to_csv(index=False),
                           file_name="golf_data_export.csv",
                           mime="text/csv", width="stretch")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:{C['gray']};font-size:.82rem;padding:8px 0;'>"
    "⛳ Golf Performance Analytics — built for varsity season prep"
    "</div>",
    unsafe_allow_html=True,
)
