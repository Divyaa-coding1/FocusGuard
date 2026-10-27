import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

from vision_engine import (
    FocusVisionProcessor,
    get_latest_vision_measurement,
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# =========================================================
# PATHS
# =========================================================

V3_DIR = Path(__file__).resolve().parent
PROJECT_DIR = V3_DIR.parent

# The existing camera_engine.py writes here.
LIVE_DATA_FILE = PROJECT_DIR / "data.json"

# V3 keeps its own data separate from the old application.
V3_DATA_DIR = V3_DIR / "data"
V3_DATA_DIR.mkdir(exist_ok=True)

HISTORY_FILE = V3_DATA_DIR / "history.json"
SETTINGS_FILE = V3_DATA_DIR / "settings.json"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FocusGuard V3",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# DESIGN SYSTEM
# =========================================================

BLUE = "#1E88E5"
DEEP_BLUE = "#0B1F3A"
GOLD = "#FBC02D"
LIGHT_BLUE = "#EAF4FF"
GREEN = "#16A34A"
RED = "#DC2626"
ORANGE = "#F97316"
TEXT = "#10213A"
MUTED = "#64748B"


st.markdown(
    f"""
    <style>
        .stApp {{
            background:
                radial-gradient(circle at 10% 5%, rgba(30,136,229,.13), transparent 27%),
                radial-gradient(circle at 90% 5%, rgba(251,192,45,.12), transparent 24%),
                #F7FAFE;
        }}

        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 4rem;
            max-width: 1450px;
        }}

        #MainMenu, footer, header {{
            visibility: hidden;
        }}

        .fg-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 20px 25px;
            background: linear-gradient(120deg, #FFFFFF, {LIGHT_BLUE});
            border: 1px solid #DCE9F8;
            border-radius: 22px;
            box-shadow: 0 16px 38px rgba(11,31,58,.08);
            color: #000000;
            margin-bottom: 18px;
        }}

        .fg-brand {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}

        .fg-logo {{
            width: 54px;
            height: 54px;
            border-radius: 16px;
            background: {LIGHT_BLUE};
            border: 1px solid #DCE9F8;
            display: grid;
            place-items: center;
            font-size: 28px;
        }}

        .fg-title {{
            font-size: 28px;
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: -.5px;
        }}

        .fg-subtitle {{
            opacity: 1;
            color: {GOLD};
            font-size: 13px;
            margin-top: 5px;
        }}

        .fg-badge {{
            background: {GOLD};
            color: #000000;
            padding: 8px 14px;
            border-radius: 999px;
            font-weight: 800;
            font-size: 12px;
            white-space: nowrap;
        }}

        .fg-card {{
            background: rgba(255,255,255,.92);
            border: 1px solid #DCE9F8;
            border-radius: 19px;
            box-shadow: 0 9px 25px rgba(24,73,120,.08);
            padding: 18px;
            margin-bottom: 14px;
        }}

        .fg-hero {{
            background: linear-gradient(135deg, #FFFFFF 0%, #EAF4FF 68%, #FFF8D9 100%);
            border: 1px solid #D8E7F8;
            border-radius: 26px;
            padding: 38px;
            box-shadow: 0 18px 45px rgba(17,71,123,.11);
            margin-top: 12px;
            margin-bottom: 20px;
        }}

        .fg-eyebrow {{
            color: {GOLD};
            font-weight: 800;
            font-size: 13px;
            letter-spacing: 1.4px;
            text-transform: uppercase;
        }}

        .fg-hero-title {{
            color: {BLUE};
            font-size: 42px;
            font-weight: 850;
            line-height: 1.08;
            max-width: 770px;
            margin: 12px 0;
        }}

        .fg-copy {{
            color: #000000;
            font-size: 17px;
            line-height: 1.65;
            max-width: 780px;
        }}

        .fg-metric-label {{
            color: {GOLD};
            font-size: 12px;
            font-weight: 750;
            letter-spacing: .8px;
            text-transform: uppercase;
        }}

        .fg-metric-value {{
            color: {BLUE};
            font-size: 28px;
            font-weight: 850;
            margin-top: 3px;
        }}

        .fg-status-focused {{
            background: #DCFCE7;
            border: 1px solid #86EFAC;
            color: {DEEP_BLUE};
            border-radius: 16px;
            padding: 14px 16px;
            font-weight: 850;
            font-size: 20px;
        }}

        .fg-status-distracted {{
            background: #FEE2E2;
            border: 1px solid #FCA5A5;
            color: #000000;
            border-radius: 16px;
            padding: 14px 16px;
            font-weight: 850;
            font-size: 20px;
        }}

        .fg-status-over {{
            background: #FEF3C7;
            border: 1px solid #FCD34D;
            color: #000000;
            border-radius: 16px;
            padding: 14px 16px;
            font-weight: 850;
            font-size: 20px;
        }}

        .fg-status-noface {{
            background: #F1F5F9;
            border: 1px solid #CBD5E1;
            color: #475569;
            border-radius: 16px;
            padding: 14px 16px;
            font-weight: 850;
            font-size: 20px;
        }}

        .fg-coach {{
            background: linear-gradient(135deg, #FFF8D3, #FFFFFF);
            border-left: 6px solid {GOLD};
            border-radius: 16px;
            padding: 17px 18px;
            color: #000000;
            box-shadow: 0 7px 20px rgba(137,104,0,.08);
            line-height: 1.55;
        }}

        .fg-blue-box {{
            background: linear-gradient(135deg, #EAF4FF, #FFFFFF);
            border-left: 6px solid {BLUE};
            border-radius: 16px;
            padding: 17px 18px;
            color: #000000;
            box-shadow: 0 7px 20px rgba(30,136,229,.08);
            line-height: 1.55;
        }}

        .fg-empty {{
            border: 2px dashed #BED4EA;
            border-radius: 18px;
            background: rgba(255,255,255,.68);
            color: #000000;
            text-align: center;
            padding: 35px 18px;
        }}

        .stButton > button {{
            border-radius: 13px;
            min-height: 44px;
            font-weight: 750;
            border: 1px solid #CFE0F2;
            box-shadow: 0 4px 13px rgba(16,54,89,.06);
        }}

        .stButton > button[kind="primary"] {{
            background: linear-gradient(110deg, {BLUE}, #1667C8);
            color: white;
            border: none;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,.92);
            border: 1px solid #DCE9F8;
            border-radius: 16px;
            padding: 14px;
            box-shadow: 0 7px 19px rgba(24,73,120,.06);
        }}

        div[data-testid="stPlotlyChart"] {{
            background: white;
            border: 1px solid #E2EAF4;
            border-radius: 18px;
            padding: 5px;
            box-shadow: 0 8px 22px rgba(20,73,120,.06);
        }}

        @media (max-width: 700px) {{
            .fg-header {{ align-items: flex-start; flex-direction: column; }}
            .fg-hero {{ padding: 24px; }}
            .fg-hero-title {{ font-size: 31px; }}
        }}

        /* ── Global Streamlit text overrides ── */
        .stApp h1, .stApp h2, .stApp h3 {{
            color: {BLUE} !important;
        }}

        .stApp p, .stApp li, .stApp label {{
            color: #000000 !important;
        }}

        .stApp span {{
            color: #000000 !important;
        }}

        /* Keep header text on-brand */
        .fg-header, .fg-header * {{
            color: #000000 !important;
        }}
        .fg-header .fg-title {{
            color: {BLUE} !important;
        }}
        .fg-subtitle {{
            color: {GOLD} !important;
        }}

        /* Secondary buttons: blue text on white */
        .stButton > button:not([kind="primary"]) {{
            background-color: #FFFFFF !important;
            color: {BLUE} !important;
            border: 1px solid {BLUE} !important;
        }}
        .stButton > button:not([kind="primary"]) span,
        .stButton > button:not([kind="primary"]) p {{
            color: {BLUE} !important;
        }}

        /* Primary buttons stay white text on blue */
        .stButton > button[kind="primary"] {{
            color: white !important;
        }}
        .stButton > button[kind="primary"] span,
        .stButton > button[kind="primary"] p {{
            color: white !important;
        }}

        /* Metric labels → gold, metric values → blue */
        div[data-testid="stMetric"] label {{
            color: {GOLD} !important;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {BLUE} !important;
        }}

        /* Streamlit widget labels */
        .stSelectbox label, .stSlider label,
        .stTextInput label, .stNumberInput label,
        .stCheckbox label, .stRadio label,
        .stMultiSelect label, .stTextArea label {{
            color: #000000 !important;
        }}

        /* Plotly chart titles */
        div[data-testid="stPlotlyChart"] text {{
            fill: #000000 !important;
        }}

        /* ── Force light backgrounds on all Streamlit containers ── */
        .stApp {{
            color-scheme: light !important;
        }}

        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        [data-testid="column"],
        [data-testid="stExpander"],
        [data-testid="stExpanderDetails"],
        [data-testid="stForm"],
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {{
            background-color: transparent !important;
        }}

        /* Streamlit widget input backgrounds */
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea,
        [data-testid="stDateInput"] > div > div > input {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }}

        /* Selectbox dropdown menu */
        [data-testid="stSelectboxVirtualDropdown"],
        ul[role="listbox"],
        ul[role="listbox"] li {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }}

        /* Expander header */
        [data-testid="stExpander"] summary {{
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }}

        /* Streamlit slider */
        .stSlider > div {{
            color: #000000 !important;
        }}

        /* Table / dataframe containers */
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {{
            background-color: #FFFFFF !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SAFE STORAGE
# =========================================================

DEFAULT_SETTINGS = {
    "blink_threshold": 25,
    "yaw_threshold": 25,
    "pitch_threshold": 25,
    "minimum_distance_cm": 45,
    "recommended_focus_minutes": 25,
    "recommended_break_minutes": 5,
    "custom_message": "Return gently to the task in front of you.",
}


def atomic_write_json(path: Path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def safe_read_json(path: Path, default):
    if not path.exists():
        return default

    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return default
        return json.loads(text)
    except (json.JSONDecodeError, OSError, TypeError):
        return default


if not SETTINGS_FILE.exists():
    atomic_write_json(SETTINGS_FILE, DEFAULT_SETTINGS)

if not HISTORY_FILE.exists():
    atomic_write_json(HISTORY_FILE, [])


def load_settings():
    saved = safe_read_json(SETTINGS_FILE, {})
    return {**DEFAULT_SETTINGS, **saved}


def load_history():
    history = safe_read_json(HISTORY_FILE, [])
    return history if isinstance(history, list) else []


def read_live_data():
    # Priority 1: live in-memory data from embedded vision engine
    try:
        vision_data = get_latest_vision_measurement()
        now_ms = int(time.time() * 1000)
        # Check if vision measurement is fresh (within 3 seconds)
        if vision_data.get("timestamp_ms", 0) > 0 and (now_ms - vision_data["timestamp_ms"]) < 3000:
            return {
                "ear": vision_data.get("ear"),
                "distance_cm": vision_data.get("distance_cm"),
                "blink_rate": float(vision_data.get("blink_rate", 0) or 0),
                "yaw": float(vision_data.get("yaw", 0) or 0),
                "pitch": float(vision_data.get("pitch", 0) or 0),
                "attention": vision_data.get("attention", "No Face / Calibrating"),
                "focus_score": float(vision_data.get("focus_score", 0.0) or 0.0),
                "face_detected": bool(vision_data.get("face_detected", False)),
                "timestamp_ms": int(vision_data.get("timestamp_ms", 0) or 0),
            }
    except Exception:
        pass

    # Priority 2: read from data.json (atomic compatibility output)
    data = safe_read_json(LIVE_DATA_FILE, {})
    if not data:
        data = safe_read_json(V3_DIR / "data.json", {})

    return {
        "ear": data.get("ear"),
        "distance_cm": data.get("distance_cm"),
        "blink_rate": float(data.get("blink_rate", 0) or 0),
        "yaw": float(data.get("yaw", 0) or 0),
        "pitch": float(data.get("pitch", 0) or 0),
        "attention": data.get("attention", "No Face / Calibrating"),
        "focus_score": float(data.get("focus_score", 0.0) or 0.0),
        "face_detected": bool(data.get("face_detected", False)),
        "timestamp_ms": int(data.get("timestamp_ms", 0) or 0),
    }


# =========================================================
# SESSION STATE
# =========================================================

STATE_DEFAULTS = {
    "page": "Home",
    "session_status": "idle",
    "session_started_at": None,
    "paused_started_at": None,
    "paused_seconds": 0.0,
    "samples": [],
    "last_timestamp_ms": 0,
    "latest_report": None,
    "streak_started_at": None,
}

for key, value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HELPERS
# =========================================================

def calculate_focus_score(data, settings):
    # If vision engine or data.json already provided a calibrated focus score
    if "focus_score" in data and data["focus_score"] is not None and data.get("face_detected", False):
        return round(float(data["focus_score"]), 3)

    if not data.get("face_detected", True) or data.get("ear") is None:
        return 0.0

    ear = float(data["ear"])
    eye_comp = float(np.clip((ear - 0.19) / 0.09, 0.0, 1.0))

    blink_threshold = max(10, settings.get("blink_threshold", 25))
    blink_rate = float(data.get("blink_rate", 0))
    if blink_rate <= 22:
        blink_comp = 1.0
    else:
        blink_comp = float(np.clip(1.0 - (blink_rate - 22) / float(blink_threshold), 0.0, 1.0))

    yaw_threshold = max(10, settings.get("yaw_threshold", 25))
    pitch_threshold = max(10, settings.get("pitch_threshold", 25))
    pose_penalty = min(
        1.0,
        (abs(data.get("yaw", 0)) / yaw_threshold + abs(data.get("pitch", 0)) / pitch_threshold) / 2.0,
    )
    pose_comp = float(np.clip(1.0 - pose_penalty, 0.0, 1.0))

    distance = data.get("distance_cm")
    min_dist = max(25, settings.get("minimum_distance_cm", 45))
    if distance is None:
        dist_comp = 0.8
    elif distance >= min_dist:
        dist_comp = 1.0
    else:
        dist_ratio = distance / float(min_dist)
        dist_comp = float(np.clip(dist_ratio ** 1.6, 0.0, 1.0))

    score = (
        0.35 * eye_comp
        + 0.30 * pose_comp
        + 0.20 * dist_comp
        + 0.15 * blink_comp
    )
    return round(float(np.clip(score, 0.0, 1.0)), 3)


def normalize_attention(data, settings):
    # Direct smoothed state from vision engine or data.json
    raw_state = str(data.get("attention", "")).strip()
    if raw_state in {"Focused", "Distracted", "Over-engaged", "No Face / Calibrating"}:
        return raw_state

    if not data.get("face_detected", True) or data.get("ear") is None:
        return "No Face / Calibrating"

    score = calculate_focus_score(data, settings)
    yaw_thresh = settings.get("yaw_threshold", 25)
    pitch_thresh = settings.get("pitch_threshold", 25)
    min_dist = settings.get("minimum_distance_cm", 45)
    blink_thresh = settings.get("blink_threshold", 25)

    if (
        abs(data.get("yaw", 0)) > yaw_thresh
        or abs(data.get("pitch", 0)) > pitch_thresh
    ):
        return "Distracted"

    if (
        (data.get("distance_cm") is not None and data["distance_cm"] < min_dist)
        or data.get("blink_rate", 0) > blink_thresh
        or score < 0.35
    ):
        return "Over-engaged"

    if score >= 0.65:
        return "Focused"

    return "Distracted"


def active_duration_seconds():
    if not st.session_state.session_started_at:
        return 0.0

    end = time.time()

    current_pause = 0.0
    if (
        st.session_state.session_status == "paused"
        and st.session_state.paused_started_at
    ):
        current_pause = end - st.session_state.paused_started_at

    return max(
        0.0,
        end
        - st.session_state.session_started_at
        - st.session_state.paused_seconds
        - current_pause,
    )


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def ai_message(state, score, data, settings):
    focused_minutes = settings.get("recommended_focus_minutes", 25)

    if state == "Focused":
        return (
            f"Your attention pattern is stable and your current focus score is "
            f"{score * 100:.0f}%. Continue the current work block, but plan a "
            f"brief recovery break near the {focused_minutes}-minute mark."
        )

    if state == "Distracted":
        return (
            "FocusGuard detected an attention shift or head-position change. "
            "Recenter your gaze, relax your shoulders, and bring your focus back to the task."
        )

    if state == "Over-engaged":
        return (
            "Your current pattern suggests visual fatigue or unusually close "
            "screen engagement. Lean back to a comfortable distance, blink naturally, "
            "and give your eyes a brief rest."
        )

    return (
        "FocusGuard is waiting for stable face data. Keep your face centered, "
        "well-lit, and visible to the camera."
    )


def status_html(state):
    if state == "Focused":
        css_class = "fg-status-focused"
        icon = "●"
    elif state == "Distracted":
        css_class = "fg-status-distracted"
        icon = "◆"
    elif state == "Over-engaged":
        css_class = "fg-status-over"
        icon = "▲"
    else:
        css_class = "fg-status-noface"
        icon = "○"

    return f'<div class="{css_class}">{icon} {state}</div>'


def start_session():
    st.session_state.session_status = "running"
    st.session_state.session_started_at = time.time()
    st.session_state.paused_started_at = None
    st.session_state.paused_seconds = 0.0
    st.session_state.samples = []
    st.session_state.last_timestamp_ms = 0
    st.session_state.latest_report = None
    st.session_state.streak_started_at = None
    st.session_state.page = "Session"


def pause_session():
    if st.session_state.session_status == "running":
        st.session_state.session_status = "paused"
        st.session_state.paused_started_at = time.time()


def resume_session():
    if st.session_state.session_status == "paused":
        if st.session_state.paused_started_at:
            st.session_state.paused_seconds += (
                time.time() - st.session_state.paused_started_at
            )
        st.session_state.paused_started_at = None
        st.session_state.session_status = "running"


def create_report():
    samples = st.session_state.samples

    if not samples:
        return {
            "id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": active_duration_seconds(),
            "average_focus": 0,
            "focused_percent": 0,
            "distracted_percent": 0,
            "over_engaged_percent": 0,
            "average_distance_cm": None,
            "longest_focus_streak_seconds": 0,
            "sample_count": 0,
            "recommendation": "Run a longer session to generate a detailed review.",
            "samples": [],
        }

    total = len(samples)
    counts = Counter(sample["state"] for sample in samples)

    average_focus = sum(sample["focus_score"] for sample in samples) / total

    distances = [
        sample["distance_cm"]
        for sample in samples
        if sample["distance_cm"] is not None
    ]

    longest_streak = 0.0
    streak_start = None

    for sample in samples:
        if sample["state"] == "Focused":
            if streak_start is None:
                streak_start = sample["captured_at"]
        elif streak_start is not None:
            longest_streak = max(
                longest_streak,
                sample["captured_at"] - streak_start,
            )
            streak_start = None

    if streak_start is not None:
        longest_streak = max(
            longest_streak,
            samples[-1]["captured_at"] - streak_start,
        )

    focused_percent = 100 * counts.get("Focused", 0) / total
    distracted_percent = 100 * counts.get("Distracted", 0) / total
    over_percent = 100 * counts.get("Over-engaged", 0) / total

    if average_focus >= 0.75 and focused_percent >= 70:
        recommendation = (
            "This was a strong session with consistent attention. Take a short "
            "break now, then reuse the same environment and work rhythm for the "
            "next focus block."
        )
    elif distracted_percent > over_percent:
        recommendation = (
            "Attention shifts were the main interruption pattern. For the next "
            "session, reduce nearby distractions and begin with one clearly "
            "defined task."
        )
    else:
        recommendation = (
            "Visual fatigue or close screen engagement affected this session. "
            "Take an eye break, reset posture, and increase viewing distance "
            "before the next focus block."
        )

    return {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(active_duration_seconds(), 2),
        "average_focus": round(average_focus, 3),
        "focused_percent": round(focused_percent, 1),
        "distracted_percent": round(distracted_percent, 1),
        "over_engaged_percent": round(over_percent, 1),
        "average_distance_cm": (
            round(sum(distances) / len(distances), 1) if distances else None
        ),
        "longest_focus_streak_seconds": round(longest_streak, 2),
        "sample_count": total,
        "recommendation": recommendation,
        "samples": samples,
    }


def stop_session():
    if st.session_state.session_status == "paused":
        resume_session()

    report = create_report()
    history = load_history()
    history.append(report)
    atomic_write_json(HISTORY_FILE, history)

    st.session_state.latest_report = report
    st.session_state.session_status = "stopped"
    st.session_state.page = "Summary"


# =========================================================
# SHARED HEADER AND NAVIGATION
# =========================================================

st.markdown(
    """
    <div class="fg-header">
        <div class="fg-brand">
            <div class="fg-logo">🧠🛡️</div>
            <div>
                <div class="fg-title">FocusGuard</div>
                <div class="fg-subtitle">
                    Real-time computer vision for healthier focus habits
                </div>
            </div>
        </div>
        <div class="fg-badge">AI FOCUS INTELLIGENCE</div>
    </div>
    """,
    unsafe_allow_html=True,
)


NAV_ITEMS = [
    ("Home", "🏠"),
    ("Session", "🎯"),
    ("Analytics", "📊"),
    ("History", "📚"),
    ("Settings", "⚙️"),
    ("About", "ℹ️"),
]

nav_columns = st.columns(len(NAV_ITEMS))

for column, (name, icon) in zip(nav_columns, NAV_ITEMS):
    with column:
        if st.button(
            f"{icon} {name}",
            use_container_width=True,
            type="primary" if st.session_state.page == name else "secondary",
            key=f"nav_{name}",
        ):
            st.session_state.page = name
            st.rerun()


# =========================================================
# HOME
# =========================================================

def render_home():
    st.markdown(
        """
        <div class="fg-hero">
            <div class="fg-eyebrow">Student-focused digital wellness</div>
            <div class="fg-hero-title">
                Understand your attention. Build healthier focus.
            </div>
            <div class="fg-copy">
                FocusGuard uses real-time facial landmarks, eye behavior, head
                position, and viewing distance to help students recognize
                distraction and visual fatigue during study sessions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown("### Ready for a focused work block?")

        if st.button(
            "▶ Start Focus Session",
            type="primary",
            use_container_width=True,
            key="home_start",
        ):
            start_session()
            st.rerun()

        quick1, quick2 = st.columns(2)

        with quick1:
            if st.button(
                "📊 View Analytics",
                use_container_width=True,
                key="home_analytics",
            ):
                st.session_state.page = "Analytics"
                st.rerun()

        with quick2:
            if st.button(
                "📚 Session History",
                use_container_width=True,
                key="home_history",
            ):
                st.session_state.page = "History"
                st.rerun()

    with right:
        history = load_history()
        completed = len(history)

        if history:
            latest = history[-1]
            average = latest.get("average_focus", 0) * 100
            last_duration = format_duration(
                latest.get("duration_seconds", 0)
            )
        else:
            average = 0
            last_duration = "No sessions"

        metric1, metric2 = st.columns(2)
        metric1.metric("Completed Sessions", completed)
        metric2.metric("Last Session Focus", f"{average:.0f}%")

        st.markdown(
            f"""
            <div class="fg-blue-box">
                <strong>Latest activity</strong><br>
                Last session duration: {last_duration}<br>
                Camera engine: {"Data detected" if LIVE_DATA_FILE.exists() else "Waiting for data"}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### How FocusGuard helps")

    feature1, feature2, feature3 = st.columns(3)

    with feature1:
        st.markdown(
            """
            <div class="fg-card">
                <div style="font-size:28px">👁️</div>
                <h4>Visual Attention</h4>
                <p>Monitors eye openness, blinking patterns, and head position.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feature2:
        st.markdown(
            """
            <div class="fg-card">
                <div style="font-size:28px">📏</div>
                <h4>Healthy Distance</h4>
                <p>Detects unusually close screen engagement and posture drift.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with feature3:
        st.markdown(
            """
            <div class="fg-card">
                <div style="font-size:28px">🧠</div>
                <h4>AI Coaching</h4>
                <p>Transforms live measurements into clear, useful guidance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# LIVE SESSION
# =========================================================

def render_session():
    settings = load_settings()

    if st.session_state.session_status in {"idle", "stopped"}:
        st.markdown(
            """
            <div class="fg-empty">
                <h3>No active focus session</h3>
                <p>Start a session to collect live analytics and coaching.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "▶ Start Session",
            type="primary",
            use_container_width=True,
            key="session_start_empty",
        ):
            start_session()
            st.rerun()
        return

    controls = st.columns([1, 1, 1, 2])

    if st.session_state.session_status == "running":
        with controls[0]:
            if st.button(
                "⏸ Pause",
                use_container_width=True,
                key="session_pause",
            ):
                pause_session()
                st.rerun()
    else:
        with controls[0]:
            if st.button(
                "▶ Resume",
                type="primary",
                use_container_width=True,
                key="session_resume",
            ):
                resume_session()
                st.rerun()

    with controls[1]:
        if st.button(
            "⏹ End Session",
            use_container_width=True,
            key="session_stop",
        ):
            stop_session()
            st.rerun()

    with controls[2]:
        if st.button(
            "↻ Reset",
            use_container_width=True,
            key="session_reset",
        ):
            start_session()
            st.rerun()

    session_status = st.session_state.session_status.upper()

    with controls[3]:
        st.info(
            f"Session status: {session_status} • "
            f"Active time: {format_duration(active_duration_seconds())}"
        )

    live_panel(settings)


def create_focus_gauge(score: float, state: str) -> go.Figure:
    score_pct = int(round(score * 100))
    if score_pct >= 65:
        bar_color = GREEN
    elif score_pct >= 35:
        bar_color = GOLD
    else:
        bar_color = RED

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score_pct,
            number={
                "suffix": "%",
                "font": {"size": 36, "color": BLUE},
            },
            title={
                "text": f"<b>FOCUS SCORE</b><br><span style='font-size:14px;color:{bar_color};font-weight:700'>{state}</span>",
                "font": {"size": 13, "color": MUTED},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#94A3B8",
                    "tickvals": [0, 35, 65, 100],
                    "tickfont": {"size": 10},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "steps": [
                    {"range": [0, 34], "color": "#FEE2E2"},   # red/orange
                    {"range": [34, 64], "color": "#FEF3C7"},  # gold
                    {"range": [64, 100], "color": "#DCFCE7"}, # green/blue
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 4},
                    "thickness": 0.75,
                    "value": score_pct,
                },
            },
        )
    )

    gauge.update_layout(
        height=230,
        margin=dict(l=15, r=15, t=40, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT},
    )
    return gauge


if hasattr(st, "fragment"):

    @st.fragment(run_every=0.25)
    def live_panel(settings):
        render_live_panel_contents(settings)

else:

    def live_panel(settings):
        st.warning(
            "Automatic live refresh requires Streamlit 1.37 or newer. "
            "Use the refresh button below or upgrade Streamlit."
        )
        if st.button("Refresh live data", key="manual_live_refresh"):
            st.rerun()
        render_live_panel_contents(settings)


def render_live_panel_contents(settings):
    data = read_live_data()
    state = normalize_attention(data, settings)
    score = calculate_focus_score(data, settings)

    # When running, record valid session samples and track focus streak
    if (
        st.session_state.session_status == "running"
        and data["timestamp_ms"] > st.session_state.last_timestamp_ms
    ):
        sample = {
            "captured_at": time.time(),
            "timestamp_ms": data["timestamp_ms"],
            "focus_score": score,
            "state": state,
            "ear": data["ear"],
            "blink_rate": data["blink_rate"],
            "distance_cm": data["distance_cm"],
            "yaw": data["yaw"],
            "pitch": data["pitch"],
        }

        st.session_state.samples.append(sample)
        st.session_state.last_timestamp_ms = data["timestamp_ms"]

        if state == "Focused":
            if st.session_state.streak_started_at is None:
                st.session_state.streak_started_at = time.time()
        else:
            st.session_state.streak_started_at = None

    # Top layout: LEFT = embedded live camera, RIGHT = compact gauge + state card + coach
    top_left, top_right = st.columns([1.15, 1.0])

    with top_left:
        st.markdown(
            """
            <div style="display:inline-flex; align-items:center; gap:6px; background:#FEE2E2; border:1px solid #FCA5A5; color:#DC2626; border-radius:999px; padding:3px 10px; font-size:11px; font-weight:800; margin-bottom:8px;">
                <span style="width:7px; height:7px; background:#DC2626; border-radius:50%; display:inline-block;"></span>
                LIVE CAMERA FEED
            </div>
            """,
            unsafe_allow_html=True,
        )

        webrtc_ctx = webrtc_streamer(
            key="focusguard_cam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=FocusVisionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.update_settings(settings)

    with top_right:
        sub_col1, sub_col2 = st.columns([1.1, 0.9])
        with sub_col1:
            gauge = create_focus_gauge(score, state)
            st.plotly_chart(
                gauge,
                width="stretch",
                config={"displayModeBar": False},
            )

        with sub_col2:
            st.markdown("### Attention State")
            st.markdown(status_html(state), unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="fg-coach" style="margin-top: 6px;">
                <strong>✨ FocusGuard Coach</strong><br><br>
                {ai_message(state, score, data, settings)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Below: Blink Rate, Approx. Distance, Head Position, Focus Streak
    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "👁 Blink Rate",
        f"{data['blink_rate']:.0f}/min",
    )
    metric2.metric(
        "📏 Approx. Distance",
        (
            f"{data['distance_cm']:.1f} cm"
            if data.get("distance_cm") is not None
            else "No face"
        ),
    )

    yaw = data.get("yaw", 0.0)
    pitch = data.get("pitch", 0.0)
    if not data.get("face_detected", False):
        head_pos = "No Face"
    elif abs(yaw) <= 12 and abs(pitch) <= 12:
        head_pos = "Centered"
    elif abs(yaw) > 12:
        head_pos = f"{'Turned Left' if yaw < 0 else 'Turned Right'} ({abs(yaw):.0f}°)"
    else:
        head_pos = f"{'Looking Down' if pitch > 0 else 'Looking Up'} ({abs(pitch):.0f}°)"

    metric3.metric(
        "🧭 Head Position",
        head_pos,
    )

    current_streak = 0.0
    if state == "Focused" and st.session_state.streak_started_at:
        current_streak = time.time() - st.session_state.streak_started_at

    metric4.metric(
        "🔥 Focus Streak",
        format_duration(current_streak),
    )

    samples = st.session_state.samples

    if samples:
        frame = pd.DataFrame(samples)
        frame["elapsed_seconds"] = (
            frame["captured_at"] - frame["captured_at"].iloc[0]
        )

        recent = frame.tail(100)

        chart1, chart2 = st.columns(2)

        with chart1:
            focus_chart = px.line(
                recent,
                x="elapsed_seconds",
                y="focus_score",
                title="Focus Score Over Time",
                markers=False,
            )
            focus_chart.update_traces(line_color=BLUE, line_width=3)
            focus_chart.update_yaxes(range=[0, 1])
            focus_chart.update_layout(
                height=260,
                xaxis_title="Session seconds",
                yaxis_title="Focus",
                margin=dict(l=25, r=20, t=45, b=25),
            )
            st.plotly_chart(
                focus_chart,
                width="stretch",
                config={"displayModeBar": False},
            )

        with chart2:
            valid_dist = recent.dropna(subset=["distance_cm"])
            if not valid_dist.empty:
                distance_chart = px.line(
                    valid_dist,
                    x="elapsed_seconds",
                    y="distance_cm",
                    title="Viewing Distance Trend",
                )
                distance_chart.update_traces(line_color=GOLD, line_width=3)
                distance_chart.update_layout(
                    height=260,
                    xaxis_title="Session seconds",
                    yaxis_title="Distance (cm)",
                    margin=dict(l=25, r=20, t=45, b=25),
                )
                st.plotly_chart(
                    distance_chart,
                    width="stretch",
                    config={"displayModeBar": False},
                )
            else:
                st.info("Collecting distance trend samples...")
    else:
        st.markdown(
            """
            <div class="fg-empty">
                Live graphs will begin when camera frames are detected during an active session.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# ANALYTICS
# =========================================================

def render_analytics():
    samples = st.session_state.samples

    if not samples:
        history = load_history()
        if history:
            samples = history[-1].get("samples", [])

    if not samples:
        st.markdown(
            """
            <div class="fg-empty">
                <h3>No analytics yet</h3>
                <p>Complete a session to generate focus analytics.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    frame = pd.DataFrame(samples)

    if "captured_at" in frame.columns:
        frame["elapsed_seconds"] = (
            frame["captured_at"] - frame["captured_at"].iloc[0]
        )
    else:
        frame["elapsed_seconds"] = range(len(frame))

    average_focus = frame["focus_score"].mean()
    counts = frame["state"].value_counts()

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Average Focus", f"{average_focus * 100:.0f}%")
    metric2.metric("Focused Samples", int(counts.get("Focused", 0)))
    metric3.metric("Distracted Samples", int(counts.get("Distracted", 0)))
    metric4.metric(
        "Over-engaged Samples",
        int(counts.get("Over-engaged", 0)),
    )

    chart1, chart2 = st.columns(2)

    with chart1:
        focus_chart = px.area(
            frame,
            x="elapsed_seconds",
            y="focus_score",
            title="Focus Performance",
        )
        focus_chart.update_traces(line_color=BLUE, fillcolor="#CFE8FF")
        focus_chart.update_yaxes(range=[0, 1])
        focus_chart.update_layout(height=360)
        st.plotly_chart(
            focus_chart,
            width="stretch",
            config={"displayModeBar": False},
        )

    with chart2:
        distribution = (
            frame["state"]
            .value_counts()
            .rename_axis("state")
            .reset_index(name="count")
        )

        pie = px.pie(
            distribution,
            names="state",
            values="count",
            title="Attention Distribution",
            hole=0.62,
            color="state",
            color_discrete_map={
                "Focused": GREEN,
                "Distracted": RED,
                "Over-engaged": GOLD,
                "No Face / Calibrating": MUTED,
                "No Face": MUTED,
            },
        )
        pie.update_layout(height=360)
        st.plotly_chart(
            pie,
            width="stretch",
            config={"displayModeBar": False},
        )

    chart3, chart4 = st.columns(2)

    with chart3:
        blink_chart = px.line(
            frame,
            x="elapsed_seconds",
            y="blink_rate",
            title="Blink Activity",
        )
        blink_chart.update_traces(line_color=ORANGE, line_width=3)
        blink_chart.update_layout(height=330)
        st.plotly_chart(
            blink_chart,
            width="stretch",
            config={"displayModeBar": False},
        )

    with chart4:
        valid_distance = frame.dropna(subset=["distance_cm"])

        if not valid_distance.empty:
            distance_chart = px.line(
                valid_distance,
                x="elapsed_seconds",
                y="distance_cm",
                title="Viewing Distance",
            )
            distance_chart.update_traces(line_color=GOLD, line_width=3)
            distance_chart.update_layout(height=330)
            st.plotly_chart(
                distance_chart,
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.info("No distance samples available.")


# =========================================================
# SUMMARY
# =========================================================

def render_summary():
    report = st.session_state.latest_report

    if not report:
        st.info("There is no completed session report yet.")
        return

    st.success("Session completed and saved to history.")

    st.markdown("## 🧠 FocusGuard Session Report")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Duration",
        format_duration(report["duration_seconds"]),
    )
    metric2.metric(
        "Average Focus",
        f"{report['average_focus'] * 100:.0f}%",
    )
    metric3.metric(
        "Focused",
        f"{report['focused_percent']:.0f}%",
    )
    metric4.metric(
        "Longest Streak",
        format_duration(report["longest_focus_streak_seconds"]),
    )

    st.markdown(
        f"""
        <div class="fg-coach">
            <strong>✨ AI Session Review</strong><br><br>
            {report["recommendation"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "📊 View Full Analytics",
            type="primary",
            use_container_width=True,
            key="summary_analytics",
        ):
            st.session_state.page = "Analytics"
            st.rerun()

    with col2:
        if st.button(
            "🏠 Return Home",
            use_container_width=True,
            key="summary_home",
        ):
            st.session_state.page = "Home"
            st.rerun()


# =========================================================
# HISTORY
# =========================================================

def render_history():
    history = list(reversed(load_history()))

    st.markdown("## 📚 Session History")

    if not history:
        st.markdown(
            """
            <div class="fg-empty">
                <h3>No completed sessions yet</h3>
                <p>Completed focus sessions will appear here automatically.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for index, report in enumerate(history):
        completed = report.get("completed_at", "Unknown date")
        duration = format_duration(report.get("duration_seconds", 0))
        focus = report.get("average_focus", 0) * 100

        with st.expander(
            f"Session {len(history) - index} • {completed} • "
            f"{duration} • {focus:.0f}% focus"
        ):
            metric1, metric2, metric3, metric4 = st.columns(4)

            metric1.metric("Average Focus", f"{focus:.0f}%")
            metric2.metric(
                "Focused",
                f"{report.get('focused_percent', 0):.0f}%",
            )
            metric3.metric(
                "Distracted",
                f"{report.get('distracted_percent', 0):.0f}%",
            )
            metric4.metric(
                "Over-engaged",
                f"{report.get('over_engaged_percent', 0):.0f}%",
            )

            st.markdown(
                f"""
                <div class="fg-blue-box">
                    <strong>Session recommendation</strong><br><br>
                    {report.get("recommendation", "No recommendation available.")}
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# SETTINGS
# =========================================================

def render_settings():
    settings = load_settings()

    st.markdown("## ⚙️ FocusGuard Settings")

    with st.form("settings_form"):
        col1, col2 = st.columns(2)

        with col1:
            blink_threshold = st.slider(
                "High blink-rate threshold",
                10,
                60,
                int(settings["blink_threshold"]),
            )

            yaw_threshold = st.slider(
                "Horizontal head-turn sensitivity",
                10,
                50,
                int(settings["yaw_threshold"]),
            )

            pitch_threshold = st.slider(
                "Vertical head-position sensitivity",
                10,
                50,
                int(settings["pitch_threshold"]),
            )

            minimum_distance = st.slider(
                "Minimum recommended viewing distance (cm)",
                20,
                100,
                int(settings["minimum_distance_cm"]),
            )

        with col2:
            focus_minutes = st.slider(
                "Recommended focus block (minutes)",
                10,
                90,
                int(settings["recommended_focus_minutes"]),
            )

            break_minutes = st.slider(
                "Recommended break (minutes)",
                1,
                20,
                int(settings["recommended_break_minutes"]),
            )

            custom_message = st.text_area(
                "Custom refocus message",
                value=settings["custom_message"],
                height=115,
            )

        submitted = st.form_submit_button(
            "💾 Save Settings",
            type="primary",
            use_container_width=True,
        )

        if submitted:
            updated = {
                "blink_threshold": blink_threshold,
                "yaw_threshold": yaw_threshold,
                "pitch_threshold": pitch_threshold,
                "minimum_distance_cm": minimum_distance,
                "recommended_focus_minutes": focus_minutes,
                "recommended_break_minutes": break_minutes,
                "custom_message": custom_message,
            }

            atomic_write_json(SETTINGS_FILE, updated)
            st.success("Settings saved successfully.")


# =========================================================
# ABOUT
# =========================================================

def render_about():
    st.markdown("## ℹ️ About FocusGuard")

    st.markdown(
        """
        <div class="fg-blue-box">
            <strong>FocusGuard</strong> is a student-built computer-vision
            application designed to help learners recognize distraction,
            visual fatigue, and unhealthy screen engagement during focused work.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How the system works")

    step1, step2, step3, step4 = st.columns(4)

    step1.markdown(
        """
        <div class="fg-card">
            <h4>1. Detect</h4>
            <p>MediaPipe identifies real-time facial landmarks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    step2.markdown(
        """
        <div class="fg-card">
            <h4>2. Measure</h4>
            <p>FocusGuard measures eye, head, blink, and distance signals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    step3.markdown(
        """
        <div class="fg-card">
            <h4>3. Interpret</h4>
            <p>The attention engine evaluates focus and fatigue patterns.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    step4.markdown(
        """
        <div class="fg-card">
            <h4>4. Coach</h4>
            <p>The dashboard delivers actionable focus guidance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Privacy-first design

        FocusGuard processes camera information locally. The V3 dashboard stores
        numerical session measurements and reports in its own local data folder.
        Before public release, a formal privacy policy and user data controls
        should be added.
        """
    )


# =========================================================
# ROUTER
# =========================================================

page = st.session_state.page

if page == "Home":
    render_home()
elif page == "Session":
    render_session()
elif page == "Analytics":
    render_analytics()
elif page == "History":
    render_history()
elif page == "Settings":
    render_settings()
elif page == "About":
    render_about()
elif page == "Summary":
    render_summary()
else:
    st.session_state.page = "Home"
    st.rerun()
