"""
╔══════════════════════════════════════════════════════════════════╗
║           🌿 PlantAI v3.0 — Python Backend (Streamlit)          ║
║         AI-Powered Plant Disease Detection System                ║
║         MSc IT Capstone Project                                  ║
╚══════════════════════════════════════════════════════════════════╝
REQUIREMENTS: 
    pip install streamlit torch torchvision pillow google-generativeai
    pip install pandas matplotlib plotly requests opencv-python
RUN:
    streamlit run app.py
"""
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import io
import time
import hashlib
import datetime
from pathlib import Path
import os
import sqlite3
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

#  SQLITE3 DATABASE SETUP
DB_FILE = "plantai.db"
def get_db():
    """Get SQLite3 connection."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    c = conn.cursor()
    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name     TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    # Scan history table
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT NOT NULL,
            label     TEXT NOT NULL,
            conf      REAL NOT NULL,
            healthy   INTEGER NOT NULL,
            severity  TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    # Reviews table
    c.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            user    TEXT NOT NULL,
            rating  INTEGER NOT NULL,
            comment TEXT NOT NULL,
            time    TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# ── USER HELPERS ──
def load_users_db():
    conn = get_db()
    rows = conn.execute("SELECT username, name, password FROM users").fetchall()
    conn.close()
    return {r["username"]: {"name": r["name"], "password": r["password"]} for r in rows}
def save_user_db(username, name, password):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO users (username, name, password) VALUES (?, ?, ?)",
        (username, name, password))
    conn.commit()
    conn.close()
def get_user_db(username):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None

# ── HISTORY HELPERS ──
def load_history_db():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM history ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def insert_history_db(entry):
    conn = get_db()
    conn.execute(
        "INSERT INTO history (username, label, conf, healthy, severity, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (entry["username"], entry["label"], entry["conf"],
         1 if entry["healthy"] else 0, entry["severity"], entry["timestamp"])
    )
    conn.commit()
    conn.close()

# ── REVIEW HELPERS ──
def load_reviews_db():
    conn = get_db()
    rows = conn.execute("SELECT * FROM reviews ORDER BY id DESC").fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "user":    r["user"],
            "rating":  r["rating"],
            "comment": r["comment"],
            "time":    r["time"],
        })
    return result

def insert_review_db(review):
    conn = get_db()
    conn.execute(
        "INSERT INTO reviews (user, rating, comment, time) VALUES (?, ?, ?, ?)",
        (review["user"], review["rating"], review["comment"], review["time"])
    )
    conn.commit()
    conn.close()
init_db()

# PAGE CONFIG
st.set_page_config(
    page_title="🌿 PlantAI — Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  PREMIUM GREEN THEME - CSS
st.markdown("""
<style>
    /* ═══════════════════════════════════════
       ANIMATIONS
    ═══════════════════════════════════════ */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes floatEmoji {
        0%, 100% { transform: translateY(0px); }
        50%       { transform: translateY(-8px); }
    }
    @keyframes pulseGreen {
        0%, 100% { box-shadow: 0 0 0 0 rgba(22,163,74,0.45); }
        55%       { box-shadow: 0 0 0 10px rgba(22,163,74,0); }
    }
    @keyframes shimmer {
        0%   { background-position: -400px 0; }
        100% { background-position:  400px 0; }
    }
    @keyframes growBar {
        from { width: 0 !important; }
    }
    @keyframes badgePop {
        0%, 100% { transform: scale(1); }
        50%       { transform: scale(1.09); }
    }
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(-12px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes countUp {
        from { opacity: 0; transform: scale(0.7); }
        to   { opacity: 1; transform: scale(1); }
    }
    @keyframes bounceDot {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
        40%            { transform: scale(1.2); opacity: 1; }
    }
    @keyframes ripple {
        0%   { box-shadow: 0 0 0 0 rgba(22,163,74,0.3); }
        100% { box-shadow: 0 0 0 14px rgba(22,163,74,0); }
    }
    /* ═══════════════════════════════════════
       BASE — clean white-green background
    ═══════════════════════════════════════ */
    .main  { background: #f0fdf4; }
    .stApp { background: #f0fdf4; }
    /* ═══════════════════════════════════════
       HEADER — deep green + bright green top border
    ═══════════════════════════════════════ */
    .plantai-header {
        background: linear-gradient(135deg, #0B3D2E, #1a5c38, #1E8449);
        border-top: 4px solid #22c55e;
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        animation: fadeInUp 0.6s ease both;
        position: relative;
        overflow: hidden;
    }
    .plantai-header::before {
        content: '';
        position: absolute; inset: 0;
        background: linear-gradient(110deg, rgba(34,197,94,0.12) 0%, transparent 55%);
        pointer-events: none;
    }
    .plantai-header h1 {
        font-size: 2.2rem; font-weight: 900; margin: 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.22);
    }
    .plantai-header p { opacity: 0.82; margin: 0.3rem 0 0; font-size: 0.9rem; }
    /* ═══════════════════════════════════════
       CARDS — green border top accent
    ═══════════════════════════════════════ */
    .plant-card {
        background: white;
        border-radius: 14px;
        padding: 1.25rem;
        border: 1px solid #bbf7d0;
        border-top: 3px solid #16a34a;
        box-shadow: 0 2px 16px rgba(22,163,74,0.08);
        margin-bottom: 1rem;
        animation: fadeInUp 0.5s ease both;
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }
    .plant-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(22,163,74,0.18);
    }
    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #bbf7d0;
        border-top: 3px solid #16a34a;
        box-shadow: 0 2px 8px rgba(22,163,74,0.07);
        animation: countUp 0.5s ease both;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 26px rgba(22,163,74,0.20);
    }
    .metric-box .val {
        font-size: 2rem; font-weight: 800; color: #0B3D2E;
        animation: countUp 0.6s ease both;
    }
    .metric-box .lbl {
        font-size: 0.7rem; color: #9ca3af;
        text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px;
    }
    /* ═══════════════════════════════════════
       BADGES
    ═══════════════════════════════════════ */
    .badge-healthy {
        background: linear-gradient(135deg, #16a34a, #22c55e);
        color: #f0fdf4;
        padding: 4px 14px; border-radius: 99px;
        font-size: 12px; font-weight: 700;
        display: inline-block;
        animation: badgePop 2.5s ease-in-out infinite;
    }
    .badge-diseased {
        background: #fee2e2; color: #b91c1c;
        padding: 4px 14px; border-radius: 99px;
        font-size: 12px; font-weight: 700;
        display: inline-block;
        animation: badgePop 2.5s ease-in-out infinite 0.35s;
    }
    .badge-warning {
        background: #dcfce7; color: #14532d;
        padding: 4px 14px; border-radius: 99px;
        font-size: 12px; font-weight: 700;
    }
    /* ═══════════════════════════════════════
       CONFIDENCE BARS
    ═══════════════════════════════════════ */
    .conf-bar-fill {
        height: 100%;
        border-radius: 99px;
        animation: growBar 1s ease both;
    }
    /* ═══════════════════════════════════════
       SOLUTION BOX — green left border
    ═══════════════════════════════════════ */
    .solution-box {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-left: 4px solid #16a34a;
        border-radius: 0 14px 14px 0;
        padding: 1.25rem;
        font-size: 14px;
        line-height: 1.9;
        animation: fadeInUp 0.7s ease both;
        transition: border-left-color 0.3s ease;
    }
    .solution-box:hover { border-left-color: #22c55e; }
    /* ═══════════════════════════════════════
       HISTORY ROWS
    ═══════════════════════════════════════ */
    .hist-row-healthy {
        background: #f0fdf4; border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;
        animation: slideInRight 0.4s ease both;
        transition: transform 0.18s ease;
    }
    .hist-row-healthy:hover  { transform: translateX(5px); }
    .hist-row-diseased {
        background: #fff5f5; border: 1px solid #fecaca;
        border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;
        animation: slideInRight 0.4s ease both;
        transition: transform 0.18s ease;
    }
    .hist-row-diseased:hover { transform: translateX(5px); }
    /* ═══════════════════════════════════════
       SIDEBAR — dark green + bright green right border
    ═══════════════════════════════════════ */
    .css-1d391kg { background: linear-gradient(180deg, #0B3D2E, #145A32) !important; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B3D2E, #145A32);
        border-right: 3px solid #22c55e;
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    /* ═══════════════════════════════════════
       BUTTONS — premium green gradient
    ═══════════════════════════════════════ */
    .stButton > button {
        background: linear-gradient(135deg, #14532d, #16a34a, #22c55e);
        color: #f0fdf4 !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.5rem 1.5rem;
        transition: all 0.22s ease;
        animation: pulseGreen 2.8s ease-in-out infinite;
    }
    .stButton > button:hover {
        opacity: 0.92;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(22,163,74,0.42);
        animation: none;
    }
    .stButton > button:active { transform: scale(0.97); }
    /* ═══════════════════════════════════════
       LOADING DOTS — green
    ═══════════════════════════════════════ */
    .dot-loader { display:flex; gap:5px; align-items:center; justify-content:center; padding:12px; }
    .dot-loader span {
        width:9px; height:9px; background:#22c55e; border-radius:50%;
        animation: bounceDot 1.3s ease-in-out infinite;
    }
    .dot-loader span:nth-child(2) { animation-delay:0.22s; }
    .dot-loader span:nth-child(3) { animation-delay:0.44s; }
    /* ═══════════════════════════════════════
       SHIMMER SKELETON — green shimmer
    ═══════════════════════════════════════ */
    .shimmer {
        background: linear-gradient(90deg, #f0fdf4 25%, #bbf7d0 50%, #f0fdf4 75%);
        background-size: 400px 100%;
        animation: shimmer 1.6s linear infinite;
        border-radius: 8px;
    }
    /* ═══════════════════════════════════════
       REVIEW CARDS
    ═══════════════════════════════════════ */
    div[style*="border-left:4px solid #16a34a"] {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: fadeInUp 0.45s ease both;
    }
    div[style*="border-left:4px solid #16a34a"]:hover {
        transform: translateX(5px);
        box-shadow: 4px 4px 16px rgba(22,163,74,0.14);
    }
    /* ═══════════════════════════════════════
       FILE UPLOADER — green ripple
    ═══════════════════════════════════════ */
    [data-testid="stFileUploader"] {
        animation: ripple 2.5s ease-in-out infinite;
        border-radius: 12px;
    }
    /* ═══════════════════════════════════════
       COMPARE SLIDER
    ═══════════════════════════════════════ */
    .compare-container {
        background: white;
        border-radius: 14px;
        border: 1px solid #bbf7d0;
        border-top: 3px solid #16a34a;
        padding: 1.2rem;
        box-shadow: 0 2px 16px rgba(22,163,74,0.08);
        margin-bottom: 1rem;
    }
    .compare-label {
        font-size: 11px;
        font-weight: 700;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .diff-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 99px;
        font-size: 12px;
        font-weight: 700;
        margin: 4px 4px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ── PDF GENERATOR ──
def generate_pdf_report(result, solution):
    GD = colors.HexColor('#0B3D2E')
    GM = colors.HexColor('#16a34a')
    GL = colors.HexColor('#f0fdf4')
    BL = colors.HexColor('#1d4ed8')
    RD = colors.HexColor('#b91c1c')
    OR = colors.HexColor('#d97706')
    GR = colors.HexColor('#6b7280')
    WH = colors.white
    BD = colors.HexColor('#bbf7d0')
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,topMargin=2*cm, bottomMargin=2*cm)
    def ps(name, **kw): return ParagraphStyle(name, **kw)
    title_s  = ps('t',  fontSize=20, textColor=WH,  fontName='Helvetica-Bold', alignment=TA_CENTER)
    sub_s    = ps('s',  fontSize=9,  textColor=colors.HexColor('#f0fdf4'), fontName='Helvetica', alignment=TA_CENTER)
    h2_s     = ps('h2', fontSize=12, textColor=GD,  fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=5)
    body_s   = ps('b',  fontSize=10, textColor=colors.HexColor('#374151'), fontName='Helvetica', leading=15, spaceAfter=3)
    head_s   = ps('sh', fontSize=11, textColor=GD,  fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=4)
    bullet_s = ps('bl', fontSize=10, textColor=colors.HexColor('#374151'), fontName='Helvetica', leading=15, leftIndent=14, spaceAfter=3)

    def tbl(data, widths, style_extra=None):
        t = Table(data, colWidths=widths)
        base = [('TOPPADDING',(0,0),(-1,-1),8), ('BOTTOMPADDING',(0,0),(-1,-1),8),
                ('LEFTPADDING',(0,0),(-1,-1),10), ('RIGHTPADDING',(0,0),(-1,-1),10)]
        if style_extra: base += style_extra
        t.setStyle(TableStyle(base))
        return t
    story = []

    # Header
    h = tbl([[Paragraph('🌿  PlantAI Disease Detection Report', title_s)]], [17*cm],
             [('BACKGROUND',(0,0),(-1,-1),GD),
              ('TOPPADDING',(0,0),(-1,-1),16),('BOTTOMPADDING',(0,0),(-1,-1),12)])
    story += [h, Spacer(1,4)]

    s = tbl([[Paragraph(f"MSc IT Capstone Project  ·  Generated: {result.get('timestamp','')}", sub_s)]], [17*cm],
             [('BACKGROUND',(0,0),(-1,-1),GM),
              ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])
    story += [s, Spacer(1,16)]

    # Diagnosis cards
    story += [Paragraph('📋  Diagnosis Summary', h2_s),
              HRFlowable(width='100%', thickness=1.5, color=GM, spaceAfter=8)]

    is_healthy = result.get('healthy', False)
    severity   = result.get('severity', 'N/A')
    conf       = result.get('conf', 0) * 100
    sev_color  = GM if severity == 'None' else (RD if severity == 'High' else OR)
    st_color   = GM if is_healthy else RD
    st_text    = '✓  Healthy' if is_healthy else '⚠  Diseased'

    lbl = lambda t: Paragraph(t, ps('l', fontSize=8, fontName='Helvetica', textColor=GR, alignment=TA_CENTER))
    val = lambda t,c=GD,sz=11: Paragraph(t, ps('v', fontSize=sz, fontName='Helvetica-Bold', textColor=c, alignment=TA_CENTER))
    cards = [
        [lbl('DIAGNOSIS'), lbl('CONFIDENCE'), lbl('SEVERITY'), lbl('STATUS')],
        [val(result.get('label','N/A'), GD, 9), val(f'{conf:.1f}%', BL, 14),
         val(severity, sev_color, 13), val(st_text, st_color, 11)],
    ]
    ct = Table(cards, colWidths=[5*cm, 3.5*cm, 3.5*cm, 4*cm])
    ct.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),GL), ('BACKGROUND',(0,1),(-1,1),WH),
        ('BOX',(0,0),(-1,-1),1,BD), ('INNERGRID',(0,0),(-1,-1),.5,BD),
        ('TOPPADDING',(0,0),(-1,-1),10), ('BOTTOMPADDING',(0,0),(-1,-1),10),
        ('LEFTPADDING',(0,0),(-1,-1),8), ('RIGHTPADDING',(0,0),(-1,-1),8),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]))
    story += [ct, Spacer(1,16)]
    # Report details
    story += [Paragraph('👤  Report Details', h2_s),
              HRFlowable(width='100%', thickness=1.5, color=GM, spaceAfter=8)]

    def row(k, v): return [Paragraph(f'<b>{k}</b>', body_s), Paragraph(v, body_s)]
    info = Table([
        row('Farmer / User', result.get('username','N/A')),
        row('Date & Time',   result.get('timestamp','N/A')),
        row('Disease Name',  result.get('label','N/A')),
        row('AI Confidence', f"{conf:.1f}%"),
        row('Severity Level', severity),
        row('Plant Status',  'Healthy ✓' if is_healthy else 'Diseased ⚠'),
    ], colWidths=[5*cm, 12*cm])
    info.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1),GL), ('BOX',(0,0),(-1,-1),1,BD),
        ('INNERGRID',(0,0),(-1,-1),.5,colors.HexColor('#e5e7eb')),
        ('TOPPADDING',(0,0),(-1,-1),8), ('BOTTOMPADDING',(0,0),(-1,-1),8),
        ('LEFTPADDING',(0,0),(-1,-1),10), ('RIGHTPADDING',(0,0),(-1,-1),10),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[WH,GL]),
    ]))
    story += [info, Spacer(1,16)]

    # Treatment plan
    story += [Paragraph('🤖  AI Treatment Plan', h2_s),
              HRFlowable(width='100%', thickness=1.5, color=GM, spaceAfter=8)]
    sol_items = []
    for line in solution.strip().split('\n'):
        line = line.strip()
        if not line: sol_items.append(Spacer(1,4))
        elif line.startswith('**') and line.endswith('**'): sol_items.append(Paragraph(line.replace('**',''), head_s))
        elif line.startswith('- ') or line.startswith('* '): sol_items.append(Paragraph('•  '+line[2:], bullet_s))
        elif line.startswith('#'): sol_items.append(Paragraph(line.lstrip('#').strip(), head_s))
        else: sol_items.append(Paragraph(line, body_s))
    story += sol_items
    story.append(Spacer(1,20))

    # Footer
    ft = Table([[Paragraph(
        '🌿  PlantAI v3.0  ·  Built for Indian Farmers  ·  PlantVillage Dataset · 38 Disease Classes',
        ps('f', fontSize=8, textColor=WH, fontName='Helvetica', alignment=TA_CENTER)
    )]], colWidths=[17*cm])
    ft.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),GD),
        ('TOPPADDING',(0,0),(-1,-1),10), ('BOTTOMPADDING',(0,0),(-1,-1),10)
    ]))
    story.append(ft)
    doc.build(story)
    return buf.getvalue()

# ── DISEASE CLASSES (PlantVillage Dataset — 38 classes) ──
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

HEALTHY_CLASSES = [c for c in DISEASE_CLASSES if 'healthy' in c.lower()]
SEVERITY_MAP = {
    'Apple___Apple_scab': 'High',
    'Apple___Black_rot': 'High',
    'Apple___Cedar_apple_rust': 'Medium',
    'Cherry_(including_sour)___Powdery_mildew': 'Medium',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Medium',
    'Corn_(maize)___Common_rust_': 'Medium',
    'Corn_(maize)___Northern_Leaf_Blight': 'High',
    'Grape___Black_rot': 'High',
    'Grape___Esca_(Black_Measles)': 'High',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Medium',
    'Orange___Haunglongbing_(Citrus_greening)': 'High',
    'Peach___Bacterial_spot': 'Medium',
    'Pepper,_bell___Bacterial_spot': 'Medium',
    'Potato___Early_blight': 'Medium',
    'Potato___Late_blight': 'High',
    'Squash___Powdery_mildew': 'Low',
    'Strawberry___Leaf_scorch': 'Medium',
    'Tomato___Bacterial_spot': 'Medium',
    'Tomato___Early_blight': 'Medium',
    'Tomato___Late_blight': 'High',
    'Tomato___Leaf_Mold': 'Low',
    'Tomato___Septoria_leaf_spot': 'Medium',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Medium',
    'Tomato___Target_Spot': 'Medium',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'High',
    'Tomato___Tomato_mosaic_virus': 'High',
}

# ── SESSION STATE INIT ──
def init_state():
    defaults = {
        'logged_in': False,
        'username': '',
        'user_name': '',
        'scan_history': load_history_db(),
        'api_key': 'AIzaSyAKJROEULQlxVenNgUsCpkRorFswtBXkG8',
        'model': None,
        'reviews': load_reviews_db(),
        'users': load_users_db(),
        'page': 'login',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# ── MODEL LOADER ──
@st.cache_resource(show_spinner=False)
def load_model(model_path: str | None = None):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(DISEASE_CLASSES))

    if model_path and os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            st.toast("✅ Model weights loaded!", icon="🌿")
        except Exception as e:
            st.warning(f"🚫 Could not load weights: {e}. Using demo mode.")
    else:
        st.info("‼️ No model weights found. Running in **demo mode** (random predictions).")

    model.eval()
    return model

# ── IMAGE PREPROCESSING ──
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def enhance_image(pil_img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(pil_img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = ImageEnhance.Color(img).enhance(1.1)
    return img

def preprocess(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert('RGB')
    return TRANSFORM(img).unsqueeze(0)

# ── INFERENCE ──
def predict(pil_img: Image.Image, model: nn.Module):
    tensor = preprocess(pil_img)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top5_idx  = torch.topk(probs, 5).indices.tolist()
    top5_prob = torch.topk(probs, 5).values.tolist()

    results = []
    for idx, prob in zip(top5_idx, top5_prob):
        label   = DISEASE_CLASSES[idx]
        display = label.replace('___', ' — ').replace('_', ' ')
        healthy = label in HEALTHY_CLASSES
        results.append({
            'label':    display,
            'raw':      label,
            'conf':     round(prob, 4),
            'healthy':  healthy,
            'severity': 'None' if healthy else SEVERITY_MAP.get(label, 'Medium'),
        })
    return results

# ── GEMINI AI ──
def get_gemini_solution(disease_name: str, api_key: str) -> str:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""You are an expert agriculture scientist helping Indian farmers.
A farmer's plant has been diagnosed with: {disease_name}

Provide a detailed treatment guide with:
1. **Cause** — What causes this disease
2. **Symptoms** — Key visual indicators
3. **Organic Treatment** — Natural remedies with steps
4. **Chemical Treatment** — Product names with dosage (available in India)
5. **Prevention Tips** — How to prevent in future
6. **Indian Market Products** — 3-5 specific brand names a farmer can buy locally

Keep language simple and practical. Format with bullet points."""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {str(e)}\n\nPlease check your API key at https://aistudio.google.com/app/apikey"

# ── STATIC SOLUTIONS (offline fallback) ──
STATIC_SOLUTIONS = {
    'Apple — Apple scab': """
**Cause:** Fungus *Venturia inaequalis*. Spreads in cool, wet spring weather.

**Organic Treatment:**
- Neem oil spray (5 ml/L water) every 7 days
- Baking soda solution (1 tsp/L) spray weekly
- Remove and burn all infected leaves and fruit

**Chemical Treatment:**
- Mancozeb 75% WP (2 g/L) — apply every 10–14 days
- Captan 50% WP (2 g/L) — before and after rain

**Prevention:**
- Prune trees for good air circulation
- Avoid overhead irrigation
- Rake and destroy fallen leaves in autumn

**Indian Market Products:** Dithane M-45, Captaf, Indofil M-45, Blitox-50
""",
    'Apple — Black rot': """
**Cause:** Fungus *Guignardia bidwellii*. Hot and humid conditions.

**Organic Treatment:**
- Bordeaux mixture spray (1%)
- Copper-based fungicide
- Remove all mummified fruits from tree and ground

**Chemical Treatment:**
- Myclobutanil (1.5 ml/L) — every 10 days
- Captan 50% WP (2 g/L)

**Prevention:**
- Orchard sanitation — remove dead wood
- Proper pruning to reduce humidity
- Avoid wounding fruits

**Indian Market Products:** Rally, Captaf, Blitox, Bavistin
""",
    'Tomato — Late blight': """
**Cause:** Oomycete *Phytophthora infestans*. Spreads in cool, moist conditions.

**Organic Treatment:**
- Copper fungicide spray every 5–7 days
- Compost tea spray to boost immunity
- Remove infected plants immediately

**Chemical Treatment:**
- Metalaxyl + Mancozeb (2.5 g/L)
- Cymoxanil + Mancozeb (2 g/L)

**Prevention:**
- Use certified disease-free seeds
- Avoid overhead watering
- Maintain plant spacing for airflow

**Indian Market Products:** Ridomil Gold, Curzate M, Master, Equation Pro
""",
    'Potato — Late blight': """
**Cause:** *Phytophthora infestans* — same pathogen as tomato late blight.

**Organic Treatment:**
- Destroy infected plants immediately
- Apply Bordeaux mixture (1%)
- Improve drainage around crop

**Chemical Treatment:**
- Metalaxyl 8% + Mancozeb 64% WP (2.5 g/L)
- Cymoxanil 8% + Mancozeb 64% (2 g/L)

**Prevention:**
- Use certified seed potatoes
- Hilling to protect tubers
- Monitor weather for blight-favourable conditions

**Indian Market Products:** Ridomil Gold MZ, Curzate M8, Kavach
""",
}
def get_static_solution(disease_display: str) -> str:
    for key, sol in STATIC_SOLUTIONS.items():
        if key.lower() in disease_display.lower():
            return sol
    return (" Detailed solution not available offline for this disease. "
            "Please add your **Gemini API key** in the sidebar for AI-powered treatment plans.")

# ── AUTH HELPERS ──
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def do_login(username, password):
    user = get_user_db(username)
    if user and user["password"] == password:
        return True
    return False

def do_register(name, username, password, confirm):
    if not all([name, username, password, confirm]):
        return False, "All fields are required."
    if password != confirm:
        return False, "Passwords do not match."
    existing = get_user_db(username)
    if existing:
        return False, "Username already exists"
    save_user_db(username, name, password)
    return True, "Registration successful!"

# ── EXPORT HELPERS ──
def history_to_df() -> pd.DataFrame:
    if not st.session_state.scan_history:
        return pd.DataFrame()
    rows = []
    for h in st.session_state.scan_history:
        rows.append({
            'Date':       h.get('timestamp', ''),
            'User':       h.get('username', ''),
            'Disease':    h.get('label', ''),
            'Confidence': f"{h.get('conf', 0)*100:.1f}%",
            'Severity':   h.get('severity', ''),
            'Healthy':    'Yes' if h.get('healthy') else 'No',
        })
    return pd.DataFrame(rows)

#  AUTH PAGE
def render_auth():
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("""
        <div style="background:linear-gradient(150deg,#0B3D2E,#145A32,#1a5c38);
                    border-top:4px solid #22c55e;
                    padding:3rem 2.5rem;border-radius:20px;min-height:600px;
                    display:flex;flex-direction:column;justify-content:center">
          <div style="font-size:52px">🌿</div>
          <h1 style="font-size:2.4rem;font-weight:900;color:#fff;line-height:1.2;
                     letter-spacing:-1px;margin:16px 0 12px">
              PlantAI<br>Disease Detection
          </h1>
          <p style="font-size:15px;color:rgba(255,255,255,.72);line-height:1.8;margin-bottom:2rem">
            AI-powered plant health diagnosis for Indian farmers.<br>
            Upload a leaf photo and get instant results with treatment plans.
          </p>
          <div style="border-top:1px solid rgba(34,197,94,0.3);padding-top:1rem">
            <p style="color:rgba(255,255,255,.85);font-size:13px;margin:8px 0">⚙️ Deep Learning — EfficientNet</p>
            <p style="color:rgba(255,255,255,.85);font-size:13px;margin:8px 0">🧠 Google Gemini 3 AI Treatment Plans</p>
            <p style="color:rgba(255,255,255,.85);font-size:13px;margin:8px 0">📊 Analytics Dashboard & Exportable Reports</p>
            <p style="color:rgba(255,255,255,.85);font-size:13px;margin:8px 0">⛅ Weather-Based Disease Alerts</p>
            <p style="color:rgba(255,255,255,.85);font-size:13px;margin:8px 0">💬 Plant Care Chatbot Assistant</p>
            <p style="color:rgba(255,255,255,.85);font-size:13px;margin:8px 0">🌿 Tailored for Indian Agriculture</p>
          </div>
          <p style="font-size:11px;color:rgba(34,197,94,0.5);margin-top:2rem">MSc.IT Capstone Project · PlantAI v3.0</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        tab_login, tab_reg = st.tabs(["🔐 Login", "👤 Register"])
        with tab_login:
            st.markdown("### Welcome back 🙂")
            st.caption("Sign in to your PlantAI account")
            username = st.text_input("Username", key="l_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="l_pass",
                                     placeholder="Enter your password")
            if st.button("Sign In →", key="btn_login", use_container_width=True):
                if do_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    user = get_user_db(username)
                    st.session_state.user_name = user["name"]
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password. Please try again.")
        with tab_reg:
            st.markdown("### Create account 🌱")
            st.caption("Join PlantAI — it's free")
            r_name = st.text_input("Full Name", placeholder="e.g. Ramesh Patel")
            r_user = st.text_input("Username", placeholder="Choose username")
            r_pass = st.text_input("Password", type="password", placeholder="Min 6 characters")
            r_conf = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
            if st.button("Create Account ✨", key="btn_reg", use_container_width=True):
                ok, msg = do_register(r_name, r_user, r_pass, r_conf)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

#  SIDEBAR
def render_sidebar():
    with st.sidebar:
        # ── Profile ──
        st.markdown(f"""
        <div style="padding:1rem 0.5rem;border-bottom:1px solid rgba(34,197,94,0.3)">
          <div style="display:flex;align-items:center;gap:10px">
            <div style="width:44px;height:44px;border-radius:50%;background:rgba(34,197,94,0.2);
                        display:flex;align-items:center;justify-content:center;
                        font-size:18px;font-weight:800;color:#fff;border:2px solid #22c55e">
              {st.session_state.user_name[0].upper()}
            </div>
            <div>
              <div style="font-size:14px;font-weight:700;color:#fff">{st.session_state.user_name}</div>
              <div style="font-size:11px;color:rgba(34,197,94,0.7)">@{st.session_state.username}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        # ── Stats ──
        my_scans  = [h for h in st.session_state.scan_history
                     if h.get('username') == st.session_state.username]
        healthy_c = sum(1 for h in my_scans if h.get('healthy'))
        sick_c    = len(my_scans) - healthy_c
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Scans",   len(my_scans))
            st.metric("Healthy", healthy_c)
        with c2:
            st.metric("Users",    len(load_users_db()))
            st.metric("Diseased", sick_c)

        st.markdown("---")
        # ── Navigation ──
        NAV = [
            ("MAIN", [
                ("📊", "Dashboard",       "dashboard"),
                ("🔎", "Detect Disease",  "predict"),
                ("📋", "Scan History",    "history"),
                ("📈", "Analytics",       "analytics"),
            ]),
            ("SMART FEATURES", [
                ("🌾", "Crop Monitor",    "cropmonitor"),
                ("💬", "Plant Chatbot",   "chatbot"),
                ("🔀", "Image Compare",   "compare"),
                ("⛅", "Weather Alerts",  "weather"),
                ("🌱", "Fertilizer Guide","fertilizer"),
            ]),
            ("COMMUNITY", [
                ("⭐", "Reviews",         "reviews"),
                ("ℹ️", "About",           "about"),
            ]),
        ]
        cur = st.session_state.get('page', 'dashboard')
        for section, items in NAV:
            st.markdown(
                f'<div style="font-size:13px;font-weight:700;color:rgba(255,255,255,0.85);'
                f'letter-spacing:1px;text-transform:uppercase;padding:8px 4px 3px">'
                f'{section}</div>',
                unsafe_allow_html=True
            )
            for icon, label, key in items:
                active = cur == key
                bg = "background:rgba(255,255,255,0.18);" if active else ""
                fw = "font-weight:700;" if active else ""
                if st.button(f"{icon}  {label}", key=f"nav_{key}",
                             use_container_width=True):
                    st.session_state.page = key
                    st.rerun()

        st.markdown("---")
        # Logout
        if st.button("🚪  Logout", key="logout_btn", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username  = ''
            st.session_state.page      = 'login'
            st.rerun()
            
#  DASHBOARD
def render_dashboard():
    st.markdown("""
    <div class="plantai-header">
      <h1>📊 Dashboard</h1>
      <p>Your PlantAI overview</p>
    </div>
    """, unsafe_allow_html=True)

    my_scans  = [h for h in st.session_state.scan_history
                 if h.get('username') == st.session_state.username]
    healthy_c = sum(1 for h in my_scans if h.get('healthy'))
    sick_c    = len(my_scans) - healthy_c
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, lbl in [
        (c1, "🧪", len(my_scans), "Total Scans"),
        (c2, "✅", healthy_c,    "Healthy"),
        (c3, "⚠️", sick_c,       "Diseased"),
        (c4, "⭐", len(st.session_state.reviews), "Reviews"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
              <div style="font-size:28px">{icon}</div>
              <div class="val">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 🕒 Recent Scans")
        if my_scans:
            for h in my_scans[:7]:
                badge = "badge-healthy" if h['healthy'] else "badge-diseased"
                label = "✓ Healthy" if h['healthy'] else "⚠ Diseased"
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            align-items:center;padding:8px 0;
                            border-bottom:1px solid #bbf7d0">
                  <span style="font-size:13px;font-weight:600">{h['label']}</span>
                  <span class="{badge}">{label}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📄 No scans yet. Go to **Detect Disease** to start!")
    with col_r:
        st.markdown("#### 🌟 Quick Actions")
        if st.button("🔎 Detect Disease",       key="qd_detect",  use_container_width=True):
            st.session_state.page = 'predict'; st.rerun()
        if st.button("💬 Ask Plant Chatbot",    key="qd_chat",    use_container_width=True):
            st.session_state.page = 'chatbot'; st.rerun()
        if st.button("⛅ Check Weather Alerts", key="qd_weather", use_container_width=True):
            st.session_state.page = 'weather'; st.rerun()

#  DETECT DISEASE
def render_predict():
    st.markdown("""
    <div class="plantai-header">
      <h1>🔎 Detect Plant Disease</h1>
      <p>Upload a clear leaf photo for instant AI diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🗂️ Upload Leaf Image")
    uploaded = st.file_uploader(
        "Choose a leaf image (JPG, PNG · Max 10 MB)",
        type=['jpg', 'jpeg', 'png'],
        key="file_upload"
    )
    if not uploaded:
        return

    pil_img = Image.open(uploaded).convert('RGB')
    col_img, col_opts = st.columns([1, 1])

    with col_img:
        st.image(pil_img, caption=uploaded.name, use_container_width=True)
    with col_opts:
        st.markdown("#### ⚙️ Analysis Options")
        auto_enhance = st.checkbox("🛠️ Auto Image Enhancement", value=True)
        get_solution = st.checkbox("🧠 Get AI Treatment Plan",  value=True)

    if not st.button("🔎 Analyze Leaf Now", key="btn_analyze", use_container_width=True):
        return
    analysis_img = enhance_image(pil_img) if auto_enhance else pil_img

    if st.session_state.model is None:
        with st.spinner("Loading AI model..."):
            model_path = st.session_state.get('model_path', 'plantvillage_efficientnet_b0.pth')
            st.session_state.model = load_model(model_path)

    with st.spinner("🔬 Analyzing leaf..."):
        progress = st.progress(0, text="Enhancing image...")
        time.sleep(0.4); progress.progress(25, text="Running AI model...")
        results = predict(analysis_img, st.session_state.model)
        time.sleep(0.3); progress.progress(75, text="Calculating confidence...")
        time.sleep(0.2); progress.progress(100, text="Generating results...")
        time.sleep(0.2); progress.empty()

    top = results[0]
    entry = {
        "username":  st.session_state.username,
        "label":     top['label'],
        "conf":      top['conf'],
        "healthy":   top['healthy'],
        "severity":  top['severity'],
        "timestamp": datetime.datetime.now().strftime('%d-%m-%Y %I:%M %p')
    }

    # Save to SQLite and refresh session state
    insert_history_db(entry)
    st.session_state.scan_history = load_history_db()
    st.success("✅ Analysis complete!")

    # Result cards
    c1, c2, c3 = st.columns(3)
    color = "#0B3D2E" if top['healthy'] else "#b91c1c"
    bg    = "#f0fdf4"  if top['healthy'] else "#fff5f5"
    icon  = "✅" if top['healthy'] else "⚠️"

    with c1:
        st.markdown(f"""
        <div class="plant-card" style="background:{bg};text-align:center">
          <div style="font-size:30px">{icon}</div>
          <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:1px">Diagnosis</div>
          <div style="font-size:18px;font-weight:800;color:{color}">{top['label']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="plant-card" style="background:#f0fdf4;text-align:center">
          <div style="font-size:30px">🎯</div>
          <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:1px">Confidence</div>
          <div style="font-size:24px;font-weight:800;color:#16a34a">{top['conf']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        sev   = top['severity']
        s_col = "#0B3D2E" if sev == "None" else ("#b91c1c" if sev == "High" else "#d97706")
        s_bg  = "#f0fdf4"  if sev == "None" else ("#fff5f5" if sev == "High" else "#fffbeb")
        st.markdown(f"""
        <div class="plant-card" style="background:{s_bg};text-align:center">
          <div style="font-size:30px">{'🌿' if top['healthy'] else '🏥'}</div>
          <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:1px">Severity</div>
          <div style="font-size:24px;font-weight:800;color:{s_col}">{sev}</div>
        </div>""", unsafe_allow_html=True)

    # Confidence bars
    st.markdown("#### 📋 All Predictions")
    for r in results:
        pct     = r['conf'] * 100
        bar_col = "#22c55e" if r['healthy'] else ("#dc2626" if r['conf'] > 0.7 else "#d97706")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
          <div style="min-width:220px;font-size:13px;color:#1a1a1a;overflow:hidden;
                      text-overflow:ellipsis;white-space:nowrap">{r['label']}</div>
          <div style="flex:1;background:#dcfce7;border-radius:99px;height:10px;overflow:hidden">
            <div style="width:{pct:.1f}%;height:100%;background:{bar_col};border-radius:99px"></div>
          </div>
          <div style="font-size:12px;color:#9ca3af;min-width:44px;text-align:right;font-weight:700">{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # AI Treatment Plan
    st.markdown("#### 🧠 AI Treatment Plan")
    if top['healthy']:
        st.success("✅ Your plant looks healthy! Continue regular monitoring, proper watering, and ensure adequate sunlight.")
    elif get_solution:
        if st.session_state.api_key:
            with st.spinner("Getting AI treatment plan from Gemini..."):
                solution = get_gemini_solution(top['label'], st.session_state.api_key)
        else:
            solution = get_static_solution(top['label'])
            st.warning("⚠️ Add your Gemini API key in the sidebar for detailed AI solutions.")

        st.markdown(f'<div class="solution-box">{solution}</div>', unsafe_allow_html=True)

        st.markdown("#### 📄 Export Report")
        ex1, ex2, _ = st.columns(3)
        with ex1:
            try:
                pdf_bytes = generate_pdf_report(entry, solution)
                st.download_button("📕 Download PDF Report", data=pdf_bytes,
                                   file_name=f"PlantAI_Report_{int(time.time())}.pdf",
                                   mime="application/pdf")
            except Exception as e:
                st.error(f"PDF Error: {e}")
        with ex2:
            df = history_to_df()
            if not df.empty:
                st.download_button("💾 Download CSV", df.to_csv(index=False),
                                   file_name=f"PlantAI_History_{int(time.time())}.csv",
                                   mime="text/csv")

#  SCAN HISTORY
def render_history():
    st.markdown("""
    <div class="plantai-header">
      <h1>📜 Scan History</h1>
      <p>All your previous plant scans</p>
    </div>
    """, unsafe_allow_html=True)

    my_scans  = [h for h in st.session_state.scan_history
                 if h.get('username') == st.session_state.username]
    healthy_c = sum(1 for h in my_scans if h.get('healthy'))
    sick_c    = len(my_scans) - healthy_c

    c1, c2, c3 = st.columns(3)
    for col, icon, val, lbl in [
        (c1, "🧪", len(my_scans), "Total"),
        (c2, "✅", healthy_c,    "Healthy"),
        (c3, "⚠️", sick_c,       "Diseased"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
              <div style="font-size:24px">{icon}</div>
              <div class="val">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    col_hdr, col_exp = st.columns([3, 1])
    with col_hdr:
        st.markdown("#### 📋 All Scans")
    with col_exp:
        df = history_to_df()
        if not df.empty:
            st.download_button("📤 Export CSV", df.to_csv(index=False),
                               file_name="PlantAI_History.csv", mime="text/csv")

    if not my_scans:
        st.info("📄 No scans yet. Upload a leaf to start!")
    else:
        for h in my_scans:
            row_cls = "hist-row-healthy" if h['healthy'] else "hist-row-diseased"
            badge   = "badge-healthy"    if h['healthy'] else "badge-diseased"
            label   = "✓ Healthy"        if h['healthy'] else "⚠ Diseased"
            st.markdown(f"""
            <div class="{row_cls}" style="display:flex;justify-content:space-between;align-items:center">
              <div>
                <div style="font-size:14px;font-weight:600">{h['label']}</div>
                <div style="font-size:11px;color:#9ca3af">🕐 {h.get('timestamp','')}</div>
              </div>
              <div style="display:flex;align-items:center;gap:10px">
                <span style="font-size:12px;color:#9ca3af">{h.get('conf',0)*100:.1f}%</span>
                <span class="{badge}">{label}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

#  ANALYTICS
def render_analytics():
    st.markdown("""
    <div class="plantai-header">
      <h1>📶 Analytics</h1>
      <p>Your prediction history graphs and trends</p>
    </div>
    """, unsafe_allow_html=True)

    my_scans = [h for h in st.session_state.scan_history
                if h.get('username') == st.session_state.username]
    if not my_scans:
        st.info("📂 No data yet. Run some scans first!")
        return
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 💹 Disease Distribution")
        counts = {}
        for h in my_scans:
            counts[h['label']] = counts.get(h['label'], 0) + 1
        df_dist = pd.DataFrame(list(counts.items()), columns=['Disease', 'Count'])
        fig = px.bar(df_dist, x='Count', y='Disease', orientation='h',
                     color='Count',
                     color_continuous_scale=['#dcfce7', '#16a34a'],
                     template='plotly_white')
        fig.update_layout(showlegend=False, height=300,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### 📉 Scans Over Time")
        colours = ['#22c55e' if h['healthy'] else "#ca0c0c" for h in my_scans[-14:]]
        fig2 = go.Figure(go.Bar(
            x=list(range(1, len(my_scans[-14:])+1)),
            y=[h['conf']*100 for h in my_scans[-14:]],
            marker_color=colours,))
        fig2.update_layout(
            xaxis_title="Scan #", yaxis_title="Confidence %",
            template='plotly_white', height=300,
            margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    healthy_c = sum(1 for h in my_scans if h['healthy'])
    hr    = round(healthy_c / len(my_scans) * 100)
    avg_c = sum(h['conf'] for h in my_scans) / len(my_scans) * 100
    with col_l2:
        st.markdown(f"""
        <div class="plant-card" style="text-align:center;padding:2rem">
          <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:1px">Health Rate</div>
          <div style="font-size:56px;font-weight:900;color:#16a34a">{hr}%</div>
          <div style="font-size:13px;color:#9ca3af">of your plants are healthy</div>
        </div>
        """, unsafe_allow_html=True)
    with col_r2:
        st.markdown(f"""
        <div class="plant-card" style="text-align:center;padding:2rem">
          <div style="font-size:10px;color:#9ca3af;text-transform:uppercase;letter-spacing:1px">Avg Confidence</div>
          <div style="font-size:56px;font-weight:900;color:#0B3D2E">{avg_c:.1f}%</div>
          <div style="font-size:13px;color:#9ca3af">average AI confidence</div>
        </div>
        """, unsafe_allow_html=True)

#  CROP MONITOR  
def render_cropmonitor():
    st.markdown("""
    <div class="plantai-header">
      <h1>🌾 Crop Monitor</h1>
      <p>Track your crop health trends over time</p>
    </div>
    """, unsafe_allow_html=True)

    my_scans = [h for h in st.session_state.scan_history
                if h.get('username') == st.session_state.username]

    if not my_scans:
        st.info("🌱 No scan data yet. Start scanning plants to see crop health trends!")
    else:
        # ── Disease frequency ──
        st.markdown("#### 📊 Disease Frequency Monitor")
        counts = {}
        for h in my_scans:
            counts[h['label']] = counts.get(h['label'], 0) + 1

        for disease, count in sorted(counts.items(), key=lambda x: -x[1]):
            is_h = 'healthy' in disease.lower()
            pct  = round(count / len(my_scans) * 100)
            col  = "#16a34a" if is_h else "#dc2626"
            icon = "✅" if is_h else "⚠️"
            st.markdown(f"""
            <div style="background:{'#f0fdf4' if is_h else '#fff5f5'};border-radius:12px;
                        padding:.9rem 1rem;border:1px solid {'#bbf7d0' if is_h else '#fecaca'};
                        margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <span style="font-size:13px;font-weight:700">{icon} {disease}</span>
                <span style="background:{'#dcfce7' if is_h else '#fee2e2'};
                             color:{col};padding:2px 10px;border-radius:99px;
                             font-size:11px;font-weight:700">{count} scans</span>
              </div>
              <div style="background:#e5e7eb;border-radius:99px;height:7px;overflow:hidden">
                <div style="width:{pct}%;height:100%;border-radius:99px;background:{col}"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

    # ── Stress indicators + Schedule — always shown ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🌡️ Plant Stress Indicators")
        for sym, diag, bc in [
            ("🟡 Yellowing Leaves", "Nutrient Deficiency", "#fef3c7;color:#92400e"),
            ("🟤 Brown Spots",      "Fungal/Bacterial",    "#fee2e2;color:#b91c1c"),
            ("⬛ Black Lesions",    "Severe Infection",    "#fee2e2;color:#b91c1c"),
            ("⚪ White Powder",     "Powdery Mildew",      "#fef3c7;color:#92400e"),
            ("🟢 Uniform Green",   "Healthy",             "#dcfce7;color:#15803d"),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:7px 0;border-bottom:1px solid #f3f4f6">
              <span style="font-size:13px">{sym}</span>
              <span style="background:{bc};padding:2px 10px;border-radius:99px;
                           font-size:11px;font-weight:700">{diag}</span>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("#### 📅 Monitoring Schedule")
        for stage, freq, fc in [
            ("🌱 Seedling Stage",   "Every 3 days", "#15803d"),
            ("🌿 Vegetative Stage", "Every 7 days", "#15803d"),
            ("🌸 Flowering Stage",  "Every 5 days", "#d97706"),
            ("🍎 Fruiting Stage",   "Every 5 days", "#d97706"),
            ("🌧️ After Rain",       "Immediately",  "#dc2626"),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:7px 0;border-bottom:1px solid #f3f4f6">
              <span style="font-size:13px">{stage}</span>
              <span style="font-weight:700;font-size:13px;color:{fc}">{freq}</span>
            </div>""", unsafe_allow_html=True)

    # ── Crop health summary chart ──
    if my_scans:
        st.markdown("---")
        st.markdown("#### 📈 Health vs Disease Summary")
        healthy_c = sum(1 for h in my_scans if h.get('healthy'))
        sick_c    = len(my_scans) - healthy_c

        fig = go.Figure(go.Pie(
            labels=['Healthy', 'Diseased'],
            values=[healthy_c, sick_c],
            hole=0.5,
            marker_colors=['#22c55e', '#dc2626'],
            textinfo='label+percent',
            hoverinfo='label+value',))
        fig.update_layout(
            showlegend=True,
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            template='plotly_white',
        )
        st.plotly_chart(fig, use_container_width=True)

#  IMAGE COMPARE  
def render_compare():
    st.markdown("""
    <div class="plantai-header">
      <h1>🔀 Image Comparison</h1>
      <p>Compare two leaf images side-by-side with AI analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("Upload two leaf images to compare them side by side and run AI diagnosis on both.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="compare-label">📷 IMAGE A — First Leaf</div>
        """, unsafe_allow_html=True)
        img_a = st.file_uploader("Upload Image A", type=['jpg', 'jpeg', 'png'], key="compare_a",
                                  label_visibility="collapsed")
    with col_b:
        st.markdown("""
        <div class="compare-label">📷 IMAGE B — Second Leaf</div>
        """, unsafe_allow_html=True)
        img_b = st.file_uploader("Upload Image B", type=['jpg', 'jpeg', 'png'], key="compare_b",
                                  label_visibility="collapsed")

    if not img_a or not img_b:
        st.markdown("""
        <div style="background:#f0fdf4;border:2px dashed #bbf7d0;border-radius:14px;
                    padding:2rem;text-align:center;margin-top:1rem">
          <div style="font-size:40px;margin-bottom:10px">🔀</div>
          <div style="font-size:15px;font-weight:700;color:#0B3D2E">Upload both images to compare</div>
          <div style="font-size:13px;color:#9ca3af;margin-top:6px">
            You can compare healthy vs diseased leaves, or track disease progression
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    pil_a = Image.open(img_a).convert('RGB')
    pil_b = Image.open(img_b).convert('RGB')

    # ── Side by side preview ──
    st.markdown("#### 🖼️ Side-by-Side View")
    prev_a, prev_b = st.columns(2)
    with prev_a:
        st.image(pil_a, caption=f"Image A: {img_a.name}", use_container_width=True)
    with prev_b:
        st.image(pil_b, caption=f"Image B: {img_b.name}", use_container_width=True)

    # ── Image info comparison ──
    st.markdown("#### 📐 Image Properties")
    prop_a, prop_b = st.columns(2)
    with prop_a:
        st.markdown(f"""
        <div class="compare-container">
          <div class="compare-label">IMAGE A</div>
          <div style="font-size:13px;color:#374151;line-height:2">
            📁 <b>File:</b> {img_a.name}<br>
            📏 <b>Size:</b> {pil_a.width} × {pil_a.height} px<br>
            🎨 <b>Mode:</b> {pil_a.mode}<br>
            💾 <b>File Size:</b> {img_a.size / 1024:.1f} KB
          </div>
        </div>
        """, unsafe_allow_html=True)
    with prop_b:
        st.markdown(f"""
        <div class="compare-container">
          <div class="compare-label">IMAGE B</div>
          <div style="font-size:13px;color:#374151;line-height:2">
            📁 <b>File:</b> {img_b.name}<br>
            📏 <b>Size:</b> {pil_b.width} × {pil_b.height} px<br>
            🎨 <b>Mode:</b> {pil_b.mode}<br>
            💾 <b>File Size:</b> {img_b.size / 1024:.1f} KB
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Pixel-level colour analysis ──
    st.markdown("#### 🎨 Colour Analysis")
    arr_a = np.array(pil_a.resize((224, 224)))
    arr_b = np.array(pil_b.resize((224, 224)))

    def channel_means(arr):
        return arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()

    ra, ga, ba = channel_means(arr_a)
    rb, gb, bb = channel_means(arr_b)

    col1, col2, col3 = st.columns(3)
    for col, ch, va, vb, color in [
        (col1, "🔴 Red",   ra, rb, "#dc2626"),
        (col2, "🟢 Green", ga, gb, "#16a34a"),
        (col3, "🔵 Blue",  ba, bb, "#1d4ed8"),
    ]:
        with col:
            diff = abs(va - vb)
            st.markdown(f"""
            <div class="compare-container" style="text-align:center">
              <div class="compare-label">{ch} Channel</div>
              <div style="display:flex;justify-content:space-around;margin:8px 0">
                <div>
                  <div style="font-size:11px;color:#9ca3af">Image A</div>
                  <div style="font-size:20px;font-weight:800;color:{color}">{va:.1f}</div>
                </div>
                <div style="font-size:20px;color:#9ca3af;padding-top:12px">vs</div>
                <div>
                  <div style="font-size:11px;color:#9ca3af">Image B</div>
                  <div style="font-size:20px;font-weight:800;color:{color}">{vb:.1f}</div>
                </div>
              </div>
              <div style="font-size:11px;color:#9ca3af">Difference: <b style="color:#0B3D2E">{diff:.1f}</b></div>
            </div>
            """, unsafe_allow_html=True)

    # ── Green index (plant health proxy) ──
    st.markdown("#### 🌿 Green Index (Plant Health Proxy)")
    gi_a = ga / (ra + ga + ba + 1e-5) * 100
    gi_b = gb / (rb + gb + bb + 1e-5) * 100
    gi_col_a, gi_col_b = st.columns(2)
    for col, label, gi in [(gi_col_a, "Image A", gi_a), (gi_col_b, "Image B", gi_b)]:
        with col:
            bar_col = "#22c55e" if gi > 35 else ("#d97706" if gi > 25 else "#dc2626")
            health_label = "Likely Healthy 🌿" if gi > 35 else ("Moderate ⚠️" if gi > 25 else "Low Green / Diseased ❌")
            st.markdown(f"""
            <div class="compare-container">
              <div class="compare-label">{label} — Green Index</div>
              <div style="font-size:32px;font-weight:900;color:{bar_col};margin:8px 0">{gi:.1f}%</div>
              <div style="background:#e5e7eb;border-radius:99px;height:10px;overflow:hidden;margin-bottom:8px">
                <div style="width:{min(gi*2, 100):.1f}%;height:100%;border-radius:99px;background:{bar_col}"></div>
              </div>
              <div style="font-size:12px;font-weight:700;color:{bar_col}">{health_label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── AI Analysis on both ──
    st.markdown("---")
    st.markdown("#### 🤖 Run AI Diagnosis on Both Images")

    if st.button("🔬 Analyze Both Images with AI", key="btn_compare_analyze", use_container_width=True):
        if st.session_state.model is None:
            with st.spinner("Loading AI model..."):
                model_path = st.session_state.get('model_path', 'plantvillage_efficientnet_b0.pth')
                st.session_state.model = load_model(model_path)
        with st.spinner("🔬 Analyzing both images..."):
            enh_a    = enhance_image(pil_a)
            enh_b    = enhance_image(pil_b)
            results_a = predict(enh_a, st.session_state.model)
            results_b = predict(enh_b, st.session_state.model)
        top_a = results_a[0]
        top_b = results_b[0]
        res_col_a, res_col_b = st.columns(2)
        for col, label, top in [(res_col_a, "Image A", top_a), (res_col_b, "Image B", top_b)]:
            with col:
                is_h  = top['healthy']
                bg    = "#f0fdf4" if is_h else "#fff5f5"
                icon  = "✅" if is_h else "⚠️"
                color = "#16a34a" if is_h else "#dc2626"
                st.markdown(f"""
                <div class="compare-container" style="background:{bg}">
                  <div class="compare-label">{label} Result</div>
                  <div style="font-size:28px;margin:6px 0">{icon}</div>
                  <div style="font-size:14px;font-weight:800;color:{color};margin-bottom:4px">{top['label']}</div>
                  <div style="font-size:13px;color:#9ca3af">
                    Confidence: <b style="color:#0B3D2E">{top['conf']*100:.1f}%</b><br>
                    Severity: <b style="color:{color}">{top['severity']}</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Comparison summary ──
        st.markdown("#### 📊 Diagnosis Comparison")
        same_disease = top_a['raw'] == top_b['raw']
        both_healthy = top_a['healthy'] and top_b['healthy']
        any_diseased = not top_a['healthy'] or not top_b['healthy']
        conf_diff    = abs(top_a['conf'] - top_b['conf']) * 100
        if both_healthy:
            st.success("✅ Both plants appear **healthy**! Great crop health detected.")
        elif same_disease:
            st.warning(f"⚠️ Both images show the **same disease**: {top_a['label']}. Consider field-wide treatment.")
        else:
            msg_parts = []
            if not top_a['healthy']:
                msg_parts.append(f"Image A: **{top_a['label']}** ({top_a['severity']} severity)")
            if not top_b['healthy']:
                msg_parts.append(f"Image B: **{top_b['label']}** ({top_b['severity']} severity)")
            st.error("🔴 Different diseases detected:\n" + "\n".join(msg_parts))

        st.markdown(f"""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px">
          <span class="diff-badge" style="background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0">
            Conf Δ: {conf_diff:.1f}%
          </span>
          <span class="diff-badge" style="background:{'#f0fdf4' if same_disease else '#fff5f5'};
                color:{'#15803d' if same_disease else '#b91c1c'};
                border:1px solid {'#bbf7d0' if same_disease else '#fecaca'}">
            {'Same Disease' if same_disease else 'Different Diseases'}
          </span>
          <span class="diff-badge" style="background:#f0fdf4;color:#0B3D2E;border:1px solid #bbf7d0">
            Green Index Δ: {abs(gi_a - gi_b):.1f}%
          </span>
        </div>
        """, unsafe_allow_html=True)

#  CHATBOT
STATIC_BOT = {
    'apple scab':  '🍎 Apple Scab is caused by fungus *Venturia inaequalis*. Treat with Mancozeb (2 g/L) or Neem oil (5 ml/L). Apply every 10–14 days.',
    'black rot':   '🍇 Black Rot is caused by *Guignardia bidwellii*. Remove infected parts, spray Captan 50% WP (2 g/L). Maintain orchard hygiene.',
    'late blight': '🥔 Late Blight (*Phytophthora infestans*) — use Ridomil Gold (2.5 g/L). Destroy infected plants immediately.',
    'fertilizer':  '🌱 For most crops: NPK 10-26-26 at planting, then urea top-dress. Organic: compost + vermicompost works great!',
    'prevention':  '🛡️ Key tips: 1) Remove infected leaves, 2) Improve air circulation, 3) Avoid overhead watering, 4) Apply fungicide before monsoon.',
    'neem':        '🌿 Neem oil (5 ml/L + 1 ml/L soap) is an excellent organic fungicide and insecticide. Apply weekly in the morning.',
    'irrigation':  '💧 Water early morning (6–8 AM). Use drip irrigation. Avoid wet foliage. Check soil moisture before watering.',
}

def render_chatbot():
    st.markdown("""
    <div class="plantai-header">
      <h1>💬 Plant Care Chatbot</h1>
      <p>Ask me anything about plant diseases, treatments, or farming tips</p>
    </div>
    """, unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {'role': 'bot', 'msg': "👋 Hello! I'm your PlantAI assistant. Ask me about plant diseases, treatments, fertilizers, or any farming questions!"},
            {'role': 'bot', 'msg': "💡 Try: *'What causes Apple Scab?'* or *'How to prevent Late Blight?'* or *'Best fertilizer for tomatoes?'*"},
        ]

    st.markdown("**Quick questions:**")
    qcols = st.columns(4)
    quick = [
        ("Apple Scab",  "How to treat Apple Scab?"),
        ("Black Rot",   "What causes Black Rot?"),
        ("Fertilizer",  "Best organic fertilizer for vegetables?"),
        ("Prevention",  "How to prevent plant diseases?"),
    ]
    for col, (label, query) in zip(qcols, quick):
        with col:
            if st.button(label, key=f"quick_{label}"):
                st.session_state.chat_history.append({'role': 'user', 'msg': query})
                if st.session_state.api_key:
                    try:
                        genai.configure(api_key=st.session_state.api_key)
                        m = genai.GenerativeModel('gemini-3-flash-preview')
                        resp = m.generate_content(
                            f"You are a friendly plant care assistant for Indian farmers. "
                            f"Answer briefly and practically: {query}"
                        )
                        st.session_state.chat_history.append({'role': 'bot', 'msg': resp.text})
                    except Exception as e:
                        st.session_state.chat_history.append({'role': 'bot', 'msg': f"❌ {e}"})
                else:
                    reply = "🤖 Add your Gemini API key in the sidebar for AI-powered answers!"
                    for k, v in STATIC_BOT.items():
                        if k in query.lower():
                            reply = v; break
                    st.session_state.chat_history.append({'role': 'bot', 'msg': reply})
                st.rerun()
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;margin:6px 0">
              <div style="background:linear-gradient(135deg,#14532d,#16a34a);color:#f0fdf4;padding:10px 14px;
                          border-radius:14px 14px 4px 14px;max-width:75%;font-size:13px">
                {msg['msg']}
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-start;margin:6px 0">
              <div style="background:#fff;border:1px solid #bbf7d0;border-left:3px solid #16a34a;padding:10px 14px;
                          border-radius:14px 14px 14px 4px;max-width:75%;font-size:13px">
                {msg['msg']}
              </div>
            </div>""", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        inp_col, btn_col = st.columns([5, 1])
        with inp_col:
            user_msg = st.text_input("", placeholder="Ask about plant diseases, treatments...",
                                     label_visibility="collapsed")
        with btn_col:
            submitted = st.form_submit_button("Send")
    if submitted and user_msg.strip():
        st.session_state.chat_history.append({'role': 'user', 'msg': user_msg.strip()})
        msg = user_msg.strip()
        if st.session_state.api_key:
            try:
                genai.configure(api_key=st.session_state.api_key)
                m = genai.GenerativeModel('gemini-3-flash-preview')
                resp = m.generate_content(
                    f"You are a friendly plant care assistant for Indian farmers. "
                    f"Answer briefly and practically: {msg}"
                )
                reply = resp.text
            except Exception as e:
                reply = f"❌ {e}"
        else:
            reply = "🤖 Add your Gemini API key in the sidebar for AI-powered answers!"
            for k, v in STATIC_BOT.items():
                if k in msg.lower():
                    reply = v; break
        st.session_state.chat_history.append({'role': 'bot', 'msg': reply})
        st.rerun()

#  WEATHER
def render_weather():
    st.markdown("""
    <div class="plantai-header">
      <h1>🌦 Weather & Disease Alerts</h1>
      <p>Weather-based disease risk and seasonal alerts</p>
    </div>
    """, unsafe_allow_html=True)
    col_w, col_r = st.columns(2)
    with col_w:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0B3D2E,#1a5c38);
                    border-top:3px solid #22c55e;
                    border-radius:14px;padding:1.5rem;color:#fff">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <div style="font-size:48px;font-weight:800">38°C</div>
              <div style="font-size:13px;opacity:.8;margin-top:4px">Sunny · High Humidity</div>
              <div style="font-size:13px;opacity:.8">Ahmedabad, Gujarat</div>
            </div>
            <div style="font-size:48px">🌞</div>
          </div>
          <div style="margin-top:1rem;display:flex;gap:16px">
            <div><div style="font-size:20px;font-weight:700;color:#86efac">72%</div>
                 <div style="font-size:11px;opacity:.7">Humidity</div></div>
            <div><div style="font-size:20px;font-weight:700;color:#86efac">12 km/h</div>
                 <div style="font-size:11px;opacity:.7">Wind</div></div>
            <div><div style="font-size:20px;font-weight:700;color:#86efac">High</div>
                 <div style="font-size:11px;opacity:.7">UV Index</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        st.markdown("#### 🌡️ Disease Risk Today")
        st.markdown("""
        <div style="border-left:4px solid #dc2626;background:#fef2f2;
                    border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:10px">
          <b style="color:#dc2626">🔴 HIGH RISK</b> — Apple Scab, Black Rot<br>
          <span style="font-size:12px;color:#4b5563">High humidity (72%) + warm temperature = ideal conditions for fungal growth</span>
        </div>
        <div style="border-left:4px solid #16a34a;background:#f0fdf4;
                    border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:10px">
          <b style="color:#14532d">🟡 MODERATE</b> — Powdery Mildew<br>
          <span style="font-size:12px;color:#4b5563">Dry winds may spread spores. Monitor plants closely.</span>
        </div>
        <div style="border-left:4px solid #0B3D2E;background:#f0fdf4;
                    border-radius:0 10px 10px 0;padding:12px 16px">
          <b style="color:#0B3D2E">🟢 LOW RISK</b> — Bacterial Canker<br>
          <span style="font-size:12px;color:#4b5563">Conditions not favorable today.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 📅 Seasonal Disease Trends")
    df_seasons = pd.DataFrame({
        'Season':   ['🌷 Spring', '🌇 Summer', '⛈️ Monsoon', '🪸 Winter'],
        'Disease':  ['Apple Scab, Fire Blight', 'Black Rot, Powdery Mildew',
                     'All fungal diseases peak', 'Cedar Apple Rust'],
        'Risk':     ['High', 'Medium', 'Very High', 'Low'],
        'Action':   ['Apply copper fungicide before bud break',
                     'Ensure good air circulation, reduce humidity',
                     'Weekly fungicide spray, drainage management',
                     'Prune and dispose infected branches'],
    })
    st.dataframe(df_seasons, use_container_width=True, hide_index=True)

#  FERTILIZER
FERT_DATA = {
    '🍎 Apple':  ('NPK 12-32-16 at planting. Urea 100 g/tree after fruit set. Calcium nitrate before harvest.',
                  'Drip irrigation. 4–6 L/tree/day in summer. Avoid waterlogging.',
                  'Watch for aphids, codling moth. Use sticky traps.'),
    '🍅 Tomato': ('NPK 19-19-19 at transplant. Potassium boost during fruiting. Calcium spray to prevent blossom end rot.',
                  'Regular, consistent watering. Avoid wet foliage. 2–3 L/plant/day.',
                  'Watch for whitefly, leaf miners. Use yellow sticky traps.'),
    '🌾 Wheat':  ('Urea 60 kg/acre at sowing. DAP 50 kg/acre. Top-dress urea at tillering stage.',
                  'Crown root initiation (21 days), tillering, jointing, grain filling.',
                  'Watch for aphids, termites. Seed treatment with fungicide.'),
    '🍚 Rice':   ('Urea 40 kg/acre basal. Top-dress at tillering and panicle initiation.',
                  'Maintain 2–5 cm water depth during vegetative stage.',
                  'Watch for stem borer, brown plant hopper. Regular field monitoring.'),
    '☁️ Cotton': ('NPK 20-10-10 at planting. Foliar spray of boron at square stage.',
                  'Critical at square formation and boll development. Avoid excess water.',
                  'Watch for bollworm, whitefly, jassids. Use Bt-based pesticides.'),
}

def render_fertilizer():
    st.markdown("""
    <div class="plantai-header">
      <h1>🌱 Fertilizer & Irrigation Guide</h1>
      <p>Smart recommendations for your crops</p>
    </div>
    """, unsafe_allow_html=True)
    crop = st.selectbox("🌾 Select Your Crop", list(FERT_DATA.keys()))
    fert, irr, pest = FERT_DATA[crop]

    st.markdown(f"#### {crop}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🌿 Fertilizer**")
        st.write(fert)
    with c2:
        st.markdown("**💧 Irrigation**")
        st.write(irr)
    with c3:
        st.markdown("**🦗 Pest Watch**")
        st.write(pest)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 💧 Irrigation Tips")
        for tip in ["Water plants early morning (6–8 AM)",
                    "Avoid overwatering — check soil moisture first",
                    "Use drip irrigation for efficient water use",
                    "During disease — reduce leaf wetness",
                    "Mulching retains moisture and prevents splash"]:
            st.markdown(f"• {tip}")
    with col_r:
        st.markdown("#### 🦗 Pest Detection Tips")
        for tip in ["Check undersides of leaves for insects",
                    "Aphids: sticky residue + curling leaves",
                    "Whitefly: white cloud when plant disturbed",
                    "Mites: fine webbing on leaves",
                    "Use yellow sticky traps for monitoring"]:
            st.markdown(f"• {tip}")

#  REVIEWS
def render_reviews():
    st.markdown("""
    <div class="plantai-header">
      <h1>⭐ Farmer Reviews</h1>
      <p>Share your experience with PlantAI</p>
    </div>
    """, unsafe_allow_html=True)
    revs      = st.session_state.reviews
    avg       = sum(r['rating'] for r in revs) / len(revs) if revs else 0
    five_star = sum(1 for r in revs if r['rating'] == 5)
    c1, c2, c3 = st.columns(3)
    for col, icon, val, lbl in [
        (c1, "💬", len(revs),      "Reviews"),
        (c2, "⭐", f"{avg:.1f}",   "Avg Rating"),
        (c3, "🏆", five_star,      "5★ Reviews"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-box">
              <div style="font-size:24px">{icon}</div>
              <div class="val">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### ⭐ Give Feedback")
    rating  = st.radio("Select Rating", options=[1,2,3,4,5],
                       format_func=lambda x: "⭐" * x, horizontal=True)
    comment = st.text_area("Comment", placeholder="Share your experience with PlantAI...", height=90)
    if st.button("Submit Review ⭐", key="submit_rev"):
        if comment.strip():
            now = datetime.datetime.now()
            review = {
                'user':    st.session_state.user_name,
                'rating':  rating,
                'comment': comment.strip(),
                'time':    now.strftime('%d-%m-%Y %I:%M %p')
            }
            insert_review_db(review)
            st.session_state.reviews = load_reviews_db()
            st.success(f"✅ Review submitted! {'⭐'*rating}")
            st.rerun()
        else:
            st.warning("Please write a comment!")

    st.markdown("#### 📢 All Reviews")
    for r in revs:
        stars = '★' * r['rating'] + '☆' * (5 - r['rating'])
        st.markdown(f"""
        <div style="background:#fff;border-left:4px solid #16a34a;
                    border-radius:0 14px 14px 0;padding:1.1rem 1.3rem;
                    margin-bottom:1rem;box-shadow:0 2px 8px rgba(22,163,74,0.08)">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
            <span style="font-weight:700;font-size:14px;color:#0B3D2E">👤 {r['user']}</span>
            <span style="color:#16a34a;font-size:14px">{stars}</span>
          </div>
          <div style="font-size:11px;color:#9ca3af;margin-bottom:8px">⏱ {r['time']}</div>
          <div style="font-size:13px;color:#4b5563;line-height:1.6">{r['comment']}</div>
        </div>
        """, unsafe_allow_html=True)

#  ABOUT
def render_about():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0B3D2E,#1a5c38);
                border-top:4px solid #22c55e;
                border-radius:16px;padding:2.5rem;text-align:center;margin-bottom:1.5rem">
      <h2 style="font-size:2rem;font-weight:900;color:#fff;margin-bottom:8px">🌿 PlantAI v3.0</h2>
      <p style="color:rgba(134,239,172,0.85);font-size:14px">
        AI-Powered Plant Disease Detection System — MSc IT Capstone Project
      </p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, name, desc in [
        (c1, "🖥️", "Python + Streamlit",   "Web Interface & Dashboard"),
        (c2, "👁️", "PyTorch + TorchVision", "Disease Detection Model"),
        (c3, "💠", "Google Gemini 3",       "AI Treatment Solutions"),
        (c4, "🐼", "Plotly + Pandas",       "Analytics & Visualisations"),
    ]:
        with col:
            st.markdown(f"""
            <div class="plant-card" style="text-align:center">
              <div style="font-size:28px">{icon}</div>
              <div style="font-size:13px;font-weight:700;color:#1a1a1a">{name}</div>
              <div style="font-size:11px;color:#9ca3af">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### ✨ Features Included")
    features = [
        "✅ 38-class PlantVillage Disease Classification",
        "✅ EfficientNet-B0 Model",
        "✅ Auto Image Enhancement Pipeline",
        "✅ Severity Level Indicator",
        "✅ Google Gemini AI Treatment Plans",
        "✅ User Login & Registration",
        "✅ Scan History Tracking",
        "✅ Export PDF / CSV Reports",
        "✅ Interactive Analytics Dashboard",
        "✅ Plotly Charts & Visualisations",
        "✅ Weather-Based Disease Alerts",
        "✅ Fertilizer & Irrigation Guide",
        "✅ Plant Care Chatbot (Gemini + offline)",
        "✅ Farmer Reviews System",
        "✅ Image Comparison Tool",
        "✅ Mobile Responsive Layout",
        "✅ SQLite3 Database Storage",
    ]
    c1, c2 = st.columns(2)
    half = len(features) // 2
    with c1:
        for f in features[:half]: st.markdown(f)
    with c2:
        for f in features[half:]: st.markdown(f)
        
    st.markdown("""
<div class="plant-card" style="text-align:center;
background:linear-gradient(135deg,#E8F5E9,#F5FFF5);
padding:14px;border-radius:12px">
<p style="font-size:13px;color:#4b5563;margin:0;line-height:1.6">

Built for 🇮🇳🌾 Indian Farmers<br>

PlantVillage Dataset (Hughes & Salathé, 2015) · 38 Disease Classes<br>

EfficientNet-B0 (Fine-Tuned) · Gemini 3 Flash · Streamlit · SQLite3
</p>
</div>
""", unsafe_allow_html=True)

#  MAIN ROUTER  
def main():
    if not st.session_state.logged_in:
        render_auth()
        return
    render_sidebar() 
    page = st.session_state.get('page', 'dashboard')
    dispatch = {
        'dashboard':   render_dashboard,
        'predict':     render_predict,
        'history':     render_history,
        'analytics':   render_analytics,
        'cropmonitor': render_cropmonitor,   
        'chatbot':     render_chatbot,
        'compare':     render_compare,       
        'weather':     render_weather,
        'fertilizer':  render_fertilizer,
        'reviews':     render_reviews,
        'about':       render_about,
    }
    dispatch.get(page, render_dashboard)()
if __name__ == "__main__":
    main()