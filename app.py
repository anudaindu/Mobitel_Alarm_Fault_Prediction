import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration - Clean White Enterprise Theme
st.set_page_config(
    page_title="4G Telecom Tower Outage Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise White CSS (Zero Emojis, Clean Border Lines, Slate Containers)
st.markdown("""
    <style>
    /* Main App Background & Typography */
    .stApp {
        background-color: #FFFFFF;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Global Sidebar Style */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Header Container */
    .enterprise-header {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .enterprise-title {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .enterprise-subtitle {
        font-size: 14px;
        color: #64748B;
        margin-top: 4px;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card-white {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px;
        text-align: left;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card-value {
        font-size: 28px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .metric-card-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Risk Badges */
    .badge-critical {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-warning {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-nominal {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
    }

    /* BBU Hardware Layout Slot Cards */
    .bbu-container {
        background-color: #F8FAFC;
        border: 2px solid #CBD5E1;
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
    }
    .slot-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .slot-card-active {
        background-color: #EFF6FF;
        border: 1px solid #93C5FD;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .slot-title {
        font-size: 13px;
        font-weight: 700;
        color: #1E3A8A;
    }
    .slot-desc {
        font-size: 12px;
        color: #475569;
    }

    /* Streamlit Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        color: #64748B;
        font-size: 14px;
        font-weight: 600;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        color: #0F172A !important;
    }
    </style>
""", unsafe_allow_html=True)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(WORKSPACE_DIR, 'telecom_features_july_2026.csv')
MODEL_PATH = os.path.join(WORKSPACE_DIR, 'models', 'calibrated_xgboost_outage.pkl')
SUBSET_PATH = os.path.join(WORKSPACE_DIR, 'july_2026_subset_dataset.csv')

@st.cache_data
def load_feature_matrix():
    if os.path.exists(FEATURES_PATH):
        df = pd.read_csv(FEATURES_PATH)
        df['window_timestamp'] = pd.to_datetime(df['window_timestamp'])
        return df
    return pd.DataFrame()

@st.cache_data
def load_raw_subset():
    if os.path.exists(SUBSET_PATH):
        df = pd.read_csv(SUBSET_PATH, low_memory=False)
        df['event_time'] = pd.to_datetime(df['Occurred On (NT)'])
        return df
    return pd.DataFrame()

@st.cache_resource
def load_calibrated_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

features_df = load_feature_matrix()
raw_subset_df = load_raw_subset()
model = load_calibrated_model()

# Enterprise Header
st.markdown("""
    <div class="enterprise-header">
        <h1 class="enterprise-title">4G Telecom Tower Outage Prediction System</h1>
        <p class="enterprise-subtitle">Enterprise Machine Learning Operations Platform for Early Warning Cell Unavailability Outage Management</p>
    </div>
""", unsafe_allow_html=True)

# Navigation Tabs (Zero Emojis)
tab1, tab2, tab3 = st.tabs([
    "Highest Risk Towers (Operational Queue)",
    "eNodeB Tower Details & Hardware",
    "Model Performance & Validation"
])

# --- TAB 1: HIGHEST RISK TOWERS (OPERATIONAL QUEUE) ---
with tab1:
    if not features_df.empty and model is not None:
        # Obtain latest prediction window slice across all sites
        latest_timestamp = features_df['window_timestamp'].max()
        latest_df = features_df[features_df['window_timestamp'] == latest_timestamp].copy()
        
        feature_cols = ['total_alarms_6h', 'critical_alarms_6h', 'major_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'total_alarms_24h']
        latest_df['failure_probability'] = model.predict_proba(latest_df[feature_cols])[:, 1]
        
        latest_df.sort_values(by='failure_probability', ascending=False, inplace=True)
        
        # Risk Classification
        latest_df['risk_status'] = latest_df['failure_probability'].apply(
            lambda p: 'CRITICAL' if p >= 0.65 else ('WARNING' if p >= 0.35 else 'NOMINAL')
        )
        
        total_eval_towers = len(latest_df)
        critical_count = (latest_df['risk_status'] == 'CRITICAL').sum()
        warning_count = (latest_df['risk_status'] == 'WARNING').sum()
        hist_outage_rate = (features_df['target_outage_next_2h'].mean() * 100)
        
        # Summary Metric Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card-white"><div class="metric-card-label">Evaluated Towers</div><div class="metric-card-value">{total_eval_towers}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card-white"><div class="metric-card-label">Critical Risk Count (>=65%)</div><div class="metric-card-value" style="color: #991B1B;">{critical_count}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card-white"><div class="metric-card-label">Warning Risk Count (35-64%)</div><div class="metric-card-value" style="color: #92400E;">{warning_count}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card-white"><div class="metric-card-label">Historical Outage Rate</div><div class="metric-card-value">{hist_outage_rate:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Prioritized Operational Outage Queue (Next 2-Hour Lookahead)")
        st.write(f"Prediction Window Timestamp: **{latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}**")
        
        # Format table columns for presentation
        queue_table = latest_df[[
            'site_id', 'failure_probability', 'risk_status',
            'critical_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h',
            'major_alarms_6h', 'total_alarms_6h', 'total_alarms_24h'
        ]].copy()
        
        queue_table['failure_probability'] = (queue_table['failure_probability'] * 100).map('{:.1f}%'.format)
        queue_table.rename(columns={
            'site_id': 'Site ID (eNodeB)',
            'failure_probability': 'Outage Probability',
            'risk_status': 'Risk Status',
            'critical_alarms_6h': 'Critical Alarms (6h)',
            'rru_alarms_6h': 'RRU Alarms (6h)',
            'bbu_alarms_6h': 'BBU Alarms (6h)',
            'major_alarms_6h': 'Major Alarms (6h)',
            'total_alarms_6h': 'Total Alarms (6h)',
            'total_alarms_24h': 'Total Alarms (24h)'
        }, inplace=True)

        # Highlight Critical Towers
        def highlight_critical(row):
            if row['Risk Status'] == 'CRITICAL':
                return ['background-color: #FEF2F2; font-weight: bold; color: #991B1B'] * len(row)
            elif row['Risk Status'] == 'WARNING':
                return ['background-color: #FFFBEB; font-weight: bold; color: #92400E'] * len(row)
            return [''] * len(row)

        st.dataframe(queue_table.style.apply(highlight_critical, axis=1), height=420, width='stretch')
    else:
        st.error("Engineered feature matrix or calibrated model artifact missing.")

# --- TAB 2: ENODEB TOWER DETAILS & HARDWARE ---
with tab2:
    if not features_df.empty:
        site_list = sorted(features_df['site_id'].unique().tolist())
        selected_site = st.selectbox("Select 4G Tower Site ID (eNodeB ID):", site_list)
        
        site_features = features_df[features_df['site_id'] == selected_site].sort_values('window_timestamp')
        latest_site_row = site_features.iloc[-1]
        
        feature_cols = ['total_alarms_6h', 'critical_alarms_6h', 'major_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'total_alarms_24h']
        input_data = pd.DataFrame([latest_site_row[feature_cols].to_dict()])
        prob = model.predict_proba(input_data)[0, 1] if model is not None else 0.0
        
        col_site_1, col_site_2 = st.columns([1, 2])
        with col_site_1:
            st.markdown("### Site Live Risk Assessment")
            st.markdown(f'<div class="metric-card-white"><div class="metric-card-label">Target Site</div><div class="metric-card-value">{selected_site}</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card-white"><div class="metric-card-label">Predicted Outage Probability</div><div class="metric-card-value">{prob*100:.1f}%</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if prob >= 0.65:
                st.markdown('<div class="badge-critical" style="font-size: 14px; width: 100%; text-align: center;">CRITICAL OUTAGE RISK DETECTED</div>', unsafe_allow_html=True)
            elif prob >= 0.35:
                st.markdown('<div class="badge-warning" style="font-size: 14px; width: 100%; text-align: center;">WARNING OUTAGE RISK DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-nominal" style="font-size: 14px; width: 100%; text-align: center;">NOMINAL OPERATIONAL STATUS</div>', unsafe_allow_html=True)

        with col_site_2:
            st.markdown("### Huawei BBU Subrack Hardware Layout Mapping")
            st.markdown("""
                <div class="bbu-container">
                    <div style="font-weight: 700; font-size: 14px; color: #0F172A; margin-bottom: 12px;">BBU3900 / BBU3910 Subrack Slot Assignment Overview</div>
                    <div class="slot-card-active">
                        <div class="slot-title">Slot 7: UMPT Board (Main Control & Transmission)</div>
                        <div class="slot-desc">Main control unit handling S1/X2 interface signaling, IP path routing, and system clock synchronization.</div>
                    </div>
                    <div class="slot-card-active">
                        <div class="slot-title">Slots 2 / 3: UBBP Boards (Universal Baseband Processing Unit)</div>
                        <div class="slot-desc">LTE baseband processing unit handling physical channel encoding, CPRI interface optical links to RRU.</div>
                    </div>
                    <div class="slot-card">
                        <div class="slot-title">Slot 16: FAN Unit (Forced Air Cooling)</div>
                        <div class="slot-desc">Subrack ventilation fan tray regulating thermal dissipation and temperature thresholds.</div>
                    </div>
                    <div class="slot-card-active">
                        <div class="slot-title">Slots 18 / 19: UPEU Power Module (Universal Power & Environment Extension)</div>
                        <div class="slot-desc">Converts -48V DC power supply, provides environmental monitoring and external alarm input interfaces.</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Historical Raw Alarm Event Stream (Selected Tower)")
        
        if not raw_subset_df.empty:
            tower_alarms = raw_subset_df[raw_subset_df['site_id'] == selected_site].sort_values('event_time', ascending=False)
            display_cols = ['event_time', 'Severity', 'alarm_name', 'MO Name', 'Location Information']
            present_cols = [c for c in display_cols if c in tower_alarms.columns]
            st.dataframe(tower_alarms[present_cols].head(100), height=350, width='stretch')
        else:
            st.info("Raw subset alarm logs not loaded.")

# --- TAB 3: MODEL PERFORMANCE & VALIDATION ---
with tab3:
    st.subheader("Holdout Model Validation & Feature Importance")
    st.write("Validation Period: **July 23, 2026 – July 31, 2026** (3,780 Holdout Test Samples)")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown('<div class="metric-card-white"><div class="metric-card-label">PR-AUC Score</div><div class="metric-card-value">0.6760</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown('<div class="metric-card-white"><div class="metric-card-label">ROC-AUC Score</div><div class="metric-card-value">0.8902</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown('<div class="metric-card-white"><div class="metric-card-label">Precision (Thresh 0.5)</div><div class="metric-card-value">0.7948</div></div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown('<div class="metric-card-white"><div class="metric-card-label">Recall (Thresh 0.5)</div><div class="metric-card-value">0.4513</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_fig1, c_fig2 = st.columns(2)
    with c_fig1:
        st.write("### Feature Importance Scores (PR-AUC Impact)")
        imp_data = pd.DataFrame({
            'Feature': ['critical_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'major_alarms_6h', 'total_alarms_24h', 'total_alarms_6h'],
            'Importance': [0.3847, 0.2763, 0.0675, 0.0472, 0.0452, 0.0162]
        }).sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(imp_data, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues', template='plotly_white')
        fig_imp.update_layout(showlegend=False, xaxis_title="PR-AUC Permutation Impact", yaxis_title="Feature")
        st.plotly_chart(fig_imp, use_container_width=True)

    with c_fig2:
        st.write("### Calibrated Outage Probability Distribution")
        if not features_df.empty and model is not None:
            feature_cols = ['total_alarms_6h', 'critical_alarms_6h', 'major_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'total_alarms_24h']
            probs = model.predict_proba(features_df[feature_cols])[:, 1]
            fig_hist = px.histogram(probs, nbins=50, labels={'value': 'Predicted Outage Probability'}, color_discrete_sequence=['#1E3A8A'], template='plotly_white')
            fig_hist.update_layout(xaxis_title="Calibrated Failure Probability", yaxis_title="Sample Count")
            st.plotly_chart(fig_hist, use_container_width=True)
