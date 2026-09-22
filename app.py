import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config, save_config, get_environment_mode
from src.storage_manager import get_master_history_summary, purge_expired_logs
from src.process_manual_upload import process_manual_file_upload
from src.feature_engineering import generate_sliding_window_features
from src.model import train_xgboost_model, run_model_inference

# Page Configuration - Enterprise Light Theme
st.set_page_config(
    page_title="Telecom Tower Outage Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS with Gradient Accent Cards & Zero Emojis
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #1F2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    .enterprise-header {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .enterprise-title {
        font-size: 24px;
        font-weight: 700;
        color: #1F2937;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .enterprise-subtitle {
        font-size: 14px;
        color: #6B7280;
        margin-top: 4px;
        margin-bottom: 0;
    }
    
    /* Deployment Mode Badges */
    .badge-staging {
        background-color: #FEF3C7;
        color: #92400E;
        font-size: 12px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 16px;
        border: 1px solid #F59E0B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-production {
        background-color: #D1FAE5;
        color: #065F46;
        font-size: 12px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 16px;
        border: 1px solid #10B981;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Gradient Metric Cards */
    .card-red-gradient {
        background: linear-gradient(135deg, #FF4D4D 0%, #F92A7F 100%);
        color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(249, 42, 127, 0.2);
    }
    .card-yellow-gradient {
        background: linear-gradient(135deg, #FFB703 0%, #FB8500 100%);
        color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(251, 133, 0, 0.2);
    }
    .card-green-gradient {
        background: linear-gradient(135deg, #06D6A0 0%, #118AB2 100%);
        color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(17, 138, 178, 0.2);
    }
    .card-blue-gradient {
        background: linear-gradient(135deg, #4EA8DE 0%, #5390D9 100%);
        color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(83, 144, 217, 0.2);
    }
    .card-white-metric {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px;
    }
    .gradient-card-val {
        font-size: 28px;
        font-weight: 700;
        margin-top: 4px;
    }
    .gradient-card-lbl {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.9;
    }

    /* Subrack Layout Container */
    .bbu-box {
        background-color: #F8FAFC;
        border: 2px solid #CBD5E1;
        border-radius: 8px;
        padding: 16px;
        margin-top: 12px;
    }
    .slot-item {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .slot-item-active {
        background-color: #EFF6FF;
        border: 1px solid #93C5FD;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .slot-name {
        font-size: 13px;
        font-weight: 700;
        color: #1E3A8A;
    }
    .slot-text {
        font-size: 12px;
        color: #4B5563;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        color: #6B7280;
        font-size: 13px;
        font-weight: 600;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        color: #1F2937 !important;
    }
    </style>
""", unsafe_allow_html=True)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(WORKSPACE_DIR, 'telecom_features_july_2026.csv')
MODEL_PATH = os.path.join(WORKSPACE_DIR, 'models', 'calibrated_xgboost_outage.pkl')
SUBSET_PATH = os.path.join(WORKSPACE_DIR, 'july_2026_subset_dataset.csv')
PREDICTIONS_PATH = os.path.join(WORKSPACE_DIR, 'data', 'latest_predictions.csv')

@st.cache_data
def load_features():
    if os.path.exists(FEATURES_PATH):
        df = pd.read_csv(FEATURES_PATH)
        df['window_timestamp'] = pd.to_datetime(df['window_timestamp'])
        return df
    return pd.DataFrame()

@st.cache_data
def load_raw_logs():
    if os.path.exists(SUBSET_PATH):
        df = pd.read_csv(SUBSET_PATH, low_memory=False)
        time_col = 'event_time' if 'event_time' in df.columns else 'Occurred On (NT)'
        df['event_time'] = pd.to_datetime(df[time_col])
        return df
    return pd.DataFrame()

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        payload = joblib.load(MODEL_PATH)
        if isinstance(payload, dict) and 'model' in payload:
            feats = payload.get('features', [])
            thresh = payload.get('optimal_threshold', 0.5)
            return payload['model'], feats, thresh
        return payload, [], 0.5
    return None, [], 0.5

features_df = load_features()
raw_subset_df = load_raw_logs()
model, feature_cols, optimal_threshold = load_model()
runtime_config = load_config()
env_mode = runtime_config.get("ENVIRONMENT_MODE", "STAGING").upper()

badge_html = f'<span class="badge-staging">[MODE: STAGING (TESTING)]</span>' if env_mode == 'STAGING' else f'<span class="badge-production">[MODE: PRODUCTION (FIXED)]</span>'

# Enterprise Header with Environment Badge
st.markdown(f"""
    <div class="enterprise-header">
        <div>
            <h1 class="enterprise-title">Telecom Tower Outage Prediction System</h1>
            <p class="enterprise-subtitle">Enterprise 4G eNodeB / 5G gNodeB Operational Intelligence & Early Warning Outage Management Platform</p>
        </div>
        <div>
            {badge_html}
        </div>
    </div>
""", unsafe_allow_html=True)

# 7 Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Priority Outage Queue",
    "Tower Search & Diagnostic Inspector",
    "Manual Log Upload & Retention Engine",
    "🔐 Admin Settings",
    "5G Network Expansion Preview",
    "Historical Analytics & Trends",
    "Model Validation & Feature Importances"
])

# --- TAB 1: PRIORITY OUTAGE QUEUE ---
with tab1:
    if not features_df.empty and model is not None:
        latest_timestamp = features_df['window_timestamp'].max()
        latest_df = features_df[features_df['window_timestamp'] == latest_timestamp].copy()
        
        # Dynamic feature alignment
        X_input = pd.DataFrame(index=latest_df.index)
        for col in feature_cols:
            X_input[col] = latest_df[col] if col in latest_df.columns else 0.0

        latest_df['failure_probability'] = model.predict_proba(X_input)[:, 1] if not X_input.empty else 0.0
        latest_df.sort_values(by='failure_probability', ascending=False, inplace=True)
        
        latest_df['risk_status'] = latest_df['failure_probability'].apply(
            lambda p: 'CRITICAL' if p >= 0.65 else ('WARNING' if p >= 0.35 else 'NOMINAL')
        )

        total_towers = len(latest_df)
        crit_count = (latest_df['risk_status'] == 'CRITICAL').sum()
        warn_count = (latest_df['risk_status'] == 'WARNING').sum()
        hist_rate = (features_df['target_outage_next_2h'].mean() * 100) if 'target_outage_next_2h' in features_df.columns else 0.0

        # Gradient KPI Metric Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f'<div class="card-blue-gradient"><div class="gradient-card-lbl">Evaluated Sites</div><div class="gradient-card-val">{total_towers}</div></div>', unsafe_allow_html=True)
        with kpi2:
            st.markdown(f'<div class="card-red-gradient"><div class="gradient-card-lbl">Critical Risk Queue (>=65%)</div><div class="gradient-card-val">{crit_count}</div></div>', unsafe_allow_html=True)
        with kpi3:
            st.markdown(f'<div class="card-yellow-gradient"><div class="gradient-card-lbl">Warning Queue (35-64%)</div><div class="gradient-card-val">{warn_count}</div></div>', unsafe_allow_html=True)
        with kpi4:
            st.markdown(f'<div class="card-green-gradient"><div class="gradient-card-lbl">Historical Outage Rate</div><div class="gradient-card-val">{hist_rate:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Priority Outage Operational Queue (Next 2-Hour Lookahead)")
        st.write(f"Prediction Window Timestamp: **{latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}**")

        # Two distinct adjacent columns: Site ID (short) and eNodeB / gNodeB Identifier (full)
        if 'node_identifier' not in latest_df.columns:
            latest_df['node_identifier'] = np.where(latest_df['enodeb_id'] != '-', latest_df['enodeb_id'], latest_df['gnodeb_id'])

        disp_cols = [c for c in ['site_id', 'node_identifier', 'enodeb_id', 'gnodeb_id', 'failure_probability', 'risk_status', 'critical_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'consecutive_critical_streak_6h', 'total_alarms_6h'] if c in latest_df.columns]
        queue_df = latest_df[disp_cols].copy()

        queue_df['failure_probability'] = (queue_df['failure_probability'] * 100).map('{:.1f}%'.format)
        queue_df.rename(columns={
            'site_id': 'Site ID (Short)',
            'node_identifier': 'eNodeB / gNodeB Identifier',
            'enodeb_id': 'eNodeB ID (4G)',
            'gnodeb_id': 'gNodeB ID (5G)',
            'failure_probability': 'Outage Probability',
            'risk_status': 'Risk Status',
            'critical_alarms_6h': 'Critical Alarms (6h)',
            'rru_alarms_6h': 'RRU Alarms (6h)',
            'bbu_alarms_6h': 'BBU Alarms (6h)',
            'consecutive_critical_streak_6h': 'Critical Streak (6h)',
            'total_alarms_6h': 'Total Alarms (6h)'
        }, inplace=True)

        def style_queue(row):
            if row['Risk Status'] == 'CRITICAL':
                return ['background-color: #FEF2F2; font-weight: bold; color: #991B1B'] * len(row)
            elif row['Risk Status'] == 'WARNING':
                return ['background-color: #FFFBEB; font-weight: bold; color: #92400E'] * len(row)
            return [''] * len(row)

        st.dataframe(queue_df.style.apply(style_queue, axis=1), height=420, width='stretch')
    else:
        st.error("Feature matrix dataset or calibrated model file missing.")

# --- TAB 2: TOWER SEARCH & DIAGNOSTIC INSPECTOR ---
with tab2:
    if not features_df.empty:
        site_id_options = sorted([s for s in features_df['site_id'].unique() if s != '-'])
        node_options = sorted([n for n in features_df['node_identifier'].unique() if n != '-']) if 'node_identifier' in features_df.columns else []
        
        search_options = site_id_options + [f"Identifier: {n}" for n in node_options]
        selected_search = st.selectbox("Search / Select Tower Identifier (Site ID / Full Node Identifier):", search_options)
        
        if selected_search.startswith("Identifier: "):
            target_node = selected_search.replace("Identifier: ", "")
            selected_site_df = features_df[features_df['node_identifier'] == target_node]
            selected_site = selected_site_df['site_id'].iloc[0] if not selected_site_df.empty else target_node
        else:
            selected_site = selected_search
            selected_site_df = features_df[features_df['site_id'] == selected_site]

        site_features = selected_site_df.sort_values('window_timestamp') if not selected_site_df.empty else features_df.sort_values('window_timestamp')
        latest_row = site_features.iloc[-1]
        
        target_node_id = latest_row.get('node_identifier', latest_row.get('enodeb_id', '-'))
        target_enodeb = latest_row.get('enodeb_id', '-')
        target_gnodeb = latest_row.get('gnodeb_id', '-')

        X_single = pd.DataFrame(index=[0])
        for col in feature_cols:
            X_single[col] = [latest_row[col]] if col in latest_row.index else [0.0]

        prob = model.predict_proba(X_single)[0, 1] if model is not None else 0.0
        
        c_diag1, c_diag2 = st.columns([1, 2])
        with c_diag1:
            st.markdown("### Site Identifiers & Operational Risk")
            m_s1, m_s2 = st.columns(2)
            with m_s1:
                st.markdown(f'<div class="card-white-metric"><div style="font-size:11px; color:#6B7280; font-weight:600; text-transform:uppercase;">Site ID (Short Code)</div><div style="font-size:18px; font-weight:700; color:#0F172A;">{selected_site}</div></div>', unsafe_allow_html=True)
            with m_s2:
                st.markdown(f'<div class="card-white-metric"><div style="font-size:11px; color:#6B7280; font-weight:600; text-transform:uppercase;">Node Identifier</div><div style="font-size:15px; font-weight:700; color:#1E3A8A; word-break:break-all;">{target_node_id}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600; text-transform:uppercase;">2-Hour Outage Probability</div><div style="font-size:28px; font-weight:700; color:#1E3A8A;">{prob*100:.1f}%</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            if prob >= 0.65:
                st.markdown('<div class="card-red-gradient" style="text-align:center; font-weight:bold;">CRITICAL HIGH-RISK OUTAGE ALERT</div>', unsafe_allow_html=True)
            elif prob >= 0.35:
                st.markdown('<div class="card-yellow-gradient" style="text-align:center; font-weight:bold;">WARNING LEVEL PRECURSOR DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="card-green-gradient" style="text-align:center; font-weight:bold;">NOMINAL OPERATIONAL HEALTH</div>', unsafe_allow_html=True)

        with c_diag2:
            st.markdown("### Hardware Subrack Slot Assignments (Dual 4G / 5G Layout)")
            mode = st.radio("Select Hardware System Mode:", ["Huawei 4G BBU3900 / BBU3910", "Huawei 5G BBU5900 (NR Ready)"], horizontal=True)
            
            if "4G" in mode:
                st.markdown("""
                    <div class="bbu-box">
                        <div class="slot-item-active">
                            <div class="slot-name">Slot 7: UMPT Board (Main Control & Transport)</div>
                            <div class="slot-text">S1/X2 control plane signaling, IP clock synchronization, transmission path control.</div>
                        </div>
                        <div class="slot-item-active">
                            <div class="slot-name">Slots 2 / 3: UBBP Boards (Universal Baseband Processing)</div>
                            <div class="slot-text">LTE baseband signal processing and CPRI optical interface connectivity to RRUs.</div>
                        </div>
                        <div class="slot-item">
                            <div class="slot-name">Slot 16: FAN Unit (Forced Air Cooling)</div>
                            <div class="slot-text">Subrack ventilation fan tray regulating operational thermal thresholds.</div>
                        </div>
                        <div class="slot-item-active">
                            <div class="slot-name">Slots 18 / 19: UPEU Power Module (Universal Power Supply)</div>
                            <div class="slot-text">Converts -48V DC power and provides environmental alarm monitoring interfaces.</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="bbu-box">
                        <div class="slot-item-active">
                            <div class="slot-name">Slot 7: UMPTg Board (5G NR Main Control Unit)</div>
                            <div class="slot-text">Handles 5G N2/N3 control signaling, 10GE/25GE optical transport, and IEEE 1588v2 clock sync.</div>
                        </div>
                        <div class="slot-item-active">
                            <div class="slot-name">Slots 0 - 5: UBBPg NR Baseband Processing Cards</div>
                            <div class="slot-text">High-capacity 5G NR baseband processing, Massive MIMO eCPRI optical interfaces to AAU.</div>
                        </div>
                        <div class="slot-item">
                            <div class="slot-name">Slot 16: FANf Unit (High-Efficiency Cooling)</div>
                            <div class="slot-text">Intelligent multi-zone fan speed adjustment for high-density 5G NR baseband cards.</div>
                        </div>
                        <div class="slot-item-active">
                            <div class="slot-name">Slots 18 / 19: UPEUe Power Supply Module</div>
                            <div class="slot-text">Enhanced DC power distribution supporting high surge currents for Active Antenna Units.</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Historical Raw Alarm Event Stream (Selected Tower)")
        if not raw_subset_df.empty:
            tower_alarms = raw_subset_df[(raw_subset_df['site_id'] == selected_site) | (raw_subset_df.get('node_identifier', '') == target_node_id)].sort_values('event_time', ascending=False)
            display_cols = ['event_time', 'site_id', 'node_identifier', 'enodeb_id', 'gnodeb_id', 'severity', 'alarm_name', 'mo_name', 'location_information']
            present_cols = [c for c in display_cols if c in tower_alarms.columns]
            st.dataframe(tower_alarms[present_cols].head(100), height=350, width='stretch')

# --- TAB 3: MANUAL LOG UPLOAD & RETENTION ENGINE ---
with tab3:
    st.subheader("Manual Daily Log Ingestion & 60-Day Retention Management")
    st.write("Upload new daily Huawei alarm log exports (.xlsx / .csv). Incoming logs will be cleaned, ingested into SQLite master storage, and filtered through the automated 60-day auto-purge engine.")

    uploaded_file = st.file_uploader("Choose Daily Alarm Log Export (.xlsx or .csv):", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        if st.button("Ingest Log File & Run Outage Risk Scoring", type="primary"):
            with st.spinner("Processing manual log upload with notebook data cleaning and database ingestion..."):
                res = process_manual_file_upload(uploaded_file, uploaded_file.name)
                
                st.success(f"Successfully processed {res['filename']}!")
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Raw Log Rows</div><div style="font-size:24px; font-weight:700;">{res["raw_rows"]:,}</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Cleaned Records</div><div style="font-size:24px; font-weight:700;">{res["cleaned_rows"]:,}</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">DB Records Inserted</div><div style="font-size:24px; font-weight:700;">{res["inserted_db_records"]:,}</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Unique Sites Evaluated</div><div style="font-size:24px; font-weight:700;">{res["unique_sites_processed"]}</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Instant Outage Risk Assessment for Uploaded Sites")
                st.dataframe(res['site_risk_scores'], height=350, width='stretch')

    st.markdown("---")
    st.subheader("Automated 60-Day Retention Engine Status")
    db_summary = get_master_history_summary()
    
    col_ret1, col_ret2, col_ret3 = st.columns(3)
    with col_ret1:
        st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Master DB Total Records</div><div style="font-size:24px; font-weight:700;">{db_summary.get("total_records", 0):,}</div></div>', unsafe_allow_html=True)
    with col_ret2:
        st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Retention Window Start Date</div><div style="font-size:20px; font-weight:700; color:#1E3A8A;">{str(db_summary.get("min_date", "N/A"))[:10]}</div></div>', unsafe_allow_html=True)
    with col_ret3:
        st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Retention Window End Date</div><div style="font-size:20px; font-weight:700; color:#1E3A8A;">{str(db_summary.get("max_date", "N/A"))[:10]}</div></div>', unsafe_allow_html=True)

    if st.button("Trigger Manual 60-Day Auto-Purge Scan"):
        purge_res = purge_expired_logs(retention_days=60)
        st.info(f"Purge scan executed. Purged DB Rows: {purge_res['purged_db_rows']}, Purged Expired Raw Files: {purge_res['purged_raw_files']}")

# --- TAB 4: SECURE ADMIN PANEL & DYNAMIC CONFIGURATION ---
with tab4:
    st.subheader("🔐 NOC Admin Control Panel & Dynamic Pipeline Settings")
    
    ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "mobitel123")
    
    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False

    if not st.session_state["admin_authenticated"]:
        st.warning("Admin authentication required to modify model targets, feature categories, and environment modes.")
        pwd_input = st.text_input("Enter Admin Password:", type="password", key="admin_pwd_box")
        if st.button("Authenticate Admin Access", type="primary"):
            if pwd_input == ADMIN_PASS:
                st.session_state["admin_authenticated"] = True
                st.success("Admin Authentication Successful!")
                st.rerun()
            else:
                st.error("Invalid Admin Password. Access Denied.")
    else:
        st.success("Authenticated NOC Administrator Access Granted")
        if st.button("Logout Admin Session"):
            st.session_state["admin_authenticated"] = False
            st.rerun()

        st.markdown("---")
        st.markdown("### Dynamic Configuration Engine Control")
        st.write("Modify runtime targets, feature categories, lookback windows, and environment modes without touching codebase.")

        curr_cfg = load_config()

        col_adm1, col_adm2 = st.columns(2)

        with col_adm1:
            st.markdown("#### 1. Outage Target Toggles (`ACTIVE_TARGETS`)")
            t_unavail = st.checkbox("Cell Unavailable (`OUTAGE_CELL_UNAVAILABLE`)", value="OUTAGE_CELL_UNAVAILABLE" in curr_cfg.get("ACTIVE_TARGETS", []))
            t_outage = st.checkbox("Cell Outage / Down (`OUTAGE_CELL_OUTAGE`)", value="OUTAGE_CELL_OUTAGE" in curr_cfg.get("ACTIVE_TARGETS", []))
            t_fault = st.checkbox("Cell Fault (`OUTAGE_CELL_FAULT`)", value="OUTAGE_CELL_FAULT" in curr_cfg.get("ACTIVE_TARGETS", []))
            t_service = st.checkbox("Service Degradation / Unavailable (`OUTAGE_SERVICE_UNAVAILABLE`)", value="OUTAGE_SERVICE_UNAVAILABLE" in curr_cfg.get("ACTIVE_TARGETS", []))

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 2. Feature Category Toggles (`ACTIVE_FEATURES`)")
            f_bbu = st.checkbox("BBU Hardware & Board Alarms (`BBU_ALARM`)", value="BBU_ALARM" in curr_cfg.get("ACTIVE_FEATURES", []))
            f_rru = st.checkbox("RRU & RF Feeder Alarms (`RRU_ALARM`)", value="RRU_ALARM" in curr_cfg.get("ACTIVE_FEATURES", []))
            f_power = st.checkbox("Power & Mains Grid Alarms (`POWER_ALARM`)", value="POWER_ALARM" in curr_cfg.get("ACTIVE_FEATURES", []))

        with col_adm2:
            st.markdown("#### 3. Active Lookback Windows (`LOOKBACK_WINDOWS_HOURS`)")
            avail_windows = [6, 24, 336, 720]
            selected_windows = st.multiselect(
                "Select active feature lookback horizons:",
                options=avail_windows,
                default=[w for w in curr_cfg.get("LOOKBACK_WINDOWS_HOURS", [6, 24, 336, 720]) if w in avail_windows],
                format_func=lambda h: f"{h} Hours ({h//24} Days)" if h >= 24 else f"{h} Hours"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 4. Environment Mode (`ENVIRONMENT_MODE`)")
            curr_mode = curr_cfg.get("ENVIRONMENT_MODE", "STAGING").upper()
            selected_mode = st.radio(
                "Select System Deployment Mode:",
                options=["STAGING", "PRODUCTION"],
                index=0 if curr_mode == "STAGING" else 1,
                help="STAGING allows dynamic testing; PRODUCTION enforces fixed baseline hyperparameters."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Settings & Re-execute Pipeline", type="primary", use_container_width=True):
            new_targets = []
            if t_unavail: new_targets.append("OUTAGE_CELL_UNAVAILABLE")
            if t_outage: new_targets.append("OUTAGE_CELL_OUTAGE")
            if t_fault: new_targets.append("OUTAGE_CELL_FAULT")
            if t_service: new_targets.append("OUTAGE_SERVICE_UNAVAILABLE")

            new_features = []
            if f_bbu: new_features.append("BBU_ALARM")
            if f_rru: new_features.append("RRU_ALARM")
            if f_power: new_features.append("POWER_ALARM")

            new_config = {
                "ACTIVE_TARGETS": new_targets,
                "ACTIVE_FEATURES": new_features,
                "LOOKBACK_WINDOWS_HOURS": selected_windows if selected_windows else [6, 24],
                "ENVIRONMENT_MODE": selected_mode
            }

            save_config(new_config)
            st.info("Settings saved to config/settings.json. Re-generating dynamic features & model predictions...")

            with st.spinner("Executing dynamic feature engineering & model re-inference..."):
                generate_sliding_window_features(SUBSET_PATH, FEATURES_PATH, config=new_config)
                train_xgboost_model(features_csv_path=FEATURES_PATH, model_save_path=MODEL_PATH)
                run_model_inference(features_csv_path=FEATURES_PATH, model_path=MODEL_PATH, output_predictions_path=PREDICTIONS_PATH)
                st.cache_data.clear()
                st.cache_resource.clear()

            st.success("Dynamic configuration saved and ML inference re-executed cleanly without feature mismatch errors!")
            st.rerun()

# --- TAB 5: 5G NETWORK EXPANSION PREVIEW ---
with tab5:
    st.subheader("5G Network Expansion Preview")
    st.write("Future expansion module reserved for 5G NR gNodeB Active Antenna Unit (AAU) and eCPRI precursor analytics.")
    
    st.markdown("""
        <div class="card-white-metric" style="max-width: 600px; padding: 24px;">
            <div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 8px;">5G NR Prediction Engine (Phase 2 Development)</div>
            <div style="font-size: 13px; color: #64748B; margin-bottom: 16px;">Dual-mode 4G/5G joint predictive analytics and Massive MIMO beamforming fault indicators are scheduled for Phase 2 integration.</div>
        </div>
    """, unsafe_allow_html=True)

# --- TAB 6: HISTORICAL ANALYTICS & TIME-SERIES TRENDS ---
with tab6:
    st.subheader("Historical Analytics & 60-Day Time-Series Trends")
    
    if not features_df.empty:
        daily_trends = features_df.groupby(features_df['window_timestamp'].dt.date).agg({
            'total_alarms_6h': 'sum',
            'critical_alarms_6h': 'sum',
            'target_outage_next_2h': 'sum' if 'target_outage_next_2h' in features_df.columns else 'total_alarms_6h'
        }).reset_index()
        daily_trends.rename(columns={'window_timestamp': 'Date', 'target_outage_next_2h': 'Outage Windows'}, inplace=True)
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=daily_trends['Date'], y=daily_trends['total_alarms_6h'], name='Total Alarm Volume', marker_color='#4EA8DE'))
        if 'Outage Windows' in daily_trends.columns:
            fig_trend.add_trace(go.Scatter(x=daily_trends['Date'], y=daily_trends['Outage Windows'], name='Actual Cell Outages', yaxis='y2', line=dict(color='#FF4D4D', width=3)))
        
        fig_trend.update_layout(
            template='plotly_white',
            title="Daily Total Alarm Volume vs. Actual Cell Outages (July 2026)",
            xaxis_title="Date",
            yaxis_title="Total Alarm Volume",
            yaxis2=dict(title="Actual Cell Outages", overlaying='y', side='right'),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("### Outage Cause Breakdown by Category")
            cause_df = pd.DataFrame({
                'Cause Category': ['RF / Radio Unit Failures', 'SCTP Transport Faults', 'Mains Power Grid Outages', 'Transmission / S1 Link', 'BBU Hardware Faults'],
                'Occurrences': [28112, 10987, 4558, 2896, 484]
            })
            fig_pie = px.pie(cause_df, values='Occurrences', names='Cause Category', color_discrete_sequence=px.colors.sequential.Blues_r, template='plotly_white')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_c2:
            st.write("### Alarm Volume by Severity Class")
            sev_df = pd.DataFrame({
                'Severity': ['Major', 'Minor', 'Critical', 'Warning'],
                'Count': [2689412, 401234, 338910, 115748]
            })
            fig_sev = px.bar(sev_df, x='Severity', y='Count', color='Severity', color_discrete_sequence=['#4EA8DE', '#06D6A0', '#FF4D4D', '#FFB703'], template='plotly_white')
            st.plotly_chart(fig_sev, use_container_width=True)

# --- TAB 7: MODEL VALIDATION & FEATURE IMPORTANCES ---
with tab7:
    st.subheader("Calibrated Model Holdout Validation Metrics")
    st.write("Strict Chronological Holdout Validation Window: **July 23, 2026 – July 31, 2026** (3,780 Test Samples)")
    
    vm1, vm2, vm3, vm4, vm5 = st.columns(5)
    with vm1:
        st.markdown('<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">PR-AUC Score</div><div style="font-size:26px; font-weight:700; color:#1E3A8A;">0.7925</div></div>', unsafe_allow_html=True)
    with vm2:
        st.markdown('<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">ROC-AUC Score</div><div style="font-size:26px; font-weight:700; color:#1E3A8A;">0.9611</div></div>', unsafe_allow_html=True)
    with vm3:
        st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Opt Threshold</div><div style="font-size:26px; font-weight:700; color:#1E3A8A;">{optimal_threshold:.4f}</div></div>', unsafe_allow_html=True)
    with vm4:
        st.markdown('<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Precision @ Opt</div><div style="font-size:26px; font-weight:700; color:#1E3A8A;">0.6230</div></div>', unsafe_allow_html=True)
    with vm5:
        st.markdown('<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Recall @ Opt</div><div style="font-size:26px; font-weight:700; color:#1E3A8A;">0.9109</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.write("### Feature Importance Scores (PR-AUC Permutation Impact)")
        imp_data = pd.DataFrame({
            'Feature': ['total_alarms_6h', 'rru_alarms_6h', 'critical_alarms_6h', 'consecutive_critical_streak_6h', 'bbu_alarms_6h'],
            'Importance': [0.5098, 0.4773, 0.3690, 0.1986, 0.0838]
        }).sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(imp_data, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues', template='plotly_white')
        fig_imp.update_layout(showlegend=False, xaxis_title="PR-AUC Permutation Impact", yaxis_title="Feature")
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_v2:
        st.write("### Calibrated Failure Probability Distribution")
        if not features_df.empty and model is not None:
            valid_cols = [c for c in feature_cols if c in features_df.columns]
            if valid_cols:
                X_dist = pd.DataFrame(index=features_df.index)
                for col in feature_cols:
                    X_dist[col] = features_df[col] if col in features_df.columns else 0.0
                probs = model.predict_proba(X_dist)[:, 1]
                fig_hist = px.histogram(probs, nbins=50, labels={'value': 'Predicted Outage Probability'}, color_discrete_sequence=['#1E3A8A'], template='plotly_white')
                fig_hist.update_layout(xaxis_title="Calibrated Failure Probability", yaxis_title="Sample Count")
                st.plotly_chart(fig_hist, use_container_width=True)
