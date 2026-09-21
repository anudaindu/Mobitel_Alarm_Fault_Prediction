import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.storage_manager import get_master_history_summary, purge_expired_logs
from src.process_manual_upload import process_manual_file_upload

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
        df['event_time'] = pd.to_datetime(df['Occurred On (NT)'])
        return df
    return pd.DataFrame()

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

features_df = load_features()
raw_subset_df = load_raw_logs()
model = load_model()

# Header Section (Zero Emojis)
st.markdown("""
    <div class="enterprise-header">
        <h1 class="enterprise-title">Telecom Tower Outage Prediction System</h1>
        <p class="enterprise-subtitle">Enterprise 4G eNodeB / 5G gNodeB Operational Intelligence & Early Warning Outage Management Platform</p>
    </div>
""", unsafe_allow_html=True)

# 6 Navigation Tabs (Zero Emojis)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Priority Outage Queue",
    "Tower Search & Diagnostic Inspector",
    "Manual Log Upload & Retention Engine",
    "5G Network Readiness Preview",
    "Historical Analytics & Trends",
    "Model Validation & Feature Importances"
])

# --- TAB 1: PRIORITY OUTAGE QUEUE ---
with tab1:
    if not features_df.empty and model is not None:
        latest_timestamp = features_df['window_timestamp'].max()
        latest_df = features_df[features_df['window_timestamp'] == latest_timestamp].copy()
        
        feature_cols = ['total_alarms_6h', 'critical_alarms_6h', 'major_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'total_alarms_24h']
        latest_df['failure_probability'] = model.predict_proba(latest_df[feature_cols])[:, 1]
        latest_df.sort_values(by='failure_probability', ascending=False, inplace=True)
        
        latest_df['risk_status'] = latest_df['failure_probability'].apply(
            lambda p: 'CRITICAL' if p >= 0.65 else ('WARNING' if p >= 0.35 else 'NOMINAL')
        )

        total_towers = len(latest_df)
        crit_count = (latest_df['risk_status'] == 'CRITICAL').sum()
        warn_count = (latest_df['risk_status'] == 'WARNING').sum()
        hist_rate = (features_df['target_outage_next_2h'].mean() * 100)

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

        queue_df = latest_df[[
            'site_id', 'failure_probability', 'risk_status',
            'critical_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h',
            'major_alarms_6h', 'total_alarms_6h', 'total_alarms_24h'
        ]].copy()

        queue_df['failure_probability'] = (queue_df['failure_probability'] * 100).map('{:.1f}%'.format)
        queue_df.rename(columns={
            'site_id': 'Site ID (eNodeB / gNodeB)',
            'failure_probability': 'Outage Probability',
            'risk_status': 'Risk Status',
            'critical_alarms_6h': 'Critical Alarms (6h)',
            'rru_alarms_6h': 'RRU Alarms (6h)',
            'bbu_alarms_6h': 'BBU Alarms (6h)',
            'major_alarms_6h': 'Major Alarms (6h)',
            'total_alarms_6h': 'Total Alarms (6h)',
            'total_alarms_24h': 'Total Alarms (24h)'
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
        site_list = sorted(features_df['site_id'].unique().tolist())
        selected_site = st.selectbox("Search / Select Tower Site ID (4G eNodeB / 5G gNodeB):", site_list)
        
        site_features = features_df[features_df['site_id'] == selected_site].sort_values('window_timestamp')
        latest_row = site_features.iloc[-1]
        
        feature_cols = ['total_alarms_6h', 'critical_alarms_6h', 'major_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'total_alarms_24h']
        input_data = pd.DataFrame([latest_row[feature_cols].to_dict()])
        prob = model.predict_proba(input_data)[0, 1] if model is not None else 0.0
        
        c_diag1, c_diag2 = st.columns([1, 2])
        with c_diag1:
            st.markdown("### Site Operational Status")
            st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600; text-transform:uppercase;">Selected Site</div><div style="font-size:24px; font-weight:700; color:#0F172A;">{selected_site}</div></div>', unsafe_allow_html=True)
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
            tower_alarms = raw_subset_df[raw_subset_df['site_id'] == selected_site].sort_values('event_time', ascending=False)
            display_cols = ['event_time', 'severity', 'alarm_name', 'mo_name', 'location_information']
            present_cols = [c for c in display_cols if c in tower_alarms.columns]
            st.dataframe(tower_alarms[present_cols].head(100), height=350, width='stretch')

# --- TAB 3: MANUAL LOG UPLOAD & RETENTION ENGINE ---
with tab3:
    st.subheader("Manual Daily Log Ingestion & 60-Day Retention Management")
    st.write("Upload new daily Huawei alarm log exports (.xlsx / .csv). Incoming logs will be cleaned, ingested into DuckDB master storage, and filtered through the automated 60-day auto-purge engine.")

    uploaded_file = st.file_uploader("Choose Daily Alarm Log Export (.xlsx or .csv):", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        if st.button("Ingest Log File & Run Outage Risk Scoring", type="primary"):
            with st.spinner("Processing manual log upload with notebook data cleaning and DuckDB ingestion..."):
                res = process_manual_file_upload(uploaded_file, uploaded_file.name)
                
                st.success(f"Successfully processed {res['filename']}!")
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Raw Log Rows</div><div style="font-size:24px; font-weight:700;">{res["raw_rows"]:,}</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Cleaned Records</div><div style="font-size:24px; font-weight:700;">{res["cleaned_rows"]:,}</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">DuckDB Records Inserted</div><div style="font-size:24px; font-weight:700;">{res["inserted_db_records"]:,}</div></div>', unsafe_allow_html=True)
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
        st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">DuckDB Master Total Records</div><div style="font-size:24px; font-weight:700;">{db_summary.get("total_records", 0):,}</div></div>', unsafe_allow_html=True)
    with col_ret2:
        st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Retention Window Start Date</div><div style="font-size:20px; font-weight:700; color:#1E3A8A;">{str(db_summary.get("min_date", "N/A"))[:10]}</div></div>', unsafe_allow_html=True)
    with col_ret3:
        st.markdown(f'<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Retention Window End Date</div><div style="font-size:20px; font-weight:700; color:#1E3A8A;">{str(db_summary.get("max_date", "N/A"))[:10]}</div></div>', unsafe_allow_html=True)

    if st.button("Trigger Manual 60-Day Auto-Purge Scan"):
        purge_res = purge_expired_logs(retention_days=60)
        st.info(f"Purge scan executed. Purged DB Rows: {purge_res['purged_db_rows']}, Purged Expired Raw Files: {purge_res['purged_raw_files']}")

# --- TAB 4: 5G NETWORK ARCHITECTURE & READINESS PREVIEW ---
with tab4:
    st.subheader("5G NR Network Architecture & Readiness Preview")
    st.write("Comparative diagnostic framework evaluating precursor hardware fault signatures across 4G LTE eNodeBs versus 5G NR gNodeBs.")

    col_5g1, col_5g2 = st.columns(2)
    with col_5g1:
        st.markdown("""
            <div class="card-blue-gradient">
                <div style="font-size: 16px; font-weight: 700; margin-bottom: 8px;">4G LTE Precursor Indicators</div>
                <ul style="font-size: 13px; line-height: 1.6; margin: 0; padding-left: 20px;">
                    <li>CPRI Optical Interface Transmission Errors (1.25G to 9.8G)</li>
                    <li>RF Unit Power Amplifier Over-Temperature Trips</li>
                    <li>VSWR Transceiver Cable Impedance Mismatches</li>
                    <li>Mains Input AC Power Out of Range Grid Outages</li>
                    <li>S1/X2 IP Backhaul Control Plane Packet Loss</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with col_5g2:
        st.markdown("""
            <div class="card-red-gradient">
                <div style="font-size: 16px; font-weight: 700; margin-bottom: 8px;">5G NR Precursor Indicators (Next-Gen)</div>
                <ul style="font-size: 13px; line-height: 1.6; margin: 0; padding-left: 20px;">
                    <li>eCPRI High-Speed Optical Link Packet Drops (25GE)</li>
                    <li>AAU Active Antenna Array Beamforming Unit Faults</li>
                    <li>Sub-6GHz / mmWave RF Channel Calibration Errors</li>
                    <li>Massive MIMO Baseband Board Overload Failures</li>
                    <li>IEEE 1588v2 Clock Synchronization Phase Deviations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### 4G vs 5G Failure Precursor Comparison Table")
    comp_df = pd.DataFrame({
        'System Parameter': ['Baseband Processing Unit', 'Radio Frequency Unit', 'Optical Link Interface', 'Antenna System', 'Clock Sync Protocol'],
        '4G LTE Architecture': ['BBU3900 / BBU3910 (UBBP)', 'Remote Radio Unit (RRU)', 'CPRI (up to 9.8 Gbps)', 'Passive Antenna Feeder', 'IP Clock / GPS'],
        '5G NR Architecture': ['BBU5900 (UBBPg Baseband)', 'Active Antenna Unit (AAU)', 'eCPRI (25 Gbps)', 'Massive MIMO 64T64R Array', 'IEEE 1588v2 PTP / SyncE'],
        'Primary Precursor Warning': ['Board Maintenance Link Fault', 'RF Unit Maintenance Failure', 'Optical Module Rx Loss', 'VSWR Antenna Fault', 'Clock Reference Out of Sync']
    })
    st.table(comp_df)

# --- TAB 5: HISTORICAL ANALYTICS & TIME-SERIES TRENDS ---
with tab5:
    st.subheader("Historical Analytics & 60-Day Time-Series Trends")
    
    if not features_df.empty:
        daily_trends = features_df.groupby(features_df['window_timestamp'].dt.date).agg({
            'total_alarms_6h': 'sum',
            'critical_alarms_6h': 'sum',
            'target_outage_next_2h': 'sum'
        }).reset_index()
        daily_trends.rename(columns={'window_timestamp': 'Date', 'target_outage_next_2h': 'Outage Windows'}, inplace=True)
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=daily_trends['Date'], y=daily_trends['total_alarms_6h'], name='Total Alarm Volume', marker_color='#4EA8DE'))
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

# --- TAB 6: MODEL VALIDATION & FEATURE IMPORTANCES ---
with tab6:
    st.subheader("Calibrated Model Holdout Validation Metrics")
    st.write("Strict Chronological Holdout Validation Window: **July 23, 2026 – July 31, 2026** (3,780 Test Samples)")
    
    vm1, vm2, vm3, vm4 = st.columns(4)
    with vm1:
        st.markdown('<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">PR-AUC Score</div><div style="font-size:28px; font-weight:700; color:#1E3A8A;">0.6760</div></div>', unsafe_allow_html=True)
    with vm2:
        st.markdown('<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">ROC-AUC Score</div><div style="font-size:28px; font-weight:700; color:#1E3A8A;">0.8902</div></div>', unsafe_allow_html=True)
    with vm3:
        st.markdown('<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Precision (Thresh 0.5)</div><div style="font-size:28px; font-weight:700; color:#1E3A8A;">0.7948</div></div>', unsafe_allow_html=True)
    with vm4:
        st.markdown('<div class="card-white-metric"><div style="font-size:12px; color:#6B7280; font-weight:600;">Recall (Thresh 0.5)</div><div style="font-size:28px; font-weight:700; color:#1E3A8A;">0.4513</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.write("### Feature Importance Scores (PR-AUC Permutation Impact)")
        imp_data = pd.DataFrame({
            'Feature': ['critical_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'major_alarms_6h', 'total_alarms_24h', 'total_alarms_6h'],
            'Importance': [0.3847, 0.2763, 0.0675, 0.0472, 0.0452, 0.0162]
        }).sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(imp_data, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues', template='plotly_white')
        fig_imp.update_layout(showlegend=False, xaxis_title="PR-AUC Permutation Impact", yaxis_title="Feature")
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_v2:
        st.write("### Calibrated Failure Probability Distribution")
        if not features_df.empty and model is not None:
            feature_cols = ['total_alarms_6h', 'critical_alarms_6h', 'major_alarms_6h', 'rru_alarms_6h', 'bbu_alarms_6h', 'total_alarms_24h']
            probs = model.predict_proba(features_df[feature_cols])[:, 1]
            fig_hist = px.histogram(probs, nbins=50, labels={'value': 'Predicted Outage Probability'}, color_discrete_sequence=['#1E3A8A'], template='plotly_white')
            fig_hist.update_layout(xaxis_title="Calibrated Failure Probability", yaxis_title="Sample Count")
            st.plotly_chart(fig_hist, use_container_width=True)
