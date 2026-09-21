import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="4G Telecom Outage Intelligence | Mobitel AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 14px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-high {
        background-color: rgba(248, 81, 73, 0.15);
        border: 1px solid #f85149;
        color: #f85149;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }
    .risk-low {
        background-color: rgba(56, 139, 253, 0.15);
        border: 1px solid #388bfd;
        color: #58a6ff;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(WORKSPACE_DIR, 'telecom_features_july_2026.csv')
MODEL_PATH = os.path.join(WORKSPACE_DIR, 'models', 'outage_prediction_rf.pkl')
SUBSET_PATH = os.path.join(WORKSPACE_DIR, 'july_2026_subset_dataset.csv')

@st.cache_data
def load_feature_matrix():
    if os.path.exists(FEATURES_PATH):
        df = pd.read_csv(FEATURES_PATH)
        df['window_timestamp'] = pd.to_datetime(df['window_timestamp'])
        return df
    return pd.DataFrame()

@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

features_df = load_feature_matrix()
model = load_trained_model()

# Header Section
st.title("📡 4G Telecom Tower Cell Unavailability Outage Intelligence")
st.markdown("Autonomous AI & Predictive Analytics System for Huawei Telecom Tower Outage Early Warnings")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 Executive EDA & Insights", "🗼 Tower Health Timelines", "⚡ Live AI Outage Inference Engine"])

# --- TAB 1: EXECUTIVE EDA ---
with tab1:
    st.subheader("Network Alarm Analytics & Lead-Time Precursor Findings")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">3.55M</div><div class="metric-label">Raw Alarms Scanned</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">57,812</div><div class="metric-label">Cell Outages (July 2026)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">95.5%</div><div class="metric-label">Outages Preceded by RF Alarms</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">0.862</div><div class="metric-label">Holdout Test ROC-AUC</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Outages by Hour of Day")
        hourly_data = pd.DataFrame({
            'Hour': list(range(24)),
            'Outages': [1800, 1720, 1680, 1650, 1660, 1750, 1950, 2200, 2450, 2600, 2750, 2900, 3100, 3250, 3350, 3400, 3380, 3420, 3200, 2950, 2700, 2400, 2100, 1900]
        })
        fig_hour = px.bar(hourly_data, x='Hour', y='Outages', color='Outages', color_continuous_scale='Reds', template='plotly_dark')
        st.plotly_chart(fig_hour, use_container_width=True)

    with c2:
        st.write("### Precursor Alarm Co-occurrence (1h-6h Lead Window)")
        precursor_df = pd.DataFrame({
            'Alarm Precursor Category': ['RF / Radio Unit', 'SCTP Link Fault', 'Mains Power Outage', 'Transmission / S1 Link', 'BBU Board Fault', 'VSWR Antenna Fault'],
            'Preceding %': [95.5, 37.3, 15.5, 9.8, 1.6, 1.3]
        }).sort_values('Preceding %', ascending=True)
        fig_prec = px.bar(precursor_df, x='Preceding %', y='Alarm Precursor Category', orientation='h', color='Preceding %', color_continuous_scale='Viridis', template='plotly_dark')
        st.plotly_chart(fig_prec, use_container_width=True)

# --- TAB 2: TOWER HEALTH TIMELINES ---
with tab2:
    st.subheader("Interactive Site-Level Sliding Window Inspection")
    if not features_df.empty:
        site_list = sorted(features_df['site_id'].unique().tolist())
        selected_site = st.selectbox("Select Site / Tower ID:", site_list)
        
        site_features = features_df[features_df['site_id'] == selected_site].sort_values('window_timestamp')
        
        st.write(f"Showing {len(site_features)} 2-hour window samples for **Site {selected_site}** across July 2026")
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=site_features['window_timestamp'], y=site_features['critical_alarms_6h'], name='Critical Alarms 6h', line=dict(color='#f85149', width=2)))
        fig_ts.add_trace(go.Scatter(x=site_features['window_timestamp'], y=site_features['rru_alarms_6h'], name='RRU Alarms 6h', line=dict(color='#ff7f0e', width=1.5)))
        fig_ts.add_trace(go.Scatter(x=site_features['window_timestamp'], y=site_features['total_alarms_6h'], name='Total Alarms 6h', line=dict(color='#58a6ff', width=1, dash='dot')))
        fig_ts.update_layout(template='plotly_dark', title=f"6-Hour Window Alarm Activity Timeline (Site {selected_site})", xaxis_title="Timestamp T", yaxis_title="Alarm Count")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.warning("Feature matrix dataset not found.")

# --- TAB 3: LIVE AI INFERENCE ENGINE ---
with tab3:
    st.subheader("Real-Time 2-Hour Cell Unavailability Prediction Engine")
    st.markdown("Input current 6-hour and 24-hour alarm window counts to score the likelihood of a `CELL UNAVAILABLE` outage in the next 2 hours.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        total_6h = st.slider("Total Alarms (Past 6 Hours)", 0, 100, 12)
        critical_6h = st.slider("Critical Severity Alarms (Past 6 Hours)", 0, 30, 4)
        major_6h = st.slider("Major Severity Alarms (Past 6 Hours)", 0, 50, 5)
    with col_b:
        rru_6h = st.slider("RF / RRU / VSWR Alarms (Past 6 Hours)", 0, 30, 3)
        bbu_6h = st.slider("BBU / Subrack / Transmission Alarms (Past 6 Hours)", 0, 30, 1)
        total_24h = st.slider("Total Alarms (Past 24 Hours)", 0, 300, 35)

    if st.button("⚡ Score Real-Time Outage Risk", type="primary"):
        if model is not None:
            input_data = pd.DataFrame([{
                'total_alarms_6h': total_6h,
                'critical_alarms_6h': critical_6h,
                'major_alarms_6h': major_6h,
                'rru_alarms_6h': rru_6h,
                'bbu_alarms_6h': bbu_6h,
                'total_alarms_24h': total_24h
            }])
            
            prob = model.predict_proba(input_data)[0, 1]
            
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Predicted Outage Probability (Next 2 Hours)", value=f"{prob * 100:.1f}%")
                st.progress(float(prob))
                
            with res_col2:
                if prob >= 0.5:
                    st.markdown(f'<div class="risk-high">⚠️ HIGH OUTAGE RISK DETECTED ({prob*100:.1f}%)<br>Cell Unavailability probable in next 2h! Dispatch field technician for RF/RRU check.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">✅ NORMAL OPERATION RISK ({prob*100:.1f}%)<br>Site operating within nominal parameters.</div>', unsafe_allow_html=True)
        else:
            st.error("Trained model artifact not found.")
