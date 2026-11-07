import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from src.config import FIGURES_DIR, MODELS_DIR, CLEANED_DATA_PATH

st.set_page_config(
    page_title="ABC Tech | ITSM Machine Learning Suite",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1f2937; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #4b5563; margin-bottom: 25px; }
    .kpi-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; text-align: center; }
    .kpi-val { font-size: 1.8rem; font-weight: 700; color: #2563eb; }
    .kpi-lbl { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">ABC Tech — ITSM Predictive Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ITIL Process Optimization, Early Warning Triage, Volume Forecasting & Asset Risk Profiling</div>', unsafe_allow_html=True)

# Sidebar Navigation
sidebar_nav = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Pillar 1: High Priority Triage",
        "Pillar 2: Volume Forecasting",
        "Pillar 3: Auto-Tagging & Routing",
        "Pillar 4: Asset Risk & RFC"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Reference**: PM-PR-0012")
st.sidebar.markdown("**Framework**: ITIL V3 + Machine Learning")
st.sidebar.markdown("**Dataset**: 46.6K Records (2012-2014)")

# Helper function to load models
@st.cache_resource
def load_artifacts():
    p1 = joblib.load(MODELS_DIR / "pillar1_high_priority_model.joblib")
    p2 = joblib.load(MODELS_DIR / "pillar2_volume_forecaster.joblib")
    p3_dept = joblib.load(MODELS_DIR / "pillar3_department_router.joblib")
    p3_prio = joblib.load(MODELS_DIR / "pillar3_priority_classifier.joblib")
    p4_rfc = joblib.load(MODELS_DIR / "pillar4_rfc_predictor.joblib")
    p4_assets = pd.read_csv(MODELS_DIR / "pillar4_asset_risk_scores.csv")
    return p1, p2, p3_dept, p3_prio, p4_rfc, p4_assets

try:
    p1, p2, p3_dept, p3_prio, p4_rfc, p4_assets = load_artifacts()
except Exception as e:
    st.error(f"Please run the modeling pipeline first (`python -m src.run_all`). Error: {e}")
    st.stop()

# -------------------------------------------------------------------------------------------------
# VIEW 1: EXECUTIVE DASHBOARD
# -------------------------------------------------------------------------------------------------
if sidebar_nav == "Executive Dashboard":
    st.subheader("Operational Health & ITIL Process Maturity Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-val">46,606</div>
            <div class="kpi-lbl">Total IT Incidents Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-val" style="color:#dc2626;">700 (1.55%)</div>
            <div class="kpi-lbl">P1 / P2 Major Outages</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-val" style="color:#d97706;">41.1%</div>
            <div class="kpi-lbl">Reassigned Incident Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-val" style="color:#16a34a;">80,021</div>
            <div class="kpi-lbl">2014 Projected Annual Load</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Priority Distribution", "Reassignment Escalation", "Historical Inflow Trend"])
    with tab1:
        st.image(str(FIGURES_DIR / "01_priority_distribution.png"), use_container_width=True)
        st.info("💡 **Key Finding**: Tickets are heavily concentrated in P4 (50.2%) and P5 (36.5%). Critical P1/P2 tickets represent just 1.55% of volume, creating severe operational risk when misidentified.")
    with tab2:
        st.image(str(FIGURES_DIR / "02_reassignments_vs_handletime.png"), use_container_width=True)
        st.info(r"💡 **Key Finding**: Tickets with $\ge 1$ reassignments experience a 114% increase in median handling duration (from 1,331 hours to 2,851 hours), proving that manual routing is ABC Tech's primary bottleneck.")
    with tab3:
        st.image(str(FIGURES_DIR / "03_incident_trend_monthly.png"), use_container_width=True)
        st.info("💡 **Key Finding**: Ticketing surged significantly starting Q4 2013, maintaining sustained high volume of 6,000 - 8,500 tickets monthly in 2014.")

# -------------------------------------------------------------------------------------------------
# VIEW 2: PILLAR 1 — HIGH PRIORITY TRIAGE
# -------------------------------------------------------------------------------------------------
elif sidebar_nav == "Pillar 1: High Priority Triage":
    st.subheader("Pillar 1 — Predicting High Priority (P1/P2) Tickets")
    st.write("Early Warning ML Engine to intercept catastrophic outages at ticket inception before triage delays.")
    
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.image(str(FIGURES_DIR / "05_pillar1_roc_pr_curves.png"), use_container_width=True)
    with col_r:
        st.image(str(FIGURES_DIR / "06_pillar1_feature_importance.png"), use_container_width=True)
        
    st.markdown("### Real-Time Incident Priority Risk Assessment")
    with st.form("p1_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            ci_cat = st.selectbox("CI Category", ["application", "subapplication", "computer", "storage", "hardware", "software", "database", "other"])
            subcat = st.selectbox("Subcategory Profile", ["Web Based Application", "Desktop Application", "Server Based Application", "SAP", "Client Based Application", "Other"])
        with f2:
            cat = st.selectbox("Ticket Category", ["incident", "request for information", "complaint", "request for change"])
            interactions = st.number_input("Related Customer Interactions at Open", min_value=0, max_value=20, value=1)
        with f3:
            hour = st.slider("Hour of Ticket Submission (0 - 23)", 0, 23, 14)
            dow = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            
        submitted = st.form_submit_button("Assess Incident Priority Risk")
        
    if submitted:
        dow_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(dow)
        mock_row = pd.DataFrame([{
            'Open_Hour': hour,
            'Open_DayOfWeek': dow_idx,
            'Is_Weekend': 1 if dow_idx >= 5 else 0,
            'Open_Month': 3,
            'No_of_Related_Interactions_Clean': interactions,
            'number_cnt': 0.5,
            'CI_Name': 'UNKNOWN',
            'CI_Subcat': subcat,
            'WBS': 'UNKNOWN',
            'KB_number': 'UNKNOWN',
            'CI_Cat': ci_cat,
            'Category': cat
        }])
        
        from src.feature_engineering import extract_features
        X_eval = extract_features(mock_row, is_train=False, encoders=p1['encoders'])
        prob = p1['model'].predict_proba(X_eval)[0, 1]
        threshold = p1['threshold']
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            if prob >= threshold:
                st.error(f"🚨 **HIGH PRIORITY ALERT (P1/P2)**\n\nProbability: **{prob*100:.1f}%** (Threshold: {threshold*100:.1f}%)")
            else:
                st.success(f"**Standard Ticket (P3 - P5)**\n\nProbability of P1/P2: **{prob*100:.1f}%**")
        with res_col2:
            st.metric("Champion Model", p1['model_name'])
            st.metric("Model PR-AUC (Test Holdout)", f"{p1['pr_auc']:.4f}")
            st.write(f"The model operates with a tuned decision threshold of **{threshold:.3f}** to maximize recall of critical service disruptions.")

# -------------------------------------------------------------------------------------------------
# VIEW 3: PILLAR 2 — VOLUME FORECASTING
# -------------------------------------------------------------------------------------------------
elif sidebar_nav == "Pillar 2: Volume Forecasting":
    st.subheader("Pillar 2 — Quarterly & Annual Incident Volume Forecasting")
    st.write("Ensemble Time-Series Forecasting (SARIMAX + Holt-Winters) for IT Resource and Capacity Planning.")
    
    st.image(str(FIGURES_DIR / "07_pillar2_volume_forecast.png"), use_container_width=True)
    
    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        st.markdown("### Quarterly Capacity Projections (2014)")
        q_data = {
            "Quarter": ["Q1 2014 (Recorded)", "Q2 2014 (Forecast)", "Q3 2014 (Forecast)", "Q4 2014 (Forecast)", "Full Year 2014 Total"],
            "Incident Volume": [f"{p2['q1_2014_actual']:,.0f}"] + [f"{v:,.0f}" for v in p2['q_projections'].values()] + [f"{p2['projected_2014_total']:,.0f}"],
            "Status": ["Historical Actual", "Projected", "Projected", "Projected", "Annual Projected Total"]
        }
        st.table(pd.DataFrame(q_data))
    with f_col2:
        st.markdown("### Annual Capacity Expansion Targets")
        st.metric("2014 Projected Total Volume", f"{p2['projected_2014_total']:,.0f} tickets", delta="+225.2% vs 2013")
        st.metric("2015 Capacity Baseline Target", f"{p2['projected_2015_total']:,.0f} tickets", delta="+5% Expansion Target")
        st.image(str(FIGURES_DIR / "08_pillar2_ci_forecasts.png"), use_container_width=True)

# -------------------------------------------------------------------------------------------------
# VIEW 4: PILLAR 3 — AUTO-TAGGING & ROUTING
# -------------------------------------------------------------------------------------------------
elif sidebar_nav == "Pillar 3: Auto-Tagging & Routing":
    st.subheader("Pillar 3 — Intelligent Auto-Tagging & SLA Delay Elimination")
    st.write("Autonomous ticket routing to eliminate manual triage hops and accelerate MTTR.")
    
    r_col1, r_col2 = st.columns([1, 1])
    with r_col1:
        st.image(str(FIGURES_DIR / "09_pillar3_department_confusion_matrix.png"), use_container_width=True)
    with r_col2:
        st.image(str(FIGURES_DIR / "10_pillar3_priority_confusion_matrix.png"), use_container_width=True)
        
    st.markdown("---")
    st.markdown("### Business ROI Calculator: Impact of Auto-Routing")
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("First-Touch Routing Accuracy", f"{p3_dept['accuracy']*100:.1f}%")
    s_col2.metric("Reassignment Reduction", "-69.4%")
    s_col3.metric("MTTR Acceleration for Misrouted", "53.3% Faster")

# -------------------------------------------------------------------------------------------------
# VIEW 5: PILLAR 4 — ASSET RISK & RFC
# -------------------------------------------------------------------------------------------------
elif sidebar_nav == "Pillar 4: Asset Risk & RFC":
    st.subheader("Pillar 4 — Request for Change (RFC) & ITSM Asset Risk Profiling")
    st.write("Proactive configuration management: identifying failure-prone assets and change triggers.")
    
    st.image(str(FIGURES_DIR / "12_pillar4_ci_asset_risk_matrix.png"), use_container_width=True)
    
    st.markdown("### Top High-Risk Configuration Items (CIs)")
    st.write("Ranked by Composite Asset Risk Index (accounting for failure volume, P1/P2 outage rate, RFC triggers, and misconfiguration errors):")
    
    top_n = st.slider("Number of Assets to Display", 5, 50, 15)
    cols = ['CI_Name', 'CI_Cat', 'CI_Subcat', 'Total_Incidents', 'P1_P2_Incidents', 'RFC_Incidents', 'Misconfig_Rate', 'Asset_Risk_Index']
    styled_df = p4_assets[cols].head(top_n).copy()
    styled_df['Misconfig_Rate'] = (styled_df['Misconfig_Rate'] * 100).map("{:.1f}%".format)
    styled_df['Asset_Risk_Index'] = styled_df['Asset_Risk_Index'].map("{:.1f}".format)
    st.dataframe(styled_df, use_container_width=True)
    
    st.markdown("---")
    st.image(str(FIGURES_DIR / "11_pillar4_rfc_roc_curve.png"), use_container_width=True)
