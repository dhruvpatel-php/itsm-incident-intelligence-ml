import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import joblib

from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR

st.set_page_config(
    page_title="ABC Tech | ITSM Machine Learning Suite",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Clean Enterprise - No Emojis)
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1f2937; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #4b5563; margin-bottom: 25px; }
    .kpi-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; text-align: center; }
    .kpi-val { font-size: 1.8rem; font-weight: 700; color: #2563eb; }
    .kpi-lbl { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .callout-info { background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 6px; color: #1e3a8a; margin-top: 10px; margin-bottom: 12px; font-size: 0.95rem; }
    .callout-success { background-color: #f0fdf4; border-left: 4px solid #22c55e; padding: 12px 16px; border-radius: 6px; color: #14532d; margin-top: 10px; margin-bottom: 12px; font-size: 0.95rem; }
    .callout-error { background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 12px 16px; border-radius: 6px; color: #7f1d1d; margin-top: 10px; margin-bottom: 12px; font-size: 0.95rem; }
    .callout-warning { background-color: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 6px; color: #78350f; margin-top: 10px; margin-bottom: 12px; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">ABC Tech — ITSM Predictive Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ITIL Process Optimization, Early Warning Triage, Volume Forecasting & Asset Risk Profiling<br><span style="font-size: 0.95rem; color: #4b5563; font-weight: 600;">Author: Dhruv Patel (Data Scientist)</span></div>', unsafe_allow_html=True)

# Helper function to load models and single source of truth metrics
@st.cache_resource
def load_artifacts():
    missing = []
    for f in ["pillar1_high_priority_model.joblib", "pillar2_volume_forecaster.joblib",
              "pillar3_department_router.joblib", "pillar3_priority_classifier.joblib",
              "pillar4_rfc_predictor.joblib", "pillar4_asset_risk_scores.csv"]:
        if not (MODELS_DIR / f).exists():
            missing.append(f)
    if not (REPORTS_DIR / "metrics.json").exists():
        missing.append("reports/metrics.json")

    if missing:
        raise FileNotFoundError(f"Missing pipeline artifacts: {', '.join(missing)}. Run `python -m src.run_all` first.")

    p1 = joblib.load(MODELS_DIR / "pillar1_high_priority_model.joblib")
    p2 = joblib.load(MODELS_DIR / "pillar2_volume_forecaster.joblib")
    p3_dept = joblib.load(MODELS_DIR / "pillar3_department_router.joblib")
    p3_prio = joblib.load(MODELS_DIR / "pillar3_priority_classifier.joblib")
    p4_rfc = joblib.load(MODELS_DIR / "pillar4_rfc_predictor.joblib")
    p4_assets = pd.read_csv(MODELS_DIR / "pillar4_asset_risk_scores.csv")
    with open(REPORTS_DIR / "metrics.json", "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    return p1, p2, p3_dept, p3_prio, p4_rfc, p4_assets, metrics

@st.cache_data
def get_dataset_summary():
    csv_path = PROJECT_ROOT / "data" / "processed" / "cleaned_itsm.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, usecols=['Priority', 'No_of_Reassignments', 'Open_DT'], low_memory=False)
    else:
        from src.data_loader import load_and_clean_data
        df, _ = load_and_clean_data(return_metrics=True)

    # P1/P2 and Priority shares
    valid_prio = df[df['Priority'].isin([1, 2, 3, 4, 5, '1', '2', '3', '4', '5'])].copy()
    prio_num = valid_prio['Priority'].astype(str).str.extract(r'(\d+)')[0].astype(int)
    denom = len(valid_prio)
    p1p2_cnt = int((prio_num <= 2).sum())
    p1p2_pct = float(p1p2_cnt / denom * 100) if denom else 0.0

    prio_counts = prio_num.value_counts()
    p4_pct = float(prio_counts[4] / denom * 100) if (denom and 4 in prio_counts) else 0.0
    p5_pct = float(prio_counts[5] / denom * 100) if (denom and 5 in prio_counts) else 0.0

    # Reassigned ratio
    total_rows = len(df)
    reassigned_cnt = int((df['No_of_Reassignments'] > 0).sum())
    reassigned_pct = float(reassigned_cnt / total_rows * 100) if total_rows else 0.0

    # Monthly range for Oct 2013 - Mar 2014 observed
    open_dt = pd.to_datetime(df['Open_DT'])
    stable_mask = (open_dt >= '2013-10-01') & (open_dt <= '2014-03-31')
    monthly = open_dt[stable_mask].dt.to_period('M').value_counts()
    min_monthly = int(monthly.min()) if len(monthly) else 0
    max_monthly = int(monthly.max()) if len(monthly) else 0
    min_year = int(open_dt.dt.year.min())
    max_year = int(open_dt.dt.year.max())

    return {
        "p1p2_count": p1p2_cnt,
        "p1p2_denom": denom,
        "p1p2_pct": p1p2_pct,
        "p4_pct": p4_pct,
        "p5_pct": p5_pct,
        "reassigned_pct": reassigned_pct,
        "min_monthly": min_monthly,
        "max_monthly": max_monthly,
        "min_year": min_year,
        "max_year": max_year,
    }

try:
    p1, p2, p3_dept, p3_prio, p4_rfc, p4_assets, metrics = load_artifacts()
    ds = get_dataset_summary()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.info("Execute: `python -m src.run_all` in your terminal to build all production artifacts.")
    st.stop()

dl_m = metrics["data_loader"]
p1_m = metrics["pillar1_early_warning"]
p2_m = metrics["pillar2_forecasting"]
p3_p_m = metrics["pillar3_priority"]
p3_r_m = metrics["pillar3_routing"]
p4_m = metrics["pillar4_rfc_and_assets"]

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
st.sidebar.markdown("**Author**: Dhruv Patel (Data Scientist)")
st.sidebar.markdown("**Framework**: ITIL Process Framework + Machine Learning")
st.sidebar.markdown(f"**Dataset**: {dl_m['total_rows'] / 1000:.1f}K Records ({ds['min_year']}-{ds['max_year']})")

# -------------------------------------------------------------------------------------------------
# VIEW 1: EXECUTIVE DASHBOARD
# -------------------------------------------------------------------------------------------------
if sidebar_nav == "Executive Dashboard":
    st.subheader("Operational Health & ITIL Process Maturity Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{dl_m['total_rows']:,}</div>
            <div class="kpi-lbl">Total IT Incidents Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val" style="color:#dc2626;">{ds['p1p2_count']:,} ({ds['p1p2_pct']:.2f}%)</div>
            <div class="kpi-lbl">P1 / P2 Major Outages (N={ds['p1p2_denom']:,})</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val" style="color:#d97706;">{ds['reassigned_pct']:.1f}%</div>
            <div class="kpi-lbl">Reassigned Incident Ratio</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val" style="color:#16a34a;">{p2_m['base_vol_2014']:,}</div>
            <div class="kpi-lbl">Projected Annual Load</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Priority Distribution", "Reassignment Escalation", "Historical Inflow Trend"])
    with tab1:
        st.image(str(FIGURES_DIR / "01_priority_distribution.png"), width="stretch")
        st.markdown(f'<div class="callout-info"><strong>Key Finding</strong>: Tickets are heavily concentrated in P4 ({ds["p4_pct"]:.1f}%) and P5 ({ds["p5_pct"]:.1f}%). Critical P1/P2 tickets represent just {ds["p1p2_pct"]:.2f}% of volume ({ds["p1p2_count"]:,} of {ds["p1p2_denom"]:,} verified tickets), creating severe operational risk when misidentified.</div>', unsafe_allow_html=True)
    with tab2:
        st.image(str(FIGURES_DIR / "02_reassignments_vs_handletime.png"), width="stretch")
        st.markdown(f'<div class="callout-info"><strong>Key Finding</strong>: Controlled econometric OLS regression proves each bounce adds <strong>+{p3_r_m["delay_penalty_pct"]:.2f}% active handle duration</strong> ({p3_r_m["delay_hours_saved_per_bounce"]:.2f} hours/bounce, p={p3_r_m["delay_p_value"]:.2e}), proving that manual routing hops are ABC Tech\'s primary MTTR bottleneck.</div>', unsafe_allow_html=True)
    with tab3:
        st.image(str(FIGURES_DIR / "03_incident_trend_monthly.png"), width="stretch")
        st.markdown(f'<div class="callout-info"><strong>Key Finding</strong>: Ticketing surged significantly post-enterprise rollout (Oct 2013 - Mar 2014 observed: {ds["min_monthly"]:,} - {ds["max_monthly"]:,} tickets monthly).</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# VIEW 2: PILLAR 1 — HIGH PRIORITY TRIAGE
# -------------------------------------------------------------------------------------------------
elif sidebar_nav == "Pillar 1: High Priority Triage":
    st.subheader("Pillar 1 — Predicting High Priority (P1/P2) Tickets")
    st.write("Early Warning ML Engine to intercept catastrophic outages at ticket inception before triage delays.")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.image(str(FIGURES_DIR / "05_pillar1_roc_pr_curves.png"), width="stretch")
    with col_r:
        st.image(str(FIGURES_DIR / "06_pillar1_feature_importance.png"), width="stretch")

    st.markdown("### Real-Time Incident Priority Risk Assessment")

    # Extract authentic CI and WBS lists from trained feature frequencies
    known_cis = sorted(list(p1['ci_freq'].keys())[:25])
    known_wbs = sorted(list(p1['wbs_freq'].keys())[:25])

    with st.form("p1_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            ci_cat = st.selectbox("CI Category", ["application", "subapplication", "computer", "storage", "hardware"])
            ci_choice = st.selectbox("Configuration Item (CI_Name)", ["Unseen / New Asset"] + known_cis)
            wbs_choice = st.selectbox("Billing Account (WBS)", ["Unseen / New WBS"] + known_wbs)
        with f2:
            cat = st.selectbox("Ticket Category", ["incident", "request for information", "complaint", "request for change"])
            interactions = st.number_input("Related Customer Interactions at Open", min_value=0, max_value=25, value=1)
            subcat = st.selectbox("Subcategory Profile", ["Desktop Application", "Web Based Application", "Server Based Application", "SAP", "Client Based Application"])
        with f3:
            hour = st.slider("Hour of Ticket Submission (0 - 23)", 0, 23, 14)
            dow = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            month = st.selectbox("Month of Ticket Submission", list(range(1, 13)), index=2)

        submitted = st.form_submit_button("Assess Incident Priority Risk")

    if submitted:
        dow_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(dow)
        is_unseen = (ci_choice == "Unseen / New Asset")

        # Build features strictly matching trained schema
        ci_freq_val = float(p1['ci_freq'][ci_choice]) if (not is_unseen and ci_choice in p1['ci_freq']) else 0.0
        wbs_freq_val = float(p1['wbs_freq'][wbs_choice]) if (wbs_choice != "Unseen / New WBS" and wbs_choice in p1['wbs_freq']) else 0.0

        mock_row = pd.DataFrame([{
            'Open_Hour': hour,
            'Open_DayOfWeek': dow_idx,
            'Is_Weekend': 1 if dow_idx >= 5 else 0,
            'Open_Month': month,
            'No_of_Related_Interactions_Clean': interactions,
            'CI_Name_Freq': ci_freq_val,
            'WBS_Freq': wbs_freq_val,
            'CI_Cat': ci_cat,
            'Category': cat
        }])

        # Categorical dummies
        X_mock = pd.get_dummies(mock_row[['CI_Cat', 'Category']], drop_first=False, dtype=float)
        X_mock['CI_Name_Freq'] = ci_freq_val
        X_mock['WBS_Freq'] = wbs_freq_val
        X_mock['No_of_Related_Interactions_Clean'] = interactions
        X_mock['Open_Hour'] = hour
        X_mock['Open_DayOfWeek'] = dow_idx
        X_mock['Open_Month'] = month
        X_mock['Is_Weekend'] = 1 if dow_idx >= 5 else 0

        # Align with trained feature columns
        train_cols = p1['train_cols']
        X_eval = X_mock.reindex(columns=train_cols, fill_value=0.0)

        raw_prob = float(p1['model'].predict_proba(X_eval)[0, 1])
        cal_prob = float(p1['calibrated_model'].predict([raw_prob])[0]) if 'calibrated_model' in p1 else raw_prob
        threshold = float(p1['threshold'])

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            if raw_prob >= threshold:
                st.markdown(f'<div class="callout-error"><strong>HIGH PRIORITY ESCALATION (P1/P2)</strong><br>Probability: <strong>{raw_prob*100:.1f}%</strong> (Threshold: {threshold*100:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="callout-success"><strong>Standard Ticket (P3 - P5)</strong><br>Probability of P1/P2: <strong>{raw_prob*100:.1f}%</strong></div>', unsafe_allow_html=True)

            if is_unseen:
                st.markdown(f'<div class="callout-warning"><strong>Unseen Asset Mode</strong>: This asset was not observed in training. Evaluated holdout performance on unseen assets: PR-AUC = <strong>{p1_m["unseen_assets"]["pr_auc"]:.4f}</strong>, Recall = <strong>{p1_m["unseen_assets"]["recall"]*100:.1f}%</strong>.</div>', unsafe_allow_html=True)
        with res_col2:
            st.metric("Champion Model", p1_m['model_name'])
            st.metric("Model PR-AUC (Holdout Test)", f"{p1_m['test_pr_auc']:.4f}", delta=f"{p1_m['statistical_lift']:.1f}x Prevalence Lift")
            st.caption(f"95% Bootstrap Confidence Interval: [{p1_m['test_pr_auc_ci_low']:.4f}, {p1_m['test_pr_auc_ci_high']:.4f}] | Validated F2 Threshold: **{threshold:.3f}** (Holdout Recall: {p1_m['test_recall']*100:.1f}%)")

# -------------------------------------------------------------------------------------------------
# VIEW 3: PILLAR 2 — VOLUME FORECASTING
# -------------------------------------------------------------------------------------------------
elif sidebar_nav == "Pillar 2: Volume Forecasting":
    st.subheader("Pillar 2 — Incident Volume & Capacity Forecasting")
    st.write("Production Ensemble (SARIMAX + Naive Persistence) with Moving-Block Bootstrapping.")

    st.image(str(FIGURES_DIR / "07_pillar2_volume_forecast.png"), width="stretch")

    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        st.markdown("### Enterprise Annual Capacity Projections")
        q_data = {
            "Horizon Period": ["Q1 2014 (Recorded)", "Remaining 39 Weeks (Forecast)", "Full Year 2014 Total (Base)", "2014 95% PI (Low Bound)", "2014 95% PI (High Bound)"],
            "Incident Volume": [f"{p2_m['q1_2014_actual']:,}", f"{p2_m['future_39w_base']:,}", f"{p2_m['base_vol_2014']:,}", f"{p2_m['low_vol_2014']:,}", f"{p2_m['high_vol_2014']:,}"],
            "Methodology": ["Observed Historical Actual", "Ensemble Forecast (SARIMAX + Naive)", "Q1 Actual + 39w Projection", "Moving-Block Bootstrap (L=4)", "Moving-Block Bootstrap (L=4)"]
        }
        st.table(pd.DataFrame(q_data))
    with f_col2:
        st.markdown("### Capacity Scenarios & Model Benchmarking")
        st.metric("Full-Year 2014 Base Forecast", f"{p2_m['base_vol_2014']:,} tickets", delta=f"95% PI: [{p2_m['low_vol_2014']:,} - {p2_m['high_vol_2014']:,}]")
        st.metric("2015 Capacity Baseline Target", f"{p2_m['scenario_2015_plus5pct']:,} tickets", delta="+5.0% Planning Trend")

        st.markdown(f"**{p2['provenance']['n_folds']}-Fold Rolling-Origin Backtest Performance (MAPE)**:")
        mape_df = pd.DataFrame([
            {"Model": k, "Backtest MAPE": f"{v:.2f}%"} for k, v in sorted(p2_m['backtest_mape'].items(), key=lambda x: x[1])
        ])
        st.dataframe(mape_df, width="stretch", hide_index=True)

# -------------------------------------------------------------------------------------------------
# VIEW 4: PILLAR 3 — AUTO-TAGGING & ROUTING
# -------------------------------------------------------------------------------------------------
elif sidebar_nav == "Pillar 3: Auto-Tagging & Routing":
    st.subheader("Pillar 3 — ITIL Priority Engine & First-Touch Department Auto-Routing")
    st.write("Deterministic 2D ITIL lookup + LightGBM impact imputer for unassigned tickets, with intake-safe routing (Variant D).")

    r_col1, r_col2 = st.columns([1, 1])
    with r_col1:
        st.image(str(FIGURES_DIR / "09_pillar3_department_confusion_matrix.png"), width="stretch")
    with r_col2:
        st.image(str(FIGURES_DIR / "10_pillar3_routing_recall.png"), width="stretch")

    st.markdown("---")
    st.markdown("### Business ROI: Recovered Engineering Capacity")
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("First-Touch Routing Accuracy (Variant D)", f"{p3_r_m['variant_d_unweighted_accuracy']*100:.2f}%", delta=f"+{p3_r_m['variant_d_unweighted_lift']*100:.2f}% vs Baseline")
    s_col2.metric("Controlled Delay Penalty per Bounce", f"+{p3_r_m['delay_penalty_pct']:.2f}%", delta=f"+{p3_r_m['delay_hours_saved_per_bounce']:.2f} hrs/hop (p={p3_r_m['delay_p_value']:.2e})", delta_color="inverse")
    s_col3.metric("Annual Engineering Hours Saved", f"{p3_r_m['annual_hours_saved_base']:,} hrs", delta=f"{p3_r_m['annual_fte_saved_base']:.2f} FTEs Recovered")

    st.markdown("#### Realized Reassignment Sensitivity Analysis")
    sens_data = [
        {"Realized Conversion": "25% Realized", "Effective Lift": f"{p3_r_m['roi_sensitivity_25']['effective_lift']*100:.2f}%", "Hours Saved": f"{p3_r_m['roi_sensitivity_25']['hours']:,} hrs", "Capacity Recovered": f"{p3_r_m['roi_sensitivity_25']['fte']:.2f} FTEs"},
        {"Realized Conversion": "50% Realized", "Effective Lift": f"{p3_r_m['roi_sensitivity_50']['effective_lift']*100:.2f}%", "Hours Saved": f"{p3_r_m['roi_sensitivity_50']['hours']:,} hrs", "Capacity Recovered": f"{p3_r_m['roi_sensitivity_50']['fte']:.2f} FTEs"},
        {"Realized Conversion": "100% Realized (Base)", "Effective Lift": f"{p3_r_m['roi_sensitivity_100']['effective_lift']*100:.2f}%", "Hours Saved": f"{p3_r_m['roi_sensitivity_100']['hours']:,} hrs", "Capacity Recovered": f"{p3_r_m['roi_sensitivity_100']['fte']:.2f} FTEs"},
    ]
    st.dataframe(pd.DataFrame(sens_data), width="stretch", hide_index=True)

# -------------------------------------------------------------------------------------------------
# VIEW 5: PILLAR 4 — ASSET RISK & RFC
# -------------------------------------------------------------------------------------------------
elif sidebar_nav == "Pillar 4: Asset Risk & RFC":
    st.subheader("Pillar 4 — Proactive RFC Detection & Empirical Bayes Asset Risk Index")
    st.write("Rigorous leakage-free evaluation of RFC signal + Bayesian-shrunk Configuration Item (CI) Risk Index.")

    st.image(str(FIGURES_DIR / "12_pillar4_ci_asset_risk_matrix.png"), width="stretch")

    st.markdown("### Top Critical Enterprise Infrastructure Assets")
    try:
        min_n = p4_m["min_incidents_threshold"]
        m_shrink = p4_m["bayes_shrinkage_m"]
        eval_cnt = p4_m["evaluated_assets_count"]
        rho_val = p4_m["min_spearman_rho"]
        rfc_prev = p4_m["rfc_prevalence"]
    except KeyError:
        st.error("metrics.json out of date - re-run python -m src.run_all")
        st.stop()

    st.markdown(f'<div class="callout-info"><strong>Bayesian Shrinkage Filter (Minimum Threshold N &ge; {min_n})</strong>: Raw incident frequency scoring exhibits severe small-sample volatility (e.g. assets with single incidents topping raw risk lists). We filter on {eval_cnt} active assets with at least {min_n} incidents and apply Empirical Bayes shrinkage (m={m_shrink:.1f}), guaranteeing rank stability (Spearman rho = {rho_val:.4f}).</div>', unsafe_allow_html=True)

    top_n = st.slider("Number of Assets to Display", 5, 50, 15)
    cols = ['CI_Name', 'CI_Cat', 'total_incidents', 'p1p2_incidents', 'rfc_incidents', 'ARI_Final', 'Risk_Tier']
    display_df = p4_assets[cols].head(top_n).copy()
    display_df.rename(columns={
        'total_incidents': 'Total Incidents',
        'p1p2_incidents': 'P1/P2 Incidents',
        'rfc_incidents': 'RFC Incidents',
        'ARI_Final': 'Asset Risk Index (0-100)',
        'Risk_Tier': 'Risk Tier'
    }, inplace=True)
    display_df['Asset Risk Index (0-100)'] = display_df['Asset Risk Index (0-100)'].map("{:.1f}".format)
    display_df['Total Incidents'] = display_df['Total Incidents'].map("{:,}".format)
    display_df['P1/P2 Incidents'] = display_df['P1/P2 Incidents'].map("{:,.0f}".format)
    display_df['RFC Incidents'] = display_df['RFC Incidents'].map("{:,.0f}".format)
    st.dataframe(display_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### Proactive RFC Prediction (Negative Result Audit)")
    rfc_c1, rfc_c2 = st.columns([1, 1])
    with rfc_c1:
        st.image(str(FIGURES_DIR / "11_pillar4_rfc_roc_curve.png"), width="stretch")
    with rfc_c2:
        st.markdown(f"""
        **Scientific Integrity Audit**:
        - **Test Holdout ROC-AUC**: **{p4_m['rfc_roc_auc']:.4f}** (Near random chance baseline)
        - **Test Holdout PR-AUC**: **{p4_m['rfc_pr_auc']:.4f}** (Prevalence baseline: ~{rfc_prev*100:.2f}%)
        
        **Why Circular Leakage Was Eliminated**:
        Historical models claimed inflated separation by leaking post-investigation fields (`Related_Change`, `No_of_Related_Changes`). In standard ITIL workflows, change identifiers are created after troubleshooting begins. Restricted strictly to intake-available metadata, tabular feeds carry zero proactive separation. Reporting this honest negative result protects engineering from deploying lookahead-dependent models into production.
        """)
