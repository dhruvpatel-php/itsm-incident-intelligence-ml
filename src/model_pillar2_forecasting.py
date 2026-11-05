import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.config import CLEANED_DATA_PATH, DAILY_VOLUME_PATH, QUARTERLY_VOLUME_PATH, FIGURES_DIR, MODELS_DIR

def build_forecasting():
    print("=" * 70)
    print("PILLAR 2: INCIDENT VOLUME FORECASTING (QUARTERLY & ANNUAL)")
    print("=" * 70)
    
    df = pd.read_csv(CLEANED_DATA_PATH, low_memory=False)
    df['Open_DT'] = pd.to_datetime(df['Open_DT'])
    
    # 1. Daily Aggregations
    daily = df.set_index('Open_DT').resample('D')['Incident_ID'].count().to_frame(name='Total_Incidents')
    # Filter starting from when continuous ticketing began in 2013
    daily_2013 = daily[daily.index >= '2013-01-01'].copy()
    daily_2013.to_csv(DAILY_VOLUME_PATH)
    print(f"[Pillar 2] Daily incident data points saved to {DAILY_VOLUME_PATH} ({len(daily_2013)} days)")
    
    # 2. Monthly Aggregation
    monthly = daily_2013.resample('ME')['Total_Incidents'].sum().to_frame(name='Monthly_Volume')
    print("\n[Monthly Volume History]:")
    print(monthly.tail(15))
    
    # 3. Quarterly Aggregation (Overall & By Field)
    quarterly_total = daily_2013.resample('QE')['Total_Incidents'].sum().to_frame(name='Total_Volume')
    
    # By CI_Cat Field
    top_fields = ['application', 'subapplication', 'computer', 'storage', 'hardware']
    quarterly_by_field = df[df['Open_DT'] >= '2013-01-01'].set_index('Open_DT').groupby(
        [pd.Grouper(freq='QE'), 'CI_Cat']
    )['Incident_ID'].count().unstack(fill_value=0)
    
    quarterly_all = quarterly_total.join(quarterly_by_field)
    quarterly_all.to_csv(QUARTERLY_VOLUME_PATH)
    print(f"[Pillar 2] Quarterly volume saved to {QUARTERLY_VOLUME_PATH}")
    print("\n[Quarterly History]:\n", quarterly_all[['Total_Volume'] + [c for c in top_fields if c in quarterly_all.columns]])
    
    # 4. Weekly Aggregation for High-Resolution Statistical Forecasting
    weekly = daily_2013.resample('W-MON')['Total_Incidents'].sum()
    # Train / Test split on weekly series (Hold out last 10 weeks of Q1 2014)
    train_weekly = weekly.iloc[:-10]
    test_weekly = weekly.iloc[-10:]
    
    print(f"\n[Pillar 2] Training on {len(train_weekly)} weeks, Testing on {len(test_weekly)} weeks")
    
    # Model A: Holt-Winters Exponential Smoothing
    hw_model = ExponentialSmoothing(
        train_weekly, trend='add', seasonal=None, initialization_method="estimated"
    ).fit()
    hw_pred = hw_model.forecast(steps=len(test_weekly))
    
    # Model B: SARIMA (p=1, d=1, q=1)
    sarima_model = SARIMAX(train_weekly, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_pred = sarima_model.forecast(steps=len(test_weekly))
    
    # Model C: Machine Learning Lag-Regression (Random Forest)
    df_lags = pd.DataFrame({'y': weekly})
    for l in [1, 2, 3, 4]:
        df_lags[f'lag_{l}'] = df_lags['y'].shift(l)
    df_lags['rolling_mean_4'] = df_lags['y'].shift(1).rolling(4).mean()
    df_lags['week_of_year'] = df_lags.index.isocalendar().week
    df_lags = df_lags.dropna()
    
    X_lags = df_lags.drop(columns=['y'])
    y_lags = df_lags['y']
    
    X_train_lags = X_lags.iloc[:-10]
    y_train_lags = y_lags.iloc[:-10]
    X_test_lags = X_lags.iloc[-10:]
    y_test_lags = y_lags.iloc[-10:]
    
    rf_ts = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_ts.fit(X_train_lags, y_train_lags)
    rf_pred = rf_ts.predict(X_test_lags)
    
    # Benchmark Evaluation
    metrics = {
        "Holt-Winters": {
            "MAE": mean_absolute_error(test_weekly, hw_pred),
            "RMSE": np.sqrt(mean_squared_error(test_weekly, hw_pred)),
            "MAPE": mean_absolute_percentage_error(test_weekly, hw_pred) * 100
        },
        "SARIMA(1,1,1)": {
            "MAE": mean_absolute_error(test_weekly, sarima_pred),
            "RMSE": np.sqrt(mean_squared_error(test_weekly, sarima_pred)),
            "MAPE": mean_absolute_percentage_error(test_weekly, sarima_pred) * 100
        },
        "Random Forest Lag Regressor": {
            "MAE": mean_absolute_error(y_test_lags, rf_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test_lags, rf_pred)),
            "MAPE": mean_absolute_percentage_error(y_test_lags, rf_pred) * 100
        }
    }
    
    print("\n[Time Series Model Comparison on Holdout Test Set]:")
    for m_name, m_vals in metrics.items():
        print(f"  {m_name:28s} | MAE: {m_vals['MAE']:.2f} | RMSE: {m_vals['RMSE']:.2f} | MAPE: {m_vals['MAPE']:.2f}%")
        
    # Fit full production model (SARIMA + Holt-Winters Ensemble) on entire historical series
    full_hw = ExponentialSmoothing(weekly, trend='add', seasonal=None, initialization_method="estimated").fit()
    full_sarima = SARIMAX(weekly, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    
    # 5. Out-of-sample Forecast for Q2, Q3, Q4 2014 (39 weeks ahead)
    forecast_weeks = 39
    future_dates = pd.date_range(start=weekly.index[-1] + pd.Timedelta(days=7), periods=forecast_weeks, freq='W-MON')
    
    pred_hw = full_hw.forecast(steps=forecast_weeks)
    pred_sarima = full_sarima.forecast(steps=forecast_weeks)
    pred_ensemble = (pred_hw + pred_sarima) / 2.0
    
    # Calculate 95% Confidence Interval based on SARIMA forecast standard error
    sarima_res = full_sarima.get_forecast(steps=forecast_weeks)
    conf_int = sarima_res.conf_int(alpha=0.05)
    lower_bound = np.maximum(0, conf_int.iloc[:, 0].values)
    upper_bound = conf_int.iloc[:, 1].values
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast_Weekly': pred_ensemble.values,
        'Lower_95': lower_bound,
        'Upper_95': upper_bound
    }).set_index('Date')
    
    # Aggregate into Quarterly Forecasts
    q_forecast = forecast_df.resample('QE').sum()
    print("\n" + "=" * 70)
    print("QUARTERLY INCIDENT VOLUME PROJECTIONS (ABC TECH 2014):")
    print("=" * 70)
    # Include Q1 2014 actual
    q1_2014_actual = quarterly_total.loc['2014-03-31', 'Total_Volume']
    print(f"Q1 2014 (Actual Recorded):  {q1_2014_actual:,.0f} incidents")
    for dt, row in q_forecast.iterrows():
        q_label = f"Q{dt.quarter} {dt.year}"
        print(f"{q_label:14s} (Projected): {row['Forecast_Weekly']:,.0f} incidents [95% CI: {row['Lower_95']:,.0f} - {row['Upper_95']:,.0f}]")
        
    # Annual 2014 Projection
    projected_2014_total = q1_2014_actual + q_forecast['Forecast_Weekly'].sum()
    print(f"\nANNUAL 2014 FULL-YEAR PROJECTION: {projected_2014_total:,.0f} incidents")
    print(f"2013 Full-Year Benchmark:         {quarterly_total.loc['2013-01-01':'2013-12-31', 'Total_Volume'].sum():,.0f} incidents")
    print(f"Year-over-Year Growth Rate:       +{((projected_2014_total / 24606) - 1)*100:.1f}%")
    
    # Annual 2015 Forecast Estimate
    projected_2015_total = projected_2014_total * 1.05 # 5% baseline trend
    print(f"ANNUAL 2015 PROJECTED CAPACITY:   {projected_2015_total:,.0f} incidents (with standard +5% ITIL baseline expansion)")
    
    # 6. Plot Forecast
    print("\n[Pillar 2] Generating 07_pillar2_volume_forecast.png...")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(weekly.index, weekly.values, label='Historical Recorded Volume (Weekly)', color='#2c7bb6', lw=2)
    ax.plot(future_dates, pred_ensemble.values, label='Ensemble Forecast (Q2-Q4 2014)', color='#d7191c', lw=2.5, linestyle='--')
    ax.fill_between(future_dates, lower_bound, upper_bound, color='#fdae61', alpha=0.35, label='95% Prediction Interval')
    ax.axvline(x=weekly.index[-1], color='gray', linestyle=':', lw=1.5, label='Forecast Horizon Boundary')
    ax.set_title("ABC Tech IT Incident Inflow Forecast with 95% Confidence Bounds", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Timeline", fontsize=12)
    ax.set_ylabel("Weekly Incident Volume", fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "07_pillar2_volume_forecast.png", dpi=300)
    plt.close(fig)

    # 7. Field Breakdown Forecast Plot
    print("[Pillar 2] Generating 08_pillar2_ci_forecasts.png...")
    fig, ax = plt.subplots(figsize=(12, 6))
    field_totals = quarterly_all[[c for c in top_fields if c in quarterly_all.columns]]
    # Format quarterly index
    q_labels = [f"Q{d.quarter} '{str(d.year)[2:]}" for d in field_totals.index]
    bottom = np.zeros(len(field_totals))
    colors = ['#386cb0', '#fdb462', '#7fc97f', '#ef3b2c', '#662506']
    for idx, col in enumerate(field_totals.columns):
        vals = field_totals[col].values
        ax.bar(q_labels, vals, bottom=bottom, label=col.capitalize(), color=colors[idx % len(colors)], edgecolor='black', alpha=0.85)
        bottom += vals
    ax.set_title("Quarterly Incident Volume Segmented by Major IT Field (CI Category)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_ylabel("Incident Count", fontsize=11)
    ax.legend(title="IT Field", loc="upper left", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "08_pillar2_ci_forecasts.png", dpi=300)
    plt.close(fig)

    # Save Forecasting Payload
    forecaster_payload = {
        "hw_model": full_hw,
        "sarima_model": full_sarima,
        "metrics": metrics,
        "q1_2014_actual": float(q1_2014_actual),
        "q_projections": {f"Q{dt.quarter}_{dt.year}": float(v) for dt, v in q_forecast['Forecast_Weekly'].items()},
        "projected_2014_total": float(projected_2014_total),
        "projected_2015_total": float(projected_2015_total)
    }
    joblib.dump(forecaster_payload, MODELS_DIR / "pillar2_volume_forecaster.joblib")
    print("[Pillar 2] Forecaster model saved to 'models/' successfully!")
    return forecaster_payload

if __name__ == "__main__":
    build_forecasting()
