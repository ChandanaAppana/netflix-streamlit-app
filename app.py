import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

st.set_page_config(page_title="Time Series Forecast", layout="wide")
st.title("📈 Time Series Forecasting App")

# Upload CSV
uploaded_file = st.file_uploader("Upload a time series CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.write("Data Preview:")
    st.write(df.head())

    # Select date and target columns
    date_col = st.selectbox("Choose the date column", df.columns)
    target_col = st.selectbox("Choose the value column", df.columns)

    # Convert date column
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df.sort_index()

    # Show line chart
    st.line_chart(df[target_col])

    # Decomposition
    st.subheader("🔍 Time Series Decomposition")
    model_type = st.radio("Select decomposition type", ["Additive", "Multiplicative"])
    try:
        result = seasonal_decompose(df[target_col], model=model_type.lower(), period=30)

        fig, ax = plt.subplots(4, 1, figsize=(10, 8))
        result.observed.plot(ax=ax[0], title='Observed')
        result.trend.plot(ax=ax[1], title='Trend')
        result.seasonal.plot(ax=ax[2], title='Seasonality')
        result.resid.plot(ax=ax[3], title='Residuals')
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Decomposition failed: {e}")

    # Forecasting
    st.subheader("🔮 Forecasting")
    model = st.selectbox("Choose model", ["ARIMA", "ETS", "Prophet"])
    forecast_horizon = st.slider("Forecast steps", 10, 100, 30)

    train = df[target_col].iloc[:-forecast_horizon]
    test = df[target_col].iloc[-forecast_horizon:]

    try:
        if model == "ARIMA":
            arima_model = ARIMA(train, order=(5, 1, 0)).fit()
            forecast = arima_model.forecast(steps=forecast_horizon)

        elif model == "ETS":
            ets_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
            forecast = ets_model.forecast(forecast_horizon)

        elif model == "Prophet":
            prophet_df = train.reset_index()
            prophet_df.columns = ['ds', 'y']
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=forecast_horizon)
            forecast_df = m.predict(future)
            forecast = forecast_df['yhat'].iloc[-forecast_horizon:].values
            test = test[:len(forecast)]  # match sizes

        # Plot
        forecast_series = pd.Series(forecast, index=test.index)
        st.line_chart(pd.DataFrame({"Actual": test, "Forecast": forecast_series}))

        # Evaluation
        st.subheader("📊 Evaluation Metrics")
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mae = mean_absolute_error(test, forecast)
        mape = np.mean(np.abs((test - forecast) / test)) * 100

        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MAPE:** {mape:.2f}%")

    except Exception as e:
        st.error(f"Model failed: {e}")
