# Netflix Time Series Forecasting App
This Streamlit web app allows users to upload and analyze time series data. It supports forecasting models such as ARIMA, ETS, and Prophet and visualizes time-based trends for better decision-making.

---

## 🚀 Features

- Upload CSV time series data (e.g., Netflix stock prices)
- Automatically detect date and value columns
- Visualize time series plots
- Decompose time series into trend, seasonality, residual
- Forecast future values with ARIMA, ETS, Prophet
- Evaluate model performance using RMSE, MAE, and MAPE
- Default sample dataset loads if user doesn't upload a file

---

## 🌐 Live App

🔗 [Click here to try it online](https://netflix-app-app-mh39ftmt2b3h8pma465dm6.streamlit.app/)

---

## 📦 How to Run Locally

```bash
git clone https://github.com/ChandanaAppana/netflix-streamlit-app.git
cd netflix-streamlit-app
pip install -r requirements.txt
streamlit run app.py
