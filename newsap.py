import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import time
import requests
from dotenv import load_dotenv
import os

# تحميل المفاتيح
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

# إعداد الصفحة
st.set_page_config(page_title="📈 نظام تحليل الأسهم", layout="wide")
st.title("📊 نظام تحليل الأسهم الأمريكي المتقدم")

# زر لتحديث البيانات
if st.button("🔄 تحديث البيانات"):
    st.cache_data.clear()

# جلب الأخبار
@st.cache_data(ttl=3600)
def get_financial_news():
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        news = newsapi.get_top_headlines(category='business', language='en', country='us')
        return news.get('articles', [])
    except:
        return []

# الأسهم الأكثر صعودًا
@st.cache_data(ttl=3600)
def get_top_gainers():
    try:
        headers = {"Authorization": f"Token {TIINGO_API_KEY}"}
        url = "https://api.tiingo.com/tiingo/daily/top"
        res = requests.get(url, headers=headers)
        return pd.DataFrame(res.json()).head(10)
    except:
        return pd.DataFrame()

# مؤشرات فنية
def calculate_indicators(df):
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"البيانات تفتقد الأعمدة التالية: {missing}")

    df = df.dropna(subset=required_cols)

    if df.empty:
        raise ValueError("بيانات السهم غير كافية بعد التنظيف.")

    df = add_all_ta_features(
        df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True
    )
    return df

# تحضير البيانات للتصنيف
def prepare_data(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(5).std()
    df = df.dropna()
    X = df[['Open', 'High', 'Low', 'Volume', 'momentum_rsi', 'trend_macd', 'Volatility']]
    y = df['Target']
    return X, y

# تدريب النموذج
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# تحليل سهم واحد
def analyze_stock(ticker):
    try:
        df = yf.download(ticker, period="6mo")
        if df.empty:
            st.warning(f"لا توجد بيانات للسهم {ticker}.")
            return

        df = calculate_indicators(df)
        X, y = prepare_data(df)

        if X.empty or y.empty:
            st.warning(f"البيانات غير كافية للسهم {ticker} بعد التحضير.")
            return

        model, acc = train_model(X, y)
        pred = model.predict(X.tail(1))[0]

        st.metric("السعر الحالي", f"{df['Close'].iloc[-1]:.2f}")
        st.metric("دقة النموذج", f"{acc*100:.2f}%")
        st.metric("التوقع", "⬆️ ارتفاع" if pred else "⬇️ انخفاض")

    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تحليل السهم {ticker}: {str(e)}")

# الإدخال
tickers = st.text_input("🔍 أدخل رموز الأسهم (مفصولة بفاصلة)", "AAPL,MSFT,TSLA")
for t in tickers.split(','):
    st.subheader(f"📌 التحليل الفني - {t.strip().upper()}")
    analyze_stock(t.strip().upper())
