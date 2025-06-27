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
#-----------------

#---------------------------------
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

# تحليل سهم واحد (النسخة المحسنة)
def analyze_stock(ticker):
    try:
        # تحميل البيانات مع فترة أطول لضمان وجود بيانات كافية
        df = yf.download(ticker, period="1y")
        
        if df.empty:
            st.warning(f"⚠️ لا توجد بيانات متاحة للسهم {ticker} على Yahoo Finance.")
            return
            
        # فحص الأعمدة المطلوبة
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"⚠️ البيانات المطلوبة غير متوفرة للسهم {ticker}. الأعمدة المفقودة: {missing_cols}")
            st.write("البيانات المتاحة:", df.columns.tolist())
            return
            
        # فحص وجود قيم فارغة
        if df[required_cols].isnull().values.any():
            st.warning(f"⚠️ يوجد قيم فارغة في بيانات السهم {ticker}. يتم تنظيف البيانات...")
            df = df.dropna(subset=required_cols)
            
        if df.empty:
            st.warning(f"⚠️ لا توجد بيانات كافية للسهم {ticker} بعد التنظيف.")
            return
            
        # حساب المؤشرات الفنية
        try:
            df = calculate_indicators(df)
        except Exception as e:
            st.error(f"❌ خطأ في حساب المؤشرات الفنية لـ {ticker}: {str(e)}")
            return
            
        # تحضير البيانات للتدريب
        try:
            X, y = prepare_data(df)
            
            if X.empty or y.empty:
                st.warning(f"⚠️ البيانات غير كافية للتدريب للسهم {ticker}.")
                return
                
            # تدريب النموذج
            model, acc = train_model(X, y)
            pred = model.predict(X.tail(1))[0]
            
            # عرض النتائج
            col1, col2, col3 = st.columns(3)
            col1.metric("السعر الحالي", f"{df['Close'].iloc[-1]:.2f}")
            col2.metric("دقة النموذج", f"{acc*100:.2f}%")
            col3.metric("التوقع غداً", "⬆️ ارتفاع" if pred else "⬇️ انخفاض")
            
            # عرض بيانات السهم
            st.subheader("📈 بيانات السهم التاريخية")
            st.dataframe(df.tail(10))
            
        except Exception as e:
            st.error(f"❌ خطأ في تحليل البيانات لـ {ticker}: {str(e)}")
            
    except Exception as e:
        st.error(f"❌ فشل تحميل بيانات السهم {ticker}: {str(e)}")

# عرض الأخبار والاسهم الصاعدة
st.sidebar.header("📰 الأخبار المالية")
news = get_financial_news()
for article in news[:5]:
    st.sidebar.write(f"### {article['title']}")
    st.sidebar.write(article['description'])
    st.sidebar.write("---")

st.sidebar.header("🚀 الأسهم الأكثر صعوداً")
gainers = get_top_gainers()
if not gainers.empty:
    st.sidebar.dataframe(gainers)
else:
    st.sidebar.warning("لا يمكن جلب بيانات الأسهم الصاعدة")

# الإدخال الرئيسي
st.header("🔍 تحليل الأسهم")
tickers = st.text_input("أدخل رموز الأسهم (مفصولة بفاصلة)", "AAPL,MSFT,TSLA,GOOGL")

if st.button("تحليل الأسهم"):
    for t in tickers.split(','):
        t = t.strip().upper()
        if t:  # تأكد من أن الرمز غير فارغ
            st.subheader(f"📌 التحليل الفني - {t}")
            analyze_stock(t)
