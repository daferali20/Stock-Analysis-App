# pe_analysis.py
import streamlit as st
import yfinance as yf
import pandas as pd
from newsap import get_financial_news  # استيراد دالة الأخبار

def display_stock_news(ticker):
    """عرض أخبار السهم"""
    st.subheader(f"📰 أخبار {ticker}")
    news = get_financial_news(ticker)
    
    if not news:
        st.warning("لا توجد أخبار متاحة حالياً")
        return
    
    for article in news[:3]:  # عرض أهم 3 أخبار
        with st.expander(article['title']):
            st.write(f"**المصدر:** {article['source']['name']}")
            st.write(f"**التاريخ:** {article['publishedAt']}")
            st.write(article['description'])
            if article['urlToImage']:
                st.image(article['urlToImage'], width=300)
            st.markdown(f"[قراءة المزيد]({article['url']})")

def analyze_low_pe_stocks(tickers, market_pe):
    """تحليل الأسهم مع إضافة قسم الأخبار"""
    low_pe_stocks = []
    
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if not ticker:
            continue
            
        pe = get_stock_pe(ticker)
        if pe is None:
            continue
            
        if pe < market_pe:
            stock_data = yf.Ticker(ticker).info
            low_pe_stocks.append({
                'السهم': ticker,
                'P/E': pe,
                'السعر': stock_data.get('currentPrice', 'N/A'),
                'القيمة السوقية': f"{stock_data.get('marketCap', 0)/1e9:.2f} مليار"
            })
            
            # عرض أخبار السهم
            display_stock_news(ticker)
            st.write("---")
    
    return pd.DataFrame(low_pe_stocks)
