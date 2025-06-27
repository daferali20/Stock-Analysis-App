# app.py
import streamlit as st
from pe_analysis import analyze_low_pe_stocks, get_market_avg_pe
from newsap import get_financial_news

# إعداد الصفحة
st.set_page_config(layout="wide")
st.title("📊 نظام تحليل الأسهم المتكامل")

# شريط جانبي
st.sidebar.header("🔍 خيارات التحليل")
analysis_type = st.sidebar.radio(
    "اختر نوع التحليل",
    ["الأسهم ذات P/E المنخفض", "الأخبار المالية"]
)

# التحليل الرئيسي
if analysis_type == "الأسهم ذات P/E المنخفض":
    st.header("📉 تحليل مكرر الأرباح")
    tickers = st.text_input("أدخل رموز الأسهم", "AAPL,MSFT,TSLA,AMZN,NVDA")
    
    if st.button("تحليل"):
        market_pe = get_market_avg_pe()
        st.write(f"متوسط P/E للسوق: {market_pe:.2f}")
        analyze_low_pe_stocks(tickers.split(','), market_pe)

else:
    st.header("📰 الأخبار المالية")
    news_query = st.text_input("ابحث عن أخبار أسهم محددة")
    news = get_financial_news(news_query if news_query else None)
    
    for article in news:
        col1, col2 = st.columns([1, 3])
        with col1:
            if article['urlToImage']:
                st.image(article['urlToImage'], width=150)
        with col2:
            st.subheader(article['title'])
            st.caption(f"{article['source']['name']} - {article['publishedAt']}")
            st.write(article['description'])
            st.markdown(f"[قراءة الخبر]({article['url']})")
        st.divider()
