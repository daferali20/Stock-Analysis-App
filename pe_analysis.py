import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

def get_market_avg_pe():
    """الحصول على متوسط مكرر أرباح السوق (S&P 500 كمثال)"""
    sp500 = yf.Ticker("^GSPC")
    info = sp500.info
    return info.get('trailingPE', 18)  # 18 كقيمة افتراضية إذا لم تتوفر البيانات

def get_stock_pe(ticker):
    """الحصول على مكرر أرباح سهم معين"""
    try:
        stock = yf.Ticker(ticker)
        pe = stock.info.get('trailingPE')
        if pe is None or np.isnan(pe):
            return None
        return pe
    except:
        return None

def analyze_low_pe_stocks(tickers, market_pe):
    """تحليل الأسهم ذات مكرر الأرباح الأقل من السوق"""
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
                'مكرر الأرباح (P/E)': pe,
                'السعر الحالي': stock_data.get('currentPrice', 'N/A'),
                'قطاع': stock_data.get('sector', 'N/A')
            })
    
    return pd.DataFrame(low_pe_stocks)

def main():
    st.title("📉 تحليل الأسهم ذات مكرر أرباح أقل من السوق")
    
    # الحصول على متوسط مكرر أرباح السوق
    market_pe = get_market_avg_pe()
    st.markdown(f"### متوسط مكرر أرباح السوق الحالي: **{market_pe:.2f}**")
    
    # إدخال رموز الأسهم
    tickers_input = st.text_input("🔍 أدخل رموز الأسهم (مفصولة بفاصلة)", "AAPL,MSFT,TSLA,GOOGL,AMZN,JNJ,JPM,PG,WMT")
    
    if st.button("تحليل"):
        tickers = tickers_input.split(',')
        with st.spinner("جاري تحليل البيانات..."):
            df = analyze_low_pe_stocks(tickers, market_pe)
            
            if not df.empty:
                st.success(f"تم العثور على {len(df)} أسهم ذات مكرر أرباح أقل من السوق")
                
                # عرض البيانات
                st.dataframe(
                    df.sort_values('مكرر الأرباح (P/E)'),
                    column_config={
                        "مكرر الأرباح (P/E)": st.column_config.NumberColumn(
                            format="%.2f",
                            help="نسبة السعر إلى الأرباح"
                        ),
                        "السعر الحالي": st.column_config.NumberColumn(
                            format="%.2f $"
                        )
                    }
                )
                
                # عرض بياني
                st.bar_chart(df.set_index('السهم')['مكرر الأرباح (P/E)'])
            else:
                st.warning("لم يتم العثور على أسهم ذات مكرر أرباح أقل من السوق")

if __name__ == "__main__":
    main()
