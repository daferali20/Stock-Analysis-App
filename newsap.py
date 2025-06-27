# newsap.py
import os
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_financial_news(query=None):
    """جلب الأخبار المالية مع فلترة حسب الاستعلام"""
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        
        if query:
            # أخبار خاصة بسهم معين
            return newsapi.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                domains='bloomberg.com,cnbc.com,marketwatch.com',
                page_size=5
            )['articles']
        else:
            # الأخبار العامة
            return newsapi.get_top_headlines(
                category='business',
                language='en',
                country='us',
                page_size=10
            )['articles']
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []
