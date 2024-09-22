# import yfinance as yf
# ticker_symbol = 'AAPL'  
# ticker = yf.Ticker(ticker_symbol)
# news_data = ticker.news
# for news_item in news_data:
#     print(f"Title: {news_item['title']}")
#     print(f"Publisher: {news_item['publisher']}")
#     print(f"Link: {news_item['link']}")
#     print(f"Published Date: {news_item['providerPublishTime']}")
#     print("\n")

import yfinance as yf
ticker_symbol = 'AAPL'
ticker = yf.Ticker(ticker_symbol)
minute_data = ticker.history(period="1d", interval="1m")
print(minute_data.head())  


