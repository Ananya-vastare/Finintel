import praw
from dotenv import load_dotenv
import os
from newsapi import NewsApiClient
import yfinance as yf

load_dotenv()


def reddit_data(limit=50):
    reddit = praw.Reddit(
        client_id=os.getenv("client_id"),
        client_secret=os.getenv("secret_key"),
        user_agent=os.getenv("USER_AGENT"),
    )
    finance_subs = [
        "wallstreetbets",
        "finance",
        "stocks",
        "investing",
        "CryptoCurrency",
        "StockMarket",
    ]
    subreddit = reddit.subreddit("+".join(finance_subs))

    reddit_posts = []

    for post in subreddit.hot(limit=50):
        reddit_posts.append(
            {
                "title": post.title,
                "body": post.selftext or "",
                "score": post.score,
            }
        )
    return reddit_posts


### market_news
def market_news():
    api_key = os.getenv("news_api_key")
    data = NewsApiClient(api_key=api_key)
    response = data.get_everything(
        q="bitcoin OR crypto OR pump OR dump",
        language="en",
        sort_by="publishedAt",
        page_size=5,
    )
    article_list = []
    for article in response["articles"]:
        article_list.append(
            {
                "title": article["title"],
                "description": article["description"][:50],
                "content": article["content"],
            }
        )
    return article_list


def yahoo_data():
    data = yf.download(
        ["BTC-USD", "AAPL", "MSFT", "GME", "AMC", "TSLA"],
        period="100d",
        interval="1d",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    data = data.dropna()

    if len(data) < 2:
        return {}

    previous_price = data["Close"].iloc[-2]
    current_price = data["Close"].iloc[-1]

    yahoo_data_dict = {}
    for ticker in previous_price.index:
        yahoo_data_dict[ticker] = {
            "previous": previous_price[ticker],
            "latest": current_price[ticker],
            "change": current_price[ticker] - previous_price[ticker],
            "pct_change": (
                (current_price[ticker] - previous_price[ticker])
                / previous_price[ticker]
            )
            * 100,
        }
    return yahoo_data_dict


def data_collection():
    data1 = yahoo_data()
    data2 = market_news()
    data3 = reddit_data()

    consolidated = {"reddit": data3, "news": data2, "yahoo": data1}

    return consolidated
