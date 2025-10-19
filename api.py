# app.py  (patch / replace your existing file contents or merge)
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import base64
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from data_fetching import yahoo_data, market_news, reddit_data
import nltk
import time
import os
import matplotlib
import numpy as np
from collections import defaultdict
from datetime import datetime

# optional for coordination detection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use("Agg")

app = Flask(__name__, static_folder="build", static_url_path="/")
CORS(app)

# Download NLTK resources once
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")

# ---------- In-memory history (simple transient store) ----------
HISTORY = defaultdict(lambda: {"volumes": [], "prices": [], "timestamps": []})
HISTORY_MAX = 200  # keep last N ticks per ticker


def update_history(yahoo_dict):
    now = datetime.utcnow().isoformat()
    for ticker, rec in yahoo_dict.items():
        hist = HISTORY[ticker]
        hist["volumes"].append(float(rec.get("volume", 0) or 0))
        # prefer explicit 'price' else derive from pct_change and prior price
        price = rec.get("price")
        if price is None:
            price = rec.get("last_price") or 0
        hist["prices"].append(float(price or 0))
        hist["timestamps"].append(now)
        # trim history
        if len(hist["volumes"]) > HISTORY_MAX:
            hist["volumes"] = hist["volumes"][-HISTORY_MAX:]
            hist["prices"] = hist["prices"][-HISTORY_MAX:]
            hist["timestamps"] = hist["timestamps"][-HISTORY_MAX:]


# ---------- Utility / detection functions ----------
def zscore(arr):
    arr = np.array(arr, dtype=float)
    if arr.size < 2:
        return 0.0
    mu = arr.mean()
    sigma = arr.std(ddof=0)
    return float((arr[-1] - mu) / (sigma if sigma != 0 else 1.0))


def rolling_series(values, window=20):
    return values[-window:] if values else []


def detect_volume_spike(ticker):
    vols = rolling_series(HISTORY[ticker]["volumes"], window=20)
    if not vols:
        return 0.0
    return max(0.0, zscore(vols))  # positive z-score -> spike


def detect_price_jump(ticker):
    prices = rolling_series(HISTORY[ticker]["prices"], window=20)
    if len(prices) < 2:
        return 0.0
    prev, last = prices[-2], prices[-1]
    if prev == 0:
        return 0.0
    pct = (last - prev) / prev  # fraction
    returns = np.diff(prices) / np.where(
        np.array(prices[:-1]) == 0, 1, np.array(prices[:-1])
    )
    vol = float(np.std(returns)) if returns.size > 1 else 0.0
    score = abs(pct) / (vol if vol > 0 else 1.0)
    return float(score)


def detect_price_volume_divergence(ticker, yahoo_record):
    vols = rolling_series(HISTORY[ticker]["volumes"], window=20)
    if not vols:
        return 0.0
    avg_vol = float(np.mean(vols)) if len(vols) >= 1 else 0.0
    pct_change = float(yahoo_record.get("pct_change", 0) or 0)
    vol = float(yahoo_record.get("volume", 0) or 0)
    if avg_vol == 0:
        return 0.0
    vol_ratio = vol / (avg_vol if avg_vol != 0 else 1.0)
    divergence = 0.0
    # tweak thresholds as needed
    if abs(pct_change) > 2.0:
        if vol_ratio < 0.5:
            divergence = (abs(pct_change) / 100.0) * (1.0 / (vol_ratio + 0.01))
    return float(divergence)


def detect_social_hype(ticker, news_list, reddit_list):
    combined = (news_list or []) + (reddit_list or [])
    mention_count = 0
    texts = []
    for item in combined:
        text = " ".join(
            [str(item.get(k, "")) for k in ("title", "description", "body")]
        ).lower()
        texts.append(text)
        if ticker.lower() in text:
            mention_count += 1
    score = min(1.0, mention_count / 20.0)  # 20 mentions -> high
    return float(score), texts


def detect_message_coordination(texts):
    if not texts or len(texts) < 2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        X = vectorizer.fit_transform(texts)
        sim = cosine_similarity(X)
        n = sim.shape[0]
        if n < 2:
            return 0.0
        high = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                if sim[i, j] > 0.8:
                    high += 1
        frac = high / total if total else 0.0
        return float(frac)
    except Exception:
        return 0.0


def compute_manipulation_signals(yahoo_dict, news_list, reddit_list):
    # update transient history
    update_history(yahoo_dict)

    manipulation = {}
    for ticker, rec in yahoo_dict.items():
        vol_z = detect_volume_spike(ticker)
        price_jump = detect_price_jump(ticker)
        divergence = detect_price_volume_divergence(ticker, rec)
        social_score, texts = detect_social_hype(ticker, news_list, reddit_list)
        coord = detect_message_coordination(texts)

        # Normalize/clip signals to 0..1
        vol_signal = float(min(1.0, vol_z / 3.0))
        price_signal = float(min(1.0, price_jump / 5.0))
        divergence_signal = float(min(1.0, divergence))
        social_signal = float(min(1.0, social_score))
        coord_signal = float(min(1.0, coord))

        # Weighted manipulation score (tune weights)
        score = (
            0.30 * vol_signal
            + 0.30 * price_signal
            + 0.15 * divergence_signal
            + 0.15 * social_signal
            + 0.10 * coord_signal
        )

        manipulation[ticker] = {
            "manipulation_score": round(score, 3),
            "signals": {
                "volume_z": round(vol_z, 3),
                "vol_signal": round(vol_signal, 3),
                "price_jump": round(price_jump, 3),
                "price_signal": round(price_signal, 3),
                "divergence": round(divergence, 3),
                "divergence_signal": round(divergence_signal, 3),
                "social_mentions": int(social_score * 20),
                "social_signal": round(social_signal, 3),
                "coordination": round(coord_signal, 3),
            },
        }
    return manipulation


# ---------- your existing functions (wordcloud / yahoo graph) ----------
def generate_wordcloud():
    """Generate WordCloud and sentiment from market_news + reddit_data"""
    data1 = market_news()
    data2 = reddit_data()
    model = SentimentIntensityAnalyzer()

    text = " ".join(
        [line.get("title", "") + " " + line.get("description", "") for line in data1]
    )
    text += " ".join(
        [line.get("title", "") + " " + line.get("description", "") for line in data2]
    )

    stop_words = set(stopwords.words("english"))
    tokens = [
        w for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words
    ]
    filtered_text = " ".join(tokens) or "No data available"

    # Compute sentiment
    compound = model.polarity_scores(filtered_text)["compound"]
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        random_state=int(time.time()),
    ).generate(filtered_text)

    buf = BytesIO()
    wordcloud.to_image().save(buf, format="PNG")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return img_base64, sentiment


def generate_yahoo_graph():
    """Generate Yahoo sentiment graph and overall sentiment"""
    data = yahoo_data()
    df = pd.DataFrame(data).T
    fig, ax = plt.subplots(figsize=(10, 6))

    sentiments = []
    for stock in df.index:
        pct_change = df.loc[stock, "pct_change"]
        max_abs_change = df["pct_change"].abs().max()
        sentiment = pct_change / max_abs_change if max_abs_change != 0 else 0
        sentiments.append(sentiment)
        ax.plot([0, 1], [0, sentiment], marker="o", linestyle="-", label=stock)

    # Overall sentiment (average)
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    if avg_sentiment >= 0.05:
        overall_sentiment = "Positive"
    elif avg_sentiment <= -0.05:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    ax.set_xlabel("Updates")
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Sentiment for All Stocks (Derived from % Change)")
    ax.set_ylim(-1, 1)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8)

    buf = BytesIO()
    fig.savefig(buf, format="PNG")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return img_base64, overall_sentiment


# ---------- API endpoint: include manipulation results ----------
@app.route("/api")
def combined_endpoint():
    """Return wordcloud, yahoo graph and manipulation signals."""
    wc_img, wc_sentiment = generate_wordcloud()
    yahoo_img, yahoo_sentiment = generate_yahoo_graph()

    # raw yahoo dict + news/reddit for manipulation detection
    yahoo_dict = yahoo_data()
    news_list = market_news()
    reddit_list = reddit_data()
    manipulation = compute_manipulation_signals(yahoo_dict, news_list, reddit_list)

    return jsonify(
        {
            "wordcloud": wc_img,
            "wordcloud_sentiment": wc_sentiment,
            "yahoo": yahoo_img,
            "yahoo_sentiment": yahoo_sentiment,
            "manipulation": manipulation,
        }
    )


# --- React Build Serving ---
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
