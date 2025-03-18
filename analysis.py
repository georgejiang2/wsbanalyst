import json
import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import yfinance as yf
from tqdm import tqdm

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Load spaCy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Custom financial lexicon to supplement VADER
wsb_lexicon = {
    # Bullish terms
    "moon": 3.0, "mooning": 3.0, "to the moon": 3.0, "rocket": 2.5, "🚀": 3.0, 
    "bull": 2.0, "bullish": 2.5, "calls": 1.5, "long": 1.5, "buy": 1.5, 
    "yolo": 1.0, "tendies": 2.0, "printing": 1.5, "diamond hands": 1.5, "💎🙌": 2.0,
    
    # Bearish terms
    "drill": -2.5, "drilling": -2.5, "tank": -2.0, "tanking": -2.5, "puts": -1.5,
    "bear": -2.0, "bearish": -2.5, "short": -1.5, "sell": -1.5, "dump": -2.0,
    "bagholder": -2.0, "paper hands": -1.0, "guh": -2.5, "crash": -3.0,
    
    # WSB-specific sentiment modifiers
    "autist": 0.0, "retard": 0.0, "ape": 0.5, "smooth brain": 0.0,
    "casino": -0.5, "wendy's": -1.0, "sir": 0.0, "degen": -0.5,
}

# Add the lexicon to VADER
for word, score in wsb_lexicon.items():
    sia.lexicon[word] = score

def extract_tickers(text):
    """Extract potential stock tickers from text using regex and filtering"""
    # Pattern for tickers (1-5 capital letters, sometimes with $ prefix)
    ticker_pattern = r'[$]?[A-Z]{1,5}\b'
    potential_tickers = re.findall(ticker_pattern, text)
    # Clean tickers (remove $ if present)
    cleaned_tickers = [ticker.replace('$', '') for ticker in potential_tickers]
    
    # Filter out common non-ticker capital words
    common_words = {'CEO', 'IPO', 'EPS', 'ATH', 'PE', 'OP', 'DD', 'LMAO', 'LMFAO', 'K', 'YOU', 'IT', 'BUY', 'DIP', 'BLK', 'FTC', 'A', 'BOOM', 'O', 'AS', 'BE', 'RSI', 'FOR', 'USD', 'CAD',
                   'AM', 'PM', 'EST', 'PST', 'EDT', 'PDT', 'USA', 'IMO', 'YOLO', 'FOMO', 'FD', 'KONG', 'HIMS', 'UK'}
    
    filtered_tickers = [ticker for ticker in cleaned_tickers if ticker not in common_words]
    
    # Verify tickers exist using yfinance (optional, can slow down processing)
    verified_tickers = []
    for ticker in set(filtered_tickers):  # Use set to remove duplicates
        try:
            if ticker == "SPX":
                continue
            info = yf.Ticker(ticker).info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                verified_tickers.append(ticker)
        except:
            continue
    
    return verified_tickers

def analyze_post_sentiment(post):
    """Analyze a WSB post to determine buy/sell sentiment for each mentioned ticker"""
    # Combine title and post text
    full_text = post['title']
    if post['text']:
        full_text += " " + post['text']
    # Extract tickers
    tickers = extract_tickers(full_text)
    
    # If no tickers found, analyze comments for tickers
    # comment_text = ""
    # if not tickers and 'comments' in post:
    #     comment_text = " ".join(post['comments'])
    #     tickers = extract_tickers(comment_text)
    
    # If still no tickers, return empty result
    if not tickers:
        return {
            'post_id': post['id'],
            'title': post['title'],
            'tickers': [],
            'sentiment': 'unknown',
            'sentiment_score': 0,
            'confidence': 0
        }
    
    # Get overall sentiment score for the entire post content
    post_content = full_text
    if 'comments' in post:
        comment_text = " ".join(post['comments'][:5])  # Include first 5 comments
        post_content += " " + comment_text
    
    overall_sentiment = sia.polarity_scores(post_content)
    overall_compound = overall_sentiment['compound']
    
    # Calculate per-ticker sentiment
    ticker_sentiments = {}
    for ticker in tickers:
        # Find mentions of this ticker in the text
        ticker_pattern = re.compile(r'\b' + re.escape(ticker) + r'\b')
        
        # Get text segments around ticker mentions (30 words before and after)
        segments = []
        for text_source in [full_text, comment_text]:
            if not text_source:
                continue
                
            words = text_source.split()
            for i, word in enumerate(words):
                if ticker_pattern.search(word):
                    start = max(0, i - 30)
                    end = min(len(words), i + 30)
                    segment = " ".join(words[start:end])
                    segments.append(segment)
        
        # If no specific segments found, use overall sentiment
        if not segments:
            ticker_sentiments[ticker] = {
                'score': overall_compound,
                'sentiment': get_sentiment_label(overall_compound),
                'confidence': min(abs(overall_compound) * 100, 100)
            }
        else:
            # Calculate sentiment for segments containing this ticker
            ticker_text = " ".join(segments)
            ticker_score = sia.polarity_scores(ticker_text)['compound']
            ticker_sentiments[ticker] = {
                'score': ticker_score,
                'sentiment': get_sentiment_label(ticker_score),
                'confidence': min(abs(ticker_score) * 100, 100)
            }
    
    return {
        'post_id': post['id'],
        'title': post['title'],
        'tickers': tickers,
        'ticker_sentiments': ticker_sentiments,
        'overall_sentiment': get_sentiment_label(overall_compound),
        'overall_score': overall_compound,
        'overall_confidence': min(abs(overall_compound) * 100, 100)
    }

def get_sentiment_label(compound_score):
    """Convert compound sentiment score to a label"""
    if compound_score >= 0.2:
        return 'buy'
    elif compound_score <= -0.2:
        return 'sell'
    else:
        return 'neutral'

def analyze_from_json(json_file="wsb_posts.json"):
    """Load posts from JSON file and analyze their sentiment"""
    # Load the JSON data
    with open(json_file, 'r') as f:
        posts = json.load(f)
    
    analyzed_posts = []
    
    print(f"Analyzing {len(posts)} posts from {json_file}...")
    
    for post in tqdm(posts, desc="Processing posts"):
        # Analyze sentiment
        sentiment_analysis = analyze_post_sentiment(post)
        if len(sentiment_analysis['tickers']) == 0:
            continue
        # Add sentiment analysis to post data
        post_data = post.copy()
        post_data.update({
            'tickers': sentiment_analysis['tickers'],
            'sentiment': sentiment_analysis['overall_sentiment'],
            'sentiment_score': sentiment_analysis['overall_score'],
            'confidence': sentiment_analysis['overall_confidence']
        })
        
        analyzed_posts.append(post_data)
        
        # For posts with multiple tickers, create individual entries for each ticker
        if len(sentiment_analysis['tickers']) > 1:
            analyzed_posts.pop()
            for ticker in sentiment_analysis['tickers']:
                ticker_specific_post = post_data.copy()
                ticker_specific_post['tickers'] = [ticker]  # Set to just this ticker
                
                # Update with ticker-specific sentiment data
                ticker_sentiment = sentiment_analysis['ticker_sentiments'][ticker]
                ticker_specific_post.update({
                    'sentiment': ticker_sentiment['sentiment'],
                    'sentiment_score': ticker_sentiment['score'],
                    'confidence': ticker_sentiment['confidence']
                })

                analyzed_posts.append(ticker_specific_post)
    
    return analyzed_posts

def save_analyzed_posts(posts, filename="wsb_sentiment_analysis.csv"):
    """Save analyzed posts to CSV"""
    # Create DataFrame
    df = pd.DataFrame([
        {
            'post_id': post['id'],
            'title': post['title'],
            'tickers': ','.join(post['tickers']),
            'sentiment': post['sentiment'],
            'sentiment_score': post['sentiment_score'],
            'confidence': post['confidence'],
            'upvotes': post['upvotes'],
            'created_at': post['created_at'],
            'url': post['url']
        }
        for post in posts
    ])
    
    # Save to CSV
    df = df[df['tickers'].notna() & (df['tickers'] != '') & (df['sentiment'] != 'unknown')]

    # Remove rows where 'sentiment' is 'Unknown'
    df = df[df['sentiment'] != 'Unknown']

    df = df.drop(columns=['post_id', 'title', 'upvotes'])
    df.to_csv(filename, index=False)
    print(f"Saved {len(posts)} analyzed posts to {filename}")

# Run the analysis
if __name__ == "__main__":
    print("Analyzing WSB posts from JSON file for stock buy/sell sentiment...")
    analyzed_posts = analyze_from_json("wsb_posts.json")
    
    # Save results
    save_analyzed_posts(analyzed_posts)