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
    common_words = {'A', 'I', 'ME', 'MY', 'THE', 'OF', 'DD', 'CEO', 'IPO', 'EPS', 'ATH', 'PE', 
                   'AM', 'PM', 'EST', 'PST', 'EDT', 'PDT', 'USA', 'IMO', 'YOLO', 'FOMO', 'FD'}
    
    filtered_tickers = [ticker for ticker in cleaned_tickers if ticker not in common_words]
    
    # Verify tickers exist using yfinance (optional, can slow down processing)
    verified_tickers = []
    for ticker in set(filtered_tickers):  # Use set to remove duplicates
        try:
            info = yf.Ticker(ticker).info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                verified_tickers.append(ticker)
        except:
            continue
    
    return verified_tickers

def analyze_post_sentiment(post):
    """Analyze a WSB post to determine buy/sell sentiment for mentioned tickers"""
    # Combine title and post text
    full_text = post['title']
    
    # Extract tickers
    tickers = extract_tickers(full_text)
    
    # If no tickers found, analyze comments for tickers
    if not tickers and 'comments' in post:
        comment_text = " ".join(post['comments'])
        tickers = extract_tickers(comment_text)
    
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
    
    # Get sentiment score for the entire post content
    post_content = full_text
    if 'comments' in post:
        post_content += " " + " ".join(post['comments'][:5])  # Include first 5 comments
        
    sentiment_scores = sia.polarity_scores(post_content)
    compound_score = sentiment_scores['compound']
    
    # Determine if it's a buy or sell signal
    if compound_score >= 0.2:
        sentiment = 'buy'
    elif compound_score <= -0.2:
        sentiment = 'sell'
    else:
        sentiment = 'neutral'
    
    # Calculate confidence based on absolute value of sentiment
    confidence = min(abs(compound_score) * 100, 100)
    
    return {
        'post_id': post['id'],
        'title': post['title'],
        'tickers': tickers,
        'sentiment': sentiment,
        'sentiment_score': compound_score,
        'confidence': confidence
    }

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
        
        # Add sentiment analysis to post data
        post_data = post.copy()
        post_data.update({
            'tickers': sentiment_analysis['tickers'],
            'sentiment': sentiment_analysis['sentiment'],
            'sentiment_score': sentiment_analysis['sentiment_score'],
            'confidence': sentiment_analysis['confidence']
        })
        
        analyzed_posts.append(post_data)
    
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
            'url': post.get('url', '')
        }
        for post in posts
    ])
    
    # Save to CSV
    df = df[df['tickers'].notna() & (df['tickers'] != '')]

    # Remove rows where 'sentiment' is 'Unknown'
    df = df[df['sentiment'] != 'Unknown']

    # Drop the columns 'post_id', 'title', and 'url'
    df = df.drop(columns=['post_id', 'title', 'url', 'upvotes'])
    df.to_csv(filename, index=False)
    print(f"Saved {len(posts)} analyzed posts to {filename}")

# Run the analysis
if __name__ == "__main__":
    print("Analyzing WSB posts from JSON file for stock buy/sell sentiment...")
    analyzed_posts = analyze_from_json("wsb_posts.json")
    
    # Print some sample results
    print("\nSample Results:")
    for post in analyzed_posts[:5]:
        if post['tickers']:
            print(f"Title: {post['title']}")
            print(f"Tickers: {', '.join(post['tickers'])}")
            print(f"Sentiment: {post['sentiment'].upper()} with {post['confidence']:.1f}% confidence")
            print(f"Score: {post['sentiment_score']:.2f}")
            print("---")
    
    # Save results
    save_analyzed_posts(analyzed_posts)