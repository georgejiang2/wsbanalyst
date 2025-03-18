import praw
import time
from datetime import datetime, timedelta

# Initialize Reddit API using PRAW (Python Reddit API Wrapper)
reddit = praw.Reddit(client_id='arfcXl0wxWzmPnMBXnhPmA',
                     client_secret='9U6IMEUynxZ2hQhx8CbT1-K3vrxVGw',
                     user_agent='wsbscraper (by u/Appropriate_Still445)')


import json

def save_to_json(posts, filename="wsb_posts.json"):
    with open(filename, 'w') as f:
        json.dump(posts, f, indent=4)
    print(f"Saved {len(posts)} posts to {filename}")


# Function to get posts from WSB from the past week
def get_wsb_posts_past_week(subreddit='wallstreetbets'):
    posts = []
    
    # Calculate timestamp for one week ago
    one_week_ago = datetime.now() - timedelta(days=1)
    one_week_ago_timestamp = int(one_week_ago.timestamp())
    
    # Use search with time filter
    for submission in reddit.subreddit(subreddit).new(limit=None):
        # Stop when we reach posts older than a week
        if submission.created_utc < one_week_ago_timestamp:
            break

        post_url = submission.url
        
        # Check if it's a self post or a link post
        if submission.is_self:
            post_url = f"https://www.reddit.com{submission.permalink}"

        posts.append({
            'title': submission.title,
            'text': submission.selftext,
            'id': submission.id,
            'upvotes': submission.score,
            'comments': [comment.body for comment in submission.comments.list()[:10] if hasattr(comment, 'body')],
            'created_at': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            'url': post_url
        })
        print("currently processing date:" + datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'))
    
    return posts

wsb_posts = get_wsb_posts_past_week()
save_to_json(wsb_posts)