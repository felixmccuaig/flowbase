"""Hacker News top stories scraper."""

from datetime import datetime
from typing import List, Dict

import pandas as pd
import requests


def fetch_data_for_day(date: datetime) -> pd.DataFrame:
    """
    Fetch Hacker News top stories for a specific day.

    Args:
        date: Date to fetch data for (used for partitioning)

    Returns:
        DataFrame with story data
    """
    # Hacker News API endpoints
    base_url = "https://hacker-news.firebaseio.com/v0"

    # Get top story IDs
    top_stories_url = f"{base_url}/topstories.json"
    response = requests.get(top_stories_url)
    response.raise_for_status()

    story_ids = response.json()[:100]  # Get top 100 stories

    # Fetch details for each story
    stories = []
    for story_id in story_ids:
        story_data = fetch_story(story_id, base_url)
        if story_data:
            stories.append(story_data)

    df = pd.DataFrame(stories)

    # Add scrape date for partitioning
    df['scrape_date'] = date.strftime('%Y-%m-%d')

    return df


def fetch_story(story_id: int, base_url: str) -> Dict:
    """Fetch a single story's details."""
    try:
        story_url = f"{base_url}/item/{story_id}.json"
        response = requests.get(story_url)
        response.raise_for_status()

        story = response.json()

        if not story:
            return None

        return {
            'id': story.get('id'),
            'title': story.get('title'),
            'url': story.get('url'),
            'score': story.get('score', 0),
            'by': story.get('by'),  # Author username
            'time': story.get('time'),  # Unix timestamp
            'type': story.get('type'),
            'descendants': story.get('descendants', 0),  # Number of comments
            'text': story.get('text'),  # Story text (if self post)
        }
    except Exception as e:
        print(f"Error fetching story {story_id}: {e}")
        return None
