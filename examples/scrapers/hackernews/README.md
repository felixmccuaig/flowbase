# Hacker News Scraper Example

A simple example scraper that collects top stories from Hacker News using their public API.

## Overview

This scraper demonstrates:
- HTTP API data collection
- Daily partitioned storage
- Automatic compaction
- Clean, declarative configuration

## What It Does

Fetches the top 100 stories from Hacker News including:
- Story title, URL, and text
- Author username
- Score (upvotes)
- Number of comments (descendants)
- Story type
- Unix timestamp

## Files

```
hackernews/
├── scraper.py              # Python scraper using HN API
├── configs/
│   ├── hackernews_scraper.yaml  # Scraper config
│   └── hackernews_table.yaml    # Table storage config
└── README.md
```

## Usage

### 1. Create the table

```bash
flowbase table create examples/scrapers/hackernews/configs/hackernews_table.yaml
```

### 2. Run the scraper

```bash
# Scrape today's top stories
flowbase scraper run examples/scrapers/hackernews/configs/hackernews_scraper.yaml

# Scrape for a specific date
flowbase scraper run examples/scrapers/hackernews/configs/hackernews_scraper.yaml --date 2025-01-05
```

### 3. Query the data

```bash
# View top stories by score
flowbase table query examples/scrapers/hackernews/configs/hackernews_table.yaml \
  "SELECT title, score, by, descendants FROM hackernews_stories ORDER BY score DESC LIMIT 10"

# Count stories by type
flowbase table query examples/scrapers/hackernews/configs/hackernews_table.yaml \
  "SELECT type, COUNT(*) as count FROM hackernews_stories GROUP BY type"

# Find stories with most comments
flowbase table query examples/scrapers/hackernews/configs/hackernews_table.yaml \
  "SELECT title, descendants FROM hackernews_stories ORDER BY descendants DESC LIMIT 10"
```

### 4. Compact monthly

```bash
flowbase table compact examples/scrapers/hackernews/configs/hackernews_table.yaml \
  --period 2025-01 --delete-source
```

## Schedule

The scraper is configured to run every 6 hours (see `hackernews_scraper.yaml`):
```yaml
schedule:
  cron: "0 */6 * * *"  # Every 6 hours
  timezone: "UTC"
```

## Data Source

- **API**: [Hacker News Firebase API](https://github.com/HackerNews/API)
- **Rate Limits**: None documented, but be respectful
- **Free**: Yes, public API

## Example Analysis

After collecting a few days of data, you could:

1. **Track trending topics**: Analyze title text for common keywords
2. **Author analysis**: Find most prolific or successful authors
3. **Time patterns**: When are top stories posted?
4. **Engagement metrics**: Relationship between score and comments

## Customization

Edit `scraper.py` to:
- Change number of stories (currently 100)
- Add more fields from the API
- Fetch comments or user data
- Filter by story type

## API Reference

Hacker News API returns these fields:
- `id`: Story ID
- `title`: Story title
- `url`: External URL (if link post)
- `text`: Story text (if self post)
- `score`: Points/upvotes
- `by`: Author username
- `time`: Unix timestamp
- `type`: "story", "job", "poll"
- `descendants`: Comment count

See: https://github.com/HackerNews/API
