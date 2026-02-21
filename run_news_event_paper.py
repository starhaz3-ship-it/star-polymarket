#!/usr/bin/env python3
"""
NEWS EVENT FRONT-RUNNER — Paper Trading Bot for Polymarket

Detects breaking news, matches to Polymarket markets, paper-trades low-price
markets before they move. Based on the insight that markets go from 1-30c to
50-90c when major news drops.

Architecture:
  News Sources (30-60s poll) -> Dedup -> Market Match -> LLM Edge -> Paper Trade -> Exit Monitor

V1.0: NewsData.io + Google News RSS + Polymarket spike detection + Claude Haiku matching
V1.1: Fix deaf pipeline — lower min_overlap to 1, extend freshness to 30min, more RSS feeds, raise entry cap
"""

import asyncio
import hashlib
import httpx
import json
import os
import re
import sys
import time
import feedparser
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# PID lock
sys.path.insert(0, str(Path(__file__).parent))
from pid_lock import acquire_pid_lock

# ============================================================================
# CONFIGURATION
# ============================================================================

# News polling
POLL_INTERVAL = 30              # seconds between news checks
NEWS_MAX_AGE = 1800             # 30 min max headline age (was 600 — too tight for RSS delay)
MARKET_CACHE_TTL = 300          # 5 min market cache refresh

# Entry criteria
MAX_ENTRY_PRICE = 0.50          # only buy markets priced < 50c (was 30c — missed 34c Iran deal)
MIN_EDGE = 0.20                 # 20pp minimum edge (LLM_prob - market_price)
MIN_LLM_CONFIDENCE = 0.70      # LLM must be > 70% confident

# Position management
TRADE_SIZE = 5.0                # $5 per trade (paper)
MAX_POSITIONS = 3               # max concurrent positions
EXIT_TARGET = 0.80              # sell when price >= 80c
TRAILING_ACTIVATE = 0.50        # activate trailing stop at 50c
TRAILING_OFFSET = 0.15          # trailing stop = price - 15c
TIMEOUT_HOURS = 4               # max hold time
FLOOR_PRICE = 0.02              # cut loss if price drops to 2c

# API endpoints
GAMMA_API_URL = "https://gamma-api.polymarket.com/events"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
NEWSDATA_API_URL = "https://newsdata.io/api/1/latest"
GOOGLE_NEWS_RSS = "https://news.google.com/rss"

# API keys
NEWSDATA_API_KEY = os.environ.get("NEWSDATA_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Results file
RESULTS_FILE = Path(__file__).parent / "news_event_results.json"

# Stop words for keyword extraction
STOP_WORDS = frozenset({
    "will", "the", "a", "an", "in", "on", "at", "by", "for", "to", "of",
    "be", "is", "are", "was", "were", "been", "being", "have", "has", "had",
    "do", "does", "did", "shall", "should", "may", "might", "must", "can",
    "could", "would", "and", "or", "but", "if", "then", "than", "that",
    "this", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "my", "your", "our", "who", "whom", "which",
    "what", "when", "where", "why", "how", "not", "no", "nor", "so", "as",
    "with", "from", "into", "about", "between", "through", "after", "before",
    "above", "below", "up", "down", "out", "off", "over", "under", "again",
    "new", "says", "said", "report", "reports", "according", "sources",
})


# ============================================================================
# NEWS FEED MANAGER
# ============================================================================

class NewsFeedManager:
    """Poll multiple news sources, dedup, yield fresh headlines."""

    def __init__(self):
        self.seen: set = set()
        self._newsdata_last = 0
        self._newsdata_calls_today = 0
        self._newsdata_day = 0

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()[:16]

    def _is_fresh(self, pub_time: Optional[datetime]) -> bool:
        if not pub_time:
            return True  # if no timestamp, assume fresh
        # Ensure timezone-aware comparison
        if pub_time.tzinfo is None:
            pub_time = pub_time.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - pub_time).total_seconds()
        return age < NEWS_MAX_AGE

    async def poll_newsdata(self) -> list[dict]:
        """NewsData.io — 200 free credits/day, near real-time."""
        if not NEWSDATA_API_KEY:
            return []

        # Rate limit: max ~8 calls/hour to stay under 200/day
        now = time.time()
        today = int(now // 86400)
        if today != self._newsdata_day:
            self._newsdata_day = today
            self._newsdata_calls_today = 0
        if self._newsdata_calls_today >= 190:
            return []
        if now - self._newsdata_last < 120:  # min 2 min between calls
            return []

        articles = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(NEWSDATA_API_URL, params={
                    "apikey": NEWSDATA_API_KEY,
                    "language": "en",
                    "category": "politics,world,business",
                })
                self._newsdata_last = now
                self._newsdata_calls_today += 1
                if r.status_code == 200:
                    data = r.json()
                    for item in data.get("results", []):
                        pub_str = item.get("pubDate", "")
                        pub_dt = None
                        if pub_str:
                            try:
                                pub_dt = datetime.fromisoformat(pub_str.replace(" ", "T").replace("Z", "+00:00"))
                            except Exception:
                                pass
                        articles.append({
                            "title": item.get("title", ""),
                            "source": item.get("source_name", ""),
                            "published": pub_dt,
                            "url": item.get("link", ""),
                            "feed": "newsdata",
                        })
                else:
                    print(f"[NEWSDATA] Error {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"[NEWSDATA] Fetch error: {e}")
        return articles

    async def poll_google_rss(self) -> list[dict]:
        """Google News RSS — free, unlimited, ~5 min delay."""
        articles = []
        # V1.2: 17 RSS feeds across 9 outlets — all free, unlimited
        feeds = [
            # Google News (5 feeds)
            f"{GOOGLE_NEWS_RSS}?hl=en-US&gl=US&ceid=US:en",  # top stories
            f"{GOOGLE_NEWS_RSS}/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFZxYUdjU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en",  # politics
            f"{GOOGLE_NEWS_RSS}/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en",  # world
            f"{GOOGLE_NEWS_RSS}/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en",  # business
            f"{GOOGLE_NEWS_RSS}/search?q=breaking+news&hl=en-US&gl=US&ceid=US:en",  # breaking news search
            # BBC (3 feeds — fastest for international)
            "https://feeds.bbci.co.uk/news/world/rss.xml",
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://feeds.bbci.co.uk/news/politics/rss.xml",
            # Al Jazeera (best for Middle East — Iran, Gaza, Yemen)
            "https://www.aljazeera.com/xml/rss/all.xml",
            # CNN (2 feeds)
            "http://rss.cnn.com/rss/cnn_world.rss",
            "http://rss.cnn.com/rss/cnn_allpolitics.rss",
            # NPR (2 feeds)
            "https://feeds.npr.org/1004/rss.xml",  # world
            "https://feeds.npr.org/1014/rss.xml",  # politics
            # Sky News (2 feeds)
            "https://feeds.skynews.com/feeds/rss/world.xml",
            "https://feeds.skynews.com/feeds/rss/home.xml",
            # NYT (2 feeds — high quality analysis)
            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
        ]
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                for feed_url in feeds:
                    try:
                        r = await client.get(feed_url, headers={"User-Agent": "Mozilla/5.0"})
                        if r.status_code != 200:
                            continue
                        feed = feedparser.parse(r.text)
                        for entry in feed.entries[:40]:  # V1.1: was 20, grab more to avoid missing scroll-off
                            pub_dt = None
                            if hasattr(entry, "published_parsed") and entry.published_parsed:
                                import calendar
                                ts = calendar.timegm(entry.published_parsed)
                                pub_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                            articles.append({
                                "title": entry.get("title", ""),
                                "source": entry.get("source", {}).get("title", "Google News"),
                                "published": pub_dt,
                                "url": entry.get("link", ""),
                                "feed": "google_rss",
                            })
                    except Exception as e:
                        print(f"[RSS] Feed error: {e}")
        except Exception as e:
            print(f"[RSS] Error: {e}")
        return articles

    async def poll_all(self) -> list[dict]:
        """Poll all sources, dedup, return fresh headlines only."""
        results = await asyncio.gather(
            self.poll_newsdata(),
            self.poll_google_rss(),
            return_exceptions=True,
        )
        headlines = []
        for result in results:
            if isinstance(result, list):
                headlines.extend(result)

        # Dedup and filter
        fresh = []
        for h in headlines:
            title = h.get("title", "").strip()
            if not title or len(title) < 15:
                continue
            h_hash = self._hash(title)
            if h_hash in self.seen:
                continue
            self.seen.add(h_hash)
            if self._is_fresh(h.get("published")):
                fresh.append(h)

        # Cap seen set to prevent memory growth
        if len(self.seen) > 10000:
            self.seen = set(list(self.seen)[-5000:])

        if fresh:
            print(f"[NEWS] {len(fresh)} new headlines from {len(headlines)} total")
        return fresh


# ============================================================================
# POLYMARKET MARKET CACHE
# ============================================================================

class PolymarketCache:
    """Cache all active Polymarket markets for instant keyword matching."""

    def __init__(self):
        self.markets: list[dict] = []
        self.keyword_index: dict[str, list[int]] = {}  # keyword -> [market indices]
        self.last_refresh = 0

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract significant keywords from text."""
        words = re.findall(r'[A-Za-z][A-Za-z\']+', text)
        keywords = []
        for w in words:
            wl = w.lower()
            if wl not in STOP_WORDS and len(wl) > 2:
                keywords.append(wl)
        return keywords

    async def refresh(self):
        """Fetch all active markets from Gamma API and build keyword index."""
        now = time.time()
        if now - self.last_refresh < MARKET_CACHE_TTL:
            return

        try:
            all_markets = []
            async with httpx.AsyncClient(timeout=30) as client:
                # Fetch multiple pages of events
                for offset in range(0, 1000, 100):
                    r = await client.get(
                        GAMMA_API_URL,
                        params={
                            "active": "true",
                            "closed": "false",
                            "limit": 100,
                            "offset": offset,
                        },
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    if r.status_code != 200:
                        break
                    events = r.json()
                    if not events:
                        break
                    for event in events:
                        for m in event.get("markets", []):
                            if m.get("closed", True):
                                continue
                            q = m.get("question", "") or event.get("title", "")
                            event_title = event.get("title", "")
                            m["_question"] = q
                            m["_event_title"] = event_title
                            # V1.1: Index BOTH question AND event title for broader keyword coverage
                            combined_text = f"{q} {event_title}"
                            m["_keywords"] = self._extract_keywords(combined_text)
                            all_markets.append(m)

            # Build inverted keyword index
            keyword_index = {}
            for i, m in enumerate(all_markets):
                for kw in m["_keywords"]:
                    if kw not in keyword_index:
                        keyword_index[kw] = []
                    keyword_index[kw].append(i)

            self.markets = all_markets
            self.keyword_index = keyword_index
            self.last_refresh = now
            print(f"[CACHE] {len(all_markets)} active markets, {len(keyword_index)} keywords indexed")

        except Exception as e:
            print(f"[CACHE] Refresh error: {e}")

    # High-value entity keywords — single match is enough for these
    # Conflict zones, geopolitical hotspots, key leaders, market-moving events
    ENTITY_KEYWORDS = frozenset({
        # Conflict zones & hotspots
        "iran", "china", "russia", "ukraine", "taiwan", "korea", "israel",
        "gaza", "palestine", "hamas", "hezbollah", "houthi", "yemen",
        "syria", "iraq", "afghanistan", "libya", "sudan", "somalia",
        "belarus", "japan", "philippines", "myanmar", "venezuela",
        "cuba", "mexico", "pakistan", "india", "kashmir", "tibet",
        "crimea", "donbas", "kherson", "zaporizhzhia",
        # Key leaders & political figures
        "trump", "biden", "putin", "zelensky", "xi", "modi", "kim",
        "khamenei", "netanyahu", "erdogan", "macron", "scholz",
        "starmer", "trudeau", "milei", "maduro", "lula", "bolsonaro",
        "desantis", "vance", "musk", "zuckerberg",
        # Orgs & alliances
        "nato", "fed", "opec", "brics", "pentagon", "cia", "fbi",
        "sec", "congress", "senate", "supreme", "iaea", "who",
        # Markets & crypto
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
        "crypto", "nasdaq", "dow", "treasury", "bond",
        # Event triggers — fast-moving news
        "tariff", "tariffs", "sanction", "sanctions", "embargo",
        "nuclear", "missile", "icbm", "warhead", "enrichment", "uranium",
        "invasion", "strike", "airstrike", "bombing", "drone", "attack",
        "war", "ceasefire", "truce", "peace", "deal", "treaty", "accord",
        "impeach", "impeachment", "resign", "resignation", "indictment",
        "assassination", "assassinated", "coup", "overthrow", "martial",
        "earthquake", "hurricane", "typhoon", "tsunami", "wildfire",
        "pandemic", "outbreak", "epidemic", "quarantine",
        "default", "recession", "depression", "inflation", "crash",
        "shutdown", "blackout", "collapse", "bankrupt", "bailout",
        "election", "referendum", "vote", "inauguration",
        "hostage", "kidnap", "prisoner", "detained", "arrested",
        "hack", "cyberattack", "breach", "ransomware",
        "explosion", "derailment", "shooting", "massacre",
    })

    def find_matches(self, headline: str, min_overlap: int = 2) -> list[dict]:
        """Find markets matching headline keywords. Returns sorted by overlap score.

        V1.2: min_overlap=2 for generic headlines, 1 for entity keywords only.
        This prevents junk (credit cards, home equity) from burning LLM calls.
        """
        h_keywords = self._extract_keywords(headline)
        if not h_keywords:
            return []

        # Check if headline contains high-value entity keywords
        has_entity = any(kw in self.ENTITY_KEYWORDS for kw in h_keywords)

        # Count keyword overlaps per market
        market_scores: dict[int, int] = {}
        for kw in h_keywords:
            for idx in self.keyword_index.get(kw, []):
                market_scores[idx] = market_scores.get(idx, 0) + 1

        # Use min_overlap=1 ONLY for entity keywords, 2 for generic headlines
        effective_min = 1 if has_entity else min_overlap

        # Filter by minimum overlap and sort by score desc
        matches = []
        for idx, score in sorted(market_scores.items(), key=lambda x: -x[1]):
            if score >= effective_min:
                m = self.markets[idx].copy()
                m["_match_score"] = score
                m["_match_keywords"] = [kw for kw in h_keywords if kw in [k.lower() for k in self.markets[idx].get("_keywords", [])]]
                matches.append(m)

        return matches[:10]  # top 10 matches


# ============================================================================
# SIGNAL GENERATOR (LLM-POWERED)
# ============================================================================

class NewsSignalGenerator:
    """Evaluate headline + candidate markets for tradeable edge."""

    MAX_LLM_PER_HOUR = 30  # V1.2: Cap LLM calls to avoid API lockout

    def __init__(self):
        self._llm_calls = 0
        self._llm_timestamps: list[float] = []  # track call times for rate limiting

    async def _get_yes_price(self, market: dict) -> Optional[float]:
        """Get current YES price from CLOB orderbook."""
        token_ids = market.get("clobTokenIds", [])
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        if not token_ids:
            return None
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(CLOB_BOOK_URL, params={"token_id": token_ids[0]})
                if r.status_code != 200:
                    return None
                book = r.json()
                asks = book.get("asks", [])
                if asks:
                    return min(float(a["price"]) for a in asks)
                # Fallback to outcomePrices
                prices = market.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    prices = json.loads(prices)
                return float(prices[0]) if prices else None
        except Exception:
            # Fallback to stored prices
            prices = market.get("outcomePrices", "[]")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except Exception:
                    return None
            return float(prices[0]) if prices else None

    async def _ask_llm(self, headline: str, market_question: str, current_price: float) -> Optional[dict]:
        """Ask Claude Haiku if this news makes the market likely to resolve YES."""
        if not ANTHROPIC_API_KEY:
            return None

        prompt = f"""You are a prediction market analyst. A breaking news headline just dropped.

HEADLINE: {headline}

POLYMARKET QUESTION: {market_question}
CURRENT YES PRICE: ${current_price:.2f} ({current_price*100:.0f}% implied probability)

Does this headline significantly increase the probability of YES for this market?

Respond ONLY with JSON:
{{"probability": <0-100>, "action": "buy_yes" or "buy_no" or "skip", "reasoning": "<1 sentence>"}}

Rules:
- Only say "buy_yes" if the headline DIRECTLY and STRONGLY supports YES
- Only say "buy_no" if the headline DIRECTLY and STRONGLY contradicts YES
- Say "skip" if the connection is weak, indirect, or uncertain
- Be conservative — false positives cost real money"""

        # V1.2: Rate limit LLM calls
        now = time.time()
        self._llm_timestamps = [t for t in self._llm_timestamps if now - t < 3600]
        if len(self._llm_timestamps) >= self.MAX_LLM_PER_HOUR:
            return None

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-3-5-haiku-20241022",
                        "max_tokens": 150,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                self._llm_calls += 1
                self._llm_timestamps.append(now)
                if r.status_code == 200:
                    data = r.json()
                    text = data.get("content", [{}])[0].get("text", "{}")
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        return json.loads(text[start:end])
        except Exception as e:
            print(f"[LLM] Error: {e}")
        return None

    async def evaluate(self, headline: str, candidates: list[dict]) -> list[dict]:
        """Evaluate candidates and return tradeable signals."""
        signals = []

        for market in candidates[:3]:  # V1.2: max 3 LLM calls per headline (was 5)
            question = market.get("_question", "")
            cid = market.get("conditionId", "")

            # Get current price
            yes_price = await self._get_yes_price(market)
            if yes_price is None:
                continue

            # Skip if price already high (move already happened)
            if yes_price > MAX_ENTRY_PRICE:
                continue

            # Ask LLM
            llm_result = await self._ask_llm(headline, question, yes_price)
            if not llm_result:
                continue

            action = llm_result.get("action", "skip")
            prob = llm_result.get("probability", 0) / 100.0
            reasoning = llm_result.get("reasoning", "")

            if action == "skip":
                continue

            # Calculate edge
            if action == "buy_yes":
                edge = prob - yes_price
            elif action == "buy_no":
                edge = (1 - prob) - (1 - yes_price)  # NO price = 1 - YES price
            else:
                continue

            if edge < MIN_EDGE:
                continue

            if prob < MIN_LLM_CONFIDENCE and action == "buy_yes":
                continue

            signal = {
                "headline": headline,
                "market_question": question,
                "condition_id": cid,
                "action": action,
                "current_price": yes_price,
                "llm_probability": prob,
                "edge": edge,
                "reasoning": reasoning,
                "match_score": market.get("_match_score", 0),
                "token_ids": market.get("clobTokenIds", []),
                "market": market,
            }
            signals.append(signal)
            print(f"  [SIGNAL] {action.upper()} | {question[:60]}")
            print(f"    Price: ${yes_price:.2f} -> LLM: {prob:.0%} | Edge: {edge:+.0%} | {reasoning[:80]}")

        return signals


# ============================================================================
# POLYMARKET SPIKE DETECTOR
# ============================================================================

class SpikeDetector:
    """Detect sudden price moves on low-price Polymarket markets."""

    def __init__(self):
        self.price_history: dict[str, list[float]] = {}  # cid -> [prices]
        self.last_check = 0

    async def check_spikes(self, markets: list[dict]) -> list[dict]:
        """Find markets with sudden upward price moves from low base."""
        spikes = []
        now = time.time()

        # Only check every 60s
        if now - self.last_check < 60:
            return spikes
        self.last_check = now

        for m in markets:
            cid = m.get("conditionId", "")
            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except Exception:
                    continue
            if not prices:
                continue

            yes_price = float(prices[0])

            # Initialize history
            if cid not in self.price_history:
                self.price_history[cid] = []
            self.price_history[cid].append(yes_price)

            # Keep last 10 data points
            if len(self.price_history[cid]) > 10:
                self.price_history[cid] = self.price_history[cid][-10:]

            # Check for spike: price was < 15c and jumped > 5c
            history = self.price_history[cid]
            if len(history) >= 3:
                avg_old = sum(history[:-1]) / len(history[:-1])
                current = history[-1]
                if avg_old < 0.15 and current - avg_old > 0.05:
                    spikes.append({
                        "market": m,
                        "old_price": avg_old,
                        "new_price": current,
                        "spike_pct": (current - avg_old) / max(avg_old, 0.01) * 100,
                    })
                    print(f"[SPIKE] {m.get('question', '')[:60]} | ${avg_old:.2f} -> ${current:.2f} (+{(current-avg_old)*100:.0f}c)")

        # Limit history to prevent memory bloat
        if len(self.price_history) > 2000:
            old_keys = list(self.price_history.keys())[:-1000]
            for k in old_keys:
                del self.price_history[k]

        return spikes


# ============================================================================
# PAPER TRADER
# ============================================================================

class NewsEventTrader:
    """Paper trade news events on Polymarket."""

    def __init__(self):
        self.positions: dict[str, dict] = {}  # cid -> position
        self.resolved: list[dict] = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0}
        self._load()

    def _load(self):
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.positions = data.get("active", {})
                self.resolved = data.get("resolved", [])
                self.stats = data.get("stats", self.stats)
                print(f"[LOAD] {len(self.resolved)} resolved, {len(self.positions)} active | "
                      f"{self.stats['wins']}W/{self.stats['losses']}L | PnL: ${self.stats['pnl']:+.2f}")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        data = {
            "active": self.positions,
            "resolved": self.resolved,
            "stats": self.stats,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            RESULTS_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    async def enter_trade(self, signal: dict):
        """Paper-enter a trade based on news signal."""
        cid = signal["condition_id"]
        if cid in self.positions:
            return  # already in this market
        if len(self.positions) >= MAX_POSITIONS:
            print(f"[SKIP] Max {MAX_POSITIONS} positions reached")
            return

        entry_price = signal["current_price"]
        shares = int(TRADE_SIZE / entry_price) if entry_price > 0 else 0
        if shares < 1:
            return
        cost = round(shares * entry_price, 2)

        position = {
            "action": signal["action"],
            "entry_price": entry_price,
            "shares": shares,
            "cost": cost,
            "headline": signal["headline"][:200],
            "market_question": signal["market_question"][:200],
            "llm_probability": signal["llm_probability"],
            "edge": signal["edge"],
            "reasoning": signal["reasoning"][:200],
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "token_ids": signal.get("token_ids", []),
            "trailing_stop": None,
            "high_water": entry_price,
        }
        self.positions[cid] = position
        self._save()

        print(f"[PAPER ENTRY] {signal['action'].upper()} | ${entry_price:.2f} x {shares}sh = ${cost:.2f}")
        print(f"  Market: {signal['market_question'][:70]}")
        print(f"  Headline: {signal['headline'][:70]}")
        print(f"  Edge: {signal['edge']:+.0%} | LLM: {signal['llm_probability']:.0%}")

    async def monitor_exits(self):
        """Check all positions for exit conditions."""
        if not self.positions:
            return

        now = datetime.now(timezone.utc)
        to_close = []

        for cid, pos in self.positions.items():
            # Get current price
            token_ids = pos.get("token_ids", [])
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            if not token_ids:
                continue

            current_price = None
            try:
                async with httpx.AsyncClient(timeout=8) as client:
                    r = await client.get(CLOB_BOOK_URL, params={"token_id": token_ids[0]})
                    if r.status_code == 200:
                        book = r.json()
                        bids = book.get("bids", [])
                        if bids:
                            current_price = max(float(b["price"]) for b in bids)
            except Exception:
                pass

            if current_price is None:
                continue

            # Update high water mark
            if current_price > pos.get("high_water", 0):
                pos["high_water"] = current_price

            # Check exit conditions
            exit_reason = None

            # 1. Target exit: price >= 80c
            if current_price >= EXIT_TARGET:
                exit_reason = f"TARGET (${current_price:.2f} >= ${EXIT_TARGET:.2f})"

            # 2. Trailing stop: activated at 50c+
            elif current_price >= TRAILING_ACTIVATE:
                stop_level = pos["high_water"] - TRAILING_OFFSET
                if pos.get("trailing_stop") is None:
                    pos["trailing_stop"] = stop_level
                    print(f"  [TRAILING] Activated for {pos['market_question'][:40]} | stop=${stop_level:.2f}")
                else:
                    pos["trailing_stop"] = max(pos["trailing_stop"], stop_level)
                if current_price <= pos["trailing_stop"]:
                    exit_reason = f"TRAILING STOP (${current_price:.2f} <= ${pos['trailing_stop']:.2f})"

            # 3. Floor: price dropped to near zero
            elif current_price <= FLOOR_PRICE:
                exit_reason = f"FLOOR (${current_price:.2f} <= ${FLOOR_PRICE:.2f})"

            # 4. Timeout: held too long
            entry_time = datetime.fromisoformat(pos["entry_time"])
            age_hours = (now - entry_time).total_seconds() / 3600
            if age_hours >= TIMEOUT_HOURS:
                exit_reason = f"TIMEOUT ({age_hours:.1f}h >= {TIMEOUT_HOURS}h)"

            if exit_reason:
                pnl = round((current_price - pos["entry_price"]) * pos["shares"], 2)
                if pos["action"] == "buy_no":
                    pnl = round(((1 - current_price) - (1 - pos["entry_price"])) * pos["shares"], 2)

                pos["exit_price"] = current_price
                pos["exit_time"] = now.isoformat()
                pos["exit_reason"] = exit_reason
                pos["pnl"] = pnl
                pos["status"] = "closed"

                won = pnl > 0
                self.stats["wins" if won else "losses"] += 1
                self.stats["pnl"] += pnl

                tag = "WIN" if won else "LOSS"
                w, l = self.stats["wins"], self.stats["losses"]
                wr = w / (w + l) * 100 if (w + l) > 0 else 0
                print(f"[{tag}] ${pnl:+.2f} | {exit_reason}")
                print(f"  {pos['market_question'][:60]}")
                print(f"  Entry: ${pos['entry_price']:.2f} -> Exit: ${current_price:.2f} | "
                      f"{w}W/{l}L {wr:.0f}%WR | Total: ${self.stats['pnl']:+.2f}")

                self.resolved.append(pos)
                to_close.append(cid)

        for cid in to_close:
            del self.positions[cid]

        if to_close:
            self._save()


# ============================================================================
# MAIN LOOP
# ============================================================================

async def main():
    acquire_pid_lock("news_event_paper")

    print("=" * 70)
    print("NEWS EVENT FRONT-RUNNER V1.0 - PAPER MODE")
    print("=" * 70)
    print(f"News sources: NewsData.io ({('ACTIVE' if NEWSDATA_API_KEY else 'NO KEY')}) + Google News RSS")
    print(f"LLM: Claude Haiku ({('ACTIVE' if ANTHROPIC_API_KEY else 'NO KEY')})")
    print(f"Entry: price < ${MAX_ENTRY_PRICE:.2f} | edge > {MIN_EDGE:.0%} | LLM conf > {MIN_LLM_CONFIDENCE:.0%}")
    print(f"Exit: target ${EXIT_TARGET:.2f} | trailing @${TRAILING_ACTIVATE:.2f} (-${TRAILING_OFFSET:.2f}) | timeout {TIMEOUT_HOURS}h")
    print(f"Size: ${TRADE_SIZE:.2f}/trade | Max positions: {MAX_POSITIONS}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print("=" * 70)

    feeds = NewsFeedManager()
    cache = PolymarketCache()
    signals = NewsSignalGenerator()
    trader = NewsEventTrader()
    spikes = SpikeDetector()

    cycle = 0
    while True:
        try:
            cycle += 1
            now = datetime.now(timezone.utc)

            # 1. Refresh market cache
            await cache.refresh()

            # 2. Poll news sources
            headlines = await feeds.poll_all()

            if cycle <= 3 or cycle % 10 == 0:
                print(f"[POLL] Cycle {cycle} | {len(headlines)} headlines | "
                      f"NewsData: {feeds._newsdata_calls_today}/190 today")

            # 3. Process headlines
            for h in headlines:
                title = h.get("title", "")
                source = h.get("source", "?")
                feed = h.get("feed", "?")
                print(f"[HEADLINE] [{feed}] {title[:80]} ({source})")

                # Find matching markets
                candidates = cache.find_matches(title)
                if not candidates:
                    continue

                print(f"  -> {len(candidates)} market matches (top: {candidates[0].get('_question', '')[:50]})")

                # Evaluate with LLM
                tradeable = await signals.evaluate(title, candidates)
                for sig in tradeable:
                    await trader.enter_trade(sig)

            # 4. Check for price spikes on low-price markets
            if cache.markets and cycle % 2 == 0:
                spike_list = await spikes.check_spikes(cache.markets)
                for spike in spike_list:
                    m = spike["market"]
                    q = m.get("question", m.get("_question", ""))
                    # Log spike but don't auto-trade (need headline confirmation)
                    print(f"[SPIKE ALERT] {q[:60]} | ${spike['old_price']:.2f} -> ${spike['new_price']:.2f}")

            # 5. Monitor exits
            await trader.monitor_exits()

            # 6. Periodic status
            if cycle % 20 == 0:  # every ~10 min
                w, l = trader.stats["wins"], trader.stats["losses"]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"[STATUS] Cycle {cycle} | {len(trader.positions)} open | "
                      f"{w}W/{l}L {wr:.0f}%WR | PnL: ${trader.stats['pnl']:+.2f} | "
                      f"LLM calls: {signals._llm_calls} | "
                      f"Markets cached: {len(cache.markets)} | "
                      f"NewsData calls today: {feeds._newsdata_calls_today}")

        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Saving state...")
            trader._save()
            break
        except Exception as e:
            print(f"[ERROR] Main loop: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
