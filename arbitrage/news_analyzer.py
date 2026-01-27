"""LLM-powered news analysis for probability updates."""
import asyncio
import httpx
import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class NewsSignal:
    """A news-derived trading signal."""
    headline: str
    source: str
    timestamp: datetime
    affected_markets: list[str]
    probability_impact: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    urgency: str  # 'immediate', 'short_term', 'long_term'


@dataclass
class MarketNewsAnalysis:
    """Analysis of news impact on a specific market."""
    market_id: str
    market_question: str
    current_price: float
    news_items: list[dict]
    suggested_probability: float
    edge: float  # suggested - current
    action: str  # 'buy_yes', 'buy_no', 'hold'
    reasoning: str


class NewsAnalyzer:
    """Analyze news for trading signals using LLM."""

    # News API endpoints
    NEWS_SOURCES = {
        "newsapi": "https://newsapi.org/v2/everything",
        "gnews": "https://gnews.io/api/v4/search",
    }

    def __init__(self, news_api_key: Optional[str] = None, anthropic_key: Optional[str] = None):
        self.news_api_key = news_api_key or os.getenv("NEWS_API_KEY")
        self.anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = httpx.AsyncClient(timeout=60)
        self.signals: list[NewsSignal] = []

    async def fetch_news(self, query: str, hours_back: int = 24) -> list[dict]:
        """Fetch recent news articles."""
        articles = []

        # Try NewsAPI
        if self.news_api_key:
            try:
                from_date = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
                resp = await self.client.get(
                    self.NEWS_SOURCES["newsapi"],
                    params={
                        "q": query,
                        "from": from_date,
                        "sortBy": "publishedAt",
                        "apiKey": self.news_api_key,
                        "language": "en",
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    articles.extend(data.get("articles", []))
            except Exception as e:
                logger.warning(f"NewsAPI error: {e}")

        # Fallback: Use web search simulation
        if not articles:
            articles = await self._fetch_from_web(query)

        return articles

    async def _fetch_from_web(self, query: str) -> list[dict]:
        """Fallback news fetching using public sources."""
        # This would use web scraping or public APIs
        # For now, return empty - real implementation would scrape news sites
        return []

    async def analyze_with_llm(self, market: dict, news: list[dict]) -> Optional[MarketNewsAnalysis]:
        """Use LLM to analyze news impact on market."""
        if not self.anthropic_key:
            return self._rule_based_analysis(market, news)

        try:
            question = market.get("question", "")
            current_price = self._get_yes_price(market)

            news_text = "\n".join([
                f"- {n.get('title', '')} ({n.get('source', {}).get('name', 'Unknown')})"
                for n in news[:10]
            ])

            prompt = f"""Analyze how this news affects the prediction market.

Market Question: {question}
Current YES Price: ${current_price:.2f} ({current_price*100:.0f}% implied probability)

Recent News:
{news_text}

Based on this news, provide:
1. Your estimated probability (0-100%)
2. Whether to BUY YES, BUY NO, or HOLD
3. Brief reasoning (1-2 sentences)

Respond in JSON format:
{{"probability": 75, "action": "buy_yes", "reasoning": "..."}}"""

            resp = await self.client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 256,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )

            if resp.status_code == 200:
                data = resp.json()
                content = data.get("content", [{}])[0].get("text", "{}")

                # Extract JSON from response
                try:
                    # Find JSON in response
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        analysis = json.loads(content[start:end])

                        suggested_prob = analysis.get("probability", current_price * 100) / 100
                        edge = suggested_prob - current_price

                        return MarketNewsAnalysis(
                            market_id=market.get("conditionId", ""),
                            market_question=question,
                            current_price=current_price,
                            news_items=news,
                            suggested_probability=suggested_prob,
                            edge=edge,
                            action=analysis.get("action", "hold"),
                            reasoning=analysis.get("reasoning", ""),
                        )
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.error(f"LLM analysis error: {e}")

        return self._rule_based_analysis(market, news)

    def _rule_based_analysis(self, market: dict, news: list[dict]) -> Optional[MarketNewsAnalysis]:
        """Fallback rule-based analysis."""
        question = market.get("question", "").lower()
        current_price = self._get_yes_price(market)

        # Count positive/negative sentiment keywords
        positive_kw = ["win", "victory", "confirmed", "success", "leads", "ahead"]
        negative_kw = ["lose", "defeat", "unlikely", "fails", "behind", "drops"]

        positive_count = 0
        negative_count = 0

        for article in news:
            title = article.get("title", "").lower()
            for kw in positive_kw:
                if kw in title:
                    positive_count += 1
            for kw in negative_kw:
                if kw in title:
                    negative_count += 1

        # Calculate sentiment adjustment
        if positive_count + negative_count > 0:
            sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            adjustment = sentiment * 0.1  # Max 10% adjustment
        else:
            adjustment = 0

        suggested_prob = min(max(current_price + adjustment, 0.05), 0.95)
        edge = suggested_prob - current_price

        action = "hold"
        if edge > 0.05:
            action = "buy_yes"
        elif edge < -0.05:
            action = "buy_no"

        return MarketNewsAnalysis(
            market_id=market.get("conditionId", ""),
            market_question=market.get("question", ""),
            current_price=current_price,
            news_items=news,
            suggested_probability=suggested_prob,
            edge=edge,
            action=action,
            reasoning=f"Rule-based: {positive_count} positive, {negative_count} negative signals",
        )

    def _get_yes_price(self, market: dict) -> float:
        """Extract YES price from market."""
        try:
            prices = market.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)
            return float(prices[0]) if prices else 0.5
        except:
            return 0.5

    async def scan_markets_for_news_edge(self, markets: list[dict]) -> list[MarketNewsAnalysis]:
        """Scan markets for news-based trading opportunities."""
        analyses = []

        for market in markets:
            question = market.get("question", "")

            # Extract key terms for news search
            search_terms = self._extract_search_terms(question)

            if not search_terms:
                continue

            # Fetch relevant news
            news = await self.fetch_news(search_terms, hours_back=48)

            if news:
                analysis = await self.analyze_with_llm(market, news)
                if analysis and abs(analysis.edge) > 0.03:  # 3% edge threshold
                    analyses.append(analysis)

        # Sort by edge magnitude
        analyses.sort(key=lambda x: -abs(x.edge))
        return analyses

    def _extract_search_terms(self, question: str) -> str:
        """Extract search terms from market question."""
        # Remove common words
        stop_words = {"will", "the", "a", "an", "in", "on", "at", "by", "for", "to", "of", "be"}

        words = question.replace("?", "").split()
        terms = [w for w in words if w.lower() not in stop_words]

        # Take first 4-5 significant words
        return " ".join(terms[:5])


class RealTimeNewsMonitor:
    """Monitor news in real-time for market-moving events."""

    def __init__(self, analyzer: NewsAnalyzer):
        self.analyzer = analyzer
        self.seen_headlines: set[str] = set()
        self.callbacks: list = []

    def on_signal(self, callback):
        """Register callback for new signals."""
        self.callbacks.append(callback)

    async def monitor(self, keywords: list[str], interval: int = 300):
        """Monitor news continuously."""
        while True:
            for keyword in keywords:
                news = await self.analyzer.fetch_news(keyword, hours_back=1)

                for article in news:
                    headline = article.get("title", "")

                    if headline and headline not in self.seen_headlines:
                        self.seen_headlines.add(headline)
                        logger.info(f"New headline: {headline}")

                        # Notify callbacks
                        for cb in self.callbacks:
                            try:
                                await cb(article)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

            await asyncio.sleep(interval)


async def main():
    """Test news analyzer."""
    analyzer = NewsAnalyzer()

    print("=" * 70)
    print("NEWS ANALYZER TEST")
    print("=" * 70)

    # Test with a sample market
    sample_market = {
        "question": "Will the Seattle Seahawks win Super Bowl 2026?",
        "conditionId": "test123",
        "outcomePrices": "[0.67, 0.33]",
    }

    print(f"\nAnalyzing: {sample_market['question']}")
    print("Fetching news...")

    news = await analyzer.fetch_news("Seattle Seahawks Super Bowl", hours_back=48)
    print(f"Found {len(news)} articles")

    if news:
        analysis = await analyzer.analyze_with_llm(sample_market, news)
        if analysis:
            print(f"\nAnalysis Result:")
            print(f"  Current Price: ${analysis.current_price:.2f}")
            print(f"  Suggested Prob: {analysis.suggested_probability*100:.0f}%")
            print(f"  Edge: {analysis.edge*100:+.1f}%")
            print(f"  Action: {analysis.action}")
            print(f"  Reasoning: {analysis.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
