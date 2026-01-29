# Polymarket GitHub Research

**Last Updated:** 2026-01-28
**Source:** GitHub repository analysis

---

## Official Polymarket Repositories

### 1. [Polymarket/agents](https://github.com/Polymarket/agents)
**AI Agent Framework for Polymarket Trading**

A developer framework enabling autonomous trading using AI agents.

**Key Components:**
- `Chroma.py` - Vector database for news and API data
- `Gamma.py` - Interfaces with Polymarket Gamma API for market metadata
- `Polymarket.py` - Core trading class (order building, signing, execution)
- `Objects.py` - Pydantic data models for trades, markets, events

**Features:**
- RAG (Retrieval-Augmented Generation) capabilities
- LLM tools for prompt engineering and market analysis
- Data sourcing from betting services, news providers, web search
- CLI interface for querying markets and executing trades

**Usage:**
```bash
python scripts/python/cli.py get-all-markets
python agents/application/trade.py
```

---

### 2. [Polymarket/poly-market-maker](https://github.com/Polymarket/poly-market-maker)
**Official Market Maker Keeper**

Automated market maker for CLOB markets.

**Strategies:**
- **Bands**: Places orders at specified distance bands from midpoint
- **AMM**: Algorithmic pricing approach

**Sync Cycle (default 30s):**
1. Retrieve current midpoint from CLOB
2. Calculate optimal order positions
3. Compare desired vs existing orders
4. Execute cancellations and new placements

**Configuration:**
- `CONDITION_ID`: Market identifier (hex)
- `STRATEGY`: "Bands" or "AMM"
- `CONFIG`: Strategy-specific config file path
- `sync_interval`: Cycle timing

**Requirements:** Python 3.10, Docker Compose

---

## Community Trading Bots

### 3. [CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot](https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot)
**Cross-Platform Arbitrage (Polymarket vs Kalshi)**

Exploits price differences between prediction markets.

**How It Works:**
1. Polls Polymarket CLOB and Kalshi API (1-second intervals)
2. Normalizes prices to 0.00-1.00 probability format
3. Compares strike prices across platforms
4. When `Total Cost < $1.00` → arbitrage opportunity

**Profit Calculation:**
- Polymarket strike > Kalshi: Check "Down + Yes" combo
- Kalshi strike > Polymarket: Check "Up + No" combo
- Binary options settle at $1.00 max → profit = $1.00 - entry cost

**Architecture:**
- Backend: Python + FastAPI
- Frontend: Next.js + TypeScript + shadcn/ui

---

### 4. [Trust412/Polymarket-spike-bot-v1](https://github.com/Trust412/Polymarket-spike-bot-v1)
**High-Frequency Spike Detection Bot**

Capitalizes on market inefficiencies through:
- Real-time price monitoring
- Automated spike detection
- Smart order execution
- Advanced threading for optimal performance

**Tech Stack:** Python, Web3

---

### 5. [warproxxx/poly-maker](https://github.com/warproxxx/poly-maker)
**Community Market Maker with Google Sheets Config**

Provides liquidity by maintaining orders on both sides.

**Google Sheets Integration:**
- `Selected Markets` - User-chosen trading venues
- `All Markets` - Complete Polymarket database
- `Hyperparameters` - Adjustable trading parameters

**Features:**
- Real-time order book monitoring via WebSockets
- Position management with risk controls
- `poly_merger` module for consolidating positions (reduces gas)

**Warning:** Author notes bot is "not profitable" in current conditions due to competition.

---

### 6. [djienne/Polymarket-bot](https://github.com/djienne/Polymarket-bot)
**HFT Bot with Kelly Criterion**

- Dual-strategy engine
- Kelly Criterion position sizing
- Real-time auto-optimization
- Exploits YES + NO < $1.00 inefficiencies

---

### 7. [vladmeer/polymarket-arbitrage-bot](https://github.com/vladmeer/polymarket-arbitrage-bot)
**Dutch Book Arbitrage Bot**

Detects guaranteed-profit opportunities in binary markets.

**Strategy:**
- When UP + DOWN tokens sum < 1.0
- Simultaneously buy both
- Lock in risk-free profit on expiration

**Performance:**
- 5-40ms detection latency
- Real-time WebSocket monitoring

---

### 8. [0xalberto/polymarket-arbitrage-bot](https://github.com/0xalberto/polymarket-arbitrage-bot)
**Single & Multi-Market Arbitrage**

Targets BTC 15-minute up/down markets.

**Claimed Performance (simulated):**
- Average $25/hour
- $500-700/day from $200 deposit
- $764 profit in one day (live trading Dec 21st)

**Warning:** Strategy loses money when deployed across multiple markets simultaneously.

---

## Research Findings

### IMDEA Networks Study (April 2024 - April 2025)
**$39.59M total arbitrage extracted from Polymarket**

| Metric | Value |
|--------|-------|
| Total arbitrage profit | $39.59M |
| Top arbitrageur profit | $2,009,631 |
| Top trader transactions | 4,049 |
| Average profit per trade | $496 |
| Daily trade frequency | 11+ trades |

**NegRisk Markets Dominance:**
- Generated 73% of total arbitrage ($28.99M)
- 29× capital efficiency advantage
- Markets with N≥4 conditions most profitable

---

## Key Takeaways

### Profitable Strategies (Ranked)
1. **NegRisk Multi-Outcome** - 73% of all arbitrage profit
2. **Cross-Platform (Polymarket vs Kalshi)** - Requires monitoring both APIs
3. **Same-Market (YES + NO < $1)** - High competition, shrinking spreads
4. **Spike/Flash Crash** - Requires low-latency infrastructure
5. **Market Making** - Difficult to profit due to competition

### Technical Requirements
- **Latency**: 5-40ms detection for competitive edge
- **APIs**: Polymarket CLOB, Gamma API, WebSocket RTDS
- **Languages**: Python (most common), TypeScript
- **Infrastructure**: Docker, FastAPI, WebSockets

### Warnings
- "Bot can potentially lose real money"
- Market making "not profitable" in current conditions
- Increased competition compressing spreads
- Need "significant" customization for profitability

---

## Useful Links

- [Polymarket/agents](https://github.com/Polymarket/agents) - Official AI framework
- [Polymarket/poly-market-maker](https://github.com/Polymarket/poly-market-maker) - Official MM
- [polymarket-arbitrage topic](https://github.com/topics/polymarket-arbitrage) - Community bots
- [py-clob-client](https://github.com/Polymarket/py-clob-client) - Python SDK
