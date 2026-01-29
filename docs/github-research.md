# Polymarket GitHub Research

**Last Updated:** 2026-01-28
**Source:** GitHub repository analysis (comprehensive 10-minute search)

---

## Table of Contents
1. [Official Polymarket Repositories](#official-polymarket-repositories)
2. [Community Trading Bots](#community-trading-bots)
3. [Copy Trading & Whale Tracking](#copy-trading--whale-tracking)
4. [ML & AI Trading Systems](#ml--ai-trading-systems)
5. [Market Making Bots](#market-making-bots)
6. [WebSocket & Real-Time Tools](#websocket--real-time-tools)
7. [Arbitrage Strategies](#arbitrage-strategies)
8. [External Tools & Platforms](#external-tools--platforms)
9. [Research Findings](#research-findings)
10. [Key Takeaways](#key-takeaways)

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

---

## Copy Trading & Whale Tracking

### [Trust412/polymarket-copy-trading-bot-version-3](https://github.com/Trust412/polymarket-copy-trading-bot-version-3)
**Most sophisticated copy trading implementation**

**Features:**
- Scans target wallet every 4 seconds (configurable)
- Proportional position mirroring with constraints
- Max 20% portfolio per position (configurable)
- Automatic position redemption every 2 hours
- RPC rotation for uptime
- Blacklist support for excluded assets

**Tech Stack:** Node.js, TypeScript, MongoDB, Polygon RPC

---

### [aarora4/Awesome-Prediction-Market-Tools](https://github.com/aarora4/Awesome-Prediction-Market-Tools)
**Curated list of all prediction market tools**

#### Whale Tracking Platforms:
| Tool | URL | Features |
|------|-----|----------|
| Stand | stand.trade | Copy trade whales, lightning alerts |
| Polymarket Bros | brosonpm.trade | Tracks trades >$4,000 |
| Polycool | polycool.live | Top 0.5% wallet identification |
| Whale Tracker Livid | whale-tracker-livid.vercel.app | Tiered alerts ($29/mo pro) |
| PolyTracker | t.me/polytracker0_bot | Telegram wallet monitoring |
| Polylerts | t.me/Polylerts_bot | Track 15 wallets with alerts |

#### Arbitrage Tools:
| Tool | URL | Function |
|------|-----|----------|
| ArbBets | getarbitragebets.com | AI-driven +EV detection |
| Eventarb | eventarb.com | Free cross-platform arb |
| Polytrage | t.me/polytrage | 15-min automated signals |
| PolyScalping | polyscalping.org | 60-second market scanning |

---

## ML & AI Trading Systems

### [NavnoorBawa/polymarket-prediction-system](https://github.com/NavnoorBawa/polymarket-prediction-system)
**ML Ensemble for Outcome Prediction**

**Models Used:**
- XGBoost
- LightGBM
- Stacking ensembles

**Features/Indicators:**
- RSI (Relative Strength Index)
- Volatility measurements
- Order book imbalance
- Expected value calculations

**Output:** STRONG BUY, BUY, HOLD signals with confidence scores

**Position Sizing:** Kelly criterion with terminal risk adjustments

---

### [polymarket-trading-ai-agent](https://github.com/polymarket-trading-ai-agent/polymarket-trading-ai-agent)
**Multi-LLM Autonomous Trader**

**Supported LLMs:**
- ChatGPT
- DeepSeek
- Claude
- Gemini
- Grok

**Example Decision:**
> Market: 38% | AI Estimate: 47.5% | Edge: +9.5% EV
> Kelly Allocation: 2.8% of capital

---

### [llSourcell/Poly-Trader](https://github.com/llSourcell/Poly-Trader)
**Siraj Raval's Autonomous Agent**

- ChatGPT for event analysis
- Edge detection (AI vs market consensus)
- Kelly Criterion bankroll management
- Automated execution

---

## Market Making Bots

### [lorine93s/polymarket-market-maker-bot](https://github.com/lorine93s/polymarket-market-maker-bot)
**Production-Ready Market Maker**

**Spread Capture:**
- Min spread: 10 basis points
- Stepping: 5 bps increments
- Cancel/replace: 500-1000ms cycles
- Exploits Polymarket's 500ms taker delay

**Inventory Management:**
- Separate YES/NO tracking
- 30% skew threshold triggers size reduction
- Configurable net exposure limits

**Risk Controls:**
- Max position/order size caps
- Pre-trade validation
- Stop-loss percentages
- Auto-halt on breach

**Optimizations:**
- Gas batching (30-50% cost reduction)
- Sub-50ms WebSocket latency
- Batch cancellations

---

### [elielieli909/polymarket-marketmaking](https://github.com/elielieli909/polymarket-marketmaking)
**Band-Based Market Making**

Uses `buyBands` and `sellBands` around market price:
- Maintains minAmount to maxAmount within margins
- Reads market price every second
- Creates new bands when price moves

---

## WebSocket & Real-Time Tools

### [Polymarket/real-time-data-client](https://github.com/Polymarket/real-time-data-client)
**Official WebSocket Client**

Subscribe to:
- `trades` messages from `activity` topic
- All comments messages

---

### [nevuamarkets/poly-websockets](https://github.com/nevuamarkets/poly-websockets)
**TypeScript WebSocket Library**

Events:
- `book` - Order book updates
- `price_change` - Price movements
- `tick_size_change` - Tick changes
- `last_trade_price` - Trade prices

Features: Auto-reconnect, easy subscription management

---

### WebSocket Endpoints

| Endpoint | Purpose |
|----------|---------|
| `wss://ws-subscriptions-clob.polymarket.com/ws/` | Orders, trades, market data |
| `wss://ws-live-data.polymarket.com` | Real-time price streaming |

---

## Arbitrage Strategies

### [runesatsdev/polymarket-arbitrage-bot](https://github.com/runesatsdev/polymarket-arbitrage-bot)
**Research-Based Arbitrage ($39.59M documented)**

#### Strategy 1: Single-Condition
- YES + NO ≠ $1.00
- $10.58M extracted (7,051 conditions)
- Avg profit: $1,500/opportunity

#### Strategy 2: NegRisk Rebalancing (MOST PROFITABLE)
- Multi-outcome markets where sum ≠ 100%
- $28.99M extracted (662 markets)
- **29× capital efficiency**
- Example: 5-outcome at 98% total = $0.02 profit per $1

#### Strategy 3: Whale Tracking
- Follow >$5K positions
- 61-68% prediction accuracy
- Top performer: $2.01M in 12 months

**Recommended Allocation:**
| Strategy | Allocation |
|----------|------------|
| NegRisk Rebalancing | 40% |
| Single-Condition | 30% |
| Event-Driven | 20% |
| Whale Following | 10% |

---

### [djienne/Polymarket-bot](https://github.com/djienne/Polymarket-bot)
**APEX PREDATOR v7.2 - Dual Strategy**

#### Gabagool (Arbitrage):
- YES + NO < $0.975
- Markets under 4 hours
- Guaranteed settlement profit

#### Smart Ape (Momentum):
- BTC 15-min Up/Down markets
- Analyzes first 2-minute window
- Triggers on >15% downward movement
- Targets 1.5x+ asymmetric payouts

**Kelly Implementation:**
```
f* = (p × b - q) / b
```
Options: 1/8, 1/4, 1/2, Full Kelly

**Performance:** 200-500ms execution, 85% opportunity capture

---

### [discountry/polymarket-trading-bot](https://github.com/discountry/polymarket-trading-bot)
**Flash Crash Strategy**

**Entry Conditions:**
- 0.30 probability drop
- Within 10-second window
- Buy crashed side

**Exit:**
- Take profit: +$0.10
- Stop loss: -$0.05

**Markets:** BTC, ETH, SOL, XRP 15-minute Up/Down

---

### [ent0n29/polybot](https://github.com/ent0n29/polybot)
**Strategy Reverse Engineering**

Analyzes any user's trading history to identify:
- Entry/exit signals
- Sizing rules
- Timing patterns

**Architecture:**
- Executor Service (8080) - Order execution
- Strategy Service (8081) - Signal generation
- Ingestor Service (8082) - Data collection
- Analytics Service (8083) - ClickHouse analytics

**Stack:** Java 21+, Kafka, ClickHouse, Grafana

---

## NegRisk (Multi-Outcome) Markets

### [Polymarket/neg-risk-ctf-adapter](https://github.com/Polymarket/neg-risk-ctf-adapter)
**Official NegRisk Smart Contracts**

Unifies mutually exclusive binary markets into single multi-outcome structure.

**Use Case:** Election markets where only one candidate can win.

**Mechanism:** Convert NO token collections into YES tokens + collateral.

**Known Issues (py-clob-client):**
- Issue #79: Invalid signature on NegRisk orders
- Issue #138: Different exchange in signature
- Issue #188: No built-in conversion function

---

## External Tools & Platforms

### Analytics
| Tool | URL | Features |
|------|-----|----------|
| Polymarket Analytics | polymarketanalytics.com | Trader data, positions |
| Polysights | app.polysights.xyz | 30+ metrics, AI insights |
| Hashdive | hashdive.com | Smart Score |
| TREMOR | tremor.live | SQL terminal, 140K+ markets |

### Trading Terminals
| Tool | URL | Features |
|------|-----|----------|
| okbet | tryokbet.com | Telegram, copy trading |
| Polymtrade | polym.trade | Mobile, AI insights |
| Polyterm | github.com/NYTEMODEONLY/polyterm | Terminal-based monitoring |

### Data Infrastructure
| Tool | URL | Purpose |
|------|-----|---------|
| Goldsky | goldsky.com | Blockchain data |
| Dome | domeapi.io | Unified APIs |
| PolyRouter | polyrouter.io | Normalized cross-platform |

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

### Strategy Profitability Ranking
1. **NegRisk Multi-Outcome** - 73% of all profit, 29× efficiency
2. **Cross-Platform (Polymarket vs Kalshi)** - Risk-free when total < $1
3. **Same-Market (YES + NO < $1)** - $10.58M extracted, high competition
4. **Flash Crash/Spike** - Requires <50ms latency
5. **Whale Copying** - 61-68% accuracy, passive income
6. **Market Making** - Difficult due to competition

### Technical Requirements
| Requirement | Specification |
|-------------|---------------|
| Detection Latency | <50ms competitive edge |
| Execution Latency | 200-500ms acceptable |
| WebSocket | Required for real-time |
| Languages | Python, TypeScript, Java |
| Infrastructure | Docker, Kafka, ClickHouse |

### Position Sizing Best Practices
- Kelly Criterion standard
- Use fractional Kelly (1/4 to 1/2)
- Max 20% per position
- 30% inventory skew triggers reduction

### Warnings
- Market making "not profitable" in current conditions
- Spreads compressing with institutional entry
- NegRisk signature issues with py-clob-client
- Many repos are sales pitches without code

---

## Useful Links

- [Polymarket/agents](https://github.com/Polymarket/agents) - Official AI framework
- [Polymarket/poly-market-maker](https://github.com/Polymarket/poly-market-maker) - Official MM
- [Polymarket/neg-risk-ctf-adapter](https://github.com/Polymarket/neg-risk-ctf-adapter) - NegRisk contracts
- [Polymarket/real-time-data-client](https://github.com/Polymarket/real-time-data-client) - WebSocket client
- [polymarket-arbitrage topic](https://github.com/topics/polymarket-arbitrage) - Community bots
- [py-clob-client](https://github.com/Polymarket/py-clob-client) - Python SDK
- [Awesome-Prediction-Market-Tools](https://github.com/aarora4/Awesome-Prediction-Market-Tools) - Curated list
