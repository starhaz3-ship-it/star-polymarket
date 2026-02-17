# Claude Context - Star Polymarket

**Project:** ML-Optimized Prediction Market Trading Bot
**Owner:** Star
**Resume Command:** "Access Star-Polymarket"

## CRITICAL RULES
- **NEVER** launch copy traders (`run_copy_live.py`, `run_copy_k9Q2.py`) without Star's explicit consent
- **NEVER** run directional traders live (`run_ta_live.py`, `run_ema_rsi_5m.py`, `run_trend_scalp_15m.py`, `run_momentum_5m.py`). Directional trading has 40% WR and -$214 lifetime losses.
- **EXCEPTION (2026-02-16)**: `run_momentum_15m_live.py --live` is APPROVED by Star. Monitor closely — kill if WR drops below 50% over 20+ trades or daily loss limit hit.
- **NEVER** run `run_maker.py` — KILLED permanently. Lost ~$300 on 1c spreads with no edge. Do NOT restart, do NOT suggest restarting.
- **ON SESSION START**: Verify NO unauthorized directional traders are running. `run_momentum_15m_live.py` is the only allowed live trader.
- Allowed live: `run_momentum_15m_live.py --live` (momentum directional) — ONLY live trader permitted
- Paper-only processes (safe): `run_ta_paper.py`, `run_15m_strategies.py`, `run_fib_fib_paper.py`, `run_momentum_5m.py --paper`

## CORE STACK (launch by default on "Access Star-Polymarket")
On session start, offer to launch any that aren't already running:
1. **Momentum 15M LIVE** — `python -u run_momentum_15m_live.py --live` — ONLY live trader
2. **TA Paper** — `python -u run_ta_paper.py` — paper trading
3. **15M Strategies** — `python -u run_15m_strategies.py` — paper 15m strategy lab
4. **5M Experiments** — `python -u run_5m_experiments.py` — whale consensus + 5m experiments
5. **Whale Watcher** — `python -u run_whale_watcher.py` — whale wallet tracker

## DO NOT KILL THESE PROCESSES
- Check PID files before killing ANY Python process in this project
- If you need to restart a trader, read the PID file first and only kill that specific PID

## TA Trading Suite (Primary)

### Quick Start Commands
```bash
# TA Paper Trader V3.5 - systematic features + ML optimization
python -u run_ta_paper.py

# Weather Paper Trader - NOAA forecast arbitrage
python -u run_weather_paper.py

# TA Live Trader - $5 flat bets (CURRENTLY OFF - needs POLYMARKET_PASSWORD)
python -u run_ta_live.py

# Check paper results
cat ta_paper_results.json | python -m json.tool
cat weather_paper_results.json | python -m json.tool
```

### System Names
- **TA Paper** (`run_ta_paper.py`) - Paper trading V3.5, systematic features, ML optimization
- **TA Live** (`run_ta_live.py`) - Live trading V3.12b, $5 flat bets (CURRENTLY OFF)
- **Weather Paper** (`run_weather_paper.py`) - NOAA forecast vs Polymarket temperature arbitrage
- **Whale Watcher** (`run_whale_watcher.py`) - 39 whale wallets tracked (incl. 2 weather whales)

### Current Status (Last Updated: 2026-02-09)
- **Live Trader: OFF** — Shut down due to zero mid-price liquidity on 15-min order books
- Paper Trader: **V3.5** RUNNING — Systematic trading features, 5 new ML-tunable features
- Weather Trader: **v1.0** RUNNING — NOAA forecast arbitrage on temperature markets
- Strategy: TA + Bregman + Kelly + NYU Vol + ML + Systematic V3.5 (overnight/invvol/MTF/volspike/volregime)
- Markets: BTC/SOL/ETH paper (15-min Up/Down) + Weather (daily temperature)
- **PID locks**: All traders locked via shared `pid_lock.py` (ta_paper, weather_paper, whale_watcher, daily_king, ta_live)
- **V3.5 Features**: BTC overnight seasonality, inverse vol sizing, multi-TF confirmation, volume spike filter, vol regime routing
- **Realistic paper fills**: +$0.03 spread offset (order books show zero mid-price liquidity)
- Skip hours: {0, 1, 8} UTC (reopened 22,23 for overnight seasonality)
- **Account**: ~$74 live (not trading), paper only until Monday liquidity check

### Standing Orders
- **AUTO-TUNE**: Continuously analyze trade data and auto-adjust settings for max profit, then win rate. Don't ask - just optimize. Log changes made.
- **FULL AUTONOMY**: Star has granted full ownership of the trading pipeline. Research, backtest, deploy, and ML-refine continuously. Cut losers fast, promote winners immediately.

## EVOLUTION PROTOCOL (run every session on "Access Star-Polymarket")
1. **Health Check**: Verify live trader, paper trader, and whale watcher are running. Restart any that are down.
2. **Read Fresh Data**: Load `latest_analysis.json` (auto-generated every 6h), `ta_live_results.json`, `ta_paper_results.json`
3. **Performance Audit**: Find new losers to cut, winners to boost, parameter drift
4. **Skip-Hour Shadows**: Check `skip_hour_stats` in paper results — any hour with 55%+ WR and 10+ trades = recommend REOPEN
5. **Research Backlog**: Read `RESEARCH_BACKLOG.md` — pick highest-impact untested hypothesis, test it
6. **Deploy Improvements**: Apply findings to live trader immediately. Log all changes.
7. **Update Memory**: Write findings to memory files (`proven_edges.md`, `failed_experiments.md`)
8. **Report to Star**: Brief summary of what changed and why

### Reminders
- **POL gas balance**: Alert Star when signer wallet POL drops below 100 redemptions (~22 POL). Signer: `0xD375494Fd97366F543DAB3CB88684EFE738DCd40`.
- **2026-02-21**: Revisit TREND_FOLLOW strategy (blacklisted, 14% WR live — 2-week cooling-off).

### Key Files
- `run_ta_paper.py` - Paper trading runner V3.5
- `run_ta_live.py` - Live trading V3.12b (currently off)
- `run_weather_paper.py` - Weather arbitrage paper trader
- `run_daily_king.py` - Daily Up/Down trader (kingofcoinflips strategy)
- `pid_lock.py` - Shared PID lock utility (prevents duplicate instances)
- `ta_paper_results.json` - Paper trade history
- `ta_live_results.json` - Live trade history
- `weather_paper_results.json` - Weather paper trade history
- `arbitrage/ta_signals.py` - TA signal generator
- `arbitrage/bregman_optimizer.py` - Bregman divergence optimizer
- `arbitrage/copy_trader.py` - Whale wallet list (39 wallets)

### Growth Plan
- **Phase 1** (current): 15-minute markets, $3 bets, prove 75%+ WR
- **Phase 2** (when PnL > $200): Add daily Up/Down (kingofcoinflips strategy) live
- **Phase 3** (when PnL > $500): Scale up bet sizes, add micro-arbitrage

## Arbitrage Suite (Legacy)

```bash
# Paper trade arbitrage
python -m arbitrage.main --dry-run --verbose

# Live trade arbitrage (REAL MONEY)
python -m arbitrage.main --verbose
```

### Strategies
1. **BTC Latency** - Spot price vs Polymarket odds lag
2. **Same-Market** - Buy YES + NO when sum < $1.00
3. **NegRisk** - Multi-outcome markets sum < 100%
4. **Endgame** - 95%+ probability near resolution
5. **Cross-Platform** - Polymarket vs Kalshi price differences

## Critical Config (Magic Wallet)

```python
# REQUIRED for trading - user uses Magic (email) wallet
ClobClient(
    signature_type=1,  # Magic wallet
    funder="0x4f3456c5b05b14D8aFc40E0299d6d68DE0fF7d7e",  # Proxy wallet
)
```

## Environment Variables
- `POLYMARKET_PRIVATE_KEY` - Encrypted private key (ENC:...)
- `POLYMARKET_KEY_SALT` - Salt for decryption
- `POLYMARKET_PASSWORD` - Password for key decryption (required for live trading)
- `POLYMARKET_PROXY_ADDRESS` - Proxy wallet address

## Whale Watcher (Paper Only)
- `run_whale_watcher.py` - Monitors 18 whale wallets, paper trades, logs to SQLite
- `whale_watcher.db` - SQLite database for ML analysis (trades, snapshots, stats)
- No real money, no execution - observation and logging only

## File Structure
- `arbitrage/` - Core trading modules
- `run_ta_paper.py` - TA paper trader
- `run_ta_live.py` - TA live trader with ML
- `run_whale_watcher.py` - Whale paper trader + SQLite logger
- `.env` - Encrypted credentials

## GitHub
- Bot: https://github.com/starhaz3-ship-it/star-polymarket
- Research: https://github.com/starhaz3-ship-it/polymarket-knowledge-base

## Full Context
See: `C:\Users\Star\.claude\star-polymarket-context.md`
