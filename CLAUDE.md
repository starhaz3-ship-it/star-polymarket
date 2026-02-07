# Claude Context - Star Polymarket

**Project:** ML-Optimized Prediction Market Trading Bot
**Owner:** Star
**Resume Command:** "Access Star-Polymarket"

## CRITICAL RULES
- **NEVER** launch copy traders (`run_copy_live.py`, `run_copy_k9Q2.py`) without Star's explicit consent
- Only auto-launch: `run_ta_live.py`, `run_ta_paper.py`, `run_whale_watcher.py`

## TA Trading Suite (Primary)

### Quick Start Commands
```bash
# TA Paper Trader - paper trading with $100 positions
python run_ta_paper.py

# TA Live Trader - live trading with $10 positions (needs POLYMARKET_PASSWORD)
python run_ta_live.py

# Check paper results
cat ta_paper_results.json | python -m json.tool
```

### System Names
- **TA Paper** (`run_ta_paper.py`) - Paper trading, $100 positions, Bregman optimization
- **TA Live** (`run_ta_live.py`) - Live trading, $10 positions, ML optimization

### Current Status (Last Updated: 2026-02-07)
- Live Trader: RUNNING — $5 base bets, $70.12 bankroll, multi-RPC redeem
- Paper Trader: RUNNING — $10 bets, skip-hour shadow tracking active
- Strategy: TA + Bregman + Kelly + ATR + NYU Vol + ML Scoring + 200 EMA
- Markets: BTC/ETH/SOL 15-minute Up/Down via Polymarket API (`tag_slug=15M`)
- Position sizing: $3-$8, quarter-Kelly, Bayesian hourly multiplier (0.5x-1.5x)
- Skip hours: {0, 1, 8, 22, 23} UTC — shadow-tracked for re-evaluation
- Trend bias: 200 EMA → counter-trend gets 85% size (softer than old 30%)
- Auto-redeem: Multi-RPC fallback (1rpc.io > publicnode > drpc > polygon-rpc)
- **Live session**: 2W/0L (+$9.63) from $70.12 start

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
- `run_ta_paper.py` - Paper trading runner
- `run_ta_live.py` - Live trading with ML
- `run_ta_experiment.py` - 6-strategy experimental lab (paper)
- `run_daily_king.py` - Daily Up/Down trader (kingofcoinflips strategy, PAPER until $200+ profit)
- `ta_paper_results.json` - Paper trade history
- `ta_live_results.json` - Live trade history
- `daily_king_results.json` - Daily king paper trade history
- `arbitrage/ta_signals.py` - TA signal generator
- `arbitrage/bregman_optimizer.py` - Bregman divergence optimizer

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
