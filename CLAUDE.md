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

### Current Status (Last Updated: 2026-02-04 late evening)
- Live Trader: PAUSED for optimization - backtested 78.7% WR
- Strategy: TA + Bregman + Kelly + ATR + RSI + strict UP + 200 EMA
- Markets: BTC/ETH/SOL 15-minute Up/Down via Polymarket API (`tag_slug=15M`)
- Position sizing: $1-$10, minimum $1 until WR proven, quarter-Kelly
- **BACKTESTED OPTIMAL FILTERS (v3)** — tested 5096 combinations:
  - Edge minimum: 10% (backtested: 0.10 + strict UP = 78.7% WR, 2.8 trades/hr)
  - ATR(14) on 1m candles: skip if recent 3 bars > 1.5x ATR
  - RSI confirmation: DOWN requires RSI<55, UP requires RSI>45
  - UP trades restricted: need 65% confidence, 30% edge, RSI>55
  - Skip hours: {6,7,8,14,15,16,19,20} UTC (8 hours skipped, 16 active)
  - Max entry price: $0.55
  - No volatility cap needed (RSI + skip hours handle it per backtest)
- Trend bias: 200 EMA → with-trend 70% capital, counter-trend 30%
- Auto-redeem: Claims winnings on-chain after every win
- ML threshold: 0.50 (adaptive: 0.30 when winning >60%, 1.0 when losing <40%)
- **Data findings** (250 trades): DOWN 66.4% WR vs UP 41.1% WR
- **Actual PnL discrepancy**: Bot tracked +$194, actual wallet -$803 (includes $780 manual Iran bets + $209 XRP copy trader + fill tracking bugs)

### Standing Orders
- **AUTO-TUNE**: Continuously analyze trade data and auto-adjust settings for max profit, then win rate. Don't ask - just optimize. Log changes made.
- ML should tune trend bias ratios (70/30) over time based on actual results
- Review and adjust SKIP_HOURS_UTC, asset list, edge thresholds as data accumulates

### Reminders
- **2026-02-04 ~6PM MST (12h report)**: Generate full performance report with charts. Compare pre-optimization (before 5AM MST) vs post-optimization metrics. Include: PnL by asset, side, hour, trend bias effectiveness, strong-signal dampening results. Auto-tweak any underperforming settings. Run this analysis script: `python analyze_performance.py` (create if needed).
- **2026-02-07**: Review ML threshold — if profitable, lower ML threshold again (currently 0.50 default, 0.30 when winning >60%, 1.0 when losing <40%). Check `get_min_score_threshold()` in `run_ta_live.py`.
- **POL gas balance**: Alert Star when signer wallet POL drops below 100 redemptions (~22 POL). Signer: `0xD375494Fd97366F543DAB3CB88684EFE738DCd40`. Auto-warning built into `auto_redeem_winnings()` in `run_ta_live.py`.

### Key Files
- `run_ta_paper.py` - Paper trading runner
- `run_ta_live.py` - Live trading with ML
- `ta_paper_results.json` - Paper trade history
- `ta_live_results.json` - Live trade history
- `arbitrage/ta_signals.py` - TA signal generator
- `arbitrage/bregman_optimizer.py` - Bregman divergence optimizer

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
