# Claude Context - Star Polymarket

**Project:** ML-Optimized Prediction Market Trading Bot
**Owner:** Star
**Resume Command:** "Access Star-Polymarket"

## DISPLAY PREFERENCES
- **Always use MST (Mountain Standard Time) when displaying times to Star.** Never use ET/Eastern. UTC is also acceptable. Format: "12:00 AM MST" or "07:00 UTC".

## CRITICAL RULES
- **NEVER** launch copy traders (`run_copy_live.py`, `run_copy_k9Q2.py`) without Star's explicit consent
- **NEVER** run directional traders live (`run_ta_live.py`, `run_ema_rsi_5m.py`, `run_trend_scalp_15m.py`, `run_momentum_5m.py`). Directional trading has 40% WR and -$214 lifetime losses.
- **EXCEPTION (2026-02-18)**: `run_momentum_15m_live.py --live` is APPROVED by Star. ONLY live strategy. V2.0 GTC orders (FOK fix applied). PROBATION $2.50/trade, auto-promotes. Paper: 9W/4L 69% WR +$13.32.
- **DEMOTED (2026-02-18)**: `run_pairs_arb.py --live --mode 15M` — demoted to paper. Live PnL: -$0.45. One exposed loss wiped all hedged gains.
- **KILLED (2026-02-18)**: `run_pairs_arb.py --live --mode 5M` — 5M bail liquidity crisis. 4/4 bails = full loss ($4.50-$4.95 each). Live PnL: -$13.30. DO NOT RESTART.
- **DEMOTED (2026-02-18)**: `run_sniper_5m_live.py --live` — demoted to paper. Live PnL: -$1.36. Oracle trades are phantom (don't fill). W:L ratio 0.34x.
- **NEVER** run `run_maker.py` — KILLED permanently. Lost ~$300 on 1c spreads with no edge. Do NOT restart, do NOT suggest restarting.
- **ON SESSION START**: Verify NO unauthorized live traders are running. Check with `_find_procs.py`.
- Allowed live: `run_momentum_15m_live.py --live` AND `run_sniper_5m_live.py --live`
- Paper-only processes (safe): `run_ta_paper.py`, `run_15m_strategies.py`, `run_5m_experiments.py`, `run_sniper_5m_paper.py`, `run_pairs_arb.py` (without --live)

## CORE STACK (launch by default on "Access Star-Polymarket")
On session start, offer to launch any that aren't already running.
All commands run from `C:/Users/Star/.local/bin/star-polymarket/` with `nohup python -u <script> > <log> 2>&1 &`

1. **Momentum 15M LIVE** — `python -u run_momentum_15m_live.py --live` → `momentum_15m_live.log` — ONLY live strategy
2. ~~Sniper 5M~~ — **DEMOTED to paper Feb 18.** Live PnL: -$1.36. Run paper only.
3. ~~Pairs Arb 15M~~ — **DEMOTED to paper Feb 18.** Live PnL: -$0.45.
4. ~~Pairs Arb 5M~~ — **KILLED Feb 18.** Bail = full loss, zero liquidity.
4. **TA Paper** — `python -u run_ta_paper.py` — paper trading
5. **15M Strategies** — `python -u run_15m_strategies.py` — paper 15m strategy lab
6. **5M Experiments** — `python -u run_5m_experiments.py` — whale consensus + 5m experiments
7. **Whale Watcher** — `python -u run_whale_watcher.py` — whale wallet tracker

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

### Current Status (Last Updated: 2026-02-17)
- **Momentum 15M LIVE** — V1.6 PROMOTED tier ($5/trade). 20W/5L (80% WR), +$41.66 PnL
- **Maker: DEAD** — Killed permanently. Lost ~$300 on 1c spreads.
- Paper traders: TA Paper, 15M Strategies, 5M Experiments all running
- Whale Watcher: Running (restarted after sqlite crash on 1.4GB DB)

### Performance Analysis (2026-02-17)
- **momentum_strong: 90% WR, +$44.86** — THE proven edge, especially BTC (92.3% WR)
- **streak_reversal: 33% WR, -$5.70** — LOSING MONEY, needs to be disabled
- **Best hours (UTC): 0-6** (7PM-1AM ET) — 100% WR at hours 0,2,3,4
- **Dead hours (UTC): 7-8** (2-3AM ET) — 0-33% WR, lose money
- **Entry price edge: <$0.50 wins, >$0.50 loses**
- **CSV vs internal gap: ~$98** — old maker positions still resolving as losses on-chain

### ACTION ITEMS (Discuss with Star on next session)
1. **Disable streak_reversal** — 33% WR coin flip, -$5.70 drag
2. **Lower max entry to $0.50** — expensive entries lose
3. **Skip hours 7-8 UTC** — dead money zone
4. **Consider BTC-only mode** — 92% WR vs ETH's 61%
5. **If WR stays >85% for 50+ trades** — promote to CHAMPION ($10/trade)

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
