# Evolution Log - Star Polymarket Trading Bot
# Append-only record of every automated improvement.
# Each entry: date, what changed, why, expected impact.

---

## 2026-02-07 — Evolution Stack Deployed (Manual Session)
- Created auto_analyze.py (6-hourly cron)
- Created RESEARCH_BACKLOG.md (11 hypotheses)
- Created memory knowledge base (proven_edges, failed_experiments, active_hypotheses)
- Added skip-hour shadow tracking to paper trader
- Added early profit-taking at 85% of max (sell_position + check_early_exit)
- Added Bayesian hourly ML position sizing
- Opened skip hours for $3 high-conviction trades (Feb 7-8 temp)
- Set up daily evolution runner for autonomous improvement

---

## 2026-02-07 — V3.4 Data-Driven Filter Upgrade (96 Paper Trade Analysis)
**Analysis**: Deep-dive into 96 paper trades revealed massive alpha leaks. Two research agents ran comprehensive hypothesis tests.

### Changes Deployed:
1. **MIN_TIME_REMAINING: 2.0 → 5.0 min** — 2-5 min entries had 46.5% WR (below breakeven); 5-12 min had 83.3% WR. This is the single biggest filter improvement.
2. **MIN_EDGE: 0.20 → 0.30** — Edge <0.30 = 36.4% WR; >=0.30 = 70.3% WR. Cuts 22 bad trades, loses only $26 PnL.
3. **KL divergence >= 0.15 filter (NEW)** — KL <0.15 = 35.7% WR (model agrees with market = no edge); >=0.15 = 67.1% WR.
4. **DOWN <$0.15 hard block** — Was only blocked during uptrends. Now blocked ALWAYS. 0% WR in paper (3 trades, 3 total losses, -$30).

### Key Findings:
- **UP >> DOWN in current regime**: UP 76.7% WR vs DOWN 50.9% (reversal from historical)
- **BTC UP = 90.9% WR** (10/11 trades) — strongest single combo
- **ETH DOWN = 40% WR, SOL DOWN = 43.8%** — weakest combos
- **$0.30-0.40 sweet spot**: 80%+ WR for both sides
- **$0.00-0.15 catastrophic**: 16.7% WR overall, 0% for DOWN
- **"Good" signal trades**: 57% WR, $1.20 avg vs "Strong" 63% WR, $7.50 avg
- **Hour 9 UTC (4-5 AM ET) = 100% WR** (4 trades, $17.15 avg)
- **Profit factor**: 4.24 ($4.24 earned per $1 lost)

### Combined Projected Impact:
- **WR: 62.5% → 75-83%** (by cutting ~40 marginal-to-bad trades)
- **PnL retained: 81-92%** (lose only $50-$70 from filtered trades)

### Research Verdicts:
- [x] KL >= 0.15 filter → **IMPLEMENT** (done)
- [x] DOWN <$0.30 mandatory → **REJECT** (data shows opposite: DOWN $0.30-0.40 = 75% WR)
- [x] SOL underperformance → **REJECT** (SOL has highest avg PnL/trade)
- [x] Time remaining sweet spot → **IMPLEMENT** (done, 5min floor)
- [x] Model confidence > 0.60 → **IMPLEMENT as edge >= 0.30** (done)
