# Research Backlog - Star Polymarket
# Ranked by expected impact. Pick top untested hypothesis each session.
# Format: - [ ] HYPOTHESIS | Expected Impact | Status

## HIGH IMPACT (test first)
- [ ] **KL divergence > 0.15 as hard filter** | Higher KL = more mispricing = higher WR? Backtest against paper trades | UNTESTED
- [ ] **Entry price < $0.30 mandatory for DOWN** | $0.20-0.30 bucket = 100% WR historically | UNTESTED
- [ ] **SOL underperformance check** | Does SOL drag down overall WR vs BTC/ETH only? | UNTESTED
- [ ] **Time remaining sweet spot** | Is 3-5 min better than 8-12 min? Backtest optimal entry window | UNTESTED
- [ ] **Model confidence > 60% hard floor** | Does cutting low-confidence trades improve net PnL? | UNTESTED

## MEDIUM IMPACT
- [ ] **Hourly sizing boost after 50+ trades** | Are Bayesian hourly multipliers helping or neutral? | COLLECTING DATA
- [ ] **ATR filter evaluation** | Shadow tracking active — is ATR filter blocking winners or losers? | COLLECTING DATA (check shadow_stats)
- [ ] **Trend bias 85% vs 70% vs 100%** | Current 85% counter-trend — is softer or harder better? | UNTESTED
- [ ] **Skip hour re-evaluation** | Paper trader shadow-tracking {0,1,8,22,23} UTC | COLLECTING DATA
- [ ] **Weekend-specific parameters** | Different edge thresholds or price limits on weekends? | UNTESTED
- [ ] **MACD-V revival** | Was removed from live — does it still add value as a filter? | UNTESTED

## LOW IMPACT / EXPLORATORY
- [ ] **Multi-asset correlation discount** | Paper trader uses 0.85 correlation — is this accurate? | UNTESTED
- [ ] **Arbitrage scanner on live** | Paper shows arb opportunities — worth live execution? | NEEDS REVIEW
- [ ] **Daily Up/Down markets** | kingofcoinflips strategy — different dynamics than 15m | PAPER ONLY
- [ ] **Cross-platform arb (Kalshi)** | Price differences between Polymarket and Kalshi | NEEDS RESEARCH

## COMPLETED
- [x] **DOWN >> UP bias** | Confirmed: DOWN 60%+ WR vs UP 44% WR | PROVEN
- [x] **Death zone $0.40-0.45** | Confirmed: 14% WR, -$321 PnL | HARDCODED SKIP
- [x] **Cheap entries win** | Confirmed: <$0.35 = highest WR bucket | PROVEN
- [x] **NYU two-parameter model** | Extreme prices = lower vol = better edge | DEPLOYED
- [x] **Skip hours data-driven** | Opened US/EU overlap, paper-validated | DEPLOYED
- [x] **Bayesian hourly sizing** | Phase 1 reduce-only, Phase 2 boost | DEPLOYED
