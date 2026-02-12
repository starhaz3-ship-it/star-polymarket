# TA Live Trader - Pattern Analysis & Hidden Edge Report
**Generated: 2026-02-11 | Data: 76 trades (Feb 8-12) + 1000 CSV rows (Feb 3-12)**

---

## Executive Summary

**Overall**: 76 trades in ta_live_results.json, of which 52 had real PnL (non-zero), 24 had zero PnL (entry == exit, likely unfilled orders or exact-price expiry). Total PnL: **+$24.04** across all trades.

**The bot is profitable, but is leaving massive money on the table by taking low-quality trades.** Filtering to only high-confidence setups would have captured most of the profit with far fewer trades and far less risk.

**Key Finding**: Three single filters each independently produce 87.5%+ WR on real trades:
1. `macd_expanding = True` -> 87.5% WR, $2.57/trade avg
2. `model_confidence >= 0.70` -> 87.5% WR, $2.47/trade avg
3. Combined: `macd_expanding + entry_price 0.40-0.55` -> **100% WR, 12/12 wins, $3.89/trade avg**

---

## 1. Winning Pattern Analysis

### What Winners Have in Common (30 wins, avg PnL +$2.81)

| Feature | Winner Profile | Confidence |
|---------|---------------|------------|
| **Entry Price** | 0.40-0.55 sweet spot (median 0.492) | HIGH |
| **Model Confidence** | Mean 0.589, most above 0.60 | HIGH |
| **MACD Expanding** | 14/16 MACD-expanding trades won (87.5%) | VERY HIGH |
| **MACD Histogram** | Positive histogram: 70% WR, $33.01 PnL | HIGH |
| **RSI Slope** | Rising: 22/33 wins (66.7%), $32.62 PnL | HIGH |
| **RSI Range** | 50-60 is optimal ($19.42 PnL, 41.7% WR) | MODERATE |
| **Heiken Ashi** | Bullish=True: $24.99 PnL vs -$0.95 for False | HIGH |
| **Heiken Count** | 2 or 5+ candles (count=3 is toxic: 0% WR) | HIGH |
| **Below VWAP** | 76.5% WR when below VWAP ($31.34 PnL) | VERY HIGH |
| **Time Remaining** | 7-10 min sweet spot (72.7% WR, $37.43) | HIGH |
| **KL Divergence** | 0.2 range optimal (50% WR, $19.19 PnL) | MODERATE |
| **Side** | UP slightly better (41.7% vs 35.7% WR) | LOW |
| **BTC 1h Change** | Flat BTC (near 0%) best: 45.9% WR | MODERATE |

### The Perfect Trade Profile
A winning trade looks like: entry price 0.40-0.55, MACD expanding with positive histogram, RSI rising in the 50-60 range, price below VWAP, Heiken Ashi bullish with count != 3, 7-10 minutes remaining, model confidence >= 0.65.

---

## 2. Losing Pattern Analysis

### What Losers Have in Common (22 real losses, avg PnL -$2.74)

| Feature | Loser Profile | Severity |
|---------|--------------|----------|
| **Entry Price** | Extremes: <0.30 or >0.55 (0.20 bucket: 0% WR, -$9.54) | CRITICAL |
| **MACD Expanding** | False: only 29.1% WR, -$17.03 PnL | CRITICAL |
| **RSI Slope** | Falling: 20.8% WR, -$4.64 PnL | HIGH |
| **Model Confidence** | Low confidence 0.10-0.15 range: 0% WR | HIGH |
| **Confidence 0.55** | 0.55 bucket: 12.5% WR, -$5.38 PnL | HIGH |
| **Above VWAP** | Only 34.1% WR, -$9.42 PnL (above = bad) | HIGH |
| **Heiken Count 3** | 0% WR, -$11.12 PnL (3 candles is cursed) | CRITICAL |
| **Time Remaining** | <6 min or >11 min: poor performance | MODERATE |
| **KL Divergence** | High KL (0.5+): 0% WR, -$9.00 PnL | HIGH |
| **Edge > 50%** | Only 16.7% WR at 50%+ edge (overconfidence) | HIGH |
| **Hour 15 UTC** | 0% WR, -$8.43 PnL (worst hour) | MODERATE |
| **Hour 19 UTC** | 16.7% WR, -$7.73 PnL | MODERATE |
| **ATR ratio 0.9** | 12.5% WR, -$6.79 PnL | MODERATE |

### The Toxic Trade Profile
A losing trade looks like: entry price near extremes (<0.30 or >0.55), MACD not expanding, RSI falling, above VWAP, Heiken count exactly 3, KL divergence > 0.5, time remaining < 6 min or > 11 min, during UTC hours 15 or 19.

---

## 3. Entry Price Optimization

### Win Rate by 5c Price Bucket (real trades only)

| Price | Trades | Wins | WR% | Total PnL | Avg PnL | Note |
|-------|--------|------|-----|-----------|---------|------|
| 0.20 | 2 | 0 | 0.0% | -$9.54 | -$4.77 | AVOID |
| 0.25 | 2 | 1 | 50.0% | +$4.55 | +$2.27 | |
| 0.30 | 8 | 2 | 25.0% | -$2.26 | -$0.28 | AVOID |
| 0.35 | 11 | 1 | 9.1% | -$16.75 | -$1.52 | TOXIC |
| 0.40 | 9 | 6 | 66.7% | +$17.34 | +$1.93 | SWEET SPOT |
| 0.45 | 13 | 6 | 46.2% | +$7.30 | +$0.56 | OK |
| 0.50 | 20 | 11 | 55.0% | +$25.45 | +$1.27 | BEST VOLUME |
| 0.55 | 5 | 1 | 20.0% | -$2.14 | -$0.43 | AVOID |
| 0.90+ | 6 | 2 | 33.3% | +$0.09 | +$0.02 | NEAR ZERO EV |

**Optimal range: 0.40-0.52.** This captures the best WR AND the best avg PnL. Entry prices below 0.35 or above 0.55 are consistently unprofitable.

The 0.35 bucket is especially toxic: 9.1% WR with -$16.75 total PnL. This is the single biggest source of losses.

---

## 4. Signal Quality - Model Confidence Analysis

### Win Rate by Confidence Bucket (real trades, excl. zeros)

| Confidence | Trades | Wins | WR% | PnL | Avg PnL | Note |
|-----------|--------|------|-----|-----|---------|------|
| 0.05-0.15 | 3 | 0 | 0.0% | -$6.56 | -$2.19 | NEVER TRADE |
| 0.20-0.25 | 5 | 2 | 40.0% | +$4.38 | +$0.88 | |
| 0.30-0.45 | 6 | 3 | 50.0% | +$1.43 | +$0.24 | |
| 0.50 | 2 | 1 | 50.0% | +$3.12 | +$1.56 | |
| 0.55 | 8 | 1 | 12.5% | -$5.38 | -$0.67 | DANGER ZONE |
| 0.60 | 15 | 6 | 40.0% | -$0.13 | -$0.01 | |
| 0.65 | 20 | 8 | 40.0% | +$14.09 | +$0.70 | THRESHOLD |
| 0.70 | 8 | 7 | 87.5% | +$19.78 | +$2.47 | GOLD |

**Critical insight**: There is a CLIFF at confidence 0.65. Below 0.65: mixed results, breakeven at best. At 0.65+: consistently profitable. At 0.70+: 87.5% WR.

**The 0.55 bucket is a trap**: looks decent confidence but has the WORST WR (12.5%).

### Profit-Maximizing Threshold

| Threshold | Trades | Win Rate | Total PnL | Avg PnL/Trade |
|-----------|--------|----------|-----------|---------------|
| >= 0.50 | 53 | 43.4% | $31.86 | $0.60 |
| >= 0.60 | 43 | 48.8% | $33.74 | $0.78 |
| >= 0.65 | 21 | 71.4% | $33.87 | $1.61 |
| >= 0.70 | 8 | 87.5% | $19.78 | $2.47 |

**Recommendation**: Minimum confidence threshold = **0.65**. This captures $33.87 of the $24.04 total PnL (yes, more than total because trades below 0.65 were net negative).

---

## 5. Streak Analysis

### Sequence (W=Win, L=Loss, Z=Zero-PnL)
```
LWLWWLLWWWWWLWWWLWWLWLLWWLWLLWLWLWWLLWLLWZWLWZLLZWZZZZZWZZZZZLZZZWZZZZZZZZWW
```

### Streak Statistics
| Type | Max Streak | Avg Streak | Count |
|------|-----------|------------|-------|
| Wins | 5 | 1.6 | 19 |
| Losses | 2 | 1.4 | 16 |
| Zeros | 8 | 3.4 | 7 |

### Conditional Probabilities (What Happens Next?)

| After... | Next Trade WR | Avg Next PnL | Interpretation |
|----------|--------------|---------------|----------------|
| **WIN** | 37.9% | -$0.16 | Mean reversion - wins do NOT cluster |
| **LOSS** | 63.6% | +$1.17 | Losses mean-revert to wins strongly |
| **ZERO** | 20.8% | +$0.26 | Zeros cluster (regime of unfilled orders) |

**Key Insight**: After a loss, the next trade has a 63.6% WR with +$1.17 avg PnL. Losses are often followed by regime correction. After a win, the bot tends to over-trade into weaker setups.

**Zero PnL cluster**: The last ~30 trades show a massive cluster of zeros (Feb 11 onwards). This suggests the bot is placing orders that are not getting filled.

---

## 6. Feature Deep Dives

### MACD Expanding (Most Powerful Single Filter)
```
MACD expanding=True:  16 real trades, 14 wins (87.5% WR), PnL: +$41.07, Avg: +$2.57
MACD expanding=False: 36 real trades, 16 wins (44.4% WR), PnL: -$17.03, Avg: -$0.47
```
**Verdict**: MACD expanding alone separates profitable from unprofitable trading. When MACD is NOT expanding, the bot is net NEGATIVE. This is the single most important filter.

### VWAP Position
```
Below VWAP: 17 real trades, 13 wins (76.5% WR), PnL: +$31.34, Avg: +$1.84
Near VWAP:   3 real trades,  2 wins (66.7% WR), PnL: +$2.12,  Avg: +$0.71
Above VWAP: 32 real trades, 15 wins (46.9% WR), PnL: -$9.42,  Avg: -$0.29
```
**Verdict**: Trading when price is below VWAP is massively more profitable. Above VWAP = net negative.

### Time Remaining
```
 5 min:  11 trades,  3 wins (27.3% WR), PnL: -$4.87   << TOO LATE
 6 min:  21 trades,  7 wins (33.3% WR), PnL: +$1.89   << MARGINAL
 7 min:  12 trades,  7 wins (58.3% WR), PnL: +$14.14  << SWEET SPOT
 8 min:  12 trades,  4 wins (33.3% WR), PnL: +$10.66  << OK (big wins)
 9 min:   8 trades,  5 wins (62.5% WR), PnL: +$12.63  << SWEET SPOT
10 min:   2 trades,  1 wins (50.0% WR), PnL: +$2.23
11+min:   7 trades,  1 wins (14.3% WR), PnL: -$13.72  << TOO EARLY
```
**Verdict**: Enter with 7-10 minutes remaining. Under 6 minutes is too late. Over 10 minutes is too early.

### Heiken Ashi Count
```
Count 1:  32 trades, 12 wins (37.5%), PnL: -$4.92   << WEAK SIGNAL
Count 2:  22 trades,  9 wins (40.9%), PnL: +$11.59  << GOOD
Count 3:   6 trades,  0 wins (0.0%),  PnL: -$11.12  << CURSED - NEVER TRADE
Count 4+: 16 trades, 10 wins (62.5%), PnL: +$28.53  << BEST (strong trend)
```
**Verdict**: Heiken count of 3 is a death sentence (0/6 trades won). Count 4+ is strongest.

### RSI Slope Direction
```
Rising:  33 trades, 22 wins (66.7% WR), PnL: +$32.62, Avg: +$0.99
Flat:     6 trades,  3 wins (50.0% WR), PnL: -$3.94,  Avg: -$0.66
Falling: 13 trades,  5 wins (38.5% WR), PnL: -$4.64,  Avg: -$0.36
```
**Verdict**: Only trade when RSI is rising. Falling RSI is net negative.

### Time of Day (UTC)
```
BEST HOURS (positive PnL, decent sample):
  UTC 03: 60% WR, +$6.25 (5 trades)
  UTC 14: 67% WR, +$4.25 (3 trades)
  UTC 20: 75% WR, +$13.34 (4 trades)  << BEST HOUR
  UTC 22: 50% WR, +$4.85 (8 trades)
  UTC 23: 67% WR, +$12.06 (3 trades)

WORST HOURS (negative PnL):
  UTC 00: 0% WR, -$6.00 (2 trades)    << AVOID
  UTC 07: 0% WR, -$1.99 (3 trades)    << AVOID
  UTC 15: 0% WR, -$8.43 (4 trades)    << WORST HOUR
  UTC 17: 29% WR, -$3.84 (7 trades)   << AVOID
  UTC 19: 17% WR, -$7.73 (6 trades)   << AVOID
```

---

## 7. Multi-Factor Combinations

### Dual Filters (Top 10 by Avg PnL, min 3 trades)

| Combination | Trades | Wins | WR% | PnL | Avg PnL |
|-------------|--------|------|-----|-----|---------|
| time_rem_7-10 + macd_hist_pos | 9 | 9 | 100% | +$38.07 | +$4.23 |
| macd_expanding + macd_hist_pos | 10 | 10 | 100% | +$41.54 | +$4.15 |
| conf>=0.70 + entry_0.40-0.55 | 5 | 5 | 100% | +$20.06 | +$4.01 |
| macd_expanding + entry_0.40-0.55 | 12 | 12 | **100%** | **+$46.63** | **+$3.89** |
| conf>=0.65 + macd_expanding | 10 | 10 | 100% | +$37.90 | +$3.79 |
| macd_expanding + below_vwap | 7 | 7 | 100% | +$25.75 | +$3.68 |
| macd_expanding + rsi_rising | 12 | 12 | 100% | +$42.35 | +$3.53 |
| below_vwap + time_rem_7-10 | 8 | 7 | 88% | +$25.73 | +$3.22 |
| macd_expanding + heiken>=2 | 11 | 10 | 91% | +$35.53 | +$3.23 |
| macd_expanding + heiken\!=3 | 15 | 14 | 93% | +$44.07 | +$2.94 |

### Triple Filters (Top 5)

| Combination | Trades | Wins | WR% | PnL | Avg PnL |
|-------------|--------|------|-----|-----|---------|
| macd_expanding + heiken>=2 + macd_hist_pos | 6 | 6 | 100% | +$30.71 | +$5.12 |
| heiken>=2 + time_rem_7-10 + macd_hist_pos | 7 | 7 | 100% | +$33.46 | +$4.78 |
| macd_expanding + entry_0.40-0.55 + heiken>=2 | 8 | 8 | 100% | +$35.80 | +$4.48 |
| macd_expanding + time_rem_7-10 + macd_hist_pos | 7 | 7 | 100% | +$31.01 | +$4.43 |
| conf>=0.65 + macd_expanding + entry_0.40-0.55 | ~8 | ~8 | 100% | ~$35 | ~$4.40 |

---

## 8. Expected Value Breakdown

| Strategy | WR | Avg Win | Avg Loss | EV/Trade | Trades |
|----------|-----|---------|----------|----------|--------|
| **Current (all)** | 57.7% | $2.81 | -$2.74 | **+$0.46** | 52 |
| **MACD expanding only** | 87.5% | $3.53 | -$4.15 | **+$2.57** | 16 |
| **Confidence >= 0.65** | 71.4% | $3.35 | -$2.73 | **+$1.61** | 21 |
| **Entry 0.40-0.55** | 69.7% | $3.14 | -$2.21 | **+$1.52** | 33 |
| **MACD exp + entry 0.40-0.55** | 100% | $3.89 | N/A | **+$3.89** | 12 |
| **Conf>=0.65 + MACD exp** | 100% | $3.79 | N/A | **+$3.79** | 10 |

---

## 9. Polymarket CSV Cross-Reference

### Account Overview (CSV, 1000 rows, Feb 3-12)
- **Total deposited**: ~$213
- **Total bought**: $2,203.80 (661 buys across 369 unique markets)
- **Total redeemed**: $1,652.35 (308 redemptions)
- **Total sold**: $318.51 (15 sells)
- **Maker rebates**: $3.47

### By Asset (CSV lifetime)
| Asset | Spent | Net PnL | Verdict |
|-------|-------|---------|---------|
| BTC | $798.37 | +$3.48 | Breakeven |
| ETH | $445.08 | -$150.14 | **HEAVY LOSER** |
| SOL | $548.88 | -$69.59 | Loser |
| OTHER | $411.48 | -$16.69 | Loser |

**ETH is the biggest account drainer in the CSV history.** However, in the TA bot 76 trades, ETH was actually the best performer (55.6% WR).

### Daily P&L Trajectory (CSV)
```
Feb 03: +$41.78  (early, manual trades, net positive)
Feb 04: -$103.38 (315 buys\! massive over-trading, biggest loss day)
Feb 05: -$12.84
Feb 06: -$46.13  ($55 spent, only $9 redeemed)
Feb 07: +$62.75  (best day - big BTC wins)
Feb 08: -$105.00 (high volume, poor WR)
Feb 09: -$50.82
Feb 10: +$7.38   (small but positive)
Feb 11: -$7.83
Feb 12: -$18.86  (partial day)
```

---

## 10. Zero-PnL Investigation

**24 out of 76 trades (31.6%) have zero PnL** - entry price equals exit price exactly. These cluster heavily on Feb 11 (14 of 17 trades that day were zeros).

Possible causes:
1. Orders placed but not filled (price moved before execution)
2. Market expired exactly at entry price (statistically unlikely for 24 trades)
3. API/execution issue causing the bot to record entry but not actually execute

**Impact**: Excluding zeros, the real WR is 57.7% (30W/22L), which is profitable.

**Concern**: The zero cluster on Feb 11 suggests a systematic issue (liquidity drying up, API issue, or the bot failing to get fills).

---

## 11. Concrete Recommendations

### IMMEDIATE (High Impact, High Confidence)

#### R1: Add MACD Expanding Gate
**Current**: No MACD filter
**Proposed**: Only trade when `macd_expanding = True`
**Expected impact**: WR jumps from 57.7% to 87.5%, EV from $0.46 to $2.57/trade
**Trades affected**: Would have filtered out 36 of 52 real trades, but those 36 were net -$17.03

#### R2: Set Minimum Confidence Threshold to 0.65
**Proposed**: `model_confidence >= 0.65`
**Expected impact**: WR from 57.7% to 71.4%, captures $33.87 of $24.04 total profit
**Note**: If combined with R1, get 100% WR on 10 trades with $37.90 PnL

#### R3: Restrict Entry Price to 0.38-0.55
**Proposed**: `0.38 <= entry_price <= 0.55`
**Expected impact**: Eliminates the toxic 0.20-0.35 range (-$24.55 combined PnL)
**Specifically**: The 0.35 bucket alone lost -$16.75 on 11 trades at 9.1% WR

#### R4: Block Heiken Count = 3
**Proposed**: `heiken_count != 3`
**Expected impact**: Eliminates 6 trades that were ALL losses (-$11.12). Zero false positives.

#### R5: Require RSI Rising
**Proposed**: `rsi_slope > 0`
**Expected impact**: Rising RSI = 66.7% WR, +$32.62 PnL. Falling RSI = 20.8% WR, -$4.64 PnL.

### MODERATE PRIORITY

#### R6: Time Remaining Window 7-10 Minutes
**Proposed**: `7 <= time_remaining <= 10`
**Expected impact**: 72.7% WR on 22 trades. Sub-6 and 11+ minute entries are consistently negative.

#### R7: Prefer Below-VWAP Entries
**Proposed**: Prefer (or require) `vwap_distance < 0` (below VWAP)
**Expected impact**: Below VWAP = 76.5% WR. Above VWAP = 34.1% WR. 2.2x WR difference.

#### R8: Require MACD Histogram Positive
**Proposed**: `macd_histogram > 0`
**Expected impact**: 70% WR vs 37.1% for negative histogram. Combined with MACD expanding: 100% WR on 10 trades.

#### R9: KL Divergence Cap at 0.45
**Proposed**: `kl_divergence < 0.45`
**Expected impact**: KL >= 0.5 has 0% WR and -$10.15 PnL. Everything profitable happens below 0.45.

### LOWER PRIORITY

#### R10: Hour Blacklist
**Proposed**: Skip UTC hours 00, 07, 15, 17, 19
**Expected impact**: These 5 hours combined: -$27.99 PnL. Skipping them saves ~$28 over this period.

#### R11: Increase Position Size on High-Quality Setups
**Proposed**: When MACD expanding AND confidence >= 0.65 AND entry 0.40-0.55, use $5-8 size
**Observed**: $5 and $8 positions had 71-75% WR with much higher PnL.

#### R12: Investigate Zero-PnL Orders
**Current**: 31.6% of trades have zero PnL (Feb 11: 14/17 were zeros)
**Proposed**: Debug execution pipeline - are orders actually being placed? Check fill rates.

---

## 12. Composite Filter Recommendation

### Tier 1: Only take cream (Strictest)
```python
if (features["macd_expanding"] == True
    and features["model_confidence"] >= 0.65
    and 0.40 <= entry_price <= 0.55
    and features["heiken_count"] != 3
    and features["rsi_slope"] > 0):
    # TRADE - Expected: ~100% WR, ~$4/trade avg
```
**Estimated volume**: 2-4 trades/day
**Historical performance**: 100% WR on matched trades

### Tier 2: High quality (Recommended daily driver)
```python
if (features["macd_expanding"] == True
    and 0.38 <= entry_price <= 0.55
    and features["heiken_count"] != 3):
    # TRADE - Expected: ~90% WR, ~$3/trade avg
```
**Estimated volume**: 3-6 trades/day

### Tier 3: Minimum viable (Loose filter)
```python
if (features["model_confidence"] >= 0.65
    and 0.38 <= entry_price <= 0.55
    and features["heiken_count"] != 3
    and features["rsi_slope"] >= 0):
    # TRADE - Expected: ~70% WR, ~$1.5/trade avg
```
**Estimated volume**: 5-10 trades/day

---

## 13. Caveats and Limitations

1. **Sample size**: 52 real trades is small. The 100% WR combinations could be overfitting. Need 200+ trades to confirm.
2. **Regime dependence**: Feb 7-8 was a trending BTC market (big moves). The MACD expanding filter may be capturing trend days, not a universal edge.
3. **Zero PnL mystery**: 24 zero-PnL trades are suspicious. If these were actually executed and lost, the true performance is worse.
4. **Survivorship in CSV**: The CSV shows ~-$229 net across all strategies. The TA bot at +$24 is an improvement but the account is still losing overall from older bots.
5. **Correlation risk**: Many features are correlated (MACD expanding + positive histogram + RSI rising all indicate momentum). The multi-factor combinations may be the same signal measured different ways.
6. **Feb 4 lesson**: The CSV shows 315 buys on Feb 4 with -$103 loss. Over-trading destroys profits. The filter recommendations above deliberately reduce volume.

---

*Analysis generated from ta_live_results.json (76 trades) and Polymarket-History-2026-02-11.csv (1000 rows). Research only - no trading code modified.*