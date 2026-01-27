# Claude Context - Star Polymarket

**Project:** ML-Optimized Prediction Market Arbitrage Bot
**Owner:** Star
**Resume Command:** "Access Star Polymarket"

## Quick Start

```bash
# Paper trade
python -m arbitrage.main --dry-run --verbose

# Live trade (REAL MONEY)
python -m arbitrage.main --verbose
```

## Critical Config (Magic Wallet)

```python
# REQUIRED for trading - user uses Magic (email) wallet
ClobClient(
    signature_type=1,  # Magic wallet
    funder="0x4f3456c5b05b14D8aFc40E0299d6d68DE0fF7d7e",  # Proxy wallet
)
```

## Strategies

1. **BTC Latency** - Spot price vs Polymarket odds lag
2. **Same-Market** - Buy YES + NO when sum < $1.00
3. **NegRisk** - Multi-outcome markets sum < 100% ($29M historical)
4. **Endgame** - 95%+ probability near resolution
5. **Cross-Platform** - Polymarket vs Kalshi price differences

## File Structure

- `arbitrage/main.py` - Bot entry point
- `arbitrage/detector.py` - Arbitrage detection
- `arbitrage/ml_optimizer.py` - ML parameter tuning
- `arbitrage/kalshi_feed.py` - Cross-platform arbitrage
- `.env` - Encrypted credentials (password: see context file)

## GitHub

- Bot: https://github.com/starhaz3-ship-it/star-polymarket
- Research: https://github.com/starhaz3-ship-it/polymarket-knowledge-base

## Full Context

See: `C:\Users\Star\.claude\star-polymarket-context.md`
