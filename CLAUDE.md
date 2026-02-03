# Claude Context - Star Polymarket

**Project:** ML-Optimized Prediction Market Trading Bot
**Owner:** Star
**Resume Command:** "Access Star-Polymarket"

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

### Current Status (Last Updated: 2026-02-03)
- Paper Trader: Running, 1W/1L, +$59.94 PnL
- Strategy: TA signals (RSI, VWAP, Heiken Ashi, MACD) + Bregman divergence + Kelly sizing
- Markets: BTC 15-minute Up/Down via Polymarket API (`tag_slug=15M`)

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

## File Structure
- `arbitrage/` - Core trading modules
- `run_ta_paper.py` - TA paper trader
- `run_ta_live.py` - TA live trader with ML
- `.env` - Encrypted credentials

## GitHub
- Bot: https://github.com/starhaz3-ship-it/star-polymarket
- Research: https://github.com/starhaz3-ship-it/polymarket-knowledge-base

## Full Context
See: `C:\Users\Star\.claude\star-polymarket-context.md`
