"""Test the ML Engine"""
import sys
sys.path.insert(0, 'C:/Users/Star/.local/bin/star-polymarket')

from arbitrage.ml_engine import MLEngine, TradeFeatures, extract_features
import json

# Initialize ML engine
engine = MLEngine()

# Load whale trade data to train the model
with open('C:/Users/Star/.local/bin/star-polymarket/data/btc_backtest_compact.json', 'r') as f:
    data = json.load(f)

print('Training ML model on historical data...')
print()

# Train on historical trades
wins = 0
losses = 0
for trade in data.get('whale_trades', []):
    features = extract_features(trade)
    pnl = float(trade.get('pnl', 0) or 0)
    won = pnl > 0

    # Record outcome for learning
    engine.record_outcome(features, won, pnl / 1000)  # Normalize

    if won:
        wins += 1
    else:
        losses += 1

print(f'Trained on {wins + losses} trades ({wins} wins, {losses} losses)')
print()

# Print learned status
engine.print_status()

# Test predictions on sample trades
print()
print('SAMPLE PREDICTIONS:')
print('-' * 60)

test_cases = [
    {'name': 'High prob YES', 'entry_price': 0.80, 'outcome': 'Yes', 'momentum': 0.20, 'whale_pnl': 2000},
    {'name': 'Low prob YES', 'entry_price': 0.30, 'outcome': 'Yes', 'momentum': -0.10, 'whale_pnl': -500},
    {'name': 'High prob NO', 'entry_price': 0.90, 'outcome': 'No', 'momentum': 0.05, 'whale_pnl': 1000},
    {'name': 'Sweet spot YES', 'entry_price': 0.70, 'outcome': 'Yes', 'momentum': 0.15, 'whale_pnl': 1500},
]

for tc in test_cases:
    features = TradeFeatures(
        entry_price=tc['entry_price'],
        momentum=tc['momentum'],
        whale_pnl=tc['whale_pnl'],
        whale_pnl_pct=tc['whale_pnl'] / 100,
        time_to_expiry_hours=48,
        outcome_is_yes=tc['outcome'] == 'Yes',
        volume=100000,
        liquidity=50000,
        price_volatility=0.1,
        whale_position_size=5000,
    )

    should_trade, reason = engine.should_trade(features)
    pred = engine.predict(features, bankroll=1000)

    status = 'TAKE' if should_trade else 'SKIP'
    print(f"{tc['name']:20s} | {status:4s} | Win: {pred.win_probability:.0%} | Kelly: ${pred.recommended_size:.0f}")
    print(f"                     | {reason[:55]}")
    print()
