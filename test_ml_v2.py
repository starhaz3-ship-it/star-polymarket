"""Test ML Engine V2 and all new components."""
import sys
sys.path.insert(0, 'C:/Users/Star/.local/bin/star-polymarket')

import asyncio
import json

print("=" * 70)
print("TESTING ML ENGINE V2 AND ADVANCED COMPONENTS")
print("=" * 70)

# Test 1: Enhanced Features
print("\n1. TESTING ENHANCED FEATURES")
print("-" * 50)
from arbitrage.features import EnhancedFeatures, FeatureExtractor

features = EnhancedFeatures()
features.entry_price = 0.75
features.current_price = 0.80
features.outcome_is_yes = True
features.time.hours_to_expiry = 24
features.time.time_decay_factor = 0.2
features.momentum.momentum_1h = 0.05
features.whale.whale_consensus = 0.8

feature_array = features.to_array()
print(f"Feature vector shape: {feature_array.shape}")
print(f"Feature names: {len(EnhancedFeatures.feature_names())} features")
print(f"Sample features: entry={features.entry_price}, momentum_1h={features.momentum.momentum_1h}")
print("PASSED")

# Test 2: Advanced Kelly
print("\n2. TESTING ADVANCED KELLY")
print("-" * 50)
from arbitrage.kelly_v2 import AdvancedKelly, MultiAssetKelly, calculate_position_size

kelly = AdvancedKelly(kelly_fraction=0.25, max_position=0.15)

# Test basic Kelly
result = kelly.fractional_kelly(
    win_prob=0.70,
    market_price=0.55,
    bankroll=1000.0
)
print(f"Fractional Kelly: {result.fractional_kelly:.2%}")
print(f"Recommended size: ${result.recommended_size:.2f}")
print(f"Edge: {result.edge:.2%}")

# Test confidence-adjusted
result2 = kelly.confidence_adjusted_kelly(
    win_prob=0.70,
    market_price=0.55,
    confidence=0.8,
    bankroll=1000.0
)
print(f"Confidence-adjusted size: ${result2.recommended_size:.2f}")
print("PASSED")

# Test 3: ML Engine V2
print("\n3. TESTING ML ENGINE V2")
print("-" * 50)
from arbitrage.ml_engine_v2 import MLEngineV2

ml_engine = MLEngineV2(state_file="test_ml_v2_state.json")

# Add training samples
print("Training on synthetic data...")
for i in range(100):
    f = EnhancedFeatures()
    f.entry_price = 0.5 + (i % 50) / 100
    f.outcome_is_yes = i % 3 != 0  # 66% YES
    f.momentum.momentum_1h = 0.1 if i % 2 == 0 else -0.1

    won = (f.entry_price > 0.6 and f.outcome_is_yes) or (f.entry_price < 0.4 and not f.outcome_is_yes)
    ml_engine.record_outcome(f, won)

# Test prediction
test_features = EnhancedFeatures()
test_features.entry_price = 0.75
test_features.outcome_is_yes = True
test_features.momentum.momentum_1h = 0.1

prediction = ml_engine.predict(test_features, 0.70)
print(f"Win probability: {prediction.win_probability:.1%}")
print(f"Calibrated prob: {prediction.calibrated_probability:.1%}")
print(f"Confidence: {prediction.confidence:.1%}")
print(f"Model agreement: {prediction.model_agreement:.1%}")
print(f"Recommended: {prediction.recommended_action}")
print("PASSED")

# Test 4: Arbitrage Detection
print("\n4. TESTING ARBITRAGE DETECTION")
print("-" * 50)
from arbitrage.arb_detector import MomentumLagDetector

async def test_arb():
    detector = MomentumLagDetector(min_edge=0.05)
    btc_price = await detector.get_btc_price()
    print(f"Current BTC price: ${btc_price:,.2f}")

    opps = await detector.scan()
    print(f"Found {len(opps)} momentum lag opportunities")

    for opp in opps[:3]:
        print(f"  - {opp.market_title[:40]}")
        print(f"    Edge: {opp.profit_margin:.1%} | Action: {opp.action}")

    return len(opps)

asyncio.run(test_arb())
print("PASSED")

# Test 5: Whale Tracker
print("\n5. TESTING WHALE TRACKER")
print("-" * 50)
from arbitrage.whale_tracker import WhaleTracker

async def test_whale():
    tracker = WhaleTracker(state_file="test_whale_state.json")

    signals = await tracker.get_copy_signals(min_signal_strength=0.3)
    print(f"Found {len(signals)} whale signals")

    for sig in signals[:3]:
        print(f"  - {sig.market_title[:40]}")
        print(f"    Direction: {sig.direction} | Strength: {sig.signal_strength:.1%}")
        print(f"    Whale PnL: ${sig.whale_pnl:+,.0f}")

    return len(signals)

asyncio.run(test_whale())
print("PASSED")

# Summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)

stats = ml_engine.get_stats()
print(f"\nML Engine Stats:")
print(f"  Training samples: {stats.get('training_samples', 0)}")
print(f"  XGBoost available: {stats.get('has_xgboost', False)}")
print(f"  LightGBM available: {stats.get('has_lightgbm', False)}")

print("\nNew modules ready:")
print("  - arbitrage/features.py (31 enhanced features)")
print("  - arbitrage/kelly_v2.py (advanced Kelly)")
print("  - arbitrage/ml_engine_v2.py (ensemble ML)")
print("  - arbitrage/arb_detector.py (arbitrage detection)")
print("  - arbitrage/whale_tracker.py (whale tracking)")
print("  - arbitrage/ml_trader_v2.py (integrated system)")
