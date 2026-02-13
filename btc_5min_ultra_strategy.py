"""
BTC 5-Min Ultra Strategy - The 'Sweet Spot' Method
====================================================
Derived from 82 live trades analysis. Targets high-probability BTC entries.

PROVEN EDGES (from ta_pattern_analysis.md):
- MACD Expanding: 87.5% WR (7/8 trades)
- Below VWAP: 76.5% WR (13/17 trades)
- BTC UP direction: 90.9% WR (10/11 trades)
- Entry $0.40-$0.52: Sweet spot for value + resolution probability

CORE RULES:
1. Entry price: $0.40 - $0.52 (reject outside this range)
2. MACD must be EXPANDING (not contracting)
3. Price must be BELOW VWAP (undervalued relative to volume-weighted avg)
4. Confidence >= 0.65
5. KL Divergence >= 0.15 (edge vs market)
6. Heiken Ashi count != 3 (avoid reversal zone)
7. Time remaining: 7-10 minutes (not too early, not too late)
8. Direction bias: BTC UP preferred (90.9% live WR)

POSITION SIZING: $3 flat per trade (Kelly suggests more but flat is safer)

TIME SKIPS (low WR hours UTC): 0, 1, 8

EXPECTED PERFORMANCE:
- Combined filter WR: ~75-85%
- Avg profit per win: ~$0.80-$1.20
- Avg loss per loss: ~$3.00
- Break-even WR at these odds: ~72%
- Edge: +3-13% above break-even
"""

# Strategy configuration for integration with ta_signals / run_ta_live
BTC_5MIN_ULTRA_CONFIG = {
    'name': 'BTC_5MIN_ULTRA',
    'asset': 'BTC',
    'timeframe': '5min',

    # Entry filters
    'min_entry_price': 0.40,
    'max_entry_price': 0.52,
    'min_confidence': 0.65,
    'min_kl_divergence': 0.15,

    # TA filters
    'require_macd_expanding': True,
    'require_below_vwap': True,
    'reject_heiken_count': [3],  # Avoid reversal zone

    # Direction bias
    'preferred_direction': 'UP',  # 90.9% WR on BTC UP
    'allow_down': True,           # Still allow DOWN but with caution
    'down_confidence_boost': 0.05, # Require +5% more confidence for DOWN

    # Timing
    'min_time_remaining_min': 7,
    'max_time_remaining_min': 10,
    'skip_hours_utc': [0, 1, 8],  # Low WR hours

    # Position sizing
    'bet_size_usd': 3.00,
    'max_concurrent_positions': 2,

    # Risk management
    'max_daily_loss_usd': 15.00,
    'max_consecutive_losses': 3,  # Pause after 3 losses in a row
    'cooldown_after_loss_streak_min': 30,
}


def should_enter_btc_ultra(signal: dict, config: dict = None) -> tuple:
    """
    Evaluate whether a BTC signal meets the Ultra Strategy criteria.

    Args:
        signal: Dict with keys: entry_price, confidence, kl_divergence,
                macd_expanding, below_vwap, heiken_count, time_remaining_min,
                direction, hour_utc
        config: Optional override config (defaults to BTC_5MIN_ULTRA_CONFIG)

    Returns:
        (should_enter: bool, reason: str)
    """
    cfg = config or BTC_5MIN_ULTRA_CONFIG

    entry = signal.get('entry_price', 0)
    conf = signal.get('confidence', 0)
    kl = signal.get('kl_divergence', 0)
    macd_exp = signal.get('macd_expanding', False)
    below_vwap = signal.get('below_vwap', False)
    heiken = signal.get('heiken_count', 0)
    time_rem = signal.get('time_remaining_min', 0)
    direction = signal.get('direction', '').upper()
    hour = signal.get('hour_utc', 0)

    # Skip hours
    if hour in cfg['skip_hours_utc']:
        return False, f"Skip hour UTC {hour}"

    # Entry price range
    if entry < cfg['min_entry_price'] or entry > cfg['max_entry_price']:
        return False, f"Entry ${entry:.2f} outside ${cfg['min_entry_price']}-${cfg['max_entry_price']}"

    # Confidence
    min_conf = cfg['min_confidence']
    if direction == 'DOWN':
        min_conf += cfg.get('down_confidence_boost', 0.05)
    if conf < min_conf:
        return False, f"Confidence {conf:.2f} < {min_conf:.2f}"

    # KL Divergence
    if kl < cfg['min_kl_divergence']:
        return False, f"KL {kl:.3f} < {cfg['min_kl_divergence']}"

    # MACD expanding
    if cfg['require_macd_expanding'] and not macd_exp:
        return False, "MACD not expanding"

    # Below VWAP
    if cfg['require_below_vwap'] and not below_vwap:
        return False, "Price not below VWAP"

    # Heiken Ashi reversal zone
    if heiken in cfg['reject_heiken_count']:
        return False, f"Heiken count {heiken} in rejection zone"

    # Time remaining
    if time_rem < cfg['min_time_remaining_min'] or time_rem > cfg['max_time_remaining_min']:
        return False, f"Time remaining {time_rem}m outside {cfg['min_time_remaining_min']}-{cfg['max_time_remaining_min']}m"

    # Direction check
    if direction == 'DOWN' and not cfg.get('allow_down', True):
        return False, "DOWN direction not allowed"

    # All filters passed
    edge_score = conf * (1 + kl)
    return True, f"ENTER {direction} @ ${entry:.2f} | conf={conf:.2f} kl={kl:.3f} edge={edge_score:.3f}"


if __name__ == '__main__':
    # Example usage
    test_signal = {
        'entry_price': 0.45,
        'confidence': 0.72,
        'kl_divergence': 0.22,
        'macd_expanding': True,
        'below_vwap': True,
        'heiken_count': 5,
        'time_remaining_min': 8,
        'direction': 'UP',
        'hour_utc': 14,
    }

    enter, reason = should_enter_btc_ultra(test_signal)
    print(f"Should enter: {enter}")
    print(f"Reason: {reason}")
    print(f"\nStrategy file: {__file__}")
