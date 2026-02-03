"""Analyze whale performance for the past week."""
import asyncio
import httpx
from datetime import datetime, timedelta

# Top whales from the tracker
WHALES = [
    ('kch123', '0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee'),
    ('bobe2', '0xed107a85a4585a381e48c7f7ca4144909e7dd2e5'),
    ('swisstony', '0x204f72f35326db932158cba6adff0b9a1da95e14'),
    ('RN1', '0x2005d16a84ceefa912d4e380cd32e7ff827875ea'),
    ('0x8dxd', '0x63ce342161250d705dc0b16df89036c8e5f9ba9a'),
    ('DrPufferfish', '0xdb27bf2ac5d428a9c63dbc914611036855a6c56e'),
    ('k9Q2mX4L8A7ZP3R', '0xd0d6053c3c37e727402d84c14069780d360993aa'),
    ('AiBird', '0xa58d4f278d7953cd38eeb929f7e242bfc7c0b9b8'),
    ('kingofcoinflips', '0xe9c689e429cc82232bb78d5598a78a4c144f5c95'),
    ('LDSIADAS', '0xd830027529b0baca2a52fd8a4dee43d366a9a592'),
    ('automatedAItradingbot', '0xd8f8c13644ea84d62e1ec88c5d1215e436eb0f11'),
    ('WickRick', '0x1c12abb42d0427e70e144ae67c92951b232f79d9'),
    ('aenews2', '0x44c1dfe43260c94ed4f1d00de2e1f80fb113ebc1'),
    ('ImJustKen', '0x9d84ce0306f8551e02efef1680475fc0f1dc1344'),
    ('willi', '0xcb2952fd655813ad0f9ee893122c54298e1a975e'),
    ('NeverYES', '0xe55108581aec68af7686bd4cbc4f53ef93680a67'),
    ('Harmless-Critic', '0x1461cC6e1A05e20710c416307Db62C28f1D122d8'),
    ('distinct-baguette', '0xe00740bce98a594e26861838885ab310ec3b548c'),
]

async def fetch_whale_data():
    results = []

    async with httpx.AsyncClient(timeout=30) as client:
        for name, address in WHALES:
            try:
                # Fetch profile for total profit
                profile_resp = await client.get(
                    f'https://polymarket.com/api/profile/{address}'
                )
                profile = profile_resp.json() if profile_resp.status_code == 200 else {}

                # Fetch positions
                pos_resp = await client.get(
                    'https://data-api.polymarket.com/positions',
                    params={'user': address, 'sizeThreshold': 0}
                )
                positions = pos_resp.json() if pos_resp.status_code == 200 else []

                # Get profit from profile
                total_profit = float(profile.get('profit', 0) or 0)
                total_volume = float(profile.get('volume', 0) or 0)
                markets_traded = int(profile.get('marketsTraded', 0) or 0)

                # Calculate metrics from positions
                total_value = sum(float(p.get('currentValue', 0) or 0) for p in positions)
                realized_pnl = sum(float(p.get('cashPnl', 0) or 0) for p in positions)
                unrealized_pnl = sum(float(p.get('percentPnl', 0) or 0) * float(p.get('currentValue', 0) or 0) for p in positions)

                # Count open positions
                open_pos_count = 0
                winning_pos = 0
                losing_pos = 0
                avg_entry = 0

                for p in positions:
                    size = float(p.get('size', 0) or 0)
                    if size > 0:
                        open_pos_count += 1
                        avg_entry += float(p.get('avgPrice', 0.5) or 0.5)
                        pct_pnl = float(p.get('percentPnl', 0) or 0)
                        if pct_pnl > 0:
                            winning_pos += 1
                        elif pct_pnl < 0:
                            losing_pos += 1

                avg_entry = avg_entry / open_pos_count if open_pos_count > 0 else 0.5

                # Estimate win rate from positions in profit vs loss
                total_positions = winning_pos + losing_pos
                win_rate = (winning_pos / total_positions * 100) if total_positions > 0 else 0

                # Also estimate from profit/volume ratio
                if total_volume > 0:
                    roi = (total_profit / total_volume) * 100
                    # Implied win rate from ROI (rough estimate)
                    # If avg entry is 0.5, winning = 100% return, losing = -100%
                    # ROI = win_rate * 100% - (1-win_rate) * 100% = 2*win_rate - 1
                    # win_rate = (ROI + 100) / 200
                    implied_wr = min(95, max(5, (roi + 100) / 2))
                else:
                    roi = 0
                    implied_wr = 50

                results.append({
                    'name': name,
                    'address': address[:10] + '...',
                    'total_profit': total_profit,
                    'total_volume': total_volume,
                    'roi': roi,
                    'markets_traded': markets_traded,
                    'total_value': total_value,
                    'realized_pnl': realized_pnl,
                    'winning_pos': winning_pos,
                    'losing_pos': losing_pos,
                    'win_rate': win_rate if win_rate > 0 else implied_wr,
                    'avg_entry': avg_entry,
                    'open_positions': open_pos_count,
                })

            except Exception as e:
                print(f'Error fetching {name}: {e}')

    return results


async def main():
    print("\nFetching whale data from Polymarket API...")
    results = await fetch_whale_data()

    # Sort by total profit
    results.sort(key=lambda x: x['total_profit'], reverse=True)

    print()
    print('=' * 100)
    print('TOP WHALE PERFORMANCE - ALL TIME (sorted by total profit)')
    print('=' * 100)
    print()
    print(f"{'Rank':<5} {'Whale':<24} {'Total Profit':>14} {'Volume':>14} {'ROI':>8} {'Markets':>8}")
    print('-' * 100)

    for i, w in enumerate(results[:15], 1):
        print(f"{i:<5} {w['name']:<24} ${w['total_profit']:>12,.0f} ${w['total_volume']:>12,.0f} {w['roi']:>7.1f}% {w['markets_traded']:>8}")

    print()
    print('=' * 100)
    print('CURRENT POSITION ANALYSIS')
    print('=' * 100)
    print()
    print(f"{'Whale':<24} {'Positions':>10} {'Win/Lose':>10} {'Win Rate':>10} {'Avg Entry':>10} {'Value':>12}")
    print('-' * 100)

    for w in results[:15]:
        wl = f"{w['winning_pos']}/{w['losing_pos']}"
        print(f"{w['name']:<24} {w['open_positions']:>10} {wl:>10} {w['win_rate']:>9.0f}% {w['avg_entry']:>10.2f} ${w['total_value']:>10,.0f}")

    print()
    print('=' * 100)
    print('EXPECTED COPY TRADE RETURNS ON $500')
    print('=' * 100)
    print()
    print(f"{'Whale':<24} {'Win Rate':>10} {'Avg Entry':>10} {'Profit/Win':>12} {'Expected':>12} {'ROI':>10}")
    print('-' * 100)

    for w in results[:15]:
        if w['avg_entry'] > 0.01:
            # If win: $500 / entry_price - $500
            profit_if_win = (500 / w['avg_entry']) - 500
            # Expected = win_rate * profit - loss_rate * 500
            wr = w['win_rate'] / 100
            expected = wr * profit_if_win - (1 - wr) * 500
            expected_roi = (expected / 500) * 100
            print(f"{w['name']:<24} {w['win_rate']:>9.0f}% {w['avg_entry']:>10.2f} ${profit_if_win:>10,.0f} ${expected:>10,.0f} {expected_roi:>9.0f}%")
        else:
            print(f"{w['name']:<24} {w['win_rate']:>9.0f}% {'N/A':>10} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

    print()
    print("Legend:")
    print("- Total Profit: All-time profit from Polymarket profile")
    print("- ROI: Total Profit / Total Volume (all-time)")
    print("- Win Rate: From current positions in profit vs loss (or estimated from ROI)")
    print("- Avg Entry: Average entry price across open positions")
    print("- Profit/Win: If $500 position wins at avg entry (pays $1 per share)")
    print("- Expected: (WinRate * Profit) - (LossRate * $500)")
    print()
    print("TOP RECOMMENDATIONS for $500 copy trading:")
    print("-" * 60)

    # Sort by expected return
    ranked = sorted(results, key=lambda x: (x['win_rate']/100 * ((500/x['avg_entry'])-500) - (1-x['win_rate']/100)*500) if x['avg_entry'] > 0.01 else -9999, reverse=True)

    for i, w in enumerate(ranked[:5], 1):
        if w['avg_entry'] > 0.01:
            profit_if_win = (500 / w['avg_entry']) - 500
            wr = w['win_rate'] / 100
            expected = wr * profit_if_win - (1 - wr) * 500
            print(f"{i}. {w['name']}: Expected ${expected:,.0f} return ({expected/5:.0f}% ROI)")
            print(f"   Win Rate: {w['win_rate']:.0f}% | Avg Entry: {w['avg_entry']:.2f} | All-time Profit: ${w['total_profit']:,.0f}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
