"""Check actual Polymarket account balance - cash + positions + trade history."""
import httpx
import json
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "0x4f3456c5b05b14D8aFc40E0299d6d68DE0fF7d7e")
print(f"Wallet: {proxy}")

client = httpx.Client(timeout=15, headers={'User-Agent': 'Mozilla/5.0'})

# === POSITIONS (via Polymarket Data API - no key needed) ===
print('\n=== OPEN POSITIONS ===')
total_value = 0
total_cost = 0
positions = []
try:
    r = client.get('https://data-api.polymarket.com/positions',
                   params={'user': proxy, 'sizeThreshold': 0})
    if r.status_code == 200:
        positions = r.json()
        for p in sorted(positions, key=lambda x: -float(x.get('currentValue', 0))):
            val = float(p.get('currentValue', 0))
            size = float(p.get('size', 0))
            avg = float(p.get('avgPrice', 0))
            cost = size * avg
            if val > 0.01:
                total_value += val
                total_cost += cost
                title = (p.get('title') or p.get('slug', '?'))[:45]
                outcome = p.get('outcome', '?')
                pnl = val - cost
                pnl_pct = (pnl / cost * 100) if cost > 0 else 0
                print(f'  ${val:>7.2f} | {outcome:>4} | {pnl:>+6.2f} ({pnl_pct:>+5.1f}%) | {title}')
except Exception as e:
    print(f'  Error fetching positions: {e}')

# === RECENT ACTIVITY (trades) ===
print('\n=== RECENT TRADES (last 20) ===')
try:
    r = client.get('https://data-api.polymarket.com/activity',
                   params={'user': proxy, 'limit': 30})
    if r.status_code == 200:
        activities = r.json()
        trade_count = 0
        for a in activities:
            action = a.get('type', '')
            if action not in ['BUY', 'SELL', 'REDEEM']:
                continue
            trade_count += 1
            if trade_count > 20:
                break

            title = (a.get('title') or a.get('slug', '?'))[:40]
            outcome = a.get('outcome', '') or a.get('side', '')
            price = float(a.get('price', 0) or 0)
            size = float(a.get('size', 0) or 0)
            usdc_size = float(a.get('usdcSize', 0) or 0)
            timestamp = a.get('timestamp', 0)

            # Parse Unix timestamp
            try:
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                time_str = dt.strftime('%H:%M')
            except:
                time_str = '?'

            if action == 'REDEEM':
                print(f'  {time_str} REDEEM ${usdc_size:>6.2f} | {title}')
            else:
                value = usdc_size if usdc_size > 0 else (price * size)
                print(f'  {time_str} {action:>4} ${value:>6.2f} @ {price:.2f} | {outcome:>4} | {title}')
except Exception as e:
    print(f'  Error fetching activity: {e}')

# === PROFIT DATA (if available) ===
print('\n=== PROFIT STATS ===')
try:
    r = client.get('https://data-api.polymarket.com/profit',
                   params={'user': proxy})
    if r.status_code == 200:
        profit_data = r.json()
        if isinstance(profit_data, dict):
            total_profit = profit_data.get('totalProfit', 0)
            print(f'  Total Profit (API): ${float(total_profit):.2f}')
        elif isinstance(profit_data, list) and profit_data:
            # Sum up profits
            total = sum(float(p.get('profit', 0)) for p in profit_data)
            print(f'  Total Profit (API): ${total:.2f}')
except Exception as e:
    print(f'  (Profit API unavailable: {e})')

# === PORTFOLIO SUMMARY ===
print(f'\n=== PORTFOLIO SUMMARY ===')
print(f'  Open Positions:   ${total_value:.2f}')
print(f'  Cost Basis:       ${total_cost:.2f}')
print(f'  Unrealized PnL:   ${total_value - total_cost:+.2f}')
print(f'  Position Count:   {len([p for p in positions if float(p.get("currentValue", 0)) > 0.01])}')

# === LIVE TRADER STATS ===
print(f'\n=== LIVE TRADER STATS ===')
try:
    with open('ta_live_results.json', 'r') as f:
        data = json.load(f)
    w = data.get('wins', 0)
    l = data.get('losses', 0)
    pnl = data.get('total_pnl', 0)
    bankroll = data.get('bankroll', 0)
    streak_w = data.get('consecutive_wins', 0)
    streak_l = data.get('consecutive_losses', 0)
    wr = w / (w + l) * 100 if (w + l) > 0 else 0
    print(f'  Win/Loss:       {w}/{l} ({wr:.1f}% WR)')
    print(f'  Tracked PnL:    ${pnl:+.2f}')
    print(f'  Bankroll:       ${bankroll:.2f}')
    print(f'  Streak:         {streak_w}W / {streak_l}L')
except Exception as e:
    print(f'  (No live trader data: {e})')
