"""Check actual Polymarket account balance - cash + positions."""
import httpx
import json
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dotenv import load_dotenv
load_dotenv()

proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "0x4f3456c5b05b14D8aFc40E0299d6d68DE0fF7d7e")
print(f"Wallet: {proxy}")

client = httpx.Client(timeout=15, headers={'User-Agent': 'Mozilla/5.0'})

# === USDC BALANCE ===
print('\n=== CASH (USDC) ===')
cash = 0
for label, addr in [
    ('USDC.e', '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174'),
    ('USDC', '0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359'),
]:
    try:
        r = client.get('https://api.polygonscan.com/api', params={
            'module': 'account', 'action': 'tokenbalance',
            'contractaddress': addr, 'address': proxy, 'tag': 'latest',
        })
        if r.status_code == 200:
            raw = r.json().get('result', '0')
            bal = int(raw) / 1e6
            cash += bal
            if bal > 0:
                print(f'  {label}: ${bal:.2f}')
    except:
        pass

if cash == 0:
    print('  (Polygonscan rate-limited - check UI for cash)')

# === POSITIONS ===
print('\n=== OPEN POSITIONS ===')
r = client.get('https://data-api.polymarket.com/positions',
               params={'user': proxy, 'sizeThreshold': 0})
total_value = 0
total_cost = 0
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
            title = (p.get('title') or p.get('slug', '?'))[:48]
            outcome = p.get('outcome', '?')
            pnl = val - cost
            print(f'  ${val:>7.2f} | {outcome:>4} | {title}')

print(f'\n=== PORTFOLIO SUMMARY ===')
print(f'  Cash (USDC):    ${cash:.2f}' + (' (check UI)' if cash == 0 else ''))
print(f'  Positions:      ${total_value:.2f}')
print(f'  Cost basis:     ${total_cost:.2f}')
print(f'  Unrealized PnL: ${total_value - total_cost:+.2f}')
print(f'  TOTAL:          ${cash + total_value:.2f}' + (' + cash' if cash == 0 else ''))
