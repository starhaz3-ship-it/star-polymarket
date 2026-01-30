"""
PAPER TRADING copy trader for k9Q2mX4L8A7ZP3R
Tracks performance without spending real money
"""

import asyncio
import os
import sys
import time
import json
from datetime import datetime
import httpx

sys.path.insert(0, '.')

TARGET = '0xd0d6053c3c37e727402d84c14069780d360993aa'
TARGET_NAME = 'k9Q2mX4L8A7ZP3R'
MAX_COPY = 100.0
MIN_COPY = 5.0
SCALE = 0.05
POLL_SEC = 15
STATE_FILE = 'paper_k9Q2_state.json'

seen = set()
paper_trades = []
paper_balance = 1000.0
starting_balance = 1000.0

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_state():
    global seen, paper_trades, paper_balance
    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
            seen = set(data.get('seen', []))
            paper_trades = data.get('paper_trades', [])
            paper_balance = data.get('paper_balance', 1000.0)
            log(f"Loaded state: {len(paper_trades)} trades, ${paper_balance:.2f} balance")
    except FileNotFoundError:
        pass

def save_state():
    with open(STATE_FILE, 'w') as f:
        json.dump({
            'seen': list(seen)[-1000:],
            'paper_trades': paper_trades[-100:],
            'paper_balance': paper_balance,
            'starting_balance': starting_balance,
            'bot_name': TARGET_NAME,
        }, f, indent=2)

async def main():
    global paper_balance, paper_trades

    log(f"PAPER TRADING - {TARGET_NAME}")
    log(f"Simulated: Max ${MAX_COPY} | Scale {SCALE*100:.0f}%")

    load_state()
    last_report = time.time()

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get("https://data-api.polymarket.com/activity",
                            params={"user": TARGET, "limit": 30})
        for t in r.json():
            seen.add(t.get("transactionHash", ""))
        log(f"Tracking {len(seen)} existing trades")

        while True:
            try:
                r = await client.get("https://data-api.polymarket.com/activity",
                                    params={"user": TARGET, "limit": 10})

                for trade in r.json():
                    tx = trade.get("transactionHash", "")
                    if tx and tx not in seen:
                        seen.add(tx)

                        title = trade.get("title", "")
                        outcome = trade.get("outcome", "")
                        price = float(trade.get("price", 0) or 0)
                        size = float(trade.get("size", 0) or 0)
                        side = trade.get("side", "")
                        token = trade.get("asset", "")
                        slug = trade.get("slug", "")

                        if side == "BUY" and price > 0:
                            their_cost = price * size
                            our_cost = min(max(their_cost * SCALE, MIN_COPY), MAX_COPY)
                            our_size = our_cost / price

                            log("=" * 50)
                            log(f"[PAPER] NEW TRADE: {outcome}")
                            log(f"Market: {title[:50]}")
                            log(f"{TARGET_NAME}: ${their_cost:.2f}")
                            log(f"Paper copy: ${our_cost:.2f} @ ${price:.4f}")
                            log("=" * 50)

                            paper_trade = {
                                'timestamp': time.time(),
                                'market': title[:50],
                                'slug': slug,
                                'outcome': outcome,
                                'entry_price': price,
                                'size': our_size,
                                'cost': our_cost,
                                'token': token,
                                'status': 'OPEN',
                                'pnl': None,
                            }
                            paper_trades.append(paper_trade)
                            paper_balance -= our_cost
                            save_state()

                # Check for resolved markets
                for trade in paper_trades:
                    if trade.get('status') != 'OPEN':
                        continue

                    try:
                        r = await client.get("https://data-api.polymarket.com/positions",
                                            params={"user": TARGET, "sizeThreshold": 0})
                        positions = r.json()

                        for pos in positions:
                            if pos.get('asset') == trade.get('token'):
                                cur_price = float(pos.get('curPrice', 0.5))
                                if cur_price <= 0.01:
                                    trade['status'] = 'CLOSED'
                                    trade['exit_price'] = 0
                                    trade['pnl'] = -trade['cost']
                                    log(f"[PAPER] LOSS: {trade['market'][:30]} -${trade['cost']:.2f}")
                                    save_state()
                                elif cur_price >= 0.99:
                                    trade['status'] = 'CLOSED'
                                    trade['exit_price'] = 1.0
                                    trade['pnl'] = trade['size'] - trade['cost']
                                    paper_balance += trade['size']
                                    log(f"[PAPER] WIN: {trade['market'][:30]} +${trade['pnl']:.2f}")
                                    save_state()
                                break
                    except:
                        pass

                # Status every 5 minutes
                if int(time.time()) % 300 < POLL_SEC:
                    closed = [t for t in paper_trades if t.get('status') == 'CLOSED']
                    open_t = [t for t in paper_trades if t.get('status') == 'OPEN']
                    log(f"Status: {len(closed)} closed, {len(open_t)} open, ${paper_balance:.2f} balance")

                await asyncio.sleep(POLL_SEC)

            except Exception as e:
                log(f"Error: {e}")
                await asyncio.sleep(30)

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"PAPER TRADING - {TARGET_NAME}")
    print(f"Wallet: {TARGET[:10]}...")
    print(f"Simulated: Max ${MAX_COPY} | Scale {SCALE*100:.0f}%")
    print(f"{'='*60}\n")
    asyncio.run(main())
