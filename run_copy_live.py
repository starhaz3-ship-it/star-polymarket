import asyncio
import os
import sys
import time
from datetime import datetime
import httpx

os.environ['POLYMARKET_PASSWORD'] = '49pIMj@*3aDwzlbqN2MqDMaX'
sys.path.insert(0, '.')

TARGET = '0x63ce342161250d705dc0b16df89036c8e5f9ba9a'
MAX_COPY = 200.0
MIN_COPY = 5.0
SCALE = 0.10
POLL_SEC = 10

seen = set()

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

async def main():
    log("LIVE COPY TRADER - 0x8dxd")
    log(f"Max: ${MAX_COPY} | Min: ${MIN_COPY} | Scale: {SCALE*100:.0f}%")
    log("Initializing executor...")
    
    from arbitrage.executor import Executor
    executor = Executor()
    
    if not executor.client:
        log("ERROR: Executor not ready")
        return
    
    log("Executor READY - monitoring for trades...")
    
    async with httpx.AsyncClient(timeout=30) as client:
        # Load existing trades
        r = await client.get("https://data-api.polymarket.com/activity",
                            params={"user": TARGET, "limit": 30})
        for t in r.json():
            seen.add(t.get("transactionHash", ""))
        log(f"Loaded {len(seen)} existing trades")
        
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
                        
                        if side == "BUY" and price > 0:
                            their_cost = price * size
                            our_cost = min(max(their_cost * SCALE, MIN_COPY), MAX_COPY)
                            our_size = our_cost / price
                            
                            log("=" * 50)
                            log("NEW TRADE DETECTED!")
                            log(f"Market: {title[:50]}")
                            log(f"Outcome: {outcome}")
                            log(f"0x8dxd: ${their_cost:.2f}")
                            log(f"Our copy: ${our_cost:.2f}")
                            log("EXECUTING...")
                            
                            try:
                                from py_clob_client.order_builder.constants import BUY as BUY_SIDE
                                order = executor.client.create_order(
                                    token_id=token,
                                    price=price,
                                    size=our_size,
                                    side=BUY_SIDE,
                                )
                                result = executor.client.post_order(order)
                                log(f"ORDER PLACED: {result}")
                            except Exception as e:
                                log(f"ORDER ERROR: {e}")
                            log("=" * 50)
                
                await asyncio.sleep(POLL_SEC)
                
            except Exception as e:
                log(f"Error: {e}")
                await asyncio.sleep(30)

asyncio.run(main())
