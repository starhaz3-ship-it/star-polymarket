import httpx
from datetime import datetime, timezone

r = httpx.get("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=24", timeout=10)
klines = r.json()
print("BTC today (hourly):")
for k in klines:
    ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
    o, c = float(k[1]), float(k[4])
    change = (c - o) / o * 100
    d = "UP" if c > o else "DN"
    print(f"  {ts.strftime('%H:%M')} UTC | ${c:>10,.2f} | {change:>+6.2f}% {d}")

r2 = httpx.get("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=30", timeout=10)
k15 = r2.json()
up = sum(1 for k in k15 if float(k[4]) > float(k[1]))
dn = len(k15) - up
print(f"\nLast 30 15m candles: {up} UP, {dn} DOWN ({100*dn/len(k15):.0f}% DOWN)")
