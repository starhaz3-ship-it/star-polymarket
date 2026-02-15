import httpx, json

proxy = "0x4f3456c5b05b14D8aFc40E0299d6d68DE0fF7d7e"

# Check positions via gamma
r = httpx.get("https://gamma-api.polymarket.com/positions",
              params={"user": proxy, "limit": 50, "sortBy": "currentValue", "sortDir": "desc"},
              headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
if r.status_code == 200:
    positions = r.json()
    total_value = sum(float(p.get("currentValue", 0)) for p in positions)
    print(f"Active positions: {len(positions)}, total value: ${total_value:.2f}")
    for p in positions[:10]:
        title = p.get("title", "")[:50]
        val = float(p.get("currentValue", 0))
        size = p.get("size", 0)
        print(f"  {title} | value=${val:.2f} | size={size}")
else:
    print(f"Positions API: {r.status_code}")

# Check activity for recent trades
r2 = httpx.get("https://gamma-api.polymarket.com/activity",
               params={"address": proxy, "limit": 10},
               headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
if r2.status_code == 200:
    activity = r2.json()
    print(f"\nRecent activity: {len(activity)} entries")
    for a in activity[:5]:
        print(f"  {a.get('type','')} | {a.get('title','')[:40]} | ${float(a.get('usdcSize',0)):.2f}")

# Check maker results
try:
    with open("maker_results.json") as f:
        data = json.load(f)
        stats = data.get("stats", {})
        print(f"\nMaker stats: total_pnl=${stats.get('total_pnl',0):.2f}")
        print(f"Volume: ${stats.get('total_volume',0):.2f}")
        print(f"Pairs: {stats.get('pairs_completed',0)} complete, {stats.get('pairs_partial',0)} partial")
except Exception as e:
    print(f"Maker results error: {e}")

# Check USDC via Polygon RPC
try:
    r3 = httpx.post("https://polygon-bor.publicnode.com", json={
        "jsonrpc": "2.0", "method": "eth_call",
        "params": [{
            "to": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
            "data": "0x70a08231000000000000000000000000" + proxy[2:].lower()
        }, "latest"],
        "id": 1
    }, timeout=15)
    result = r3.json().get("result", "0x0")
    usdc_bal = int(result, 16) / 1e6
    print(f"\nUSDC on Polygon (wallet): ${usdc_bal:.2f}")
except Exception as e:
    print(f"RPC error: {e}")
