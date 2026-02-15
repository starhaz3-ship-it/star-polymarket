import csv
from datetime import datetime, timezone
from collections import defaultdict

path = r'C:\Users\Star\Downloads\Polymarket-History-2026-02-14 (8).csv'
with open(path, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

# Reverse to chronological order (CSV is newest first)
rows.reverse()

# Group by market
markets = defaultdict(list)
for r in rows:
    mkt = r.get('marketName', '')
    if mkt:
        markets[mkt].append(r)

print("=" * 90)
print("FILL ANALYSIS: Why did partials happen?")
print("=" * 90)

# For each market, check if we got both sides or only one
paired = []
partial_one_side = []
partial_no_redeem = []

for mkt, txns in markets.items():
    buys = [t for t in txns if t['action'] == 'Buy']
    sells = [t for t in txns if t['action'] == 'Sell']
    redeems = [t for t in txns if t['action'] == 'Redeem']

    if not buys:
        continue

    buy_up = [t for t in buys if t['tokenName'] == 'Up']
    buy_down = [t for t in buys if t['tokenName'] == 'Down']

    total_buy = sum(float(t['usdcAmount']) for t in buys)
    total_sell = sum(float(t['usdcAmount']) for t in sells)
    total_redeem = sum(float(t['usdcAmount']) for t in redeems)
    pnl = total_sell + total_redeem - total_buy

    up_shares = sum(float(t['tokenAmount']) for t in buy_up)
    down_shares = sum(float(t['tokenAmount']) for t in buy_down)
    up_cost = sum(float(t['usdcAmount']) for t in buy_up)
    down_cost = sum(float(t['usdcAmount']) for t in buy_down)

    has_up = up_shares > 0
    has_down = down_shares > 0

    # Determine fill status
    if has_up and has_down and (total_redeem > 0 or total_sell > 0):
        status = "PAIRED"
        paired.append((mkt, pnl, up_shares, down_shares, up_cost, down_cost, total_sell, total_redeem, sells))
    elif has_up and not has_down:
        status = "PARTIAL-UP-ONLY"
        partial_one_side.append((mkt, pnl, 'UP', up_shares, up_cost, total_sell, total_redeem, buys, sells))
    elif has_down and not has_up:
        status = "PARTIAL-DOWN-ONLY"
        partial_one_side.append((mkt, pnl, 'DOWN', down_shares, down_cost, total_sell, total_redeem, buys, sells))
    elif has_up and has_down and total_redeem == 0 and total_sell == 0:
        status = "BOTH-BOUGHT-NO-RESOLVE"
        partial_no_redeem.append((mkt, pnl, up_shares, down_shares, up_cost, down_cost))

print(f"\nPAIRED (both sides filled): {len(paired)}")
print(f"PARTIAL (one side only):    {len(partial_one_side)}")
print(f"BOUGHT BUT NO RESOLVE:      {len(partial_no_redeem)}")

# Analyze partials
print(f"\n{'='*90}")
print("PARTIAL FILLS - ONE SIDE ONLY")
print(f"{'='*90}")
up_only = sum(1 for p in partial_one_side if p[2] == 'UP')
down_only = sum(1 for p in partial_one_side if p[2] == 'DOWN')
print(f"UP-only: {up_only} | DOWN-only: {down_only}")

total_partial_loss = 0
for mkt, pnl, side, shares, cost, sell_rev, redeem_rev, buys, sells in partial_one_side:
    total_partial_loss += pnl
    sell_info = ""
    if sells:
        for s in sells:
            sp = float(s['usdcAmount']) / float(s['tokenAmount']) if float(s['tokenAmount']) > 0 else 0
            sell_info += f" SOLD {float(s['tokenAmount']):.0f}@${sp:.2f}"
    redeem_info = f" REDEEM=${redeem_rev:.2f}" if redeem_rev > 0 else ""

    # Get buy timestamps
    buy_times = []
    for b in buys:
        ts = int(b['timestamp'])
        dt = datetime.utcfromtimestamp(ts)
        buy_times.append(dt.strftime("%H:%M:%S"))

    print(f"  ${pnl:+7.2f} | {side:4s} {shares:5.0f}sh ${cost:6.2f} |{sell_info}{redeem_info} | {mkt[:55]}")

print(f"\n  TOTAL PARTIAL LOSS: ${total_partial_loss:+.2f}")

# Analyze paired trades
print(f"\n{'='*90}")
print("PAIRED TRADES")
print(f"{'='*90}")
total_paired_pnl = 0
for mkt, pnl, up_sh, dn_sh, up_cost, dn_cost, sell_rev, redeem_rev, sells in paired:
    total_paired_pnl += pnl
    combined = (up_cost/up_sh + dn_cost/dn_sh) if up_sh > 0 and dn_sh > 0 else 0
    sell_info = ""
    if sells:
        for s in sells:
            sp = float(s['usdcAmount']) / float(s['tokenAmount']) if float(s['tokenAmount']) > 0 else 0
            sell_info += f" SOLD@${sp:.2f}"
    print(f"  ${pnl:+7.2f} | UP {up_sh:3.0f}sh + DN {dn_sh:3.0f}sh | combined=${combined:.2f} |{sell_info} | {mkt[:45]}")

print(f"\n  TOTAL PAIRED PnL: ${total_paired_pnl:+.2f}")

# No resolve
if partial_no_redeem:
    print(f"\n{'='*90}")
    print("BOUGHT BOTH SIDES BUT NO RESOLVE (still pending or lost)")
    print(f"{'='*90}")
    for mkt, pnl, up_sh, dn_sh, up_cost, dn_cost in partial_no_redeem:
        combined = (up_cost/up_sh + dn_cost/dn_sh) if up_sh > 0 and dn_sh > 0 else 0
        print(f"  ${pnl:+7.2f} | UP {up_sh:3.0f}sh + DN {dn_sh:3.0f}sh | combined=${combined:.2f} | {mkt[:55]}")

# Time analysis - when do partials happen?
print(f"\n{'='*90}")
print("TIME ANALYSIS - When do partials vs pairs happen?")
print(f"{'='*90}")
hour_stats = defaultdict(lambda: {'paired': 0, 'partial': 0})
for mkt, pnl, up_sh, dn_sh, up_cost, dn_cost, sell_rev, redeem_rev, sells in paired:
    # Extract hour from market name
    for part in mkt.split(','):
        if 'AM' in part or 'PM' in part:
            time_str = part.strip().split('-')[0].strip()
            try:
                hour = int(time_str.split(':')[0])
                if 'PM' in part and hour != 12:
                    hour += 12
                elif 'AM' in part and hour == 12:
                    hour = 0
                hour_stats[hour]['paired'] += 1
            except:
                pass

for mkt, pnl, side, shares, cost, sell_rev, redeem_rev, buys, sells in partial_one_side:
    for part in mkt.split(','):
        if 'AM' in part or 'PM' in part:
            time_str = part.strip().split('-')[0].strip()
            try:
                hour = int(time_str.split(':')[0])
                if 'PM' in part and hour != 12:
                    hour += 12
                elif 'AM' in part and hour == 12:
                    hour = 0
                hour_stats[hour]['partial'] += 1
            except:
                pass

for hour in sorted(hour_stats.keys()):
    s = hour_stats[hour]
    total = s['paired'] + s['partial']
    pct = s['partial'] / total * 100 if total > 0 else 0
    bar = '#' * s['partial'] + '.' * s['paired']
    print(f"  {hour:2d}:00 ET | Paired: {s['paired']:3d} | Partial: {s['partial']:3d} | Partial%: {pct:5.1f}% | {bar}")

# Buy price analysis for partials vs paired
print(f"\n{'='*90}")
print("BID PRICE ANALYSIS")
print(f"{'='*90}")
paired_prices = []
for mkt, pnl, up_sh, dn_sh, up_cost, dn_cost, sell_rev, redeem_rev, sells in paired:
    if up_sh > 0:
        paired_prices.append(up_cost / up_sh)
    if dn_sh > 0:
        paired_prices.append(dn_cost / dn_sh)

partial_prices = []
for mkt, pnl, side, shares, cost, sell_rev, redeem_rev, buys, sells in partial_one_side:
    if shares > 0:
        partial_prices.append(cost / shares)

if paired_prices:
    print(f"  Paired avg buy price:  ${sum(paired_prices)/len(paired_prices):.3f} (range ${min(paired_prices):.2f} - ${max(paired_prices):.2f})")
if partial_prices:
    print(f"  Partial avg buy price: ${sum(partial_prices)/len(partial_prices):.3f} (range ${min(partial_prices):.2f} - ${max(partial_prices):.2f})")

# Multiple buys on same market (double-sizing issue)
print(f"\n{'='*90}")
print("MARKETS WITH MULTIPLE BUY BATCHES (possible double-sizing)")
print(f"{'='*90}")
for mkt, txns in markets.items():
    buys = [t for t in txns if t['action'] == 'Buy']
    if len(buys) > 4:  # More than 2 orders (1 UP + 1 DOWN = 2, so >4 means multiple batches)
        total = sum(float(t['usdcAmount']) for t in buys)
        up_buys = [t for t in buys if t['tokenName'] == 'Up']
        dn_buys = [t for t in buys if t['tokenName'] == 'Down']
        print(f"  {len(buys)} buys (${total:.2f}) | {len(up_buys)} UP + {len(dn_buys)} DN | {mkt[:55]}")
