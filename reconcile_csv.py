#!/usr/bin/env python3
"""
Polymarket On-Chain CSV vs Internal Trade Records Reconciliation
================================================================
Compares Polymarket-History CSV (on-chain truth) against internal JSON result files.
All times displayed in MST (Mountain Standard Time = UTC-7).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import csv
import json
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

# ============================================================
# CONSTANTS
# ============================================================
MST = timezone(timedelta(hours=-7))
BASELINE_TS = 1771308000  # V5.2 deploy Feb 17 ~12:00AM ET
STARTING_BALANCE = 118.89

CSV_PATH = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-19 (1).csv"
MOMENTUM_PATH = r"C:\Users\Star\.local\bin\star-polymarket\momentum_15m_results.json"
SNIPER_LIVE_PATH = r"C:\Users\Star\.local\bin\star-polymarket\sniper_5m_live_results.json"
SNIPER_ARCHIVE_PATH = r"C:\Users\Star\.local\bin\star-polymarket\archive\sniper_5m_live_results_pre_v1.1.json"
TA_LIVE_PATH = r"C:\Users\Star\.local\bin\star-polymarket\ta_live_results.json"
MAKER_PATH = r"C:\Users\Star\.local\bin\star-polymarket\maker_results.json"
PAIRS_ARB_PATH = r"C:\Users\Star\.local\bin\star-polymarket\pairs_arb_results.json"
PAIRS_ARB_5M_PATH = r"C:\Users\Star\.local\bin\star-polymarket\pairs_arb_5m_results.json"
CROSS_ARB_PATH = r"C:\Users\Star\.local\bin\star-polymarket\cross_arb_results.json"

def ts_to_mst(ts):
    """Convert unix timestamp to MST datetime string."""
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(MST)
    return dt.strftime("%Y-%m-%d %I:%M:%S %p MST")

def ts_to_date(ts):
    """Convert unix timestamp to MST date string."""
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(MST)
    return dt.strftime("%Y-%m-%d")

def iso_to_ts(iso_str):
    """Convert ISO datetime string to unix timestamp."""
    if not iso_str:
        return 0
    dt = datetime.fromisoformat(iso_str)
    return dt.timestamp()

# ============================================================
# 1. PARSE CSV
# ============================================================
print("=" * 80)
print("POLYMARKET ON-CHAIN CSV vs INTERNAL RECORDS RECONCILIATION")
print("=" * 80)
print(f"CSV: {CSV_PATH}")
print(f"Baseline timestamp: {BASELINE_TS} ({ts_to_mst(BASELINE_TS)})")
print(f"Starting balance: ${STARTING_BALANCE:.2f}")
print()

rows = []
with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['timestamp'] = int(row['timestamp'])
        row['usdcAmount'] = float(row['usdcAmount'])
        row['tokenAmount'] = float(row['tokenAmount'])
        rows.append(row)

print(f"Total CSV rows: {len(rows)}")
print(f"Date range: {ts_to_mst(min(r['timestamp'] for r in rows))} to {ts_to_mst(max(r['timestamp'] for r in rows))}")
print()

# ============================================================
# 1a. FULL CSV summary by action type
# ============================================================
print("-" * 60)
print("FULL CSV SUMMARY (ALL TIME)")
print("-" * 60)
action_counts = Counter(r['action'] for r in rows)
for action, count in sorted(action_counts.items()):
    total = sum(r['usdcAmount'] for r in rows if r['action'] == action)
    print(f"  {action:10s}: {count:5d} rows, ${total:10.2f} USDC")

# Group by day
print("\n  Trades per day:")
day_counts = defaultdict(lambda: defaultdict(int))
for r in rows:
    day = ts_to_date(r['timestamp'])
    day_counts[day][r['action']] += 1
for day in sorted(day_counts):
    parts = ", ".join(f"{a}:{c}" for a, c in sorted(day_counts[day].items()))
    total_day = sum(day_counts[day].values())
    print(f"    {day}: {total_day:4d} ({parts})")

# ============================================================
# 1b. POST-BASELINE CSV analysis
# ============================================================
post = [r for r in rows if r['timestamp'] > BASELINE_TS]
pre = [r for r in rows if r['timestamp'] <= BASELINE_TS]

print()
print("-" * 60)
print(f"POST-BASELINE CSV (after {ts_to_mst(BASELINE_TS)})")
print("-" * 60)
print(f"Post-baseline rows: {len(post)}")
print(f"Pre-baseline rows:  {len(pre)}")
print()

buys = [r for r in post if r['action'] == 'Buy']
sells = [r for r in post if r['action'] == 'Sell']
redeems = [r for r in post if r['action'] == 'Redeem']

total_buy_usdc = sum(r['usdcAmount'] for r in buys)
total_sell_usdc = sum(r['usdcAmount'] for r in sells)
total_redeem_usdc = sum(r['usdcAmount'] for r in redeems)

print(f"  Buys:    {len(buys):4d} rows, ${total_buy_usdc:10.4f} USDC spent")
print(f"  Sells:   {len(sells):4d} rows, ${total_sell_usdc:10.4f} USDC received")
print(f"  Redeems: {len(redeems):4d} rows, ${total_redeem_usdc:10.4f} USDC received")
print()

csv_net_pnl = total_redeem_usdc + total_sell_usdc - total_buy_usdc
print(f"  CSV Net PnL (post-baseline, trades only): ${csv_net_pnl:+.4f}")
print(f"  (Redeems + Sells - Buys = ${total_redeem_usdc:.4f} + ${total_sell_usdc:.4f} - ${total_buy_usdc:.4f})")
print()

# Handle non-trade rows (Deposit, Maker Rebate)
deposits = [r for r in post if r['action'] == 'Deposit']
rebates = [r for r in post if r['action'] == 'Maker Rebate']
total_deposits = sum(r['usdcAmount'] for r in deposits)
total_rebates = sum(r['usdcAmount'] for r in rebates)
if deposits:
    print(f"  Deposits: {len(deposits):4d} rows, ${total_deposits:10.4f} USDC")
if rebates:
    print(f"  Rebates:  {len(rebates):4d} rows, ${total_rebates:10.4f} USDC")

# Break down by asset
print("\n  By asset (post-baseline, Buy/Sell/Redeem only):")
asset_pnl = defaultdict(lambda: {'buys': 0, 'sells': 0, 'redeems': 0, 'buy_count': 0, 'sell_count': 0, 'redeem_count': 0})
for r in post:
    if r['action'] not in ('Buy', 'Sell', 'Redeem'):
        continue
    market = r['marketName']
    asset = 'UNKNOWN'
    if 'Bitcoin' in market:
        asset = 'BTC'
    elif 'Ethereum' in market:
        asset = 'ETH'
    elif 'Solana' in market:
        asset = 'SOL'
    elif 'XRP' in market:
        asset = 'XRP'
    else:
        asset = market[:30]

    if r['action'] == 'Buy':
        asset_pnl[asset]['buys'] += r['usdcAmount']
        asset_pnl[asset]['buy_count'] += 1
    elif r['action'] == 'Sell':
        asset_pnl[asset]['sells'] += r['usdcAmount']
        asset_pnl[asset]['sell_count'] += 1
    elif r['action'] == 'Redeem':
        asset_pnl[asset]['redeems'] += r['usdcAmount']
        asset_pnl[asset]['redeem_count'] += 1

for asset in sorted(asset_pnl):
    d = asset_pnl[asset]
    net = d['redeems'] + d['sells'] - d['buys']
    print(f"    {asset:6s}: Buy ${d['buys']:8.2f} ({d['buy_count']}), "
          f"Sell ${d['sells']:8.2f} ({d['sell_count']}), "
          f"Redeem ${d['redeems']:8.2f} ({d['redeem_count']}), "
          f"Net: ${net:+.2f}")

# ============================================================
# 1c. Zero-value redeems (losing trades)
# ============================================================
zero_redeems = [r for r in post if r['action'] == 'Redeem' and r['usdcAmount'] == 0]
nonzero_redeems = [r for r in post if r['action'] == 'Redeem' and r['usdcAmount'] > 0]
print(f"\n  Zero-value redeems (losses): {len(zero_redeems)}")
print(f"  Non-zero redeems (wins):     {len(nonzero_redeems)}, total: ${sum(r['usdcAmount'] for r in nonzero_redeems):.4f}")

# ============================================================
# 1d. Per-day post-baseline
# ============================================================
print("\n  Post-baseline by day:")
day_pnl = defaultdict(lambda: {'buys': 0, 'sells': 0, 'redeems': 0, 'buy_count': 0})
for r in post:
    day = ts_to_date(r['timestamp'])
    if r['action'] == 'Buy':
        day_pnl[day]['buys'] += r['usdcAmount']
        day_pnl[day]['buy_count'] += 1
    elif r['action'] == 'Sell':
        day_pnl[day]['sells'] += r['usdcAmount']
    elif r['action'] == 'Redeem':
        day_pnl[day]['redeems'] += r['usdcAmount']

for day in sorted(day_pnl):
    d = day_pnl[day]
    net = d['redeems'] + d['sells'] - d['buys']
    print(f"    {day}: {d['buy_count']:3d} buys (${d['buys']:.2f}), "
          f"redeems ${d['redeems']:.2f}, sells ${d['sells']:.2f}, net: ${net:+.2f}")

# ============================================================
# 2. LOAD INTERNAL TRADE RECORDS
# ============================================================
print()
print("=" * 80)
print("INTERNAL TRADE RECORDS")
print("=" * 80)

# --- Momentum 15M ---
with open(MOMENTUM_PATH, 'r') as f:
    momentum_data = json.load(f)
mom_resolved = momentum_data.get('resolved', [])
mom_active = momentum_data.get('active', {})
print(f"\n[MOMENTUM 15M] Resolved: {len(mom_resolved)}, Active: {len(mom_active)}")
mom_pnl = sum(t.get('pnl', 0) for t in mom_resolved)
mom_wins = sum(1 for t in mom_resolved if t.get('pnl', 0) > 0)
mom_losses = sum(1 for t in mom_resolved if t.get('pnl', 0) <= 0)
print(f"  PnL: ${mom_pnl:+.2f}, W/L: {mom_wins}/{mom_losses}")
for t in mom_resolved:
    etime = t.get('entry_time', '')
    ts = iso_to_ts(etime) if etime else 0
    title = t.get('title', '')
    side = t.get('side', '')
    pnl = t.get('pnl', 0)
    cost = t.get('size_usd', 0)
    print(f"  [{ts_to_mst(ts) if ts else 'N/A':>35s}] {side:5s} ${cost:.2f} -> PnL ${pnl:+.2f}  {title}")

# --- Sniper 5M Live ---
with open(SNIPER_LIVE_PATH, 'r') as f:
    sniper_data = json.load(f)
sniper_resolved = sniper_data.get('resolved', [])
sniper_live = [t for t in sniper_resolved if t.get('live', False)]
sniper_paper = [t for t in sniper_resolved if not t.get('live', False)]
print(f"\n[SNIPER 5M LIVE] Resolved: {len(sniper_resolved)} (live: {len(sniper_live)}, paper: {len(sniper_paper)})")
sniper_live_pnl = sum(t.get('pnl', 0) for t in sniper_live)
sniper_wins = sum(1 for t in sniper_live if t.get('result') == 'WIN')
sniper_losses = sum(1 for t in sniper_live if t.get('result') == 'LOSS')
print(f"  Live PnL: ${sniper_live_pnl:+.2f}, W/L: {sniper_wins}/{sniper_losses}")
for t in sniper_live:
    etime = t.get('entry_time', '')
    ts = iso_to_ts(etime) if etime else 0
    title = t.get('question', '')
    side = t.get('side', '')
    pnl = t.get('pnl', 0)
    cost = t.get('cost', 0)
    result = t.get('result', '')
    print(f"  [{ts_to_mst(ts) if ts else 'N/A':>35s}] {side:5s} ${cost:.2f} -> PnL ${pnl:+.2f} ({result})  {title}")

# --- Sniper 5M Archive (pre v1.1) ---
try:
    with open(SNIPER_ARCHIVE_PATH, 'r') as f:
        sniper_archive = json.load(f)
    sniper_arch_resolved = sniper_archive.get('resolved', [])
    sniper_arch_live = [t for t in sniper_arch_resolved if t.get('live', False)]
    print(f"\n[SNIPER 5M ARCHIVE] Resolved: {len(sniper_arch_resolved)} (live: {len(sniper_arch_live)})")
    sniper_arch_pnl = sum(t.get('pnl', 0) for t in sniper_arch_live)
    print(f"  Live PnL: ${sniper_arch_pnl:+.2f}")
    for t in sniper_arch_live:
        etime = t.get('entry_time', '')
        ts = iso_to_ts(etime) if etime else 0
        title = t.get('question', '')
        side = t.get('side', '')
        pnl = t.get('pnl', 0)
        cost = t.get('cost', 0)
        result = t.get('result', '')
        print(f"  [{ts_to_mst(ts) if ts else 'N/A':>35s}] {side:5s} ${cost:.2f} -> PnL ${pnl:+.2f} ({result})  {title}")
except Exception as e:
    sniper_arch_live = []
    print(f"\n[SNIPER 5M ARCHIVE] Could not load: {e}")

# --- TA Live ---
with open(TA_LIVE_PATH, 'r') as f:
    ta_data = json.load(f)
ta_trades = list(ta_data.get('trades', {}).values()) if isinstance(ta_data.get('trades'), dict) else ta_data.get('trades', [])
ta_live_post = [t for t in ta_trades if t.get('status') == 'closed' and iso_to_ts(t.get('entry_time', '')) > BASELINE_TS]
print(f"\n[TA LIVE] Total trades: {len(ta_trades)}, Post-baseline: {len(ta_live_post)}")
ta_pnl = sum(t.get('pnl', 0) for t in ta_trades if t.get('status') == 'closed')
ta_post_pnl = sum(t.get('pnl', 0) for t in ta_live_post)
print(f"  All-time PnL: ${ta_pnl:+.2f}")
print(f"  Post-baseline PnL: ${ta_post_pnl:+.2f}")
if ta_live_post:
    for t in ta_live_post:
        etime = t.get('entry_time', '')
        ts = iso_to_ts(etime) if etime else 0
        title = t.get('market_title', '')
        side = t.get('side', '')
        pnl = t.get('pnl', 0)
        cost = t.get('size_usd', 0)
        print(f"  [{ts_to_mst(ts) if ts else 'N/A':>35s}] {side:5s} ${cost:.2f} -> PnL ${pnl:+.2f}  {title}")

# --- Maker ---
with open(MAKER_PATH, 'r') as f:
    maker_data = json.load(f)
maker_resolved = maker_data.get('resolved', [])
maker_post = [t for t in maker_resolved if t.get('resolved_at', '') and iso_to_ts(t.get('resolved_at', '')) > BASELINE_TS]
print(f"\n[MAKER] Total resolved: {len(maker_resolved)}, Post-baseline: {len(maker_post)}")
maker_total_pnl = sum(t.get('pnl', 0) for t in maker_resolved)
maker_post_pnl = sum(t.get('pnl', 0) for t in maker_post)
print(f"  All-time PnL: ${maker_total_pnl:+.2f}")
print(f"  Post-baseline PnL: ${maker_post_pnl:+.2f}")

# --- Pairs Arb (15M) ---
with open(PAIRS_ARB_PATH, 'r') as f:
    pairs_data = json.load(f)
pairs_resolved = pairs_data.get('resolved', [])
# Check for live orders (order_id starts with 0x, not paper_)
pairs_live = []
for t in pairs_resolved:
    up_oid = t.get('up_order', {}).get('order_id', '')
    down_oid = t.get('down_order', {}).get('order_id', '')
    if up_oid.startswith('0x') or down_oid.startswith('0x'):
        pairs_live.append(t)
pairs_live_post = [t for t in pairs_live if t.get('entry_time', 0) > BASELINE_TS]
print(f"\n[PAIRS ARB 15M] Total: {len(pairs_resolved)}, Live: {len(pairs_live)}, Post-baseline live: {len(pairs_live_post)}")
pairs_live_pnl = sum(t.get('pnl', 0) for t in pairs_live)
pairs_post_pnl = sum(t.get('pnl', 0) for t in pairs_live_post)
print(f"  Live PnL: ${pairs_live_pnl:+.2f}")
print(f"  Post-baseline live PnL: ${pairs_post_pnl:+.2f}")
for t in pairs_live_post:
    etime = t.get('entry_time', 0)
    title = t.get('question', '')
    pnl = t.get('pnl', 0)
    status = t.get('status', '')
    print(f"  [{ts_to_mst(etime):>35s}] PnL ${pnl:+.2f} ({status})  {title}")

# --- Pairs Arb 5M ---
with open(PAIRS_ARB_5M_PATH, 'r') as f:
    pairs5_data = json.load(f)
pairs5_resolved = pairs5_data.get('resolved', [])
pairs5_live = []
for t in pairs5_resolved:
    up_oid = t.get('up_order', {}).get('order_id', '')
    down_oid = t.get('down_order', {}).get('order_id', '')
    if up_oid.startswith('0x') or down_oid.startswith('0x'):
        pairs5_live.append(t)
pairs5_live_post = [t for t in pairs5_live if t.get('entry_time', 0) > BASELINE_TS]
print(f"\n[PAIRS ARB 5M] Total: {len(pairs5_resolved)}, Live: {len(pairs5_live)}, Post-baseline live: {len(pairs5_live_post)}")
pairs5_live_pnl = sum(t.get('pnl', 0) for t in pairs5_live)
pairs5_post_pnl = sum(t.get('pnl', 0) for t in pairs5_live_post)
print(f"  Live PnL: ${pairs5_live_pnl:+.2f}")
print(f"  Post-baseline live PnL: ${pairs5_post_pnl:+.2f}")
for t in pairs5_live_post:
    etime = t.get('entry_time', 0)
    title = t.get('question', '')
    pnl = t.get('pnl', 0)
    status = t.get('status', '')
    print(f"  [{ts_to_mst(etime):>35s}] PnL ${pnl:+.2f} ({status})  {title}")

# --- Cross Arb ---
with open(CROSS_ARB_PATH, 'r') as f:
    cross_data = json.load(f)
cross_resolved = cross_data.get('resolved', [])
cross_paper = [t for t in cross_resolved if t.get('paper', False)]
cross_live_count = len(cross_resolved) - len(cross_paper)
print(f"\n[CROSS ARB] Total: {len(cross_resolved)}, Paper: {len(cross_paper)}, Live: {cross_live_count}")

# ============================================================
# 3. MATCH CSV BUYS TO INTERNAL RECORDS
# ============================================================
print()
print("=" * 80)
print("TRADE MATCHING: CSV Buys <-> Internal Records")
print("=" * 80)

# Build lookup of all internal live trades with order_ids
# Key: order_id -> trade info
internal_by_order_id = {}

# Momentum 15M
for t in mom_resolved:
    oid = t.get('order_id', '')
    if oid:
        internal_by_order_id[oid] = {'source': 'MOMENTUM_15M', 'trade': t}

# Sniper 5M Live
for t in sniper_live:
    oid = t.get('order_id', '')
    if oid:
        internal_by_order_id[oid] = {'source': 'SNIPER_5M_LIVE', 'trade': t}

# Sniper 5M Archive
for t in sniper_arch_live:
    oid = t.get('order_id', '')
    if oid:
        internal_by_order_id[oid] = {'source': 'SNIPER_5M_ARCHIVE', 'trade': t}

# TA Live
for t in ta_trades:
    oid = t.get('order_id', '')
    if oid:
        internal_by_order_id[oid] = {'source': 'TA_LIVE', 'trade': t}

# Maker (has bid_id, ask_id etc - different structure)
# Maker trades may not have simple order_ids

# Pairs Arb
for t in pairs_live:
    up_oid = t.get('up_order', {}).get('order_id', '')
    down_oid = t.get('down_order', {}).get('order_id', '')
    if up_oid and up_oid.startswith('0x'):
        internal_by_order_id[up_oid] = {'source': 'PAIRS_ARB_15M', 'trade': t, 'leg': 'UP'}
    if down_oid and down_oid.startswith('0x'):
        internal_by_order_id[down_oid] = {'source': 'PAIRS_ARB_15M', 'trade': t, 'leg': 'DOWN'}

for t in pairs5_live:
    up_oid = t.get('up_order', {}).get('order_id', '')
    down_oid = t.get('down_order', {}).get('order_id', '')
    if up_oid and up_oid.startswith('0x'):
        internal_by_order_id[up_oid] = {'source': 'PAIRS_ARB_5M', 'trade': t, 'leg': 'UP'}
    if down_oid and down_oid.startswith('0x'):
        internal_by_order_id[down_oid] = {'source': 'PAIRS_ARB_5M', 'trade': t, 'leg': 'DOWN'}

print(f"Total internal order_ids tracked: {len(internal_by_order_id)}")

# Now try matching CSV buys by market title + timestamp proximity + amount
# Since CSV has tx hashes not order_ids, we match by:
#   - Market name (title)
#   - Side (tokenName in CSV = side in internal)
#   - Amount proximity
#   - Timestamp proximity

def normalize_title(title):
    """Normalize market title for matching."""
    return title.strip().lower().replace('[btc] ', '').replace('[eth] ', '').replace('[sol] ', '').replace('[xrp] ', '')

# Build internal trade list for matching
all_internal_live = []

for t in mom_resolved:
    etime = t.get('entry_time', '')
    ts = iso_to_ts(etime) if etime else 0
    all_internal_live.append({
        'source': 'MOMENTUM_15M',
        'title': normalize_title(t.get('title', '')),
        'side': t.get('side', '').upper(),
        'cost': t.get('size_usd', 0),
        'timestamp': ts,
        'pnl': t.get('pnl', 0),
        'order_id': t.get('order_id', ''),
        'raw': t
    })

for t in sniper_live:
    etime = t.get('entry_time', '')
    ts = iso_to_ts(etime) if etime else 0
    all_internal_live.append({
        'source': 'SNIPER_5M_LIVE',
        'title': normalize_title(t.get('question', '')),
        'side': t.get('side', '').upper(),
        'cost': t.get('cost', 0),
        'timestamp': ts,
        'pnl': t.get('pnl', 0),
        'order_id': t.get('order_id', ''),
        'raw': t
    })

for t in sniper_arch_live:
    etime = t.get('entry_time', '')
    ts = iso_to_ts(etime) if etime else 0
    # Flag phantom oracle trades (entered after market closed)
    is_phantom = t.get('time_remaining_sec', 999) < 0 or t.get('strategy') == 'oracle'
    all_internal_live.append({
        'source': 'SNIPER_5M_ARCHIVE' + (' (PHANTOM)' if is_phantom else ''),
        'title': normalize_title(t.get('question', '')),
        'side': t.get('side', '').upper(),
        'cost': t.get('cost', 0),
        'timestamp': ts,
        'pnl': t.get('pnl', 0),
        'order_id': t.get('order_id', ''),
        'phantom': is_phantom,
        'raw': t
    })

for t in ta_trades:
    etime = t.get('entry_time', '')
    ts = iso_to_ts(etime) if etime else 0
    if t.get('status') == 'closed':
        all_internal_live.append({
            'source': 'TA_LIVE',
            'title': normalize_title(t.get('market_title', '')),
            'side': t.get('side', '').upper(),
            'cost': t.get('size_usd', 0),
            'timestamp': ts,
            'pnl': t.get('pnl', 0),
            'order_id': t.get('order_id', ''),
            'raw': t
        })

# Pairs arb creates multiple legs; each buy is separate
for t in pairs_live:
    etime = t.get('entry_time', 0)
    up_oid = t.get('up_order', {}).get('order_id', '')
    down_oid = t.get('down_order', {}).get('order_id', '')
    up_cost = t.get('up_order', {}).get('cost', 0)
    down_cost = t.get('down_order', {}).get('cost', 0)
    title = normalize_title(t.get('question', ''))
    if t.get('up_filled'):
        all_internal_live.append({
            'source': 'PAIRS_ARB_15M',
            'title': title,
            'side': 'UP',
            'cost': up_cost,
            'timestamp': etime,
            'pnl': t.get('pnl', 0),
            'order_id': up_oid,
            'raw': t
        })
    if t.get('down_filled'):
        all_internal_live.append({
            'source': 'PAIRS_ARB_15M',
            'title': title,
            'side': 'DOWN',
            'cost': down_cost,
            'timestamp': etime,
            'pnl': t.get('pnl', 0),
            'order_id': down_oid,
            'raw': t
        })

for t in pairs5_live:
    etime = t.get('entry_time', 0)
    up_oid = t.get('up_order', {}).get('order_id', '')
    down_oid = t.get('down_order', {}).get('order_id', '')
    up_cost = t.get('up_order', {}).get('cost', 0)
    down_cost = t.get('down_order', {}).get('cost', 0)
    title = normalize_title(t.get('question', ''))
    if t.get('up_filled'):
        all_internal_live.append({
            'source': 'PAIRS_ARB_5M',
            'title': title,
            'side': 'UP',
            'cost': up_cost,
            'timestamp': etime,
            'pnl': t.get('pnl', 0),
            'order_id': up_oid,
            'raw': t
        })
    if t.get('down_filled'):
        all_internal_live.append({
            'source': 'PAIRS_ARB_5M',
            'title': title,
            'side': 'DOWN',
            'cost': down_cost,
            'timestamp': etime,
            'pnl': t.get('pnl', 0),
            'order_id': down_oid,
            'raw': t
        })

# Filter to post-baseline internal trades
all_internal_post = [t for t in all_internal_live if t['timestamp'] > BASELINE_TS]
print(f"Post-baseline internal live trades: {len(all_internal_post)}")

# Count by source
src_counts = Counter(t['source'] for t in all_internal_post)
for src, cnt in sorted(src_counts.items()):
    pnl = sum(t['pnl'] for t in all_internal_post if t['source'] == src)
    print(f"  {src}: {cnt} trades, PnL ${pnl:+.2f}")

# ============================================================
# 3a. Match CSV buys to internal trades
# ============================================================
print()
print("-" * 60)
print("MATCHING LOGIC: CSV Buy rows -> Internal trades")
print("-" * 60)

# For each post-baseline CSV buy, try to find a matching internal trade
matched_csv = []
unmatched_csv = []
matched_internal = set()  # track which internal trades were matched

for csv_row in buys:
    csv_title = normalize_title(csv_row['marketName'])
    csv_side = csv_row['tokenName'].upper() if csv_row['tokenName'] else ''
    csv_amount = csv_row['usdcAmount']
    csv_ts = csv_row['timestamp']

    best_match = None
    best_score = float('inf')

    for idx, itrade in enumerate(all_internal_post):
        if idx in matched_internal:
            continue

        # Title must match
        if itrade['title'] != csv_title:
            continue

        # Side must match (if CSV has tokenName)
        if csv_side and itrade['side'] and csv_side != itrade['side']:
            continue

        # Amount proximity (within $1)
        amount_diff = abs(csv_amount - itrade['cost'])
        if amount_diff > 2.0:
            continue

        # Timestamp proximity (within 5 minutes = 300s)
        ts_diff = abs(csv_ts - itrade['timestamp'])
        if ts_diff > 300:
            continue

        score = amount_diff + ts_diff / 100
        if score < best_score:
            best_score = score
            best_match = (idx, itrade)

    if best_match:
        idx, itrade = best_match
        matched_internal.add(idx)
        matched_csv.append({
            'csv': csv_row,
            'internal': itrade,
            'amount_diff': abs(csv_amount - itrade['cost']),
            'ts_diff': abs(csv_ts - itrade['timestamp'])
        })
    else:
        unmatched_csv.append(csv_row)

# Unmatched internal trades
unmatched_internal = [t for idx, t in enumerate(all_internal_post) if idx not in matched_internal]

print(f"\nMatched:            {len(matched_csv)} CSV buys -> internal trades")
print(f"Unmatched CSV buys: {len(unmatched_csv)} (on-chain but NOT in internal records)")
print(f"Unmatched internal: {len(unmatched_internal)} (in records but NOT in CSV)")

# ============================================================
# 3b. Unmatched CSV buys (leaked trades from killed bots?)
# ============================================================
if unmatched_csv:
    print()
    print("-" * 60)
    print("UNMATCHED CSV BUYS (on-chain, not in internal records)")
    print("These may be from: maker bot, killed bots, manual trades, or old bots")
    print("-" * 60)

    # Identify likely source by analyzing patterns
    # Maker: typically $3-$10 bets on multi-asset or deep bids, often pairs (Up+Down same market)
    # Pairs arb: buys both sides of same market, often $4.50-$5 each
    # Sniper: $3-$5 single direction BTC-only

    # Group unmatched by market to find paired buys
    unmatched_by_market = defaultdict(list)
    for r in unmatched_csv:
        unmatched_by_market[r['marketName']].append(r)

    unmatched_by_source = defaultdict(list)
    classified_rows = set()

    for market, market_rows in unmatched_by_market.items():
        sides = set(r['tokenName'].upper() for r in market_rows if r['tokenName'])
        has_both_sides = 'UP' in sides and 'DOWN' in sides

        for r in market_rows:
            rid = r['hash'] + str(r['timestamp'])
            if rid in classified_rows:
                continue
            classified_rows.add(rid)

            amount = r['usdcAmount']
            is_btc = 'Bitcoin' in market
            is_5m = any(x in market for x in ['AM-', 'PM-'] if ':' in market.split('-')[-1] if len(market.split('-')[-1].strip()) < 10) if '-' in market else False

            if amount < 0.1:
                source = "DUST (<$0.10)"
            elif has_both_sides and is_btc:
                # Both sides bought = pairs arb or maker
                if amount > 8:
                    source = "MAKER (deep bid, >$8)"
                else:
                    source = "PAIRS_ARB (both sides bought)"
            elif not is_btc:
                source = "MAKER (non-BTC asset)"
            elif amount > 8:
                source = "MAKER (deep bid BTC, >$8)"
            else:
                source = "UNATTRIBUTED (BTC single-side)"

            unmatched_by_source[source].append(r)

    total_unmatched_usdc = sum(r['usdcAmount'] for r in unmatched_csv)
    print(f"Total USDC in unmatched buys: ${total_unmatched_usdc:.4f}")
    print()

    for source, trades in sorted(unmatched_by_source.items()):
        total = sum(r['usdcAmount'] for r in trades)
        print(f"  {source}: {len(trades)} buys, ${total:.4f} USDC")

    print(f"\n  First 30 unmatched buys:")
    for i, r in enumerate(unmatched_csv[:30]):
        print(f"    [{ts_to_mst(r['timestamp']):>35s}] ${r['usdcAmount']:8.4f} {r['tokenName']:5s} | {r['marketName'][:60]}")
    if len(unmatched_csv) > 30:
        print(f"    ... and {len(unmatched_csv) - 30} more")

# ============================================================
# 3c. Unmatched internal trades (phantom fills?)
# ============================================================
if unmatched_internal:
    print()
    print("-" * 60)
    print("UNMATCHED INTERNAL TRADES (in records, NOT in CSV)")
    print("These may be: phantom fills, or CSV doesn't cover full history")
    print("-" * 60)

    total_phantom_cost = sum(t['cost'] for t in unmatched_internal)
    print(f"Total cost of unmatched internal: ${total_phantom_cost:.4f}")

    for t in unmatched_internal:
        ts = t['timestamp']
        print(f"    [{ts_to_mst(ts) if ts else 'N/A':>35s}] {t['source']:20s} {t['side']:5s} ${t['cost']:8.4f}  {t['title'][:55]}")

# ============================================================
# 4. RECONCILIATION SUMMARY
# ============================================================
print()
print("=" * 80)
print("RECONCILIATION SUMMARY")
print("=" * 80)

# Internal PnL (post-baseline only)
internal_total_pnl = 0
internal_real_pnl = 0
phantom_pnl = 0
print("\n  Internal PnL by bot (post-baseline live trades only):")
for src in sorted(src_counts):
    pnl = sum(t['pnl'] for t in all_internal_post if t['source'] == src)
    # For pairs arb, pnl is per-pair not per-leg, so deduplicate
    if 'PAIRS' in src:
        seen_conditions = set()
        dedup_pnl = 0
        for t in all_internal_post:
            if t['source'] == src:
                cid = t['raw'].get('condition_id', id(t['raw']))
                if cid not in seen_conditions:
                    seen_conditions.add(cid)
                    dedup_pnl += t['raw'].get('pnl', 0)
        pnl = dedup_pnl
    internal_total_pnl += pnl
    is_phantom = 'PHANTOM' in src
    if is_phantom:
        phantom_pnl += pnl
    else:
        internal_real_pnl += pnl
    flag = " *** PHANTOM - never settled on-chain" if is_phantom else ""
    print(f"    {src:30s}: ${pnl:+.2f}{flag}")

# Also include maker post-baseline that wasn't matched above
if maker_post:
    print(f"    {'MAKER':30s}: ${maker_post_pnl:+.2f} (from maker_results.json)")
    internal_total_pnl += maker_post_pnl
    internal_real_pnl += maker_post_pnl

print(f"    {'':30s}  --------")
print(f"    {'TOTAL (all)':30s}: ${internal_total_pnl:+.2f}")
print(f"    {'TOTAL (real, excl phantom)':30s}: ${internal_real_pnl:+.2f}")
print(f"    {'TOTAL (phantom only)':30s}: ${phantom_pnl:+.2f}")

print(f"\n  CSV Net PnL (post-baseline):               ${csv_net_pnl:+.4f}")
print(f"  Internal Net PnL (all, incl phantom):      ${internal_total_pnl:+.2f}")
print(f"  Internal Net PnL (real only):              ${internal_real_pnl:+.2f}")
gap_all = csv_net_pnl - internal_total_pnl
gap_real = csv_net_pnl - internal_real_pnl
print(f"  DISCREPANCY (CSV - Internal all):          ${gap_all:+.2f}")
print(f"  DISCREPANCY (CSV - Internal real):         ${gap_real:+.2f}")

print(f"\n  Unmatched CSV buys:   {len(unmatched_csv)} trades, ${sum(r['usdcAmount'] for r in unmatched_csv):.2f} USDC")
print(f"  Unmatched internals:  {len(unmatched_internal)} trades, ${sum(t['cost'] for t in unmatched_internal):.2f} USDC")

# Check for any $0 redeems that might be losses from maker bot
maker_redeems_post = [r for r in post if r['action'] == 'Redeem' and r['usdcAmount'] == 0]
print(f"\n  Zero-value redeems (100% losses): {len(maker_redeems_post)}")
if maker_redeems_post:
    # See what markets they come from
    loss_assets = Counter()
    for r in maker_redeems_post:
        market = r['marketName']
        if 'Bitcoin' in market:
            loss_assets['BTC'] += 1
        elif 'Ethereum' in market:
            loss_assets['ETH'] += 1
        elif 'Solana' in market:
            loss_assets['SOL'] += 1
        elif 'XRP' in market:
            loss_assets['XRP'] += 1
        else:
            loss_assets['OTHER'] += 1
    print(f"    By asset: {dict(loss_assets)}")

csv_net_with_deposits = csv_net_pnl + total_deposits + total_rebates
print(f"\n  Starting balance:                  ${STARTING_BALANCE:.2f}")
print(f"  + CSV Net PnL (trades):            ${csv_net_pnl:+.2f}")
print(f"  + Deposits (post-baseline):        ${total_deposits:+.2f}")
print(f"  + Maker Rebates (post-baseline):   ${total_rebates:+.2f}")
print(f"  = Implied current balance:         ${STARTING_BALANCE + csv_net_with_deposits:.2f}")

# ============================================================
# 5. POTENTIAL ISSUES / FLAGS
# ============================================================
print()
print("=" * 80)
print("FLAGS / POTENTIAL ISSUES")
print("=" * 80)

# Check for multi-buy markets (same market, multiple buys = likely maker/pairs)
market_buy_counts = Counter()
for r in buys:
    market_buy_counts[r['marketName']] += 1

multi_buy_markets = {m: c for m, c in market_buy_counts.items() if c > 1}
if multi_buy_markets:
    print(f"\n  Markets with multiple buys (likely pairs arb or maker): {len(multi_buy_markets)}")
    for m, c in sorted(multi_buy_markets.items(), key=lambda x: -x[1])[:15]:
        total = sum(r['usdcAmount'] for r in buys if r['marketName'] == m)
        print(f"    {c}x buys (${total:.2f}): {m[:65]}")

# Check for very large individual buys
large_buys = [r for r in buys if r['usdcAmount'] > 5]
if large_buys:
    print(f"\n  Large buys (>$5): {len(large_buys)}")
    for r in large_buys[:10]:
        print(f"    [{ts_to_mst(r['timestamp']):>35s}] ${r['usdcAmount']:8.2f} {r['tokenName']:5s} | {r['marketName'][:55]}")

# Check for sells (unusual - maker selling?)
if sells:
    print(f"\n  Sell transactions (post-baseline): {len(sells)}")
    for r in sells[:10]:
        print(f"    [{ts_to_mst(r['timestamp']):>35s}] ${r['usdcAmount']:8.2f} {r['tokenName']:5s} | {r['marketName'][:55]}")

print()
print("=" * 80)
print("END OF RECONCILIATION")
print("=" * 80)
