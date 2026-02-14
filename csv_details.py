"""Show deposits, rebates, other markets, check truncation."""
import csv, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CSV_PATH = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-13 (3).csv"
rows = list(csv.DictReader(open(CSV_PATH, encoding='utf-8-sig')))

print(f"Total rows: {len(rows)}")
print("CSV likely TRUNCATED at 1000!" if len(rows) == 1000 else f"Full export ({len(rows)} rows)")

# Deposits
print("\nDEPOSITS:")
dep_total = 0
for r in rows:
    if r['action'] == 'Deposit':
        amt = float(r['usdcAmount']) if r['usdcAmount'] else 0
        dep_total += amt
        print(f"  ${amt:>8.2f} | {r['marketName'][:60]}")
print(f"  Total deposits: ${dep_total:.2f}")

# Maker rebates
print("\nMAKER REBATES:")
reb_total = 0
for r in rows:
    if r['action'] == 'Maker Rebate':
        amt = float(r['usdcAmount']) if r['usdcAmount'] else 0
        reb_total += amt
        print(f"  ${amt:>8.4f} | {r['marketName'][:60]}")
print(f"  Total rebates: ${reb_total:.4f}")

# Other category (non-updown)
print("\n'OTHER' MARKETS (non-UpDown buys):")
other_spent = 0
for r in rows:
    name = r['marketName'].lower()
    if 'up or down' not in name and r['action'] == 'Buy':
        amt = float(r['usdcAmount']) if r['usdcAmount'] else 0
        other_spent += amt
        print(f"  Buy ${amt:>8.2f} | {r['marketName'][:70]}")
print(f"  Total other spent: ${other_spent:.2f}")

# Unredeemed check - buys without matching redeems
print("\nUNREDEEMED POSITIONS (markets with buys but no redeems):")
buy_markets = set()
redeem_markets = set()
for r in rows:
    if r['action'] == 'Buy':
        buy_markets.add(r['marketName'])
    elif r['action'] == 'Redeem':
        redeem_markets.add(r['marketName'])
unredeemed = buy_markets - redeem_markets
if unredeemed:
    for m in sorted(unredeemed)[:20]:
        print(f"  {m[:70]}")
    print(f"  Total unredeemed markets: {len(unredeemed)}")
else:
    print("  All markets redeemed")
