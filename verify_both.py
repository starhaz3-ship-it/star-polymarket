"""Cross-check on-chain data-api vs CSV export."""
import json, csv
from collections import defaultdict

# Load on-chain activity
with open("onchain_activity.json") as f:
    activity = json.load(f)

# Load CSV
csv_rows = []
with open("C:/Users/Star/Downloads/Polymarket-History-2026-02-17 (3).csv", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_rows.append(row)

# Group on-chain by tx hash
onchain_by_hash = {}
for a in activity:
    h = a.get("transactionHash", "")
    if h:
        onchain_by_hash[h] = a

# Group CSV by hash
csv_by_hash = {}
for r in csv_rows:
    h = r.get("hash", "").strip()
    if h:
        csv_by_hash[h] = r

# Compare
only_onchain = set(onchain_by_hash.keys()) - set(csv_by_hash.keys())
only_csv = set(csv_by_hash.keys()) - set(onchain_by_hash.keys())
both = set(onchain_by_hash.keys()) & set(csv_by_hash.keys())

print(f"On-chain activity entries: {len(activity)}")
print(f"CSV rows: {len(csv_rows)}")
print(f"Matching tx hashes: {len(both)}")
print(f"Only in on-chain (not in CSV): {len(only_onchain)}")
print(f"Only in CSV (not in on-chain): {len(only_csv)}")
print()

# CSV total PnL
csv_by_market = defaultdict(list)
for r in csv_rows:
    csv_by_market[r["marketName"].strip()].append(r)

csv_spent = 0.0
csv_redeemed = 0.0
for r in csv_rows:
    if r["action"] == "Buy":
        csv_spent += float(r["usdcAmount"])
    elif r["action"] == "Redeem":
        csv_redeemed += float(r["usdcAmount"])

print(f"CSV total spent (Buys): ${csv_spent:.2f}")
print(f"CSV total redeemed: ${csv_redeemed:.2f}")
print(f"CSV net PnL: ${csv_redeemed - csv_spent:+.2f}")
print()

# On-chain totals
oc_trades = [a for a in activity if a["type"] == "TRADE"]
oc_redeems = [a for a in activity if a["type"] == "REDEEM"]
oc_spent = sum(a["usdcSize"] for a in oc_trades)
oc_redeemed = sum(a["usdcSize"] for a in oc_redeems)

print(f"On-chain total spent (Trades): ${oc_spent:.2f}")
print(f"On-chain total redeemed: ${oc_redeemed:.2f}")
print(f"On-chain net PnL: ${oc_redeemed - oc_spent:+.2f}")
print()

# Show CSV-only entries (things in CSV not in on-chain)
if only_csv:
    print(f"--- CSV entries NOT in on-chain data ---")
    for h in list(only_csv)[:10]:
        r = csv_by_hash[h]
        print(f"  {r['action']:6} {r['tokenName']:5} ${r['usdcAmount']:>8} | {r['marketName'][:55]}")
