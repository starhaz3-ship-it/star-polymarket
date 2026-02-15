"""
Polymarket Partial Fill Analyzer
Analyzes last 4 hours of trades from CSV to find partial fills causing losses.
"""
import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone

CSV_PATH = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-14 (6).csv"

# Current time ~Feb 15 01:15 UTC. Last 4 hours = since ~21:15 UTC on Feb 14
# That's timestamp >= 1739567700 ... but let's compute properly
# Feb 14, 2026 21:15 UTC
# The CSV timestamps are Unix epoch seconds
# Let's use the user's cutoff: 1771103700
CUTOFF_TS = 1771103700

def ts_to_str(ts):
    """Convert unix timestamp to human-readable UTC string"""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

def shorten_market(name):
    """Shorten market name for display"""
    # "Bitcoin Up or Down - February 14, 8:55PM-9:00PM ET" -> "BTC 8:55-9:00PM"
    name = name.strip()
    asset = name.split(" Up or Down")[0] if " Up or Down" in name else name[:10]
    asset_map = {"Bitcoin": "BTC", "Ethereum": "ETH", "XRP": "XRP", "Solana": "SOL"}
    short_asset = asset_map.get(asset, asset[:5])

    # Extract time range
    if ", " in name:
        time_part = name.split(", ")[-1].replace(" ET", "")
    else:
        time_part = ""

    return f"{short_asset} {time_part}"

def main():
    # Read CSV
    rows = []
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row['timestamp'])
            row['timestamp_int'] = ts
            row['usdcAmount'] = float(row['usdcAmount'])
            row['tokenAmount'] = float(row['tokenAmount'])
            rows.append(row)

    print(f"Total CSV rows: {len(rows)}")
    print(f"Time range: {ts_to_str(min(r['timestamp_int'] for r in rows))} to {ts_to_str(max(r['timestamp_int'] for r in rows))}")
    print(f"Cutoff: {ts_to_str(CUTOFF_TS)}")
    print()

    # Filter to last 4 hours
    recent = [r for r in rows if r['timestamp_int'] >= CUTOFF_TS]
    print(f"Rows in last 4 hours: {len(recent)}")
    if not recent:
        print("No trades in last 4 hours!")
        return
    print(f"Recent range: {ts_to_str(min(r['timestamp_int'] for r in recent))} to {ts_to_str(max(r['timestamp_int'] for r in recent))}")
    print()

    # Group by market
    markets = defaultdict(lambda: {
        'buys_up': [], 'buys_down': [],
        'sells': [], 'redeems': [],
        'buy_usdc_up': 0, 'buy_usdc_down': 0,
        'buy_tokens_up': 0, 'buy_tokens_down': 0,
        'sell_usdc': 0, 'redeem_usdc': 0,
        'first_ts': float('inf'), 'last_ts': 0
    })

    for r in recent:
        mkt = r['marketName']
        m = markets[mkt]
        ts = r['timestamp_int']
        m['first_ts'] = min(m['first_ts'], ts)
        m['last_ts'] = max(m['last_ts'], ts)

        action = r['action']
        usdc = r['usdcAmount']
        tokens = r['tokenAmount']
        token_name = r['tokenName'].strip()

        if action == 'Buy':
            if token_name == 'Up':
                m['buys_up'].append(r)
                m['buy_usdc_up'] += usdc
                m['buy_tokens_up'] += tokens
            elif token_name == 'Down':
                m['buys_down'].append(r)
                m['buy_usdc_down'] += usdc
                m['buy_tokens_down'] += tokens
            else:
                print(f"  WARNING: Unknown token name '{token_name}' in Buy for {mkt}")
        elif action == 'Sell':
            m['sells'].append(r)
            m['sell_usdc'] += usdc
        elif action == 'Redeem':
            m['redeems'].append(r)
            m['redeem_usdc'] += usdc

    # Analyze each market
    print("=" * 120)
    print(f"{'MARKET':<35} {'SIDES':<12} {'BUY UP':>10} {'BUY DN':>10} {'TOTAL BUY':>10} {'SELL':>10} {'REDEEM':>10} {'NET PnL':>10} {'STATUS':<12}")
    print("=" * 120)

    paired_count = 0
    paired_pnl = 0
    partial_count = 0
    partial_pnl = 0
    invisible_loss_count = 0
    invisible_loss_usdc = 0
    partial_up_only = 0
    partial_down_only = 0

    # Sort markets by first timestamp
    sorted_markets = sorted(markets.items(), key=lambda x: x[1]['first_ts'])

    for mkt_name, m in sorted_markets:
        short = shorten_market(mkt_name)
        buy_up = m['buy_usdc_up']
        buy_dn = m['buy_usdc_down']
        total_buy = buy_up + buy_dn
        total_sell = m['sell_usdc']
        total_redeem = m['redeem_usdc']
        total_received = total_sell + total_redeem
        net_pnl = total_received - total_buy

        has_up = buy_up > 0
        has_dn = buy_dn > 0
        has_sell = total_sell > 0
        has_redeem = total_redeem > 0

        if has_up and has_dn:
            sides = "BOTH"
            status = "PAIRED"
            paired_count += 1
            paired_pnl += net_pnl
        elif has_up:
            sides = "UP ONLY"
            partial_up_only += 1
            if not has_sell and not has_redeem:
                status = "INVISIBLE"
                invisible_loss_count += 1
                invisible_loss_usdc += total_buy
            else:
                status = "PARTIAL"
            partial_count += 1
            partial_pnl += net_pnl
        elif has_dn:
            sides = "DN ONLY"
            partial_down_only += 1
            if not has_sell and not has_redeem:
                status = "INVISIBLE"
                invisible_loss_count += 1
                invisible_loss_usdc += total_buy
            else:
                status = "PARTIAL"
            partial_count += 1
            partial_pnl += net_pnl
        else:
            # Only redeems, no buys in window (buys were earlier)
            sides = "REDEEM"
            status = "REDEEM-ONLY"
            # Still count the redeem income
            paired_pnl += net_pnl  # will be positive since total_buy=0
            continue  # skip from display unless significant

        pnl_color = "+" if net_pnl >= 0 else ""
        print(f"{short:<35} {sides:<12} ${buy_up:>8.2f} ${buy_dn:>8.2f} ${total_buy:>8.2f} ${total_sell:>8.2f} ${total_redeem:>8.2f} {pnl_color}${net_pnl:>8.2f} {status:<12}")

    # Now let's look for redeem-only entries (markets where buys were before the window)
    redeem_only_markets = [(n, m) for n, m in sorted_markets
                           if m['buy_usdc_up'] == 0 and m['buy_usdc_down'] == 0
                           and (m['redeem_usdc'] > 0 or m['sell_usdc'] > 0)]

    if redeem_only_markets:
        print(f"\n--- Redeem/Sell-only in window (buys were before 4hr cutoff): {len(redeem_only_markets)} markets ---")
        for mkt_name, m in redeem_only_markets:
            short = shorten_market(mkt_name)
            if m['redeem_usdc'] > 0 or m['sell_usdc'] > 0:
                print(f"  {short:<35} Redeem: ${m['redeem_usdc']:.2f}  Sell: ${m['sell_usdc']:.2f}")

    # =========================================================================
    # DETAILED BREAKDOWN OF PAIRED vs PARTIAL
    # =========================================================================
    print("\n" + "=" * 120)
    print("DETAILED MARKET BREAKDOWN")
    print("=" * 120)

    for mkt_name, m in sorted_markets:
        buy_up = m['buy_usdc_up']
        buy_dn = m['buy_usdc_down']
        if buy_up == 0 and buy_dn == 0:
            continue

        short = shorten_market(mkt_name)
        total_buy = buy_up + buy_dn
        total_received = m['sell_usdc'] + m['redeem_usdc']
        net = total_received - total_buy

        has_up = buy_up > 0
        has_dn = buy_dn > 0

        print(f"\n  {short}")
        print(f"    Time: {ts_to_str(m['first_ts'])} - {ts_to_str(m['last_ts'])}")

        if has_up:
            avg_price_up = buy_up / m['buy_tokens_up'] if m['buy_tokens_up'] > 0 else 0
            print(f"    UP:   {len(m['buys_up'])} fills, ${buy_up:.4f} for {m['buy_tokens_up']:.2f} tokens (avg ${avg_price_up:.4f}/token)")
        if has_dn:
            avg_price_dn = buy_dn / m['buy_tokens_down'] if m['buy_tokens_down'] > 0 else 0
            print(f"    DOWN: {len(m['buys_down'])} fills, ${buy_dn:.4f} for {m['buy_tokens_down']:.2f} tokens (avg ${avg_price_dn:.4f}/token)")

        if has_up and has_dn:
            # For paired trades: combined cost should be < token amount for profit
            # If you buy X Up tokens at price p_up and X Down tokens at price p_dn,
            # cost = X * p_up + X * p_dn. If p_up + p_dn < 1.0, you profit X * (1 - p_up - p_dn)
            total_tokens = min(m['buy_tokens_up'], m['buy_tokens_down'])
            combined_avg = (buy_up / m['buy_tokens_up']) + (buy_dn / m['buy_tokens_down']) if m['buy_tokens_up'] > 0 and m['buy_tokens_down'] > 0 else 999
            spread_profit = 1.0 - combined_avg
            print(f"    PAIRED: Combined avg price: ${combined_avg:.4f} (spread profit: ${spread_profit:.4f}/token)")
            if m['buy_tokens_up'] != m['buy_tokens_down']:
                print(f"    WARNING: Uneven fill! UP={m['buy_tokens_up']:.2f} tokens vs DOWN={m['buy_tokens_down']:.2f} tokens")
                excess_side = "UP" if m['buy_tokens_up'] > m['buy_tokens_down'] else "DOWN"
                excess_tokens = abs(m['buy_tokens_up'] - m['buy_tokens_down'])
                print(f"    EXCESS: {excess_tokens:.2f} unmatched {excess_side} tokens = directional exposure!")

        if m['sells']:
            print(f"    SELL:   ${m['sell_usdc']:.4f} ({len(m['sells'])} txns)")
        if m['redeems']:
            nonzero_redeems = [r for r in m['redeems'] if r['usdcAmount'] > 0]
            zero_redeems = [r for r in m['redeems'] if r['usdcAmount'] == 0]
            if nonzero_redeems:
                print(f"    REDEEM: ${m['redeem_usdc']:.4f} ({len(nonzero_redeems)} winning, {len(zero_redeems)} losing)")
            elif zero_redeems:
                print(f"    REDEEM: $0.00 ({len(zero_redeems)} losing redeems)")

        if not m['sells'] and not m['redeems']:
            print(f"    EXIT:   NONE - still open or invisible loss")

        pnl_str = f"+${net:.4f}" if net >= 0 else f"-${abs(net):.4f}"
        print(f"    NET:    {pnl_str}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 120)
    print("SUMMARY - LAST 4 HOURS")
    print("=" * 120)

    total_markets_with_buys = sum(1 for _, m in sorted_markets if m['buy_usdc_up'] > 0 or m['buy_usdc_down'] > 0)
    total_usdc_spent = sum(m['buy_usdc_up'] + m['buy_usdc_down'] for _, m in sorted_markets)
    total_usdc_received = sum(m['sell_usdc'] + m['redeem_usdc'] for _, m in sorted_markets)

    print(f"\n  Total markets traded:     {total_markets_with_buys}")
    print(f"  Total USDC spent (buys):  ${total_usdc_spent:.2f}")
    print(f"  Total USDC received:      ${total_usdc_received:.2f}")
    print(f"  Net PnL:                  ${total_usdc_received - total_usdc_spent:+.2f}")

    print(f"\n  PAIRED trades (both sides):")
    print(f"    Count:      {paired_count}")
    print(f"    Net PnL:    ${paired_pnl:+.2f}")
    if paired_count > 0:
        print(f"    Avg PnL:    ${paired_pnl/paired_count:+.2f}")

    print(f"\n  PARTIAL fills (one side only):")
    print(f"    Count:      {partial_count}")
    print(f"    Net PnL:    ${partial_pnl:+.2f}")
    if partial_count > 0:
        print(f"    Avg PnL:    ${partial_pnl/partial_count:+.2f}")
    print(f"    UP-only:    {partial_up_only}")
    print(f"    DOWN-only:  {partial_down_only}")

    print(f"\n  INVISIBLE losses (buy but no sell/redeem):")
    print(f"    Count:      {invisible_loss_count}")
    print(f"    USDC lost:  ${invisible_loss_usdc:.2f}")

    # =========================================================================
    # TOKEN BALANCE ANALYSIS (uneven fills in PAIRED trades)
    # =========================================================================
    print(f"\n  {'='*80}")
    print(f"  UNEVEN FILL ANALYSIS (paired trades with mismatched token amounts)")
    print(f"  {'='*80}")

    uneven_count = 0
    total_excess_exposure = 0

    for mkt_name, m in sorted_markets:
        if m['buy_usdc_up'] > 0 and m['buy_usdc_down'] > 0:
            up_tokens = m['buy_tokens_up']
            dn_tokens = m['buy_tokens_down']
            if abs(up_tokens - dn_tokens) > 0.5:  # More than 0.5 token mismatch
                uneven_count += 1
                excess = abs(up_tokens - dn_tokens)
                side = "UP" if up_tokens > dn_tokens else "DOWN"
                excess_cost = excess * (m['buy_usdc_up']/up_tokens if side == "UP" else m['buy_usdc_down']/dn_tokens)
                total_excess_exposure += excess_cost
                short = shorten_market(mkt_name)
                print(f"    {short:<35} UP:{up_tokens:.1f} DN:{dn_tokens:.1f} -> {excess:.1f} excess {side} (${excess_cost:.2f} directional)")

    if uneven_count == 0:
        print(f"    None found - all paired trades have matching token amounts")
    else:
        print(f"\n    Total uneven paired trades: {uneven_count}")
        print(f"    Total directional exposure: ${total_excess_exposure:.2f}")

    # =========================================================================
    # PRICE ANALYSIS: Are we buying at fair prices?
    # =========================================================================
    print(f"\n  {'='*80}")
    print(f"  PRICE ANALYSIS: Average buy prices")
    print(f"  {'='*80}")
    print(f"  (For maker: combined UP+DOWN price should be < $1.00 to profit)")
    print()

    for mkt_name, m in sorted_markets:
        if m['buy_usdc_up'] > 0 and m['buy_usdc_down'] > 0:
            short = shorten_market(mkt_name)
            avg_up = m['buy_usdc_up'] / m['buy_tokens_up']
            avg_dn = m['buy_usdc_down'] / m['buy_tokens_down']
            combined = avg_up + avg_dn
            edge = 1.0 - combined
            edge_pct = edge * 100
            marker = "PROFIT" if edge > 0 else "LOSS"
            print(f"    {short:<35} UP:${avg_up:.4f} + DN:${avg_dn:.4f} = ${combined:.4f}  edge:{edge_pct:+.2f}%  {marker}")


if __name__ == '__main__':
    main()
