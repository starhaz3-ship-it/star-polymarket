#!/usr/bin/env python3
import json, os, sys
from datetime import datetime, timezone

BASE = "C:/Users/Star/.local/bin/star-polymarket"
LOG = os.path.join(BASE, "overnight_evolution.log")

def fmt(v, d=2):
    return f"${v:.{d}f}"

def main():
    now = datetime.now(timezone.utc)
    print(f"\n{'='*70}")
    print(f"EVOLUTION CHECK {now.isoformat()}")
    print(f"{'='*70}")
    
    fill_rate = 0; stats = {}; resolved = []; active = []
    ta = {}; wr = 0; total_trades = 0; hourly = {}
    
    try:
        with open(os.path.join(BASE, "maker_results.json")) as f:
            mk = json.load(f)
        stats = mk["stats"]; resolved = mk["resolved"]; active = mk["active_positions"]
        print("\n--- MAKER BOT ---")
        print(f"Total PnL: {fmt(stats['total_pnl'],4)}")
        print(f"Paired: {fmt(stats['paired_pnl'],4)} | Partial: {fmt(stats['partial_pnl'],4)}")
        print(f"Pairs: {stats['pairs_attempted']} att, {stats['pairs_completed']} comp, {stats['pairs_partial']} part")
        fill_rate = (stats['pairs_completed']/stats['pairs_attempted']*100) if stats['pairs_attempted'] > 0 else 0
        print(f"Fill rate: {fill_rate:.1f}% | Active: {len(active)}")
        for a in ['BTC','ETH','SOL']:
            at = [r for r in resolved if r['asset']==a]
            if at:
                print(f"  {a}: {len(at)}T {fmt(sum(t['pnl'] for t in at))} p:{sum(1 for t in at if t['paired'])} part:{sum(1 for t in at if t.get('partial',False))}")
        pt = [r for r in resolved if r.get('partial',False)]
        if pt:
            print(f"\nPARTIALS: {fmt(sum(t['pnl'] for t in pt))}")
            for t in pt: print(f"  {t['asset']} {t['outcome']} {fmt(t['pnl'])} UP:{t['up_price']} DN:{t['down_price']}")
        print(f"\nACTIVE ({len(active)}):")
        for p in active: print(f"  {p['asset']} {p['status']} UP:{p['up_filled']} DN:{p['down_filled']} off:{p.get('bid_offset_used','?')}")
    except Exception as e: print(f"MAKER ERROR: {e}")
    
    try:
        with open(os.path.join(BASE, "maker_ml_state.json")) as f: ml = json.load(f)
        print("\n--- ML STATE ---")
        for h in sorted(ml.keys(), key=int):
            for o, s in ml[h].items():
                if s['attempts'] > 0:
                    r = s['paired']/s['attempts']*100
                    print(f"  H{int(h):02d} off={o} {s['attempts']}att {s['paired']}pair ({r:.0f}%) {s['partial']}part PnL:{fmt(s['total_pnl'])}")
    except Exception as e: print(f"ML ERROR: {e}")
    
    try:
        with open(os.path.join(BASE, "ta_live_results.json")) as f: ta = json.load(f)
        total_trades = ta['wins']+ta['losses']
        wr = ta['wins']/total_trades*100 if total_trades > 0 else 0
        print("\n--- TA LIVE ---")
        print(f"PnL: {fmt(ta['total_pnl'])} | {ta['wins']}W/{ta['losses']}L ({wr:.1f}%)")
        print(f"Bank: {fmt(ta['bankroll'])} | Streak: {ta['consecutive_wins']}W/{ta['consecutive_losses']}L")
        hourly = ta.get('hourly_stats',{})
        print("\nHOURLY:")
        hd = []
        for h, hs in hourly.items():
            t = hs['wins']+hs['losses']
            if t > 0: hd.append((int(h),t,hs['wins'],hs['losses'],hs['pnl']))
        for h,t,w,l,p in sorted(hd,key=lambda x:x[4],reverse=True):
            hw = w/t*100
            f = " ***" if p < -3 and t >= 3 else ""
            if p > 3 and hw >= 70: f = " +++"
            print(f"  UTC{h:>3}: {t}T ({w}W/{l}L) {hw:.0f}% {fmt(p)}{f}")
        print("\nHYDRA:")
        for n,s in ta.get('hydra_live',{}).get('per_strategy',{}).items():
            if s.get('trades',0) > 0:
                sw = s['wins']/(s['wins']+s['losses'])*100 if (s['wins']+s['losses']) > 0 else 0
                print(f"  {n}: {s['trades']}T ({s['wins']}W/{s['losses']}L) {sw:.0f}% {fmt(s['pnl'])} [{s.get('status','?')}]")
        print("\nFILTERS:")
        for n,s in ta.get('filter_stats',{}).items():
            if s['blocked'] > 0:
                t = s['wins']+s['losses']
                if t > 0:
                    fw = s['wins']/t*100
                    a = ""
                    if s['pnl'] > 10 and fw > 52 and t >= 20: a = " LOOSEN?"
                    elif s['pnl'] < -10 and fw < 45: a = " GOOD"
                    print(f"  {n}: {s['blocked']}blk {t}T ({s['wins']}W/{s['losses']}L) {fw:.0f}% {fmt(s['pnl'])}{a}")
    except Exception as e: print(f"TA ERROR: {e}")
    
    print("\n--- OPTIMIZATION ---")
    changes = []
    try:
        if stats.get('pairs_attempted',0) >= 20:
            if fill_rate < 60: print(f"MAKER: Fill {fill_rate:.1f}% < 60%. Tighten?")
            elif fill_rate > 90: print(f"MAKER: Fill {fill_rate:.1f}% > 90%. Widen?")
        else: print(f"MAKER: {stats.get('pairs_attempted',0)} pairs, need 20+")
        sp = [r for r in resolved if r['asset']=='SOL' and r.get('partial',False)]
        if len(sp) >= 2: print(f"MAKER: SOL {len(sp)} partials losing {fmt(sum(t['pnl'] for t in sp))}")
    except: pass
    try:
        if total_trades >= 20:
            if wr > 55: print(f"TA: {wr:.1f}% > 55%. Good.")
            elif wr < 35: print(f"TA: {wr:.1f}% < 35%. TIGHTEN!")
            else: print(f"TA: {wr:.1f}% ok.")
        h15 = hourly.get('15',{})
        h15t = h15.get('wins',0)+h15.get('losses',0)
        if h15t >= 3 and h15.get('wins',0)==0:
            print(f"TA: UTC15 0%WR ({h15t}T) {fmt(h15.get('pnl',0))}. ADD SKIP")
        v33 = ta.get('filter_stats',{}).get('v33_conviction',{})
        if v33.get('blocked',0) > 100:
            vt = v33.get('wins',0)+v33.get('losses',0)
            if vt > 0:
                vw = v33['wins']/vt*100
                if vw > 52 and v33['pnl'] > 20:
                    print(f"TA: v33_conviction {v33['blocked']}blk ({vw:.0f}% WR, {fmt(v33['pnl'])}). Too aggressive?")
    except: pass
    
    print("\n--- SUMMARY ---")
    print(f"Maker: {fmt(stats.get('total_pnl',0))} | {stats.get('pairs_completed',0)} paired | {fill_rate:.0f}% fill")
    print(f"TA: {fmt(ta.get('total_pnl',0))} | {wr:.1f}% WR | Best:UTC23/3/14 Worst:UTC15/0/7")
    print(f"Changes: {changes if changes else 'None'}")
    
    with open(LOG, "a") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"EVO CHECK {now.isoformat()}\n")
        f.write(f"Maker: {fmt(stats.get('total_pnl',0),4)} | {stats.get('pairs_completed',0)}/{stats.get('pairs_attempted',0)} ({fill_rate:.0f}%)\n")
        f.write(f"TA: {fmt(ta.get('total_pnl',0))} | {ta.get('wins',0)}W/{ta.get('losses',0)}L ({wr:.1f}%)\n")
        f.write(f"Active: {len(active)} | Bank: {fmt(ta.get('bankroll',0))}\n")
        f.write(f"Changes: {changes if changes else 'None'}\n")
    print(f"\nLogged to {LOG}")

if __name__ == "__main__": main()
