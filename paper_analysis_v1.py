#!/usr/bin/env python3
import json, os, sys
from collections import defaultdict
from datetime import datetime

DATA = r"C:\Users\Star\.local\bin\star-polymarket\ta_paper_results.json"

def load():
    with open(DATA,"r") as f: return json.load(f)

def asset(title):
    t=title.lower()
    if "bitcoin" in t: return "BTC"
    if "ethereum" in t: return "ETH"
    if "solana" in t: return "SOL"
    return "OTHER"

def hour_utc(s):
    try:
        ds=s.split("+")[0] if "+" in s else s.replace("Z","")
        return datetime.fromisoformat(ds).hour
    except: return -1

def pb(p):
    if p<0.15: return "<0.15"
    if p<0.25: return "0.15-0.25"
    if p<0.35: return "0.25-0.35"
    if p<0.45: return "0.35-0.45"
    if p<0.55: return "0.45-0.55"
    return "0.55+"

def main():
    d=load()
    td=d.get("trades",{})
    hs=d.get("hourly_stats",{})
    shs=d.get("skip_hour_shadows",{})
    shst=d.get("skip_hour_stats",{})
    ss=d.get("systematic_stats",{})
    trades=[]
    for tid,t in td.items():
        t["_k"]=tid; t["_a"]=asset(t.get("market_title",""))
        t["_w"]=t.get("pnl",0)>0; t["_h"]=hour_utc(t.get("entry_time",""))
        trades.append(t)
    N=len(trades)
    wins=[t for t in trades if t["_w"]]
    losses=[t for t in trades if not t["_w"]]
    tpnl=sum(t.get("pnl",0) for t in trades)
    sep="="*80

    # SECTION 1: BASIC STATS
    print(sep)
    print("    COMPREHENSIVE PAPER TRADING ANALYSIS")
    print(sep)
    print(f"  Period: {d.get('start_time','?')} to {d.get('last_updated','?')}")
    print()
    print(sep); print("  1. BASIC STATS"); print(sep)
    wr=len(wins)/N*100 if N else 0
    aw=sum(t["pnl"] for t in wins)/len(wins) if wins else 0
    al=sum(t["pnl"] for t in losses)/len(losses) if losses else 0
    ts=sum(t.get("size_usd",0) for t in trades)
    wt=sum(t["pnl"] for t in wins)
    lt=sum(t["pnl"] for t in losses)
    pf=abs(wt/lt) if lt!=0 else float("inf")
    print(f"  Trades: {N} | Wins: {len(wins)} | Losses: {len(losses)} | WR: {wr:.1f}%")
    print(f"  Total PnL: ${tpnl:+.2f} | Avg Win: ${aw:+.2f} | Avg Loss: ${al:+.2f}")
    print(f"  Profit Factor: {pf:.2f} | Total Risked: ${ts:.0f} | ROI: {tpnl/ts*100:.1f}%")
    print(f"  Signals: {d.get('signals_count','?')} | Trade rate: {N/max(d.get('signals_count',1),1)*100:.1f}%")
    print()

    # By Asset
    print("  --- By Asset ---")
    hdr=f"  {'Asset':<8}{'#':>4}{'W':>4}{'L':>4}{'WR%':>7}{'PnL':>10}{'Avg':>8}"
    print(hdr); print("  "+"-"*50)
    for a in ["BTC","ETH","SOL"]:
        at=[t for t in trades if t["_a"]==a]
        if not at: continue
        w=sum(1 for t in at if t["_w"]); l=len(at)-w
        p=sum(t["pnl"] for t in at)
        print(f"  {a:<8}{len(at):>4}{w:>4}{l:>4}{w/len(at)*100:>6.1f}%${p:>+8.2f}${p/len(at):>+6.2f}")
    print()

    # By Side
    print("  --- By Side ---")
    print(f"  {'Side':<8}{'#':>4}{'W':>4}{'L':>4}{'WR%':>7}{'PnL':>10}{'Avg':>8}")
    print("  "+"-"*50)
    for s in ["UP","DOWN"]:
        st=[t for t in trades if t.get("side")==s]
        if not st: continue
        w=sum(1 for t in st if t["_w"]); l=len(st)-w
        p=sum(t["pnl"] for t in st)
        print(f"  {s:<8}{len(st):>4}{w:>4}{l:>4}{w/len(st)*100:>6.1f}%${p:>+8.2f}${p/len(st):>+6.2f}")
    print()

    # Side x Asset
    print("  --- Side x Asset ---")
    print(f"  {'Combo':<12}{'#':>4}{'W':>4}{'WR%':>7}{'PnL':>10}")
    print("  "+"-"*40)
    for s in ["UP","DOWN"]:
        for a in ["BTC","ETH","SOL"]:
            ct=[t for t in trades if t.get("side")==s and t["_a"]==a]
            if not ct: continue
            cw=sum(1 for t in ct if t["_w"])
            cp=sum(t["pnl"] for t in ct)
            print(f"  {s+'/'+a:<12}{len(ct):>4}{cw:>4}{cw/len(ct)*100:>6.1f}%${cp:>+8.2f}")
    print()

    # SECTION 2: FEATURE ANALYSIS
    print(sep); print("  2. FEATURE ANALYSIS"); print(sep)
    nf=["edge_at_entry","kl_divergence","kelly_fraction","entry_price","size_usd","guaranteed_profit"]
    print(f"  {'Feature':<22}{'Wins Avg':>10}{'Loss Avg':>10}{'Delta':>10}{'Pred?':>8}")
    print("  "+"-"*62)
    for feat in nf:
        wv=[t.get(feat,0) for t in wins if t.get(feat) is not None]
        lv=[t.get(feat,0) for t in losses if t.get(feat) is not None]
        wa=sum(wv)/len(wv) if wv else 0
        la=sum(lv)/len(lv) if lv else 0
        dd=wa-la
        pr="YES" if abs(dd)>0.02 else "WEAK"
        print(f"  {feat:<22}{wa:>10.4f}{la:>10.4f}{dd:>+10.4f}{pr:>8}")
    print()

    # Systematic stats
    print("  --- Systematic Stats ---")
    for sk,sv in ss.items():
        if isinstance(sv,dict) and "trades" in sv:
            st=sv.get("trades",0); sw=sv.get("wins",0); sl=sv.get("losses",0); sp=sv.get("pnl",0)
            swr=sw/st*100 if st else 0
            f=" ** WIN" if swr>=70 and st>=3 else (" ** LOSE" if swr<=30 and st>=3 else "")
            print(f"    {sk:<22}{st:>3}t | {sw}W/{sl}L ({swr:.0f}%) | ${sp:+.2f}{f}")
        else:
            print(f"    {sk:<22}{sv}")
    print()

    # SECTION 3: PRICE BUCKETS
    print(sep); print("  3. PRICE BUCKET ANALYSIS"); print(sep)
    bk=defaultdict(lambda:{"t":0,"w":0,"l":0,"p":0.0,"s":0.0})
    for t in trades:
        b=pb(t.get("entry_price",0))
        bk[b]["t"]+=1; bk[b]["p"]+=t.get("pnl",0); bk[b]["s"]+=t.get("size_usd",0)
        if t["_w"]: bk[b]["w"]+=1
        else: bk[b]["l"]+=1
    print(f"  {'Bucket':<12}{'#':>4}{'W':>4}{'L':>4}{'WR%':>7}{'PnL':>10}{'Avg':>8}{'ROI%':>7}")
    print("  "+"-"*58)
    for b in ["<0.15","0.15-0.25","0.25-0.35","0.35-0.45","0.45-0.55","0.55+"]:
        x=bk[b]
        if x["t"]==0: continue
        wr=x["w"]/x["t"]*100; roi=x["p"]/x["s"]*100 if x["s"] else 0
        fl=" <<<" if wr<40 else (" +++" if wr>=80 else "")
        print(f"  {b:<12}{x['t']:>4}{x['w']:>4}{x['l']:>4}{wr:>6.1f}%${x['p']:>+8.2f}${x['p']/x['t']:>+6.2f}{roi:>+6.1f}%{fl}")
    print()

    # Price bucket by side
    for side in ["UP","DOWN"]:
        print(f"  {side}:")
        sb=defaultdict(lambda:{"t":0,"w":0,"p":0.0})
        for t in trades:
            if t.get("side")!=side: continue
            b=pb(t.get("entry_price",0))
            sb[b]["t"]+=1; sb[b]["p"]+=t.get("pnl",0)
            if t["_w"]: sb[b]["w"]+=1
        print(f"  {'Bucket':<12}{'#':>4}{'W':>4}{'WR%':>7}{'PnL':>10}")
        print("  "+"-"*40)
        for b in ["<0.15","0.15-0.25","0.25-0.35","0.35-0.45","0.45-0.55","0.55+"]:
            x=sb[b]
            if x["t"]==0: continue
            wr=x["w"]/x["t"]*100
            print(f"  {b:<12}{x['t']:>4}{x['w']:>4}{wr:>6.1f}%${x['p']:>+8.2f}")
        print()

    # SECTION 4: TIME ANALYSIS
    print(sep); print("  4. TIME ANALYSIS (UTC)"); print(sep)
    hh=defaultdict(lambda:{"t":0,"w":0,"l":0,"p":0.0})
    for t in trades:
        h=t["_h"]; hh[h]["t"]+=1; hh[h]["p"]+=t.get("pnl",0)
        if t["_w"]: hh[h]["w"]+=1
        else: hh[h]["l"]+=1
    print(f"  {'Hour':>6}{'#':>4}{'W':>4}{'L':>4}{'WR%':>7}{'PnL':>10}")
    print("  "+"-"*38)
    for h in sorted(hh.keys()):
        x=hh[h]; wr=x["w"]/x["t"]*100 if x["t"] else 0
        fl=" *** SKIP" if x["p"]<-10 else (" +++ BEST" if x["p"]>20 else "")
        print(f"  {h:>6}{x['t']:>4}{x['w']:>4}{x['l']:>4}{wr:>6.1f}%${x['p']:>+8.2f}{fl}")
    print()

    # From file hourly_stats
    print("  --- File hourly_stats ---")
    ah={h:v for h,v in hs.items() if v.get("wins",0)+v.get("losses",0)>0}
    for h in sorted(ah.keys(),key=int):
        v=ah[h]; w,l,p=v.get("wins",0),v.get("losses",0),v.get("pnl",0)
        wr=w/(w+l)*100 if (w+l) else 0
        print(f"    Hour {h:>2}: {w}W/{l}L ({wr:.0f}%) ${p:+.2f}")
    print()

    # SECTION 5: EDGE/KL
    print(sep); print("  5. EDGE / KL THRESHOLD ANALYSIS"); print(sep)
    print("  --- Edge at Entry ---")
    print(f"  {'Thresh':>8}{'#':>4}{'W':>4}{'WR%':>7}{'PnL':>10}{'Avg':>8}")
    print("  "+"-"*42)
    for th in [0.30,0.33,0.35,0.38,0.40,0.45,0.50,0.55]:
        et=[t for t in trades if t.get("edge_at_entry",0)>=th]
        if not et: continue
        ew=sum(1 for t in et if t["_w"]); ep=sum(t["pnl"] for t in et)
        print(f"  >={th:<6.2f}{len(et):>4}{ew:>4}{ew/len(et)*100:>6.1f}%${ep:>+8.2f}${ep/len(et):>+6.2f}")
    print()

    print("  --- KL Divergence ---")
    print(f"  {'Thresh':>8}{'#':>4}{'W':>4}{'WR%':>7}{'PnL':>10}{'Avg':>8}")
    print("  "+"-"*42)
    for th in [0.20,0.25,0.28,0.30,0.35,0.40,0.50,0.60]:
        et=[t for t in trades if t.get("kl_divergence",0)>=th]
        if not et: continue
        ew=sum(1 for t in et if t["_w"]); ep=sum(t["pnl"] for t in et)
        print(f"  >={th:<6.2f}{len(et):>4}{ew:>4}{ew/len(et)*100:>6.1f}%${ep:>+8.2f}${ep/len(et):>+6.2f}")
    print()

    # Every trade sorted by edge
    print("  --- Trades by Edge (desc) ---")
    print(f"  {'Edge':>7}{'KL':>7}{'Side':>5}{'Coin':>5}{'Ent':>6}{'Exit':>6}{'PnL':>9}{'Res':>5}")
    print("  "+"-"*55)
    for t in sorted(trades,key=lambda x:x.get("edge_at_entry",0),reverse=True):
        r="W" if t["_w"] else "L"
        print(f"  {t.get('edge_at_entry',0):>7.3f}{t.get('kl_divergence',0):>7.3f}{t.get('side','?'):>5}{t['_a']:>5}{t.get('entry_price',0):>6.2f}{t.get('exit_price',0):>6.2f}${t.get('pnl',0):>+7.2f}  {r}")
    print()

    # SECTION 6: STREAKS
    print(sep); print("  6. STREAK ANALYSIS"); print(sep)
    sbt=sorted(trades,key=lambda t:t.get("entry_time",""))
    streaks=[]; cs,ct=0,None
    for t in sbt:
        w=t["_w"]
        if ct is None: ct,cs=w,1
        elif w==ct: cs+=1
        else: streaks.append((ct,cs)); ct,cs=w,1
    if ct is not None: streaks.append((ct,cs))
    mws=max((s[1] for s in streaks if s[0]),default=0)
    mls=max((s[1] for s in streaks if not s[0]),default=0)
    print(f"  Max Win Streak:  {mws}")
    print(f"  Max Loss Streak: {mls}")
    seq=" ".join(f"{'W' if s[0] else 'L'}{s[1]}" for s in streaks)
    print(f"  Sequence: {seq}")
    print()

    # After patterns
    alw,all_,aww,awl=0,0,0,0
    for i in range(1,len(sbt)):
        if not sbt[i-1]["_w"]:
            if sbt[i]["_w"]: alw+=1
            else: all_+=1
        else:
            if sbt[i]["_w"]: aww+=1
            else: awl+=1
    if alw+all_>0: print(f"  After loss: {alw}W/{all_}L ({alw/(alw+all_)*100:.0f}% WR)")
    if aww+awl>0: print(f"  After win:  {aww}W/{awl}L ({aww/(aww+awl)*100:.0f}% WR)")
    print()

    # Cumulative PnL
    print("  --- Cumulative PnL ---")
    cum=0
    for i,t in enumerate(sbt):
        cum+=t.get("pnl",0); r="W" if t["_w"] else "L"
        bl=max(1,int(abs(cum)/3))
        bar=("#"*bl) if cum>=0 else ("-"*bl)
        print(f"  {i+1:>2}. {r} ${t.get('pnl',0):>+8.2f}  Cum: ${cum:>+8.2f}  |{bar}")
    print()

    # Max drawdown
    peak=0; mdd=0; cum=0
    for t in sbt:
        cum+=t.get("pnl",0)
        if cum>peak: peak=cum
        dd=peak-cum
        if dd>mdd: mdd=dd
    print(f"  Max Drawdown: ${mdd:.2f} (from peak ${peak:.2f})")
    print()

    # SECTION 7: HOURLY SUMMARY
    print(sep); print("  7. HOURLY SUMMARY"); print(sep)
    if ah:
        bh=max(ah,key=lambda h:ah[h].get("pnl",0))
        wh=min(ah,key=lambda h:ah[h].get("pnl",0))
        print(f"  Best hour:  {bh} UTC (${ah[bh].get('pnl',0):+.2f})")
        print(f"  Worst hour: {wh} UTC (${ah[wh].get('pnl',0):+.2f})")
        prh=sum(1 for v in ah.values() if v.get("pnl",0)>0)
        loh=sum(1 for v in ah.values() if v.get("pnl",0)<0)
        print(f"  Profitable hours: {prh}, Losing hours: {loh}")
    print()

    # SECTION 8: SHADOWS
    print(sep); print("  8. SHADOW / FILTER STATS"); print(sep)
    if shs:
        for h,sv in sorted(shs.items()): print(f"    Shadow hour {h}: {json.dumps(sv)}")
    else:
        print("  No skip_hour_shadows data (empty dict).")
    askip={h:v for h,v in shst.items() if v.get("wins",0)+v.get("losses",0)>0}
    if askip:
        for h,sv in sorted(askip.items()): print(f"    Skip hour {h}: {sv}")
    else:
        print("  No skip-hour trades recorded.")
    print()

    # SECTION 9: ADVANCED
    print(sep); print("  9. ADVANCED INSIGHTS"); print(sep)
    wsz=[t.get("size_usd",0) for t in wins]
    lsz=[t.get("size_usd",0) for t in losses]
    if wsz: print(f"  Avg win size:  ${sum(wsz)/len(wsz):.1f} | Max: ${max(wsz):.0f}")
    if lsz: print(f"  Avg loss size: ${sum(lsz)/len(lsz):.1f} | Max: ${max(lsz):.0f}")
    print()

    pnls=sorted([t.get("pnl",0) for t in trades])
    print(f"  PnL: Min=${min(pnls):+.2f}, Max=${max(pnls):+.2f}, Med=${pnls[len(pnls)//2]:+.2f}")
    print(f"  Top 5 wins:   {[round(p,2) for p in sorted(pnls,reverse=True)[:5]]}")
    print(f"  Top 5 losses: {[round(p,2) for p in sorted(pnls)[:5]]}")
    print()

    # GP analysis
    hgp=[(t.get("guaranteed_profit",0),t["_w"],t.get("pnl",0)) for t in trades if t.get("guaranteed_profit",0)>0.4]
    lgp=[(t.get("guaranteed_profit",0),t["_w"],t.get("pnl",0)) for t in trades if t.get("guaranteed_profit",0)<=0.4]
    if hgp:
        hw=sum(1 for g in hgp if g[1])
        print(f"  GP>0.4: {hw}/{len(hgp)} ({hw/len(hgp)*100:.0f}%) ${sum(g[2] for g in hgp):+.2f}")
    if lgp:
        lw=sum(1 for g in lgp if g[1])
        print(f"  GP<=0.4: {lw}/{len(lgp)} ({lw/len(lgp)*100:.0f}%) ${sum(g[2] for g in lgp):+.2f}")
    print()

    # Loss autopsy
    print("  --- LOSS AUTOPSY ---")
    print(f"  {'#':>3}{'Side':>5}{'Coin':>5}{'Ent':>6}{'Exit':>6}{'PnL':>9}{'Edge':>7}{'KL':>7}{'Hr':>4}")
    print("  "+"-"*55)
    for i,t in enumerate(sorted(losses,key=lambda x:x.get("pnl",0))):
        print(f"  {i+1:>3}{t.get('side','?'):>5}{t['_a']:>5}{t.get('entry_price',0):>6.2f}{t.get('exit_price',0):>6.2f}${t.get('pnl',0):>+7.2f}{t.get('edge_at_entry',0):>7.3f}{t.get('kl_divergence',0):>7.3f}{t['_h']:>4}")
    lsides=[t.get("side") for t in losses]
    lassets=[t["_a"] for t in losses]
    lp=[t.get("entry_price",0) for t in losses]
    wp=[t.get("entry_price",0) for t in wins]
    print(f"\n  Losses: UP={lsides.count('UP')}, DOWN={lsides.count('DOWN')}")
    print(f"  Loss assets: BTC={lassets.count('BTC')}, ETH={lassets.count('ETH')}, SOL={lassets.count('SOL')}")
    print(f"  Loss avg entry: ${sum(lp)/len(lp):.3f} | Win avg entry: ${sum(wp)/len(wp):.3f}")
    print()

    # SECTION 10: RECOMMENDATIONS
    print(sep); print("  10. SPECIFIC RECOMMENDATIONS"); print(sep); print()

    h5=hs.get("5",{})
    h5p=h5.get("pnl",0)
    print("  [A] ADD HOUR 5 UTC TO SKIP_HOURS")
    print(f"      Hour 5: {h5.get('wins',0)}W/{h5.get('losses',0)}L, ${h5p:+.2f}")
    print(f"      100%% loss rate. Instant save: ${abs(h5p):.2f}")
    print()

    ma=ss.get("mtf_agree",{}); md=ss.get("mtf_disagree",{})
    mat=ma.get("trades",0); maw=ma.get("wins",0); mal=ma.get("losses",0); map_=ma.get("pnl",0)
    mdt=md.get("trades",0); mdw=md.get("wins",0); mdl=md.get("losses",0); mdp=md.get("pnl",0)
    print("  [B] HARD-BLOCK MTF_DISAGREE TRADES")
    print(f"      mtf_agree:    {mat}t, {maw}W/{mal}L ({maw/max(mat,1)*100:.0f}%%) ${map_:+.2f}")
    print(f"      mtf_disagree: {mdt}t, {mdw}W/{mdl}L ({mdw/max(mdt,1)*100:.0f}%%) ${mdp:+.2f}")
    print(f"      BEST FILTER. Save ${abs(mdp):.2f}")
    print()

    vs=ss.get("volspike",{}); vn=ss.get("volspike_normal",{})
    print("  [C] REDUCE VOLSPIKE TRADE SIZE")
    print(f"      volspike: {vs.get('trades',0)}t, ${vs.get('pnl',0):+.2f}")
    print(f"      normal:   {vn.get('trades',0)}t, ${vn.get('pnl',0):+.2f}")
    print(f"      Cut volspike size 50%% or require extra confirm")
    print()

    dc=[t for t in trades if t.get("side")=="DOWN" and t.get("entry_price",0)<0.35]
    if dc:
        dcw=sum(1 for t in dc if t["_w"]); dcp=sum(t["pnl"] for t in dc)
        print("  [D] BLOCK DOWN ENTRIES < $0.35")
        print(f"      {len(dc)}t, {dcw}W/{len(dc)-dcw}L ({dcw/len(dc)*100:.0f}%%), ${dcp:+.2f}")
        print()

    ut=[t for t in trades if t.get("side")=="UP"]
    dt_=[t for t in trades if t.get("side")=="DOWN"]
    uwr=sum(1 for t in ut if t["_w"])/len(ut)*100 if ut else 0
    dwr=sum(1 for t in dt_ if t["_w"])/len(dt_)*100 if dt_ else 0
    up_=sum(t["pnl"] for t in ut); dp_=sum(t["pnl"] for t in dt_)
    print("  [E] UP OUTPERFORMS - RELAX UP REQUIREMENTS")
    print(f"      UP:   {len(ut)}t, {uwr:.0f}%% WR, ${up_:+.2f}, avg ${up_/len(ut):+.2f}/trade")
    print(f"      DOWN: {len(dt_)}t, {dwr:.0f}%% WR, ${dp_:+.2f}, avg ${dp_/len(dt_):+.2f}/trade")
    print(f"      Market regime may have shifted from historical DOWN bias")
    print()

    le=[t for t in trades if t.get("edge_at_entry",0)<0.35]
    he=[t for t in trades if t.get("edge_at_entry",0)>=0.35]
    print("  [F] EDGE THRESHOLD")
    if le:
        lewr=sum(1 for t in le if t["_w"])/len(le)*100
        print(f"      Edge<0.35: {len(le)}t, {lewr:.0f}%%, ${sum(t['pnl'] for t in le):+.2f}")
    if he:
        hewr=sum(1 for t in he if t["_w"])/len(he)*100
        print(f"      Edge>=0.35: {len(he)}t, {hewr:.0f}%%, ${sum(t['pnl'] for t in he):+.2f}")
    print()

    vm=ss.get("volregime_match",{}); vmm=ss.get("volregime_mismatch",{})
    print("  [G] VOLREGIME MISMATCH ACTUALLY WINS (KEEP IT)")
    print(f"      match:    {vm.get('trades',0)}t, {vm.get('wins',0)}W/{vm.get('losses',0)}L, ${vm.get('pnl',0):+.2f}")
    print(f"      mismatch: {vmm.get('trades',0)}t, {vmm.get('wins',0)}W/{vmm.get('losses',0)}L, ${vmm.get('pnl',0):+.2f}")
    print(f"      Do NOT penalize mismatch - may capture regime transitions")
    print()

    # EXECUTIVE SUMMARY
    print(sep); print("  EXECUTIVE SUMMARY"); print(sep); print()
    h5s=abs(h5p); mtfs=abs(mdp); tots=h5s+mtfs
    rt=N-h5.get("losses",0)-md.get("losses",0)
    nwr=len(wins)/rt*100 if rt else 0
    print(f"  Current:   {N}t, {len(wins)}W/{len(losses)}L ({wr:.1f}%%), ${tpnl:+.2f}")
    print(f"  Projected: {rt}t, {len(wins)}W/{rt-len(wins)}L ({nwr:.1f}%%), ${tpnl+tots:+.2f}")
    print(f"  PnL boost: +${tots:.2f} from blocking hour5 + mtf_disagree")
    print()
    print("  TOP 3 IMMEDIATE ACTIONS:")
    print(f"    1. skip_hours.add(5)               -> saves ${h5s:.2f}")
    print(f"    2. Block mtf_disagree trades        -> saves ${mtfs:.2f}")
    print(f"    3. Reduce volspike size by 50%%     -> reduces variance")
    print()
    print("  PARAMETER CHANGES:")
    print("    - skip_hours: add 5")
    print("    - mtf_filter: HARD BLOCK when mtf_agree=False")
    print("    - volspike_size_mult: 0.5")
    print("    - DOWN max_entry: $0.55 -> $0.50")
    print("    - UP: relax requirements (higher max_price, lower min_edge)")
    print("    - Keep vol_regime_mismatch trades")
    print()

if __name__=="__main__":
    main()
