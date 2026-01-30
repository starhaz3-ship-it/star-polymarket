"""
Bot Comparison Report with GUI Graph
Compares paper trading performance of 0x8dxd vs k9Q2mX4L8A7ZP3R
"""

import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

# State files
STATE_FILES = {
    '0x8dxd': 'paper_0x8dxd_state.json',
    'k9Q2mX4L8A7ZP3R': 'paper_k9Q2_state.json',
}

STARTING_BALANCE = 1000.0

def load_bot_data(name, filename):
    """Load bot state from file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return {
                'name': name,
                'balance': data.get('paper_balance', STARTING_BALANCE),
                'trades': data.get('paper_trades', []),
                'starting': data.get('starting_balance', STARTING_BALANCE),
            }
    except FileNotFoundError:
        return {
            'name': name,
            'balance': STARTING_BALANCE,
            'trades': [],
            'starting': STARTING_BALANCE,
        }

def calculate_stats(bot_data):
    """Calculate statistics for a bot."""
    trades = bot_data['trades']
    closed = [t for t in trades if t.get('status') == 'CLOSED']
    open_trades = [t for t in trades if t.get('status') == 'OPEN']

    wins = [t for t in closed if t.get('pnl', 0) > 0]
    losses = [t for t in closed if t.get('pnl', 0) <= 0]

    total_pnl = sum(t.get('pnl', 0) for t in closed)
    win_rate = len(wins) / max(1, len(closed)) * 100
    roi = (bot_data['balance'] - bot_data['starting']) / bot_data['starting'] * 100

    avg_win = sum(t.get('pnl', 0) for t in wins) / max(1, len(wins))
    avg_loss = sum(t.get('pnl', 0) for t in losses) / max(1, len(losses))

    return {
        'name': bot_data['name'],
        'balance': bot_data['balance'],
        'starting': bot_data['starting'],
        'total_trades': len(trades),
        'closed_trades': len(closed),
        'open_trades': len(open_trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'roi': roi,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trades': trades,
    }

def build_equity_curve(trades, starting_balance):
    """Build equity curve over time."""
    if not trades:
        return [datetime.now()], [starting_balance]

    # Sort trades by timestamp
    sorted_trades = sorted(trades, key=lambda t: t.get('timestamp', 0))

    times = [datetime.fromtimestamp(sorted_trades[0].get('timestamp', 0))]
    equity = [starting_balance]

    balance = starting_balance
    for trade in sorted_trades:
        ts = datetime.fromtimestamp(trade.get('timestamp', 0))
        cost = trade.get('cost', 0)
        pnl = trade.get('pnl', 0) if trade.get('status') == 'CLOSED' else 0

        balance -= cost
        if trade.get('status') == 'CLOSED':
            balance += cost + pnl

        times.append(ts)
        equity.append(balance)

    return times, equity

def generate_report():
    """Generate the comparison report with GUI."""
    print("=" * 60)
    print("BOT COMPARISON REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load data
    bots = {}
    for name, filename in STATE_FILES.items():
        bots[name] = load_bot_data(name, filename)

    # Calculate stats
    stats = {}
    for name, data in bots.items():
        stats[name] = calculate_stats(data)

    # Print text report
    print("\n" + "-" * 60)
    print("PERFORMANCE SUMMARY")
    print("-" * 60)

    for name, s in stats.items():
        print(f"\n{name}:")
        print(f"  Balance: ${s['balance']:.2f} (started ${s['starting']:.2f})")
        print(f"  ROI: {s['roi']:+.2f}%")
        print(f"  Total PnL: ${s['total_pnl']:+.2f}")
        print(f"  Trades: {s['total_trades']} ({s['closed_trades']} closed, {s['open_trades']} open)")
        print(f"  Wins: {s['wins']} | Losses: {s['losses']} | Win Rate: {s['win_rate']:.1f}%")
        print(f"  Avg Win: ${s['avg_win']:.2f} | Avg Loss: ${s['avg_loss']:.2f}")

    # Determine winner
    print("\n" + "=" * 60)
    winner = max(stats.values(), key=lambda x: x['roi'])
    loser = min(stats.values(), key=lambda x: x['roi'])

    if winner['roi'] > 0:
        print(f"WINNER: {winner['name']} with {winner['roi']:+.2f}% ROI")
    else:
        print(f"BEST PERFORMER: {winner['name']} with {winner['roi']:+.2f}% ROI")
        print(f"(Both bots lost money)")

    print(f"UNDERPERFORMER: {loser['name']} with {loser['roi']:+.2f}% ROI")
    print("=" * 60)

    # Create GUI graph
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Copy Trading Bot Comparison - 24 Hour Paper Trading', fontsize=14, fontweight='bold')

    colors = {'0x8dxd': '#2196F3', 'k9Q2mX4L8A7ZP3R': '#4CAF50'}

    # 1. Equity Curve
    ax1 = axes[0, 0]
    for name, data in bots.items():
        times, equity = build_equity_curve(data['trades'], STARTING_BALANCE)
        ax1.plot(times, equity, label=name, color=colors.get(name, 'gray'), linewidth=2)
    ax1.axhline(y=STARTING_BALANCE, color='red', linestyle='--', alpha=0.5, label='Starting Balance')
    ax1.set_title('Equity Curve Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Balance ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. ROI Comparison Bar Chart
    ax2 = axes[0, 1]
    names = list(stats.keys())
    rois = [stats[n]['roi'] for n in names]
    bars = ax2.bar(names, rois, color=[colors.get(n, 'gray') for n in names])
    ax2.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    ax2.set_title('Return on Investment (ROI)')
    ax2.set_ylabel('ROI (%)')
    for bar, roi in zip(bars, rois):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{roi:+.1f}%', ha='center', va='bottom', fontweight='bold')

    # 3. Win/Loss Pie Charts
    ax3 = axes[1, 0]
    ax3.set_title('Trade Outcomes')

    # Create side-by-side pie charts
    for i, name in enumerate(names):
        wins = stats[name]['wins']
        losses = stats[name]['losses']
        if wins + losses > 0:
            ax3.pie([wins, losses], labels=[f'Wins ({wins})', f'Losses ({losses})'],
                   colors=['#4CAF50', '#f44336'], autopct='%1.1f%%',
                   startangle=90, center=(i*2, 0), radius=0.8)
            ax3.text(i*2, -1.2, name, ha='center', fontweight='bold')
    ax3.set_xlim(-1, 3)
    ax3.set_ylim(-1.5, 1.5)
    ax3.axis('equal')

    # 4. Stats Table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = [
        ['Metric', '0x8dxd', 'k9Q2mX4L8A7ZP3R'],
        ['Balance', f"${stats['0x8dxd']['balance']:.2f}", f"${stats['k9Q2mX4L8A7ZP3R']['balance']:.2f}"],
        ['ROI', f"{stats['0x8dxd']['roi']:+.2f}%", f"{stats['k9Q2mX4L8A7ZP3R']['roi']:+.2f}%"],
        ['Total PnL', f"${stats['0x8dxd']['total_pnl']:+.2f}", f"${stats['k9Q2mX4L8A7ZP3R']['total_pnl']:+.2f}"],
        ['Total Trades', str(stats['0x8dxd']['total_trades']), str(stats['k9Q2mX4L8A7ZP3R']['total_trades'])],
        ['Win Rate', f"{stats['0x8dxd']['win_rate']:.1f}%", f"{stats['k9Q2mX4L8A7ZP3R']['win_rate']:.1f}%"],
        ['Avg Win', f"${stats['0x8dxd']['avg_win']:.2f}", f"${stats['k9Q2mX4L8A7ZP3R']['avg_win']:.2f}"],
        ['Avg Loss', f"${stats['0x8dxd']['avg_loss']:.2f}", f"${stats['k9Q2mX4L8A7ZP3R']['avg_loss']:.2f}"],
    ]

    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color header row
    for j in range(3):
        table[(0, j)].set_facecolor('#333333')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Highlight winner row
    winner_col = 1 if stats['0x8dxd']['roi'] > stats['k9Q2mX4L8A7ZP3R']['roi'] else 2
    for i in range(1, len(table_data)):
        table[(i, winner_col)].set_facecolor('#e8f5e9')

    plt.tight_layout()

    # Save and show
    report_file = f"bot_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(report_file, dpi=150, bbox_inches='tight')
    print(f"\nGraph saved to: {report_file}")

    plt.show()

    return stats

if __name__ == "__main__":
    generate_report()
