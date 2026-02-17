"""
Whale Watcher - Paper trade all tracked whales and log to SQLite for ML analysis.
No real trades executed. Monitors activity + positions for PnL tracking.
"""

import asyncio
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arbitrage.copy_trader import CopyTrader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whale_watcher.db")
DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
POLL_SEC = 30          # seconds between full whale scan
SNAPSHOT_SEC = 300     # position snapshots every 5 min
REPORT_SEC = 1800      # console summary every 30 min
SWEEP_SEC = 600        # resolve stale open trades every 10 min
ACTIVITY_LIMIT = 15    # recent activities per whale per poll

WHALES = CopyTrader.WHALES  # {name: address, ...}


def log(msg: str):
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {safe}", flush=True)


# ---------------------------------------------------------------------------
# Market category classifier
# ---------------------------------------------------------------------------
_CRYPTO_COINS = re.compile(r"\b(BTC|Bitcoin|ETH|Ethereum|SOL|Solana|XRP|DOGE|ADA|MATIC|AVAX|BNB)\b", re.I)
_SPORTS = re.compile(r"\b(NBA|NFL|NHL|MLB|soccer|tennis|Premier League|La Liga|Serie A|UFC|boxing|Super Bowl)\b", re.I)
_POLITICS = re.compile(r"\b(president|election|Trump|Biden|congress|senate|governor|RFK|Vance|DeSantis)\b", re.I)


def classify_market(title: str) -> str:
    t = title or ""
    up_down = bool(re.search(r"[Uu]p or [Dd]own", t))
    if up_down:
        if re.search(r"15\s*M|15-?[Mm]in", t) or re.search(r"\d{1,2}:\d{2}\s*(AM|PM)", t):
            return "crypto_15m"
        return "crypto_hourly"
    if _CRYPTO_COINS.search(t):
        return "crypto_other"
    if _SPORTS.search(t):
        return "sports"
    if _POLITICS.search(t):
        return "politics"
    return "other"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS whale_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       REAL    NOT NULL,
            whale_name      TEXT    NOT NULL,
            whale_address   TEXT    NOT NULL,
            tx_hash         TEXT    UNIQUE,
            market_title    TEXT,
            condition_id    TEXT,
            token_id        TEXT,
            outcome         TEXT,
            side            TEXT,
            price           REAL,
            size            REAL,
            cost_usd        REAL,
            paper_entry_price REAL,
            paper_exit_price  REAL,
            paper_pnl       REAL,
            status          TEXT    DEFAULT 'open',
            market_category TEXT,
            resolved_at     REAL
        );

        CREATE INDEX IF NOT EXISTS idx_wt_whale   ON whale_trades(whale_name);
        CREATE INDEX IF NOT EXISTS idx_wt_status  ON whale_trades(status);
        CREATE INDEX IF NOT EXISTS idx_wt_cat     ON whale_trades(market_category);
        CREATE INDEX IF NOT EXISTS idx_wt_cond    ON whale_trades(condition_id, whale_name);

        CREATE TABLE IF NOT EXISTS whale_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       REAL    NOT NULL,
            whale_name      TEXT    NOT NULL,
            total_positions INTEGER,
            total_value_usd REAL,
            positions_json  TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_ws_whale ON whale_snapshots(whale_name);

        CREATE VIEW IF NOT EXISTS whale_stats AS
        SELECT
            whale_name,
            market_category,
            COUNT(*)                                          AS total_trades,
            SUM(CASE WHEN paper_pnl > 0 THEN 1 ELSE 0 END)  AS wins,
            SUM(CASE WHEN paper_pnl <= 0 THEN 1 ELSE 0 END) AS losses,
            ROUND(SUM(CASE WHEN paper_pnl > 0 THEN 1.0 ELSE 0 END) * 100.0 / MAX(COUNT(*), 1), 1) AS win_rate,
            ROUND(SUM(paper_pnl), 2)                         AS total_pnl,
            ROUND(AVG(cost_usd), 2)                          AS avg_trade_size,
            ROUND(AVG(paper_pnl), 2)                         AS avg_pnl
        FROM whale_trades
        WHERE status IN ('closed', 'resolved')
        GROUP BY whale_name, market_category;
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Core watcher
# ---------------------------------------------------------------------------
class WhaleWatcher:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.seen_tx: dict[str, set[str]] = {name: set() for name in WHALES}
        self.known_positions: dict[str, set[str]] = {name: set() for name in WHALES}
        self._resolution_cache: dict[str, dict | None] = {}  # condition_id -> {outcome: price} or None
        self._load_seen()

    def _load_seen(self):
        """Load already-recorded tx hashes so we don't duplicate on restart."""
        rows = self.db.execute("SELECT whale_name, tx_hash FROM whale_trades WHERE tx_hash IS NOT NULL").fetchall()
        for name, tx in rows:
            if name in self.seen_tx:
                self.seen_tx[name].add(tx)
        total = sum(len(s) for s in self.seen_tx.values())
        log(f"Loaded {total} existing tx hashes from DB")

        # Load open positions per whale
        rows = self.db.execute(
            "SELECT whale_name, condition_id FROM whale_trades WHERE status = 'open'"
        ).fetchall()
        for name, cid in rows:
            if name in self.known_positions:
                self.known_positions[name].add(cid)

    # ----- market resolution -----

    async def resolve_market_outcome(self, client: httpx.AsyncClient, condition_id: str) -> dict | None:
        """Query gamma-api for actual market resolution outcome.

        Returns dict mapping outcome name -> final price (e.g. {"Up": 1.0, "Down": 0.0})
        or None if market is not yet resolved.
        """
        if condition_id in self._resolution_cache:
            return self._resolution_cache[condition_id]

        try:
            r = await client.get(
                f"{GAMMA_API}/markets",
                params={"condition_id": condition_id},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                data = data[0] if data else None
            if not data:
                return None

            prices = data.get("outcomePrices", "")
            outcomes = data.get("outcomes", "")
            if isinstance(prices, str):
                prices = json.loads(prices) if prices else []
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes) if outcomes else []

            if not prices or not outcomes or len(prices) < 2:
                return None

            result = {}
            resolved = False
            for i, outcome_name in enumerate(outcomes):
                if i < len(prices):
                    p = float(prices[i])
                    result[outcome_name] = p
                    if p >= 0.95 or p <= 0.05:
                        resolved = True

            if resolved:
                self._resolution_cache[condition_id] = result
                return result

            return None
        except Exception:
            return None

    # ----- activity polling -----

    async def poll_whale_activity(self, client: httpx.AsyncClient, name: str, address: str):
        """Fetch recent activity for one whale, record new trades."""
        try:
            r = await client.get(
                f"{DATA_API}/activity",
                params={"user": address, "limit": ACTIVITY_LIMIT},
            )
            r.raise_for_status()
            activities = r.json()
        except Exception as e:
            log(f"  {name}: activity error - {e}")
            return

        now = time.time()
        new_count = 0
        for act in activities:
            tx = act.get("transactionHash", "")
            if not tx or tx in self.seen_tx[name]:
                continue
            self.seen_tx[name].add(tx)
            new_count += 1

            title = act.get("title", "")
            outcome = act.get("outcome", "")
            price = float(act.get("price", 0) or 0)
            size = float(act.get("size", 0) or 0)
            side = act.get("side", "")
            token = act.get("asset", "")
            cond = act.get("conditionId", "")
            cost = price * size
            cat = classify_market(title)

            self.db.execute("""
                INSERT OR IGNORE INTO whale_trades
                (timestamp, whale_name, whale_address, tx_hash, market_title,
                 condition_id, token_id, outcome, side, price, size, cost_usd,
                 paper_entry_price, status, market_category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, name, address, tx, title,
                cond, token, outcome, side, price, size, cost,
                price, "open" if side == "BUY" else "closed", cat,
            ))

            if side == "BUY":
                self.known_positions[name].add(cond)
                log(f"  NEW: {name} BUY {outcome} @ ${price:.2f} ({title[:45]}) ${cost:.2f}")
            else:
                log(f"  SELL: {name} SELL {outcome} @ ${price:.2f} ({title[:45]})")

        if new_count:
            self.db.commit()

    # ----- position resolution -----

    async def check_positions(self, client: httpx.AsyncClient, name: str, address: str):
        """Check positions to detect closures/resolutions for PnL calc."""
        try:
            r = await client.get(
                f"{DATA_API}/positions",
                params={"user": address, "sizeThreshold": 0},
            )
            r.raise_for_status()
            positions = r.json()
        except Exception:
            return

        current_cids = {p.get("conditionId", "") for p in positions}
        closed = self.known_positions[name] - current_cids

        now = time.time()
        resolved_any = False
        retry_cids = set()
        for cid in closed:
            rows = self.db.execute(
                "SELECT id, outcome, paper_entry_price, size, side FROM whale_trades "
                "WHERE whale_name = ? AND condition_id = ? AND status = 'open'",
                (name, cid),
            ).fetchall()
            if not rows:
                continue

            # Query actual market outcome
            outcome_prices = await self.resolve_market_outcome(client, cid)
            if outcome_prices is None:
                # Market not resolved yet (whale may have sold) — keep tracking
                retry_cids.add(cid)
                continue

            for tid, outcome, entry_p, sz, side in rows:
                if side != "BUY":
                    continue
                outcome_price = outcome_prices.get(outcome, 0.0)
                if outcome_price >= 0.95:
                    exit_price = 1.0
                    pnl = (1.0 - entry_p) * sz
                else:
                    exit_price = 0.0
                    pnl = -(entry_p * sz)

                self.db.execute(
                    "UPDATE whale_trades SET status = 'resolved', paper_exit_price = ?, paper_pnl = ?, resolved_at = ? "
                    "WHERE id = ?",
                    (exit_price, pnl, now, tid),
                )
                log(f"  CLOSE: {name} {outcome} {'WIN' if exit_price > 0.5 else 'LOSS'} - Paper PnL: ${pnl:+.2f}")
                resolved_any = True

        # Remove resolved cids, keep unresolved ones for retry next cycle
        self.known_positions[name] = (self.known_positions[name] & current_cids) | retry_cids
        if resolved_any:
            self.db.commit()

    # ----- stale trade sweep -----

    async def sweep_stale_trades(self, client: httpx.AsyncClient):
        """Resolve old open trades by querying gamma-api for market outcomes."""
        cutoff = time.time() - 7200  # trades older than 2 hours
        rows = self.db.execute(
            "SELECT DISTINCT condition_id FROM whale_trades "
            "WHERE status = 'open' AND side = 'BUY' AND timestamp < ?",
            (cutoff,),
        ).fetchall()

        unique_cids = [r[0] for r in rows if r[0]]
        if not unique_cids:
            return

        log(f"[SWEEP] Checking {len(unique_cids)} stale condition_ids...")
        resolved_wins = 0
        resolved_losses = 0
        resolved_trades = 0
        now = time.time()
        api_calls = 0

        for cid in unique_cids:
            if api_calls >= 60:
                log(f"[SWEEP] Rate limit reached, continuing next cycle")
                break
            api_calls += 1

            outcome_prices = await self.resolve_market_outcome(client, cid)
            await asyncio.sleep(0.2)

            if outcome_prices is None:
                continue

            trades = self.db.execute(
                "SELECT id, whale_name, outcome, paper_entry_price, size FROM whale_trades "
                "WHERE condition_id = ? AND status = 'open' AND side = 'BUY'",
                (cid,),
            ).fetchall()

            for tid, whale_name, outcome, entry_p, sz in trades:
                outcome_price = outcome_prices.get(outcome, 0.0)
                if outcome_price >= 0.95:
                    exit_price = 1.0
                    pnl = (1.0 - entry_p) * sz
                    resolved_wins += 1
                else:
                    exit_price = 0.0
                    pnl = -(entry_p * sz)
                    resolved_losses += 1
                resolved_trades += 1

                self.db.execute(
                    "UPDATE whale_trades SET status = 'resolved', paper_exit_price = ?, paper_pnl = ?, resolved_at = ? "
                    "WHERE id = ?",
                    (exit_price, pnl, now, tid),
                )

                # Remove from known_positions if present
                if whale_name in self.known_positions:
                    self.known_positions[whale_name].discard(cid)

        if resolved_trades:
            self.db.commit()
            log(f"[SWEEP] Resolved {resolved_trades} trades ({resolved_wins}W/{resolved_losses}L) from {api_calls} markets checked")
        else:
            log(f"[SWEEP] Checked {api_calls} markets, none newly resolved")

    # ----- snapshots -----

    async def snapshot_whale(self, client: httpx.AsyncClient, name: str, address: str):
        """Take a portfolio snapshot for one whale."""
        try:
            r = await client.get(f"{DATA_API}/positions", params={"user": address, "sizeThreshold": 0})
            r.raise_for_status()
            positions = r.json()
        except Exception:
            return

        total_val = sum(float(p.get("currentValue", 0) or 0) for p in positions)
        self.db.execute(
            "INSERT INTO whale_snapshots (timestamp, whale_name, total_positions, total_value_usd, positions_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), name, len(positions), total_val, json.dumps(positions)),
        )

    # ----- reporting -----

    def print_report(self):
        rows = self.db.execute("""
            SELECT whale_name,
                   COUNT(*) AS trades,
                   SUM(CASE WHEN paper_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN paper_pnl IS NOT NULL AND paper_pnl <= 0 THEN 1 ELSE 0 END) AS losses,
                   ROUND(COALESCE(SUM(paper_pnl), 0), 2) AS pnl,
                   SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) AS open_trades
            FROM whale_trades
            GROUP BY whale_name
            ORDER BY pnl DESC
        """).fetchall()

        log("=" * 65)
        log(f"{'WHALE':<20} {'Trades':>6} {'W/L':>7} {'PnL':>10} {'Open':>5}")
        log("-" * 65)
        total_pnl = 0
        for name, trades, wins, losses, pnl, open_t in rows:
            wr = f"{wins}/{losses}"
            total_pnl += pnl or 0
            log(f"  {name:<18} {trades:>6} {wr:>7} ${pnl or 0:>+9.2f} {open_t:>5}")
        log("-" * 65)
        log(f"  {'TOTAL':<18} {'':>6} {'':>7} ${total_pnl:>+9.2f}")
        log("=" * 65)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def main():
    log("WHALE WATCHER - Paper Mode")
    log(f"Tracking {len(WHALES)} whales | DB: {DB_PATH}")
    log(f"Poll: {POLL_SEC}s | Snapshots: {SNAPSHOT_SEC}s | Reports: {REPORT_SEC}s")
    log("")

    db = init_db(DB_PATH)
    watcher = WhaleWatcher(db)

    whale_list = list(WHALES.items())
    last_snapshot = 0
    last_report = 0
    last_sweep = 0
    delay_per_whale = max(1.0, POLL_SEC / len(whale_list))  # stagger requests

    async with httpx.AsyncClient(timeout=30) as client:
        # Startup sweep: resolve trades that closed while watcher was offline
        log("[STARTUP] Running initial stale-trade sweep...")
        await watcher.sweep_stale_trades(client)

        while True:
            cycle_start = time.time()

            # --- Poll activity for each whale (staggered) ---
            for name, address in whale_list:
                await watcher.poll_whale_activity(client, name, address)
                await watcher.check_positions(client, name, address)
                await asyncio.sleep(delay_per_whale)

            # --- Periodic stale-trade sweep ---
            now = time.time()
            if now - last_sweep >= SWEEP_SEC:
                await watcher.sweep_stale_trades(client)
                last_sweep = now

            # --- Periodic snapshots ---
            now = time.time()
            if now - last_snapshot >= SNAPSHOT_SEC:
                log("Taking portfolio snapshots...")
                for name, address in whale_list:
                    await watcher.snapshot_whale(client, name, address)
                    await asyncio.sleep(0.5)
                db.commit()
                last_snapshot = now
                log(f"Snapshots saved for {len(whale_list)} whales")

            # --- Periodic report ---
            if now - last_report >= REPORT_SEC:
                watcher.print_report()
                last_report = now

            # Sleep remainder of poll interval
            elapsed = time.time() - cycle_start
            sleep_time = max(1, POLL_SEC - elapsed)
            await asyncio.sleep(sleep_time)


if __name__ == "__main__":
    from pid_lock import acquire_pid_lock, release_pid_lock
    acquire_pid_lock("whale_watcher")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Whale watcher stopped.")
    finally:
        release_pid_lock("whale_watcher")
