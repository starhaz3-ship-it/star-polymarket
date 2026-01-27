# Polymarket Development Knowledge Base

## Official Repositories (93+ repos)

### Core Trading Clients

| Repository | Language | Purpose |
|------------|----------|---------|
| [py-clob-client](https://github.com/Polymarket/py-clob-client) | Python | Official CLOB trading client |
| [clob-client](https://github.com/Polymarket/clob-client) | TypeScript | Official CLOB trading client |
| [agents](https://github.com/Polymarket/agents) | Python | AI agents framework for autonomous trading |
| [real-time-data-client](https://github.com/Polymarket/real-time-data-client) | TypeScript | WebSocket streaming client |

### API Endpoints

```
CLOB API:    https://clob.polymarket.com
Gamma API:   https://gamma-api.polymarket.com
Data API:    https://data-api.polymarket.com
WebSocket:   wss://ws-subscriptions-clob.polymarket.com/ws/
RTDS:        wss://ws-live-data.polymarket.com
```

### Authentication Types

| signature_type | Wallet Type | Description |
|----------------|-------------|-------------|
| 0 | EOA | Direct wallet (MetaMask, hardware) |
| 1 | Magic | Email/Magic wallet (delegated signing) |
| 2 | Proxy | Browser wallet proxy signatures |

### Client Initialization Patterns

**Python (Magic Wallet):**
```python
from py_clob_client.client import ClobClient

client = ClobClient(
    host="https://clob.polymarket.com",
    key=private_key,
    chain_id=137,
    signature_type=1,  # Magic wallet
    funder=proxy_address
)
creds = client.derive_api_key()
client.set_api_creds(creds)
```

**TypeScript:**
```typescript
const client = new ClobClient(host, 137, signer, credentials, signatureType, funder);
```

### Order Placement

```python
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

order_args = OrderArgs(
    price=0.50,      # Price per share (0-1)
    size=100,        # Number of shares
    side=BUY,        # BUY or SELL
    token_id=token_id
)

signed_order = client.create_order(order_args)
response = client.post_order(signed_order, OrderType.GTC)
```

### Data API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/positions?user=` | Current positions |
| `/trades?user=` | Trade history |
| `/activity?user=` | On-chain activity |
| `/holders?market=` | Top market holders |
| `/value?user=` | Total position value |

### Gamma API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/markets` | List markets (filter: active, closed, order, limit) |
| `/markets/{id}` | Get market by condition ID |
| `/events` | List events |
| `/events/{id}` | Get event details |
| `/search?query=` | Search markets/events |

### CLOB API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/price?token_id=` | Current token price |
| `/book?token_id=` | Order book |
| `/midpoint?token_id=` | Midpoint price |
| POST `/order` | Place order (auth required) |
| DELETE `/order` | Cancel order (auth required) |

---

## AI Agents Framework

**Repository:** https://github.com/Polymarket/agents

### Architecture
- `Chroma.py` - Vector database for news
- `Gamma.py` - Market metadata
- `Polymarket.py` - DEX order execution
- `cli.py` - Command line interface

### Quick Start
```bash
git clone https://github.com/Polymarket/agents.git
pip install -r requirements.txt
cp .env.example .env
# Set POLYGON_WALLET_PRIVATE_KEY and OPENAI_API_KEY
python agents/application/trade.py
```

---

## Arbitrage Tools

### Cross-Platform (Polymarket + Kalshi)
- [polymarket-kalshi-btc-arbitrage-bot](https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot)
- Monitors BTC 1-hour price markets
- Signals when combined cost < $1.00

### Single-Market
- [polymarket-arbitrage-bot](https://github.com/0xalberto/polymarket-arbitrage-bot)
- NegRisk rebalancing
- Whale tracking

**Note:** $39.59M in arbitrage extracted from Polymarket (April 2024 - April 2025)

---

## Market Making

**Repository:** https://github.com/warproxxx/poly-maker

**Warning:** Author states "This bot is not profitable and will lose money"

Features:
- Two-sided liquidity provision
- Google Sheets configuration
- Position merging (gas savings)
- WebSocket monitoring

---

## Subgraph (On-Chain Data)

**Repository:** https://github.com/Polymarket/polymarket-subgraph

Available subgraphs:
- activity-subgraph
- fpmm-subgraph
- oi-subgraph (open interest)
- orderbook-subgraph
- pnl-subgraph
- wallet-subgraph
- sports-oracle-subgraph

---

## WebSocket Channels

### CLOB WebSocket
```
wss://ws-subscriptions-clob.polymarket.com/ws/
```

**Channels:**
- `market` (public) - Order book updates
- `user` (authenticated) - Order status

### RTDS WebSocket
```
wss://ws-live-data.polymarket.com
```

**Message Types:**
- Activity (trades)
- Comments
- RFQ (requests/quotes)
- Crypto/equity prices

---

## Fee Structure

Current: **0 bps** for all volume levels

Formula: `baseRate × min(price, 1-price) × size`

---

## Important Notes

1. **Geographic Restrictions:** Trading restricted for US persons
2. **Token Allowances:** MetaMask users must approve USDC + Conditional Tokens on 3 exchange contracts
3. **Magic Wallet:** Email users skip allowance setup
4. **Settlement:** On-chain on Polygon (chain_id: 137)

---

## Documentation Links

- [Official Docs](https://docs.polymarket.com/)
- [CLOB Introduction](https://docs.polymarket.com/developers/CLOB/introduction)
- [Quickstart](https://docs.polymarket.com/developers/CLOB/quickstart)
- [API Reference](https://docs.polymarket.com/quickstart/reference/endpoints)
- [WebSocket Docs](https://docs.polymarket.com/developers/CLOB/websocket/wss-overview)
- [Data API Gist](https://gist.github.com/shaunlebron/0dd3338f7dea06b8e9f8724981bb13bf)
