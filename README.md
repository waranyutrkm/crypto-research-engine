# Binance Research Engine v3.5.2  
### Liquidity-Driven Crypto Strategy Research (Thai Edition)

An **interactive, browser-based crypto research & backtesting engine**  
designed for **liquidity-aware portfolio construction, strategy grid search,  
risk analysis, and robustness testing** — built entirely with client-side technology.

> ⚡ No backend required  
> ⚡ Free hosting via GitHub Pages  
> ⚡ Binance API + Mock fallback (CORS-safe)

---

## 🌐 Live Demo

👉 **https://trkmwaranyu.github.io/binance-research-engine/**

---

## 🎯 Project Objective

This project was built to answer a key research question:

> **“How do different portfolio construction strategies perform  
> when the tradable universe itself changes dynamically  
> based on real liquidity?”**

Instead of assuming a fixed universe, the engine:
- Continuously **re-selects assets by Dollar Volume**
- Tests strategies across **multiple universe sizes**
- Evaluates **robustness**, not just raw returns

---

## 🧠 Core Concepts

### 1. Liquidity-Based Dynamic Universe
- Scans Binance spot market
- Filters real tradable pairs (ex-stablecoins, leverage tokens)
- Ranks by **30-Day Average Dollar Volume**
- Rebuilds universe every *N days*

### 2. Strategy Grid Search
- Multiple strategies
- Multiple lookback windows
- Multiple rebalance frequencies
- Multiple universe sizes

➡️ Produces a **multi-dimensional performance landscape**

### 3. Robustness over Optimization
Instead of chasing peak Sharpe only, the engine emphasizes:
- Win rate consistency
- Drawdown behavior
- Distribution shape
- Parameter stability

---

## ⚙️ Implemented Strategies

| Strategy | Description |
|--------|------------|
| Equal | Equal-weighted portfolio |
| Rank | Momentum rank-weighted |
| Top3 | Top 3 momentum leaders |
| Top50Rank | Top 50% ranked assets |
| MomWeight | Momentum-proportional weights |
| InvVol | Inverse volatility weighting |
| AAA | Adaptive Aggressive Allocation |

---

## 📊 Metrics & Analytics

### Performance Metrics
- CAGR
- Sharpe Ratio (rolling)
- Win-Rate (30D)
- Max Drawdown
- Annualized Volatility

### Risk & Robustness
- Robust Score (Sharpe + Win-Rate blend)
- Distribution analysis (±1σ / ±2σ / ±3σ)
- Worst-month detection
- Parameter surface smoothness

### Visualization
- Equity curve (strategy vs benchmark, re-based)
- Drawdown curve
- 3D parameter landscape
- Risk distribution histogram
- Monte Carlo forward simulation

---

## 🔬 Monte Carlo Engine

- Log-normal price simulation
- Parameters inferred from historical strategy metrics
- 100 simulation paths
- Outputs:
  - Expected terminal value
  - Confidence bands
  - Expected drawdown
  - Forward CAGR estimate

---

## 🧩 System Architecture

```

Browser (Client-Side Only)
│
├─ Binance REST API (Spot / Klines / Ticker)
│   └─ Auto fallback → Mock generator
│
├─ Data Alignment Engine
│   ├─ Time-index normalization
│   └─ Dollar Volume computation
│
├─ Strategy Engine
│   ├─ Dynamic universe selection
│   ├─ Weight computation
│   └─ Transaction cost modeling
│
├─ Analytics Layer
│   ├─ Metrics
│   ├─ Robust scoring
│   └─ Monte Carlo
│
└─ Visualization Layer
├─ Chart.js
└─ Plotly (3D)

```

---

## 🛠 Tech Stack

- **HTML5 / Vanilla JavaScript**
- **Tailwind CSS**
- **Chart.js**
- **Plotly.js**
- **Binance Public API**
- **GitHub Pages (Free Hosting)**

No frameworks. No build step. No backend.

---

## 🚀 How to Use

1. Open the Live Demo
2. Click **“Scan Top 50 Liquid Pairs”**
3. Configure:
   - Universe Size Grid
   - Lookback Range
   - Rebalance Frequency
4. Click **“Run Liquid Grid”**
5. Explore:
   - Leaderboard
   - Charts
   - 3D landscape
   - Logs
   - Monte Carlo forecast

---

## ⚠️ Important Notes

- Binance API may be blocked by browser CORS  
  → The system **automatically switches to mock data**
- Results are for **research & educational purposes**
- No guarantees of future performance

---

## 🧪 Limitations

- Client-side only (CPU-bound)
- Not designed for tick-level data
- Simplified fee & slippage model
- No order book simulation

---

## 🛣 Roadmap (Future Ideas)

- Web Workers for faster grid search
- CSV / JSON export
- Walk-forward validation
- Regime detection layer
- Multi-benchmark comparison
- Python / FastAPI backend version

---

## ⚖️ Disclaimer

This project is provided **for research, learning, and experimentation only**.  
It does **not constitute financial advice**.  
Trading cryptocurrencies involves substantial risk.

---

## 👤 Author

**trkmwaranyu**  
Independent Research / Quant-Style Exploration

---

## 📜 License

MIT License  
Free to use, modify, and distribute with attribution.
``
