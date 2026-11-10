# GCN + CVaR Portfolio Optimisation (walk-forward)

A graph convolutional network learns how 30 US large caps co-move. Its node embeddings
shape the covariance matrix fed to a long-only mean-CVaR optimiser, which is re-solved
every month in a walk-forward backtest.

## Design

| Item | Setting |
|---|---|
| Universe | 5 sectors × 6 stocks = 30 (Technology, Healthcare, Financial, Consumer, Energy; first six names of each sector list, fixed in advance) |
| Data | Daily adjusted closes, 2015-01 – 2024-12 (Yahoo Finance via `yfinance`) |
| Walk-forward | Monthly rebalance, trailing 3-year (756-day) training window, no information after the rebalance date |
| Graph | Edge if \|correlation\| > 0.3 inside the training window |
| GCN | 2 layers (32 → 16), trained on monthly snapshots in the window to predict each stock's next 21-day return |
| Covariance | 50% sample covariance + 50% target built from cosine similarity of GCN embeddings × sample volatilities |
| Optimiser | Mean − 5 × CVaR(95%), Rockafellar–Uryasev LP in `cvxpy` (Clarabel), weights 0–15% |
| Costs | 10 bp one-way on turnover |

Hyper-parameters were set before the run and not tuned on the test period.

## Out-of-sample results (2018-02 – 2024-12, 83 rebalances)

| Strategy | Ann. return | Ann. vol | Sharpe | Sortino | Max DD | Daily CVaR 95% |
|---|---|---|---|---|---|---|
| GCN-CVaR | 6.6% | 17.5% | **0.46** | 0.56 | −36.3% | 2.6% |
| Sample-CVaR (same optimiser, no GCN) | 7.8% | 17.5% | 0.52 | 0.64 | −33.7% | 2.6% |
| Equal weight | 16.7% | 21.7% | 0.82 | 1.00 | −39.6% | 3.3% |

Sharpe is computed against a zero risk-free rate.

**Reading the results.** The CVaR objective dominates, so the portfolio behaves like a
minimum-tail-risk portfolio. It averaged 65% Healthcare, 27% Consumer and under 1% Technology.
It lowered daily CVaR from 3.3% to 2.6% and reduced the maximum drawdown slightly. Because it
was nearly absent from Technology during 2018–2024, it gave up most of that period's upside.
The GCN covariance did not improve on the plain sample covariance in this setup.

Full outputs are in `results/`: daily returns, monthly weights, the performance table and the run config.

## Run

```bash
pip install numpy pandas torch cvxpy yfinance
python gnn_cvar_walkforward.py      # about 5 minutes on CPU
```

Prices are downloaded on the first run and cached to `data/prices.csv`. The repo does not
include raw prices because Yahoo Finance terms do not allow redistribution.

## Other files

Earlier exploration (Mar–Jun 2025), kept for reference:
- `gnn_cvar_portfolio_v3.py`: first GCN prototype (single 80/20 split, `scipy` optimiser).
- `portfolio_final.py` / `.ipynb`: sector correlation analysis and classical portfolio baselines.
- `stock_network_analysis_v1.py`, `_v2.py`: correlation-network analysis.
- `figures/`: figures from those scripts.
- `models/best_gnn_model.pth`: weights from an earlier experiment. Its training code is not in this repo.
