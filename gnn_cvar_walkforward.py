"""GCN-based covariance + CVaR portfolio, walk-forward backtest.

Pipeline (re-run every month, using only data available at the rebalance date):
  1. Build a correlation graph from the trailing training window.
  2. Train a 2-layer GCN on that graph to predict each stock's next 21-day return
     from its trailing features. The GCN's node embeddings summarise how each stock
     co-moves with its neighbours.
  3. Covariance = sample covariance shrunk toward an embedding-based target
     (cosine similarity of GCN embeddings x sample volatilities).
  4. Long-only mean-CVaR optimisation (Rockafellar-Uryasev LP, cvxpy), with
     scenarios drawn from the training window and re-coloured to the GCN covariance.
  5. Hold for one month, charge transaction costs, roll forward.

Universe: 5 US sectors x 6 stocks = 30 stocks. The six are the first six tickers of each
sector list below, fixed before running. Hyper-parameters are fixed up front and are not
tuned on the test period.

Data: daily adjusted closes from Yahoo Finance via yfinance. Raw prices are not shipped
with the repo (Yahoo terms forbid redistribution); they are downloaded on first run and
cached in data/prices.csv (git-ignored).
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cvxpy as cp

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

SECTORS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA'],
    'Healthcare': ['JNJ', 'PFE', 'ABT', 'TMO', 'UNH', 'MRK'],
    'Financial':  ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS'],
    'Consumer':   ['AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW'],
    'Energy':     ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX'],
}
TICKERS = [t for v in SECTORS.values() for t in v]
START, END = '2015-01-01', '2024-12-31'

TRAIN_DAYS = 756          # 3-year trailing window
HORIZON = 21              # one month ahead
CORR_THRESHOLD = 0.3      # edge if |corr| > threshold
SHRINK = 0.5              # weight on GCN target in covariance
ALPHA = 0.95              # CVaR level
RISK_AVERSION = 5.0       # objective = mean - lambda * CVaR
W_MAX = 0.15              # per-stock cap
COST_BPS = 10             # one-way transaction cost
EPOCHS, HIDDEN, EMB, LR = 150, 32, 16, 0.01
RF_ANNUAL = 0.0           # Sharpe reported as excess over zero; see README

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'data', 'prices.csv')
OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data
if os.path.exists(CACHE):
    prices = pd.read_csv(CACHE, index_col=0, parse_dates=True)
else:
    import yfinance as yf
    prices = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)['Close']
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    prices.to_csv(CACHE)
prices = prices.loc[START:END, TICKERS].dropna()
rets = prices.pct_change().dropna()
print(f'prices {prices.index[0].date()} ~ {prices.index[-1].date()}, {rets.shape[1]} stocks, {len(rets)} days')


def node_features(r):
    """Per-stock features from a block of daily returns (rows = days)."""
    feats = [
        r.iloc[-21:].sum(), r.iloc[-63:].sum(), r.iloc[-252:].sum(),
        r.iloc[-21:].std(), r.iloc[-63:].std(),
        r.iloc[-63:].quantile(0.05),
    ]
    return np.stack([f.values for f in feats], axis=1)


def norm_adj(corr, thr):
    a = (corr.abs().values > thr).astype(float)
    np.fill_diagonal(a, 1.0)
    d = 1 / np.sqrt(a.sum(1))
    return torch.tensor(a * d[:, None] * d[None, :], dtype=torch.float32)


class GCN(nn.Module):
    def __init__(self, f_in):
        super().__init__()
        self.l1 = nn.Linear(f_in, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, EMB)
        self.head = nn.Linear(EMB, 1)

    def forward(self, x, A):
        h = torch.relu(A @ self.l1(x))
        z = A @ self.l2(h)
        return self.head(torch.relu(z)).squeeze(-1), z


def gcn_fit(train_r):
    """Train on monthly snapshots inside the window; return embeddings + forecast at its end."""
    A = norm_adj(train_r.corr(), CORR_THRESHOLD)
    X, Y = [], []
    for end in range(252, len(train_r) - HORIZON, HORIZON):
        X.append(node_features(train_r.iloc[:end]))
        Y.append(train_r.iloc[end:end + HORIZON].sum().values)
    X = np.array(X)
    mu, sd = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0) + 1e-8
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor(np.array(Y), dtype=torch.float32)
    model = GCN(X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = sum(nn.functional.mse_loss(model(Xt[i], A)[0], Yt[i]) for i in range(len(Xt))) / len(Xt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        x_now = torch.tensor((node_features(train_r) - mu) / sd, dtype=torch.float32)
        pred, z = model(x_now, A)
    return pred.numpy(), z.numpy()


def gcn_covariance(train_r, z):
    s = train_r.cov().values
    vol = np.sqrt(np.diag(s))
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    c = np.clip(zn @ zn.T, -0.99, 0.99)
    np.fill_diagonal(c, 1.0)
    target = c * np.outer(vol, vol)
    cov = (1 - SHRINK) * s + SHRINK * target
    w, v = np.linalg.eigh(cov)                      # keep PSD
    return (v * np.clip(w, 1e-10, None)) @ v.T


def recolor(train_r, cov):
    """Scenarios with the empirical marginals' shape but the target covariance."""
    x = train_r.values - train_r.values.mean(0)
    L_s = np.linalg.cholesky(np.cov(x.T) + 1e-10 * np.eye(x.shape[1]))
    L_t = np.linalg.cholesky(cov + 1e-10 * np.eye(x.shape[1]))
    return x @ np.linalg.inv(L_s).T @ L_t.T


def cvar_weights(scen, mu_daily):
    n = scen.shape[1]
    w, var, u = cp.Variable(n), cp.Variable(), cp.Variable(scen.shape[0])
    loss = -(scen + mu_daily) @ w
    cvar = var + cp.sum(u) / ((1 - ALPHA) * scen.shape[0])
    prob = cp.Problem(cp.Maximize(mu_daily @ w - RISK_AVERSION * cvar),
                      [u >= 0, u >= loss - var, cp.sum(w) == 1, w >= 0, w <= W_MAX])
    prob.solve(solver=cp.CLARABEL)
    return np.clip(w.value, 0, None) / np.clip(w.value, 0, None).sum()


# ---------------------------------------------------------------- walk-forward
month_ends = rets.groupby(rets.index.to_period('M')).apply(lambda d: d.index[-1])
rebal = [d for d in month_ends if rets.index.get_loc(d) >= TRAIN_DAYS - 1]
strategies = ['GCN-CVaR', 'Sample-CVaR', 'EqualWeight']
daily = {s: [] for s in strategies}
prev_w = {s: np.zeros(len(TICKERS)) for s in strategies}
w_log = []

for k, d in enumerate(rebal[:-1]):
    i = rets.index.get_loc(d)
    train = rets.iloc[i - TRAIN_DAYS + 1:i + 1]                 # data up to and including d
    hold = rets.loc[(rets.index > d) & (rets.index <= rebal[k + 1])]

    pred, z = gcn_fit(train)
    cov_g = gcn_covariance(train, z)
    mu_g = pred / HORIZON
    w = {
        'GCN-CVaR': cvar_weights(recolor(train, cov_g), mu_g),
        'Sample-CVaR': cvar_weights(train.values - train.values.mean(0), train.values.mean(0)),
        'EqualWeight': np.full(len(TICKERS), 1 / len(TICKERS)),
    }
    for s in strategies:
        cost = COST_BPS / 1e4 * np.abs(w[s] - prev_w[s]).sum()
        drift = w[s].copy()
        r_hold = []
        for j, (_, row) in enumerate(hold.iterrows()):
            r = float(drift @ row.values)
            if j == 0:
                r -= cost
            r_hold.append(r)
            drift = drift * (1 + row.values)
            drift /= drift.sum()
        daily[s].append(pd.Series(r_hold, index=hold.index))
        prev_w[s] = drift
    w_log.append(pd.Series(w['GCN-CVaR'], index=TICKERS, name=d))
    print(f'{d.date()}  top: ' + ', '.join(f'{t} {x:.0%}' for t, x in
          pd.Series(w["GCN-CVaR"], index=TICKERS).nlargest(3).items()))

# ---------------------------------------------------------------- evaluation
def stats(r):
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - RF_ANNUAL) / vol
    dn = r[r < 0].std() * np.sqrt(252)
    v = (1 + r).cumprod()
    mdd = (v / v.cummax() - 1).min()
    cvar = -r[r <= r.quantile(0.05)].mean()
    return dict(AnnReturn=ann, AnnVol=vol, Sharpe=sharpe, Sortino=r.mean() * 252 / dn,
                MaxDD=mdd, DailyCVaR95=cvar)

R = pd.DataFrame({s: pd.concat(daily[s]) for s in strategies})
table = pd.DataFrame({s: stats(R[s]) for s in strategies}).T
print(f'\nout-of-sample {R.index[0].date()} ~ {R.index[-1].date()} ({len(rebal) - 1} rebalances)')
print(table.round(3).to_string())

R.to_csv(os.path.join(OUT, 'daily_returns.csv'))
table.to_csv(os.path.join(OUT, 'performance.csv'))
pd.DataFrame(w_log).to_csv(os.path.join(OUT, 'gcn_cvar_weights.csv'))
with open(os.path.join(OUT, 'config.json'), 'w') as f:
    json.dump({k: v for k, v in globals().items() if k.isupper() and k not in ('HERE', 'CACHE', 'OUT')
               and isinstance(v, (int, float, str, list, dict))},
              f, indent=1)
