import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import statsmodels.api as sm

# ----------------------------------------
# PARAMETERS
# ----------------------------------------
top_n = 3
pval_thresh = 0.05
correlation_threshold = 0.75
lookback_days = 250

# ----------------------------------------
# TICKERS by Sector
# ----------------------------------------
tickers_info = {
    '096770': 'SK이노베이션', '010950': 'S-Oil', '003670': '포스코퓨처엠',
    '011170': '롯데케미칼', '051910': 'LG화학',
    '010140': '삼성중공업', '042660': '대우조선해양', '009540': '한국조선해양', '267250': 'HD현대'
}
sector_map = {
    '096770': 'Energy', '010950': 'Energy', '003670': 'Energy',
    '011170': 'Chemicals', '051910': 'Chemicals',
    '010140': 'Shipbuilding', '042660': 'Shipbuilding', '009540': 'Shipbuilding', '267250': 'Shipbuilding'
}

# ----------------------------------------
# FUNCTION: Download data from Naver Finance (Refactored)
# ----------------------------------------
def get_naver_stock_data(code, start_date, end_date):
    headers = {'User-Agent': 'Mozilla/5.0'}
    start_date_obj = pd.to_datetime(start_date)
    end_date_obj = pd.to_datetime(end_date)
    df_list = []
    page = 1
    max_empty_pages = 3
    empty_page_count = 0

    print(f"[INFO] Start downloading: {code} ({tickers_info.get(code, 'Unknown')})")

    while True:
        url = f"https://finance.naver.com/item/sise_day.naver?code={code}&page={page}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"[ERROR] Failed to fetch page {page} for {code}")
            break
        response.encoding = 'euc-kr'
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'type2'})
        if not table:
            empty_page_count += 1
            if empty_page_count >= max_empty_pages:
                break
            page += 1
            continue

        rows = table.find_all('tr')
        new_data_found = False
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue
            date_str = cols[0].text.strip()
            price_str = cols[1].text.strip().replace(',', '')
            try:
                date = pd.to_datetime(date_str)
                if start_date_obj <= date <= end_date_obj:
                    price = float(price_str)
                    df_list.append({'Date': date, 'Close': price})
                    new_data_found = True
                    print(f"[DATA] {code} | Date: {date.date()} | Close: {price}")
            except:
                continue

        if not new_data_found:
            empty_page_count += 1
        else:
            empty_page_count = 0

        if date < start_date_obj:
            break

        page += 1
        time.sleep(np.random.uniform(0.2, 0.5))  # slow down to avoid blocking

    df = pd.DataFrame(df_list).drop_duplicates()
    if df.empty:
        print(f"[WARNING] No data collected for {code}")
        return pd.DataFrame()
    df.set_index('Date', inplace=True)
    print(f"[INFO] Finished downloading {code} with {len(df)} records.")
    return df.sort_index()
# ----------------------------------------
# FUNCTION: Download all stock prices
# ----------------------------------------
def get_all_stock_prices(tickers_info, start_date, end_date):
    price_df = pd.DataFrame()
    for code, name in tqdm(tickers_info.items(), desc="Downloading all stocks"):
        print(f"📦 Downloading {name} ({code})...")
        df = get_naver_stock_data(code, start_date, end_date)
        if not df.empty:
            price_df[code] = df['Close']
            print(f"[INFO] {name} ({code}) data added to DataFrame with {df.shape[0]} rows.")
        else:
            print(f"[WARNING] No valid data for {name} ({code})")
    return price_df.ffill()

# ----------------------------------------
# FUNCTION: Build Correlation Network
# ----------------------------------------
def build_correlation_network(returns):
    corr = returns.corr().abs()
    G = nx.Graph()
    for i in corr.columns:
        for j in corr.columns:
            if i != j and corr.loc[i, j] >= correlation_threshold:
                G.add_edge(i, j, weight=corr.loc[i, j])
    return G

# ----------------------------------------
# FUNCTION: Build Granger Causality Network
# ----------------------------------------
def build_granger_network(returns):
    G = nx.DiGraph()
    for i in returns.columns:
        for j in returns.columns:
            if i == j:
                continue
            x = returns[j].shift(1)
            y = returns[i]
            df_pair = pd.concat([x, y], axis=1).dropna()
            if len(df_pair) < 30:
                continue
            X = sm.add_constant(df_pair.iloc[:, 0])
            model = sm.OLS(df_pair.iloc[:, 1], X).fit()
            if model.pvalues[1] < pval_thresh:
                G.add_edge(j, i, weight=model.params[1])
    return G

# ----------------------------------------
# FUNCTION: Compute Centrality Measures
# ----------------------------------------
def compute_centralities(G):
    centralities = {
        'Degree': pd.Series(nx.degree_centrality(G)),
        'Betweenness': pd.Series(nx.betweenness_centrality(G)),
        'PageRank': pd.Series(nx.pagerank(G))
    }
    try:
        centralities['Eigenvector'] = pd.Series(nx.eigenvector_centrality(G, max_iter=1000))
    except:
        centralities['Eigenvector'] = pd.Series({node: 0.0 for node in G.nodes})
    return centralities

# ----------------------------------------
# FUNCTION: Risk Parity Portfolio
# ----------------------------------------
def risk_parity_weights(cov_matrix):
    inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
    weights = inv_vol / inv_vol.sum()
    return weights

# ----------------------------------------
# FUNCTION: Backtest Strategy
# ----------------------------------------
def backtest_with_granger(price_df, endpoints):
    returns_dict = {key: [] for key in ['Degree', 'Betweenness', 'Eigenvector', 'PageRank', 'RiskParity']}
    dates = []
    for current_end in tqdm(endpoints[12:-1], desc="Backtesting"):
        window_start = current_end - pd.DateOffset(days=lookback_days)
        window_data = price_df.loc[window_start:current_end]
        if len(window_data) < 200:
            continue
        returns = window_data.pct_change().dropna()
        G = build_granger_network(returns)
        if len(G.nodes) < top_n:
            continue
        nx.write_graphml(G, f"graphs/network_{current_end.date()}.graphml")
        centralities = compute_centralities(G)
        next_end = endpoints[endpoints.get_loc(current_end) + 1]
        future_prices = price_df.loc[[current_end, next_end]]
        if future_prices.isnull().values.any():
            continue
        for key in ['Degree', 'Betweenness', 'Eigenvector', 'PageRank']:
            top_stocks = centralities[key].nlargest(top_n).index
            try:
                ret = (future_prices.loc[next_end, top_stocks] - future_prices.loc[current_end, top_stocks]) / future_prices.loc[current_end, top_stocks]
                returns_dict[key].append(ret.mean())
            except:
                continue
        try:
            cov = returns.cov()
            weights = risk_parity_weights(cov)
            future_ret = (future_prices.loc[next_end] - future_prices.loc[current_end]) / future_prices.loc[current_end]
            weighted_ret = (weights * future_ret).sum()
            returns_dict['RiskParity'].append(weighted_ret)
        except:
            continue
        dates.append(next_end)
    return pd.DataFrame(returns_dict, index=dates)

# ----------------------------------------
# FUNCTION: Performance Plot
# ----------------------------------------
def plot_performance(result_df):
    cumulative_df = (1 + result_df).cumprod()
    plt.figure(figsize=(14, 7))
    for col in cumulative_df.columns:
        plt.plot(cumulative_df[col], label=col)
    plt.title("Cumulative Returns by Centrality Measure")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("centrality_cumulative_returns.png")
    plt.show()
    return cumulative_df

# ----------------------------------------
# FUNCTION: Performance Evaluation
# ----------------------------------------
def evaluate_performance(result_df):
    cumulative_df = (1 + result_df).cumprod()
    years = len(cumulative_df) / 12
    summary = {}
    for col in result_df.columns:
        cagr = cumulative_df[col].iloc[-1] ** (1 / years) - 1
        mdd = (cumulative_df[col] / cumulative_df[col].cummax() - 1).min()
        vol = result_df[col].std() * np.sqrt(12)
        sharpe = cagr / vol if vol > 0 else 0
        summary[col] = {'CAGR': cagr, 'Max Drawdown': mdd, 'Annual Volatility': vol, 'Sharpe': sharpe}
    perf_df = pd.DataFrame(summary).T
    perf_df.to_csv("performance_metrics.csv")
    print(perf_df)
    return perf_df

# ----------------------------------------
# FUNCTION: Sector Cumulative Plot
# ----------------------------------------
def plot_sector_cumulative(price_df):
    sectors = pd.Series(sector_map)
    returns = price_df.pct_change().dropna()
    sector_cumrets = {}
    for sector in sectors.unique():
        codes = sectors[sectors == sector].index
        cumrets = (1 + returns[codes]).mean(axis=1).cumprod()
        sector_cumrets[sector] = cumrets
    plt.figure(figsize=(14, 7))
    for s, series in sector_cumrets.items():
        plt.plot(series, label=s)
    plt.title("\U0001F4CA Sector-wise Average Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sector_cumulative_returns.png")
    plt.show()

# ----------------------------------------
# FUNCTION: Sector Interaction Network Visualization
# ----------------------------------------
def visualize_sector_interaction_network(G):
    pos = nx.spring_layout(G, seed=42)
    node_colors = [
        'red' if sector_map.get(n) == 'Energy' else
        'blue' if sector_map.get(n) == 'Chemicals' else
        'green' if sector_map.get(n) == 'Shipbuilding' else 'gray'
        for n in G.nodes
    ]
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8, edge_color='gray')
    plt.title("Sector-Interaction Network")
    plt.tight_layout()
    plt.savefig("sector_interaction_network.png")
    plt.show()

# ----------------------------------------
# MAIN EXECUTION
# ----------------------------------------
start_date = "20200401"
end_date = datetime.today().strftime("%Y%m%d")
price_df = get_all_stock_prices(tickers_info, start_date, end_date)
monthly_endpoints = price_df.groupby([price_df.index.year, price_df.index.month]).apply(lambda x: x.index[-1])
monthly_endpoints = monthly_endpoints.reset_index(drop=True).sort_values()

result_df = backtest_with_granger(price_df, monthly_endpoints)
if not result_df.empty:
    result_df.to_csv("strategy_returns.csv")
    cumulative_df = plot_performance(result_df)
    cumulative_df.to_csv("cumulative_returns.csv")
    plot_sector_cumulative(price_df)
    evaluate_performance(result_df)
    
    # Visualize last correlation network (optional demo)
    returns = price_df.pct_change().dropna().iloc[-lookback_days:]
    corr_G = build_correlation_network(returns)
    visualize_sector_interaction_network(corr_G)
