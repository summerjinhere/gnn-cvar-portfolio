import numpy as np
import pandas as pd
import yfinance as yf
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import arch
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimization:
    def __init__(self, start_date='2020-01-01', end_date='2023-01-01', rebalance_freq='M'):
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_freq = rebalance_freq
        self.data = None
        self.monthly_returns = None
        self.graph = None
        self.features = None
        self.model = None
        self.cmap = cm.get_cmap('summer')

    def get_stock_data(self, stock_tickers, futures_tickers=None, options_tickers=None):
        all_tickers = list(stock_tickers)
        if futures_tickers:
            all_tickers += futures_tickers
        df = yf.download(all_tickers, start=self.start_date, end=self.end_date, auto_adjust=True)['Close'].ffill()
        if options_tickers:
            for t in options_tickers:
                try:
                    iv = yf.Ticker(t).info.get('impliedVolatility', np.nan)
                except:
                    iv = np.nan
                df[f"{t}_IV"] = iv
        self.data = df
        self.monthly_returns = df.resample(self.rebalance_freq).last().pct_change().dropna()
        return df

    def construct_graph(self, threshold=0.3):
        """
        상관관계를 기반으로 그래프를 구성합니다.
        threshold 값을 낮추면 더 많은 에지가 생성됩니다.
        """
        corr = self.monthly_returns.corr().abs()
        G = nx.Graph()
        G.add_nodes_from(corr.columns)
        
        # 에지 추가
        edge_count = 0
        for i, u in enumerate(corr.columns):
            # 자기 자신과의 에지 항상 추가 (self-loop)
            G.add_edge(u, u, weight=1.0)
            edge_count += 1
            
            for v in corr.columns[i+1:]:
                w = corr.at[u, v]
                if w > threshold:
                    G.add_edge(u, v, weight=w)
                    edge_count += 1
        
        # 그래프에 에지가 너무 적으면 threshold를 낮춤
        if edge_count < len(corr.columns) * 2:
            print(f"경고: 에지가 너무 적습니다. threshold {threshold}에서 에지 수: {edge_count}")
            # threshold를 낮춰서 다시 시도
            new_threshold = threshold * 0.8
            print(f"threshold를 {new_threshold}로 낮춰서 다시 시도합니다.")
            return self.construct_graph(threshold=new_threshold)
            
        self.graph = G
        print(f"그래프 생성 완료: 노드 {len(G.nodes())}개, 에지 {len(G.edges())}개")
        return G

    def calculate_centrality(self):
        pr = nx.pagerank(self.graph)
        ev = nx.eigenvector_centrality(self.graph, max_iter=1000)
        dg = nx.degree_centrality(self.graph)
        return pd.DataFrame({'PageRank': pr, 'Eigenvector': ev, 'Degree': dg})

    def calculate_technical_indicators(self):
        df = self.data
        frames = []
        for t in df.columns:
            s = df[t]
            delta = s.diff()
            gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rsi = 100 - (100 / (1 + avg_gain/avg_loss))
            ema12 = s.ewm(span=12).mean()
            ema26 = s.ewm(span=26).mean()
            macd = ema12 - ema26
            macd_sig = macd.ewm(span=9).mean()
            sma20, std20 = s.rolling(20).mean(), s.rolling(20).std()
            upper_bb, lower_bb = sma20 + 2*std20, sma20 - 2*std20
            momentum = s / s.shift(14)
            df_i = pd.DataFrame({
                (t, 'RSI'): rsi,
                (t, 'MACD'): macd,
                (t, 'MACD_Signal'): macd_sig,
                (t, 'Upper_Band'): upper_bb,
                (t, 'Lower_Band'): lower_bb,
                (t, 'Momentum'): momentum
            })
            frames.append(df_i)
        tech = pd.concat(frames, axis=1)
        return tech.resample(self.rebalance_freq).last().dropna()

    def estimate_volatility(self, method='hybrid', lambda_ewma=0.97):
        vol = {}
        for t in self.monthly_returns.columns:
            series = self.monthly_returns[t].dropna() * 100
            if series.empty:
                vol[t] = pd.Series(dtype=float)
                continue
            # GARCH
            if method in ['garch', 'hybrid']:
                try:
                    m = arch.arch_model(series, vol='Garch', p=1, q=1)
                    res = m.fit(disp='off')
                    v = res.conditional_volatility / 100
                except:
                    v = series.rolling(12).std() / 100
            # EWMA fallback
            if method == 'ewma' or (method == 'hybrid' and 'v' not in locals()):
                ewma = pd.Series(index=series.index, dtype=float)
                ewma.iloc[0] = series.iloc[:12].std() / 100
                for i in range(1, len(series)):
                    ewma.iloc[i] = np.sqrt(
                        lambda_ewma * ewma.iloc[i-1]**2 +
                        (1 - lambda_ewma) * (series.iloc[i-1]/100)**2
                    )
                v = ewma
            vol[t] = v
        return pd.DataFrame(vol).dropna(how='all')

    def calculate_expected_shortfall(self, alpha=0.05):
        es = {}
        for t in self.monthly_returns.columns:
            r = self.monthly_returns[t].dropna()
            if r.size < 2 or r.nunique() == 1:
                es[t] = 0.0
                continue
            cutoff = np.percentile(r, 100*alpha)
            tail = r[r <= cutoff]
            es[t] = -tail.mean() if not tail.empty else 0.0
        return pd.Series(es)

    def risk_parity_weights(self, cov, tol=1e-8, maxiter=1000):
        n = cov.shape[0]
        w = np.ones(n) / n
        for _ in range(maxiter):
            port_var = w @ cov @ w
            mc = cov @ w / np.sqrt(port_var)
            rc = w * mc
            inv = 1 / rc
            w_new = inv / inv.sum()
            if np.max(np.abs(w_new - w)) < tol:
                break
            w = w_new
        return w

    def create_feature_matrix(self, cent, tech, vol):
        """
        주식 노드의 특성 행렬을 생성하고 그래프 데이터 객체를 반환합니다.
        """
        dates = vol.index
        nodes = list(self.graph.nodes())
        es = self.calculate_expected_shortfall()
        feat = []
        
        # 특성 행렬 구성
        for d in dates:
            row = []
            for t in nodes:
                c = cent.loc[t].values
                ti = tech[t].loc[d].fillna(0).values
                v_ = vol.at[d, t]
                row.append(np.concatenate([c, ti, [v_, es[t]]]))
            feat.append(row)
        
        # 텐서 변환
        X = torch.tensor(np.array(feat[:-1]), dtype=torch.float)
        
        # 에지 인덱스 생성
        idx = {n:i for i, n in enumerate(nodes)}
        
        # 그래프에 에지가 있는지 확인
        edges = []
        for u, v in self.graph.edges():
            edges.append([idx[u], idx[v]])
            # 무방향 그래프이므로 반대 방향도 추가
            if u != v:  # 자기 루프가 아닌 경우에만
                edges.append([idx[v], idx[u]])
        
        # 에지가 없는 경우 기본 연결 추가 (완전 연결 그래프)
        if not edges:
            print("경고: 그래프에 에지가 없습니다. 기본 연결을 추가합니다.")
            for i in range(len(nodes)):
                # 최소한 자기 자신과는 연결
                edges.append([i, i])
                # 인접 노드와 연결 (예: 체인 구조)
                if i < len(nodes) - 1:
                    edges.append([i, i+1])
                    edges.append([i+1, i])
        
        # 에지 인덱스 텐서 생성
        edge_index = torch.tensor(edges, dtype=torch.long).T
        
        # 타겟 텐서 생성
        y = torch.tensor(self.monthly_returns[nodes].iloc[1:].values, dtype=torch.float)
        
        # 데이터 객체에 저장하기 전에 형태 검증
        print(f"X 텐서 형태: {X.shape}, 에지 인덱스 형태: {edge_index.shape}, Y 텐서 형태: {y.shape}")
        print(f"에지 수: {edge_index.shape[1]}")
        
        # Data 객체 생성
        return Data(x=X, edge_index=edge_index, y=y)

    def build_gnn_model(self, input_dim, hidden=64, output=1):
        class SimpleGCN(nn.Module):
            def __init__(self, in_d, h, o):
                super(SimpleGCN, self).__init__()
                self.conv1 = GCNConv(in_d, h)
                self.bn1   = nn.BatchNorm1d(h)
                self.conv2 = GCNConv(h, o)

            def forward(self, x, edge_index):
                # 빈 배치 처리
                if x.dim() == 3 and x.size(0) == 0:
                    return torch.empty(
                        (0, x.size(1), self.conv2.out_channels),
                        device=x.device,
                        dtype=x.dtype
                    )

                # 정상 입력: 시점별로 GCN 적용
                outs = []
                for t in range(x.size(0)):
                    h = F.relu(self.conv1(x[t], edge_index))
                    h = self.bn1(h)
                    h = F.dropout(h, p=0.2, training=self.training)
                    outs.append(self.conv2(h, edge_index))
                return torch.stack(outs)

        # 모델 인스턴스 생성
        self.model = SimpleGCN(input_dim, hidden, output)
        return self.model

    def train_gnn_model(self, data, epochs=50, lr=0.005, patience=10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        
        # 텐서 타입과 requires_grad 확인
        x = data.x.clone().detach().to(device).float()
        edge_index = data.edge_index.to(device)
        y = data.y.clone().detach().to(device).float()
        
        # 차원 확인 및 조정
        if y.dim() == 2:
            y = y.unsqueeze(-1)
            
        # 유효성 검사 크기 설정
        val_size = max(1, int(0.15 * len(x)))
        tx, ty = x[:-val_size], y[:-val_size]
        vx, vy = x[-val_size:], y[-val_size:]
        
        # 옵티마이저 및 스케줄러 설정
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=patience//2, verbose=False)
        best_loss, counter = float('inf'), 0
        best_state = None
        
        # 모델 훈련 루프
        for ep in range(epochs):
            # 훈련 모드 설정
            model.train()
            
            # 그래디언트 초기화
            opt.zero_grad()
            
            # 순전파
            out = model(tx, edge_index)
            
            # 손실 계산 및 역전파
            try:
                loss = F.smooth_l1_loss(out, ty)
                loss.backward()
                opt.step()
            except RuntimeError as e:
                print(f"역전파 에러 발생: {e}")
                print(f"텐서 정보 - out: {out.shape}, requires_grad: {out.requires_grad}")
                print(f"텐서 정보 - ty: {ty.shape}, requires_grad: {ty.requires_grad}")
                # 지속적 학습을 위한 오류 처리
                if best_state is not None:
                    model.load_state_dict(best_state)
                    break
                else:
                    # 첫 에폭에서 오류가 발생하면 기본 가중치 사용
                    print("초기 훈련 실패, 기본 가중치 사용")
                    return model
            
            # 검증
            model.eval()
            with torch.no_grad():
                vout = model(vx, edge_index)
                vloss = F.smooth_l1_loss(vout, vy)
            
            # 학습률 조정
            sched.step(vloss)
            
            # 조기 종료 확인
            if vloss < best_loss:
                best_loss, counter = vloss, 0
                best_state = model.state_dict().copy()  # 가중치 복사
            else:
                counter += 1
                if counter >= patience:
                    print(f"조기 종료: {ep+1}번째 에폭")
                    break
                    
        # 최적 가중치 로드
        if best_state is not None:
            model.load_state_dict(best_state)
            
        return model

    def calculate_portfolio_weights(self, preds, cent, vol, use_financial_engineering=False, method='mv'):
        nodes = list(self.graph.nodes())
        df = pd.DataFrame(index=nodes)
        arr = preds.flatten() if isinstance(preds, np.ndarray) else preds
        df['Return'] = arr[:len(nodes)]
        
        if method == 'rp':
            cov = self.monthly_returns[nodes].cov()
            w = self.risk_parity_weights(cov)
            return pd.Series(w, index=nodes)
            
        if method == 'cvar':
            R = self.monthly_returns[nodes]
            def es_obj(w):
                port = R.dot(w)
                cutoff = np.percentile(port, 5)
                return -port[port <= cutoff].mean()
            res = minimize(es_obj, np.ones(len(nodes))/len(nodes),
                           bounds=tuple((0,0.2) for _ in nodes),
                           constraints=({'type':'eq','fun':lambda w: w.sum()-1},))
            return pd.Series(res.x if res.success else np.ones(len(nodes))/len(nodes), index=nodes)
            
        if use_financial_engineering and method == 'mv':
            er = df['Return']
            cov = self.monthly_returns[nodes].cov()
            def mv_obj(w): 
                return -(w @ er - w @ cov @ w)
            
            res = minimize(mv_obj, np.ones(len(nodes))/len(nodes),
                           bounds=tuple((0,0.2) for _ in nodes),
                           constraints=({'type':'eq','fun':lambda w:w.sum()-1},))
            return pd.Series(res.x if res.success else np.ones(len(nodes))/len(nodes), index=nodes)
        
        # 기본 방식: 페이지랭크와 변동성 활용
        df['Centrality'] = cent['PageRank']
        df['Vol'] = vol.iloc[-1]
        df['Raw'] = (df['Return'] * df['Centrality'] / df['Vol']).clip(lower=0)
        
        if df['Raw'].sum() > 0:
            return pd.Series(df['Raw']/df['Raw'].sum(), index=nodes)
        else:
            return pd.Series(1/len(nodes), index=nodes)

    def backtest(self, weights, returns):
        val, vals = 1.0, [1.0]
        dates = [returns.index[0]]
        for r in returns.values:
            val *= 1 + (weights*r).sum()
            vals.append(val)
            dates.append(dates[-1] + pd.tseries.frequencies.to_offset(self.rebalance_freq))
        return pd.DataFrame({'Value': vals}, index=dates)

    def create_equal_weight_portfolio(self, returns):
        w = np.ones(len(returns.columns))/len(returns.columns)
        return self.backtest(pd.Series(w, index=returns.columns), returns)

    def plot_weights(self, weights_dict, title='Portfolio Weights Comparison'):
        df = pd.DataFrame(weights_dict)
        fig, ax = plt.subplots(figsize=(10,6))
        df.plot(kind='bar', ax=ax, colormap='summer')
        ax.set_title(title)
        ax.set_ylabel('Weight')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        return fig

    def plot_cumulative_returns(self, perf_dict, title='Cumulative Returns Comparison'):
        fig, ax = plt.subplots(figsize=(10,6))
        for name, perf in perf_dict.items():
            cum = perf['Value']/perf['Value'].iloc[0] - 1
            ax.plot(cum.index, cum.values, label=name)
        ax.set_title(title)
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        return fig

    def plot_drawdown(self, perf_df, title='Drawdown Curve'):
        run_max = perf_df['Value'].cummax()
        dd = perf_df['Value']/run_max - 1
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(dd.index, dd.values, color=self.cmap(0.6))
        ax.fill_between(dd.index, dd.values, color=self.cmap(0.6), alpha=0.3)
        ax.set_title(title)
        ax.set_ylabel('Drawdown')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        return fig

    def animate_cumulative_returns(self, perf_dict, interval=300, title='Cumulative Returns Animation'):
        fig, ax = plt.subplots(figsize=(10,6))
        strategies = list(perf_dict.keys())
        dates = perf_dict[strategies[0]].index
        cumrets = {name: perf['Value']/perf['Value'].iloc[0] - 1 for name, perf in perf_dict.items()}
        lines = {name: ax.plot([], [], label=name)[0] for name in strategies}
        ax.set_xlim(dates[0], dates[-1])
        all_vals = np.hstack([cumrets[name].values for name in strategies])
        ax.set_ylim(np.nanmin(all_vals)*1.1, np.nanmax(all_vals)*1.1)
        ax.set_title(title)
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        def init():
            for line in lines.values():
                line.set_data([], [])
            return list(lines.values())

        def update(frame):
            for name, line in lines.items():
                line.set_data(dates[:frame], cumrets[name].values[:frame])
            return list(lines.values())

        anim = FuncAnimation(fig, update, frames=len(dates)+1, init_func=init,
                             interval=interval, blit=False)
        plt.tight_layout()
        return anim

    def plot_graph(self, G, title='Graph Visualization'):
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color=self.cmap(0.5), edge_color=self.cmap(0.3), node_size=500, font_size=10)
        plt.title(title)
        return plt.gcf()

    def evaluate_performance(self, perf, bench=None):
        ret = perf['Value'].pct_change().dropna()
        tr = perf['Value'].iloc[-1]/perf['Value'].iloc[0]-1
        ar = (1+tr)**(12/len(ret))-1
        vol = ret.std()*np.sqrt(12)
        sr=ar/vol
        dd =(perf['Value']/perf['Value'].cummax()-1).min()
        dr=ret[ret<0].std()*np.sqrt(12)
        so=ar/dr if dr>0 else np.nan
        m = dict(Total=tr, Annual=ar, Vol=vol, Sharpe=sr, Sortino=so, MaxDD=dd)
        if bench is not None:
            br=bench['Value'].pct_change().dropna()
            btr=bench['Value'].iloc[-1]/bench['Value'].iloc[0]-1
            bar=(1+btr)**(12/len(br))-1
            bvol=br.std()*np.sqrt(12)
            bsr=bar/bvol
            m.update(BSharpe=bsr)
        return m

if __name__ == '__main__':
    try:
        # Define tickers
        stock_tickers   = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN',
            'META', 'TSLA', 'ASML', 'AVGO', 'ORCL',
            'IBM',  'INTC', 'CSCO', 'TXN',  'SAP',
            'TSM',  'CRM',  'QCOM', 'ADBE', 'AMD','NFLX', 'DIS', 'VZ', 'T',
            'CMCSA', 'PEP', 'KO', 'NKE', 'MCD']
        futures_tickers = ['ES=F', 'CL=F']             # S&P 500 futures, Crude Oil futures
        options_tickers = ['^VIX']                     # VIX as proxy for equity option implied vol

        print("========== 포트폴리오 최적화 시작 ==========")
        
        # Initialize portfolio object
        p = PortfolioOptimization(start_date='2020-01-01', end_date='2023-01-01')
        print(f"날짜 범위: {p.start_date} ~ {p.end_date}, 리밸런싱 주기: {p.rebalance_freq}")

        # Fetch combined data
        print("데이터 로딩 중...")
        data = p.get_stock_data(
            stock_tickers=stock_tickers,
            futures_tickers=futures_tickers,
            options_tickers=options_tickers
        )
        print(f"로드된 티커: {len(data.columns)}개")
        print(f"데이터 기간: {data.index[0]} ~ {data.index[-1]}")
        print(f"월별 수익률 형태: {p.monthly_returns.shape}")

        # Proceed with the usual pipeline
        print("\n1. 상관관계 그래프 구성 중...")
        # 기본 threshold 값을 0.2로 낮춤
        G = p.construct_graph(threshold=0.2)  
        
        print("\n2. 중심성 계산 중...")
        cent = p.calculate_centrality()
        print(f"중심성 상위 5개 노드:")
        print(cent.sort_values('PageRank', ascending=False).head(5))
        
        print("\n3. 기술적 지표 계산 중...")
        tech = p.calculate_technical_indicators()
        
        print("\n4. 변동성 추정 중...")
        vol = p.estimate_volatility(method='hybrid')
        
        print("\n5. 그래프 신경망 특성 행렬 생성 중...")
        gnn_data = p.create_feature_matrix(cent, tech, vol)
        
        # 그래프 시각화
        print("\n6. 상관관계 네트워크 시각화 중...")
        graph_fig = p.plot_graph(G, "Stock Correlation Network")
        plt.savefig('correlation_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("'correlation_network.png' 저장됨")

        # Dynamically reshape and determine feature dimension
        print("\n7. GNN 모델 구축 중...")
        dims = gnn_data.x.dim()
        if dims == 3:
            input_dim = gnn_data.x.shape[2]
            print(f"3D 입력 특성: [시간, 노드, 특성] = {gnn_data.x.shape}")
        elif dims == 2:
            input_dim = gnn_data.x.shape[1]
            print(f"2D 입력 특성: [노드, 특성] = {gnn_data.x.shape}")
        elif dims == 1:
            # Single snapshot: infer nodes and feature count, reshape to [1, nodes, features]
            total_len = gnn_data.x.shape[0]
            num_nodes = len(p.graph.nodes())
            if total_len % num_nodes != 0:
                raise ValueError(f"Cannot reshape 1D features of length {total_len} into {num_nodes} nodes")
            feature_count = total_len // num_nodes
            gnn_data.x = gnn_data.x.view(1, num_nodes, feature_count)
            input_dim = feature_count
            print(f"1D 입력 특성 재구성: [1, {num_nodes}, {feature_count}]")
        else:
            raise ValueError(f"Unexpected feature tensor dimensions: {dims}")

        # 모델 구축 및 훈련
        model = p.build_gnn_model(input_dim=input_dim)
        print(f"GNN 모델 구축 완료: 입력 차원={input_dim}, 은닉층 크기=64")
        
        print("\n8. GNN 모델 훈련 중...")
        try:
            trained = p.train_gnn_model(gnn_data, epochs=20)
            print("GNN 모델 훈련 완료")
            
            # 예측 수행
            print("\n9. 포트폴리오 수익률 예측 중...")
            with torch.no_grad():
                preds = trained(gnn_data.x, gnn_data.edge_index).cpu().numpy()[-1]
                
            # 포트폴리오 가중치 계산
            print("\n10. 포트폴리오 가중치 계산 중...")
            w_gnn   = p.calculate_portfolio_weights(preds, cent, vol, use_financial_engineering=False)
            w_mv    = p.calculate_portfolio_weights(preds, cent, vol, use_financial_engineering=True, method='mv')
            w_rp    = p.calculate_portfolio_weights(preds, cent, vol, method='rp')
            w_cvar  = p.calculate_portfolio_weights(preds, cent, vol, method='cvar')
            
            print("각 전략별 상위 3개 종목:")
            for name, weights in {'GNN': w_gnn, 'MeanVar': w_mv, 'RiskParity': w_rp, 'CVaR': w_cvar}.items():
                top3 = weights.sort_values(ascending=False).head(3)
                print(f"- {name}: {', '.join([f'{t} ({w:.1%})' for t, w in top3.items()])}")
            
            # 가중치 시각화
            print("\n11. 포트폴리오 가중치 시각화 중...")
            weights_dict = {'GNN': w_gnn, 'MeanVar': w_mv, 'RiskParity': w_rp, 'CVaR': w_cvar}
            fig_weights = p.plot_weights(weights_dict)
            plt.savefig('portfolio_weights.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("'portfolio_weights.png' 저장됨")
            
            # 백테스팅 수행
            print("\n12. 백테스트 수행 중...")
            split = int(0.8 * len(p.monthly_returns))
            test_returns = p.monthly_returns.iloc[split:]
            print(f"훈련 기간: {p.monthly_returns.index[0]} ~ {p.monthly_returns.index[split-1]}")
            print(f"테스트 기간: {test_returns.index[0]} ~ {test_returns.index[-1]}")
            
            # 각 전략별 수익률 계산
            perf_gnn = p.backtest(w_gnn, test_returns)
            perf_mv = p.backtest(w_mv, test_returns)
            perf_rp = p.backtest(w_rp, test_returns)
            perf_cvar = p.backtest(w_cvar, test_returns)
            benchmark_perf = p.create_equal_weight_portfolio(test_returns)
            
            perf_dict = {
                'GNN': perf_gnn,
                'MeanVar': perf_mv,
                'RiskParity': perf_rp,
                'CVaR': perf_cvar,
                'Baseline': benchmark_perf
            }
            
            # 성능 시각화
            print("\n13. 누적 수익률 시각화 중...")
            fig_cum = p.plot_cumulative_returns(perf_dict)
            plt.savefig('cumulative_returns.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("'cumulative_returns.png' 저장됨")
            
            print("\n14. 최대 낙폭 시각화 중...")
            fig_dd = p.plot_drawdown(perf_gnn)
            plt.savefig('drawdown.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("'drawdown.png' 저장됨")
            
            # 성능 평가
            print("\n15. 성능 지표 계산:")
            for name, perf in perf_dict.items():
                metrics = p.evaluate_performance(perf, benchmark_perf)
                print(f"\n[{name} 전략 성능]")
                for k, v in metrics.items():
                    if k in ['Annual', 'Sharpe', 'Sortino']:
                        print(f"- {k}: {v:.4f}")
                print(f"- MaxDrawdown: {metrics['MaxDD']:.2%}")
            
            # 애니메이션 생성
            print("\n16. 누적 수익률 애니메이션 생성 중...")
            try:
                anim = p.animate_cumulative_returns(perf_dict, interval=200)
                anim.save('cumulative_returns_animation.gif', writer='pillow', fps=5)
                print("'cumulative_returns_animation.gif' 저장됨")
            except Exception as e:
                print(f"애니메이션 생성 실패: {e}")
        
        except Exception as e:
            print(f"GNN 모델 훈련 또는 예측 중 오류 발생: {e}")
            print("기본 동일 가중치 포트폴리오로 대체합니다.")
            
            # 동일 가중치 포트폴리오만 생성
            split = int(0.8 * len(p.monthly_returns))
            test_returns = p.monthly_returns.iloc[split:]
            benchmark_perf = p.create_equal_weight_portfolio(test_returns)
            
            print("\n성능 지표:")
            metrics = p.evaluate_performance(benchmark_perf)
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
    
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n========== 포트폴리오 최적화 완료 ==========")
        print("생성된 파일:")
        import os
        for file in ['correlation_network.png', 'portfolio_weights.png', 
                     'cumulative_returns.png', 'drawdown.png', 
                     'cumulative_returns_animation.gif']:
            if os.path.exists(file):
                print(f"- {file}")