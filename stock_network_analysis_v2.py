import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import statsmodels.api as sm
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# ----------------------------------------
# PARAMETERS
# ----------------------------------------
top_n = 3
pval_thresh = 0.05
correlation_threshold = 0.75
lookback_days = 250
os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# [tickers_info와 sector_map 코드는 동일하게 유지]
# ----------------------------------------
# TICKERS by Sector
# Energy + Chemicals + BioMedical sectors

# Energy & Chemicals (각 20개 선정)
tickers_info = {
    '051910': 'LG화학', '096770': 'SK이노베이션', '034730': 'SK', '010950': 'S-Oil',
    '267250': 'HD현대', '011790': 'SKC', '078930': 'GS', '011780': '금호석유',
    '009830': '한화솔루션', '000880': '한화', '011170': '롯데케미칼', '02380': 'KCC',
    '014680': '한솔케미칼', '457190': '이수스페셜티케미컬', '010060': 'OCI홀딩스',
    '004000': '롯데정밀화학', '298020': '효성티앤씨', '02840': '미원상사',
    '120110': '코오롱인더', '298050': 'HS효성첨단소재',

    '090430': '아모레퍼시픽', '161390': '한국타이어앤테크놀로지', '051900': 'LG생활건강',
    '278470': '에이피알', '192820': '코스맥스', '002790': '아모레G', '05290': '동진쎄미켐',
    '161890': '한국콜마', '078520': '에이블씨엔씨', '226320': '잇츠한불',
    '086450': '동국제약', '200130': '콜마비앤에이치', '002000': '한국유나이티드제약',
    '214610': '미코바이오메드', '093370': '후성', '006650': '대한유화',
    '051360': '토니모리', '138490': '코오롱플라스틱', '214430': '에스맥',
    '007690': '국도화학', '298360': '에스디바이오센서',

    '207940': '삼성바이오로직스', '068270': '셀트리온', '196170': '알테오젠',
    '001000': '유한양행', '028300': 'HLB', '145020': '휴젤',
    '002500': '삼천당제약', '214150': '클래시스', '214450': '파마리서치',
    '141080': '리가켐바이오', '302440': 'SK바이오사이언스', '128940': '한미약품',
    '087010': '펩트론', '068760': '셀트리온제약', '310210': '보로노이',
    '007390': '네이처셀', '298380': '에이비엘바이오', '140860': '파크시스템스',
    '237690': '에스티팜', '009420': '한올바이오파마'
}

sector_map = {
    # Energy
    '051910': 'Energy', '096770': 'Energy', '034730': 'Energy', '010950': 'Energy',
    '267250': 'Energy', '011790': 'Energy', '078930': 'Energy', '011780': 'Energy',
    '009830': 'Energy', '000880': 'Energy', '011170': 'Energy', '02380': 'Energy',
    '014680': 'Energy', '457190': 'Energy', '010060': 'Energy', '004000': 'Energy',
    '298020': 'Energy', '02840': 'Energy', '120110': 'Energy', '298050': 'Energy',

    # Chemicals
    '090430': 'Chemicals', '161390': 'Chemicals', '051900': 'Chemicals',
    '278470': 'Chemicals', '192820': 'Chemicals', '002790': 'Chemicals',
    '05290': 'Chemicals', '161890': 'Chemicals', '078520': 'Chemicals', '226320': 'Chemicals',
    '086450': 'Chemicals', '200130': 'Chemicals', '002000': 'Chemicals',
    '214610': 'Chemicals', '093370': 'Chemicals', '006650': 'Chemicals',
    '051360': 'Chemicals', '138490': 'Chemicals', '214430': 'Chemicals',
    '007690': 'Chemicals', '298360': 'Chemicals',

    # BioMedical
    '207940': 'BioMedical', '068270': 'BioMedical', '196170': 'BioMedical',
    '001000': 'BioMedical', '028300': 'BioMedical', '145020': 'BioMedical',
    '002500': 'BioMedical', '214150': 'BioMedical', '214450': 'BioMedical',
    '141080': 'BioMedical', '302440': 'BioMedical', '128940': 'BioMedical',
    '087010': 'BioMedical', '068760': 'BioMedical', '310210': 'BioMedical',
    '007390': 'BioMedical', '298380': 'BioMedical', '140860': 'BioMedical',
    '237690': 'BioMedical', '009420': 'BioMedical'
}



# 병렬 데이터 수집 함수 정의
def get_naver_stock_data(code, start_date, end_date):
    headers = {'User-Agent': 'Mozilla/5.0'}
    start_date_obj = pd.to_datetime(start_date)
    end_date_obj = pd.to_datetime(end_date)
    df_list = []
    page = 1
    while True:
        url = f"https://finance.naver.com/item/sise_day.naver?code={code}&page={page}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
        except Exception:
            break
        if response.status_code != 200:
            break
        response.encoding = 'euc-kr'
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'type2'})
        if not table:
            break
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue
            try:
                date = pd.to_datetime(cols[0].text.strip())
                price = float(cols[1].text.strip().replace(',', ''))
                if start_date_obj <= date <= end_date_obj:
                    df_list.append({'Date': date, 'Close': price})
            except:
                continue
        if df_list and df_list[-1]['Date'] < start_date_obj:
            break
        page += 1
        time.sleep(0.2)
        if page > 20:
            break
    df = pd.DataFrame(df_list).drop_duplicates()
    if df.empty:
        return pd.DataFrame()
    df.set_index('Date', inplace=True)
    return df.sort_index()

def fetch_single_stock(args):
    code, name, start_date, end_date = args
    df = get_naver_stock_data(code, start_date, end_date)
    return code, df['Close'] if not df.empty else None

def get_all_stock_prices_parallel(tickers_info, start_date, end_date):
    price_df = pd.DataFrame()
    args_list = [(code, name, start_date, end_date) for code, name in tickers_info.items()]
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(fetch_single_stock, args_list), total=len(args_list), desc="데이터 수집 중"))
    for code, series in results:
        if series is not None:
            price_df[code] = series
    return price_df.ffill()

# ----------------------------------------
# NETWORK FUNCTIONS
# ----------------------------------------
def build_granger_network(returns):
    G = nx.DiGraph()
    # 모든 노드 먼저 추가
    for col in returns.columns:
        G.add_node(col)
        
    for i in tqdm(returns.columns, desc="Granger 네트워크 구축"):
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

def build_correlation_network(returns, threshold=0.75):
    corr_matrix = returns.corr()
    G = nx.Graph()
    
    # 모든 노드 먼저 추가
    for col in returns.columns:
        G.add_node(col)
        
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and abs(corr_matrix.loc[i, j]) >= threshold:
                G.add_edge(i, j, weight=corr_matrix.loc[i, j])
    return G

def compute_centralities(G):
    centralities = {
        'Degree': pd.Series(nx.degree_centrality(G)),
        'Betweenness': pd.Series(nx.betweenness_centrality(G))
    }
    
    # PageRank에 대한 예외 처리 추가
    try:
        centralities['PageRank'] = pd.Series(nx.pagerank(G, max_iter=500))
    except:
        print("PageRank 계산 중 수렴하지 않음 - 기본값 설정")
        centralities['PageRank'] = pd.Series({node: 1.0/len(G) for node in G.nodes})
    
    # Eigenvector에 대한 예외 처리
    try:
        centralities['Eigenvector'] = pd.Series(nx.eigenvector_centrality(G, max_iter=1000))
    except:
        print("Eigenvector 계산 중 수렴하지 않음 - 기본값 설정")
        centralities['Eigenvector'] = pd.Series({node: 0.0 for node in G.nodes})
    
    return centralities

def plot_network(G, centrality, title, filename):
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(G, seed=42)
    
    # 노드 크기: 중앙성 비례
    sizes = [5000 * centrality.get(n, 0.01) for n in G.nodes()]
    
    # 노드 색상: 각 노드별 중앙성 값을 순서대로
    colors = [centrality.get(n, 0.01) for n in G.nodes()]

    # 산업별 색상 지정 (뒤섞지 않고, 중앙성 컬러맵으로 표현하려면 아래 sector-based node_colors 대신 colors 사용)
    sector_colors = {
        'Energy': 'skyblue',
        'Chemicals': 'salmon',
        'BioMedical': 'limegreen'
    }
    node_sector_colors = [sector_colors.get(sector_map.get(n, 'unknown'), 'lightgray') 
                          for n in G.nodes()]

    # 라벨: 코드/종목명
    labels = {n: f"{n}/{tickers_info.get(n, '')}" for n in G.nodes()}

    # 실제 그리기: 중앙성 값으로 컬러맵
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=sizes,
        node_color=colors,       
        cmap="viridis"
    )
    if G.is_directed():
        nx.draw_networkx_edges(G, pos, alpha=0.4, arrows=True, arrowstyle='->', arrowsize=15)
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    plt.title(title)
    plt.colorbar(nodes, ax=plt.gca(), label="Centrality")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()  # plt.close() 대신에 실제로 그래프 표시

# ----------------------------------------
# BACKTEST FUNCTION
def backtest_strategy(price_df, endpoints, network_func, centrality_type='Degree', top_n=3):
    result = []
    dates = []
    
    # 백테스트 진행 상황 표시
    for i in tqdm(range(12, len(endpoints) - 1), desc=f"백테스팅 {centrality_type}"):
        start = endpoints[i - 12]
        end = endpoints[i]
        next_end = endpoints[i + 1]

        # lookback 기간 확인
        window_prices = price_df.loc[start:end]
        if window_prices.shape[0] < 200:  # 최소 거래일 수 확인
            continue

        # 수익률 계산 및 네트워크 구축
        returns = window_prices.pct_change().dropna()
        
        # 네트워크 구축 (threshold 적용)
        if network_func == build_correlation_network:
            G = network_func(returns, threshold=correlation_threshold)
        else:
            G = network_func(returns)
            
        if G.number_of_nodes() < top_n:
            continue
            
        centralities = compute_centralities(G)
        
        # top_n 종목 선택
        if centrality_type in centralities:
            top_nodes = centralities[centrality_type].nlargest(top_n).index
        else:
            print(f"Warning: {centrality_type} 중심성 지표가 없습니다")
            continue
            
        # 다음 달의 수익률 계산
        try:
            future_prices = price_df.loc[[end, next_end], top_nodes]
            if future_prices.isnull().values.any():
                continue
                
            ret = (future_prices.loc[next_end] - future_prices.loc[end]) / future_prices.loc[end]
            result.append(ret.mean())
            dates.append(next_end)
        except Exception as e:
            print(f"백테스트 오류: {e}")
            continue

    if not result:
        return pd.Series([], index=[], name=f"{centrality_type}_strategy")
        
    return pd.Series(result, index=dates, name=f"{centrality_type}_strategy")

# ----------------------------------------
# 시각화 및 결과 표시 함수
def display_numerical_results_and_visuals(price_df, returns, centralities_corr, centralities_granger, granger_backtest, correlation_backtest=None):
    plt.figure(figsize=(14, 10))
    corr_matrix = returns.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, center=0, vmin=-1, vmax=1, mask=mask)
    plt.title("Correlation Heatmap of Returns")
    plt.tight_layout()
    plt.savefig("graphs/correlation_heatmap.png")
    plt.show()  # plt.close() 대신에 실제로 그래프 표시

    for key in centralities_corr:
        print(f"\n🔎 Top 10 {key} Centrality (Correlation Network):")
        print(centralities_corr[key].sort_values(ascending=False).head(10))

        plt.figure(figsize=(10, 6))
        centralities_corr[key].sort_values(ascending=False).head(10).plot(kind='barh')
        plt.title(f"Top 10 {key} Centrality (Correlation Network)")
        plt.xlabel(f"{key} Centrality")
        plt.tight_layout()
        plt.savefig(f"graphs/top10_{key.lower()}_correlation.png")
        plt.show()  # plt.close() 대신에 실제로 그래프 표시

    for key in centralities_granger:
        print(f"\n🔎 Top 10 {key} Centrality (Granger Network):")
        print(centralities_granger[key].sort_values(ascending=False).head(10))

        plt.figure(figsize=(10, 6))
        centralities_granger[key].sort_values(ascending=False).head(10).plot(kind='barh', color='orange')
        plt.title(f"Top 10 {key} Centrality (Granger Network)")
        plt.xlabel(f"{key} Centrality")
        plt.tight_layout()
        plt.savefig(f"graphs/top10_{key.lower()}_granger.png")
        plt.show()  # plt.close() 대신에 실제로 그래프 표시

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    plt.figure(figsize=(18, 10))
    for i, t in enumerate(thresholds):
        G_t = build_correlation_network(returns, threshold=t)
        try:
            pagerank = nx.pagerank(G_t, max_iter=500)
        except:
            pagerank = {node: 1.0/len(G_t) for node in G_t.nodes()}
        
        pos = nx.spring_layout(G_t, seed=42)
        plt.subplot(2, 3, i+1)
        nx.draw_networkx_nodes(G_t, pos, node_color=list(pagerank.values()), cmap="viridis", node_size=300)
        nx.draw_networkx_edges(G_t, pos, alpha=0.4)
        plt.title(f"Threshold: {t:.2f}, Nodes: {G_t.number_of_nodes()}, Edges: {G_t.number_of_edges()}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("graphs/pagerank_threshold_comparison.png")
    plt.show()  # plt.close() 대신에 실제로 그래프 표시

    if correlation_backtest is not None and not correlation_backtest.empty and not granger_backtest.empty:
        plt.figure(figsize=(12, 6))
        
        # 누적 수익률 계산 및 그래프 그리기
        granger_cum = granger_backtest.cumsum()
        corr_cum = correlation_backtest.cumsum()
        
        granger_cum.plot(label="Granger")
        corr_cum.plot(label="Correlation")
        
        plt.title("Cumulative Return Comparison: Granger vs Correlation")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("graphs/backtest_comparison.png")
        plt.show()  # plt.close() 대신에 실제로 그래프 표시

    # 중앙성 간 상관관계 heatmap
    all_centralities = pd.DataFrame()
    for key in centralities_corr:
        all_centralities[f"{key}_Corr"] = centralities_corr[key]
    for key in centralities_granger:
        all_centralities[f"{key}_Granger"] = centralities_granger[key]
    
    if not all_centralities.empty:
        corr_matrix = all_centralities.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Centrality Measures Correlation Matrix")
        plt.tight_layout()
        plt.savefig("graphs/centrality_correlation_matrix.png")
        plt.show()  # plt.close() 대신에 실제로 그래프 표시

# MAIN EXECUTION
# ----------------------------------------
if __name__ == "__main__":
    start_date = "2022-04-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    print("데이터 수집 중...")
    price_df = get_all_stock_prices_parallel(tickers_info, start_date, end_date)
    
    if price_df.empty:
        print("주식 데이터를 가져오는데 실패했습니다.")
        exit(1)
        
    print(f"수집 완료: {len(price_df.columns)}개 종목")
    returns = price_df.pct_change().dropna()

    # Granger Network
    print("Granger 네트워크 구축 중...")
    G_granger = build_granger_network(returns)
    print(f"Granger 네트워크: {G_granger.number_of_nodes()}개 노드, {G_granger.number_of_edges()}개 엣지")
    
    centralities_granger = compute_centralities(G_granger)
    for key, ser in centralities_granger.items():
        ser.to_csv(f"results/granger_{key}_centrality.csv")
        plot_network(G_granger, ser, f"Granger Network ({key})", f"graphs/granger_network_{key}.png")

    # Correlation Network
    print("상관관계 네트워크 구축 중...")
    G_corr = build_correlation_network(returns, correlation_threshold)
    print(f"상관관계 네트워크: {G_corr.number_of_nodes()}개 노드, {G_corr.number_of_edges()}개 엣지")
    
    centralities_corr = compute_centralities(G_corr)
    for key, ser in centralities_corr.items():
        ser.to_csv(f"results/correlation_{key}_centrality.csv")
        plot_network(G_corr, ser, f"Correlation Network ({key})", f"graphs/correlation_network_{key}.png")

    # Monthly endpoints
    print("백테스팅 준비 중...")
    monthly_endpoints = price_df.groupby([price_df.index.year, price_df.index.month]).apply(lambda x: x.index[-1])
    monthly_endpoints = monthly_endpoints.reset_index(drop=True).sort_values()

    # Backtest for all centrality measures
    print("백테스팅 수행 중...")
    
    # 모든 중심성 지표에 대해 백테스트 수행
    centrality_types = ['Degree', 'Betweenness', 'PageRank', 'Eigenvector']
    
    backtest_results_granger = {}
    backtest_results_corr = {}
    
    for centrality_type in centrality_types:
        print(f"{centrality_type} 중심성에 대한 백테스트 수행 중...")
        backtest_results_granger[centrality_type] = backtest_strategy(
            price_df, monthly_endpoints, build_granger_network, 
            centrality_type=centrality_type, top_n=top_n
        )
        
        backtest_results_corr[centrality_type] = backtest_strategy(
            price_df, monthly_endpoints, build_correlation_network, 
            centrality_type=centrality_type, top_n=top_n
        )

    # 백테스트 결과 저장 및 시각화
    for centrality_type, result in backtest_results_granger.items():
        if not result.empty:
            result.to_csv(f"results/backtest_granger_{centrality_type.lower()}.csv")
            
            fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # 누적 수익률
            result.cumsum().plot(ax=ax[0], color='navy')
            ax[0].set_title(f"Granger {centrality_type} Centrality Strategy - Cumulative Return")
            ax[0].set_ylabel("Cumulative Return")
            ax[0].grid(True)

            # 월별 수익률
            result.plot(kind='bar', ax=ax[1], color='teal')
            ax[1].set_title("Monthly Returns")
            ax[1].set_ylabel("Return")
            ax[1].grid(True)
            ax[1].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(f"graphs/backtest_granger_{centrality_type.lower()}.png")
            plt.show()

            # 성과 지표 계산
            total_return = result.sum()
            annual_volatility = result.std() * np.sqrt(12)
            sharpe_ratio = result.mean() / result.std() * np.sqrt(12) if result.std() != 0 else 0
            mdd = (result.cumsum() / result.cumsum().cummax() - 1).min() if not result.cumsum().empty else 0

            print(f"\n📊 {centrality_type} 백테스트 성과 요약:")
            print(f"누적 수익률: {total_return:.2%}")
            print(f"연간 변동성: {annual_volatility:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {mdd:.2%}")
    
    # 결과 시각화 호출
    print("\n시각화 중...")
    display_numerical_results_and_visuals(
        price_df, returns,
        centralities_corr, centralities_granger,
        backtest_results_granger['Degree'], backtest_results_corr['Degree']
    )

    # 모든 중심성 지표 비교
    plt.figure(figsize=(14, 8))
    
    for centrality_type, result in backtest_results_granger.items():
        if not result.empty:
            result.cumsum().plot(label=f"Granger {centrality_type}")
    
    plt.title("Comparison of Different Centrality Measures (Granger)")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("graphs/comparison_centrality_granger.png")
    plt.show()

    print("\n✅ Analysis complete! All results are saved, backtesting also complete.")