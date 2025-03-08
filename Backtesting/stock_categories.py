"""
股票分类模块 - 用于管理不同类型的股票分类信息和相关辅助功能
"""

# 股票分类字典
TECH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 
    'TSLA', 'AMD', 'INTC', 'AVGO', 'QCOM', 'CSCO', 'ORCL', 
    'IBM', 'CRM', 'ADBE', 'NFLX', 'PYPL', 'INTC', 'AMAT',
    'MU', 'TXN', 'UBER', 'ABNB', 'SNOW', 'ZM', 'PLTR'
]

CHINA_STOCKS = [
    'BABA', 'JD', 'BIDU', 'PDD', 'NTES', 'BILI', 'TCEHY', 
    'NIO', 'LI', 'XPEV', 'TME', 'YUMC', 'TAL', 'EDU', 'HTHT',
    'ZTO', 'VIPS', 'ATHM', 'BEKE', 'FUTU', 'DADA', 'IQ', 'GDS'
]

ETF_STOCKS = [
    'QQQ', 'SPY', 'DIA', 'IWM', 'EEM', 'XLK', 'XLF', 'XLV', 
    'XLE', 'XLI', 'XLP', 'XLU', 'XLB', 'XLY', 'ARKK', 'ARKG',
    'VTI', 'VEA', 'VWO', 'BND', 'VXUS', 'VNQ', 'VIG', 'GLD',
    'SLV', 'TLT', 'IEF', 'IEMG', 'IJR', 'IJH'
]

INDEX_STOCKS = [
    '^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX', '^FTSE', 
    '^N225', '^HSI', '^GDAXI', '^FCHI', '^TNX'
]

FINANCE_STOCKS = [
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 
    'V', 'MA', 'PYPL', 'SCHW', 'CB', 'PNC', 'TFC'
]

HEALTHCARE_STOCKS = [
    'JNJ', 'PFE', 'MRK', 'ABBV', 'ABT', 'UNH', 'TMO', 
    'DHR', 'BMY', 'AMGN', 'LLY', 'GILD', 'ISRG', 'BIIB'
]

CONSUMER_STOCKS = [
    'KO', 'PEP', 'PG', 'WMT', 'COST', 'HD', 'MCD', 'SBUX', 
    'NKE', 'DIS', 'NFLX', 'AMZN', 'BABA', 'JD', 'TGT'
]

# 行业分类映射
SECTOR_MAPPINGS = {
    'tech': TECH_STOCKS,
    'china': CHINA_STOCKS,
    'etf': ETF_STOCKS,
    'index': INDEX_STOCKS,
    'finance': FINANCE_STOCKS,
    'healthcare': HEALTHCARE_STOCKS,
    'consumer': CONSUMER_STOCKS
}

# 推荐的训练集股票（代表性强且数据质量高）
RECOMMENDED_TRAINING_STOCKS = {
    'US_LARGE_CAP': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'HD', 'NVDA', 'MRK', 'DIS'],
    'US_TECH': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM', 'INTC', 'AMD', 'IBM', 'QCOM', 'TXN'],
    'CHINA': ['BABA', 'JD', 'PDD', 'BIDU', 'TCEHY', 'NIO', 'NTES', 'BILI', 'TAL', 'YUMC'],
    'INDICES': ['^GSPC', '^DJI', '^IXIC', '^RUT'],
    'ETF': ['SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLV', 'XLE', 'VTI', 'VEA', 'VWO'],
    'MIXED_OPTIMAL': ['AAPL', 'MSFT', 'AMZN', 'SPY', 'QQQ', 'BABA', 'JD', 'JPM', 'JNJ', 'XOM', 'WMT', 'PFE', 'NVDA', 'TSLA']
}

def get_stock_type(ticker):
    """
    获取股票的分类类型
    
    参数:
    ticker: 股票代码
    
    返回:
    股票类型: 'tech', 'china', 'etf', 'index', 'finance', 'healthcare', 'consumer' 或 'other'
    """
    ticker = ticker.upper()  # 规范化处理
    
    # 检查是否为指数
    if ticker.startswith('^'):
        return 'index'
    
    # 检查各个类别
    for sector, stocks in SECTOR_MAPPINGS.items():
        if ticker in stocks:
            return sector
    
    # 默认分类为其他
    return 'other'

def get_sector_features(ticker):
    """
    为给定股票返回扇区特征的独热编码字典
    
    参数:
    ticker: 股票代码
    
    返回:
    包含扇区特征的字典 {'is_tech': 0/1, 'is_china': 0/1, ...}
    """
    stock_type = get_stock_type(ticker)
    
    features = {
        'is_tech': 1 if stock_type == 'tech' else 0,
        'is_china': 1 if stock_type == 'china' else 0,
        'is_etf': 1 if stock_type == 'etf' else 0,
        'is_index': 1 if stock_type == 'index' else 0,
        'is_finance': 1 if stock_type == 'finance' else 0,
        'is_healthcare': 1 if stock_type == 'healthcare' else 0,
        'is_consumer': 1 if stock_type == 'consumer' else 0
    }
    
    return features

def get_recommended_training_set(category='MIXED_OPTIMAL', limit=15):
    """
    获取推荐的训练股票列表
    
    参数:
    category: 类别名称，可选值: 'US_LARGE_CAP', 'US_TECH', 'CHINA', 'INDICES', 'ETF', 'MIXED_OPTIMAL'
    limit: 返回的最大股票数量
    
    返回:
    推荐的股票列表
    """
    if category in RECOMMENDED_TRAINING_STOCKS:
        stocks = RECOMMENDED_TRAINING_STOCKS[category]
        return stocks[:min(limit, len(stocks))]
    else:
        # 默认返回混合优化集
        return RECOMMENDED_TRAINING_STOCKS['MIXED_OPTIMAL'][:min(limit, len(RECOMMENDED_TRAINING_STOCKS['MIXED_OPTIMAL']))]

def add_stock_to_category(ticker, category):
    """
    将新股票添加到指定分类中
    
    参数:
    ticker: 股票代码
    category: 类别名称 ('tech', 'china', 等)
    """
    ticker = ticker.upper()
    
    if category in SECTOR_MAPPINGS:
        if ticker not in SECTOR_MAPPINGS[category]:
            SECTOR_MAPPINGS[category].append(ticker)
            return True
    return False

# 测试
if __name__ == "__main__":
    # 测试股票分类
    test_stocks = ['AAPL', 'BABA', 'SPY', '^GSPC', 'JPM', 'Unknown']
    
    for stock in test_stocks:
        print(f"{stock} 的类型是: {get_stock_type(stock)}")
        features = get_sector_features(stock)
        print(f"{stock} 的特征: {features}")
    
    # 测试获取推荐训练集
    print("\n推荐的训练股票:")
    for category in RECOMMENDED_TRAINING_STOCKS:
        stocks = get_recommended_training_set(category, 5)
        print(f"{category}: {stocks}") 