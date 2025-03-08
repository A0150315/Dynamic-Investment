"""
股票分类模块 - 用于管理不同类型的股票分类信息和相关辅助功能
"""

# 股票分类字典
TECH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 
    'TSLA', 'AMD', 'INTC', 'AVGO', 'QCOM', 'CSCO', 'ORCL', 
    'IBM', 'CRM', 'ADBE', 'NFLX', 'PYPL', 'INTC', 'AMAT',
    'MU', 'TXN', 'UBER', 'ABNB', 'SNOW', 'ZM', 'PLTR', "SPOTY"
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
    'NKE', 'DIS', 'NFLX', 'AMZN', 'BABA', 'JD', 'TGT', "SPOTY"
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

US_TECH = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM', 
        'INTC', 'AMD', 'IBM', 'QCOM', 'TXN', 'TSLA', 'NFLX', 'PYPL', 'ORCL', 'AMAT',
        'MU', 'KLAC', 'NOW', 'LRCX', 'SNPS', 'CDNS', 'INTU', 'ADI', 'UBER', 'ABNB'
    ]

INDICES = [
        '^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX', '^FTSE', '^N225', '^HSI', '^GDAXI', '^FCHI'
    ]

US_FINANCE = [
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA', 
        'SCHW', 'USB', 'PNC', 'TFC', 'BK', 'COF', 'AIG', 'MET', 'PRU', 'CB'
    ]

US_HEALTHCARE = [
        'JNJ', 'PFE', 'MRK', 'ABBV', 'ABT', 'UNH', 'TMO', 'DHR', 'BMY', 'AMGN', 
        'LLY', 'GILD', 'ISRG', 'BIIB', 'CVS', 'VRTX', 'REGN', 'MRNA', 'ZTS', 'BSX'
    ]

US_CONSUMER = [
        'KO', 'PEP', 'PG', 'WMT', 'COST', 'HD', 'MCD', 'SBUX', 'NKE', 'DIS',
        'TGT', 'LOW', 'YUM', 'MDLZ', 'CL', 'EL', 'MO', 'PM', 'AMZN', 'EBAY'
    ]

# 推荐的训练集股票（代表性强且数据质量高）
RECOMMENDED_TRAINING_STOCKS = {
    
    'MY': [
        'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'NVDA', 
        'INTC', 'ADBE', 'NFLX', "TSLA", "TSM", "AMD", "MCD","SPOTY",
        "NTES", "TCEHY", "BABA", "BIDU"
    ],

    'US_LARGE_CAP': [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'JNJ', 'V', 'PG', 'JPM', 
        'UNH', 'HD', 'NVDA', 'MRK', 'DIS', 'PFE', 'KO', 'PEP', 'CSCO', 'VZ', 
        'INTC', 'CMCSA', 'ADBE', 'NFLX', 'CRM', 'ABT', 'TMO', 'CVX', 'XOM', 'COST'
    ],
    
    'US_TECH': US_TECH,
    "TECH":US_TECH,
    
    'US_FINANCE': US_FINANCE,
    "FINANCE":US_FINANCE,
    
    'US_HEALTHCARE': US_HEALTHCARE,
    "HEALTHCARE":US_HEALTHCARE,
    
    
    'US_CONSUMER': US_CONSUMER,
    "CONSUMER":US_CONSUMER,
    
    
    'CHINA': [
        'BABA', 'JD', 'PDD', 'BIDU', 'TCEHY', 'NIO', 'NTES', 'BILI', 'TAL', 'YUMC',
        'LI', 'XPEV', 'TME', 'HTHT', 'ZTO', 'VIPS', 'ATHM', 'BEKE', 'FUTU', 'DADA',
        'IQ', 'GDS', 'BGNE', 'EDU', 'VNET', 'WB', 'YY', 'JOBS', 'MOMO', 'NOAH'
    ],
    
    'CHINA_TECH': [
        'BABA', 'JD', 'BIDU', 'PDD', 'TCEHY', 'NTES', 'BILI', 'FUTU', 'GDS', 'IQ',
        'WB', 'YY', 'VNET', 'JOYY', 'KC', 'ATHM', 'QTT', 'HUYA', 'DOYU', 'WUBA'
    ],
    
    'CHINA_CONSUMER': [
        'JD', 'PDD', 'YUMC', 'HTHT', 'VIPS', 'BEKE', 'NIU', 'ZTO', 'DADA', 'YSG',
        'LX', 'GOTU', 'LKNCY', 'ZH', 'TCOM', 'SXTC', 'EDU', 'BEDU', 'LAIX', 'TAL'
    ],
    
    "INDEX":INDICES,
    'INDICES': INDICES,
    
    'ETF': [
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLV', 'XLE', 'VTI', 'VEA', 
        'VWO', 'VNQ', 'VIG', 'VOO', 'ARKK', 'LQD', 'TLT', 'HYG', 'AGG', 'GLD'
    ],
    
    'MIXED_OPTIMAL': [
        # 美国大型科技
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA',
        # 中国科技
        'BABA', 'JD', 'PDD', 'BIDU', 'TCEHY',
        # 金融
        'JPM', 'V', 'BAC', 
        # 医疗保健
        'JNJ', 'PFE', 'UNH',
        # 消费
        'WMT', 'KO', 'MCD',
        # 指数和ETF
        'SPY', 'QQQ', '^GSPC', '^VIX',
        # 能源
        'XOM', 'CVX'
    ],
    
    'CHINA_FOCUSED': [
        # 中国股票
        'BABA', 'JD', 'PDD', 'BIDU', 'TCEHY', 'NIO', 'NTES', 'BILI', 'TAL', 'YUMC',
        'LI', 'XPEV', 'TME', 'HTHT', 'ZTO', 'VIPS', 'FUTU', 'DADA',
        # 相关指数和ETF
        '^HSI', 'MCHI', 'KWEB', 'FXI', 'CQQQ', 'CHIQ', 'CHIX',
        # 几个美国大盘股作为对照
        'AAPL', 'MSFT', 'AMZN', 'SPY'
    ],
    
    'TECH_FOCUSED': [
        # 美国科技
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
        'ADBE', 'PYPL', 'NFLX', 'CSCO', 'AVGO', 'QCOM', 'MU', 'AMAT', 'TXN', 'KLAC',"SPOTY",
        # 中国科技
        'BABA', 'JD', 'BIDU', 'PDD', 'TCEHY', 'NTES', 'BILI',
        # 相关ETF
        'XLK', 'QQQ', 'ARKK', 'SMH', 'SOXX'
    ],
    
    'BALANCED_GLOBAL': [
        # 美国各行业
        'AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'WMT', 'XOM', 'PG', 'DIS', 'KO',
        # 中国
        'BABA', 'JD', 'PDD', 'BIDU', 'TCEHY',
        # 欧洲
        'VOD', 'BP', 'GSK', 'BTI', 'UL',
        # 全球ETF
        'SPY', 'EFA', 'EEM', 'VEU', 'ACWI',
        # 行业ETF
        'XLK', 'XLF', 'XLV', 'XLE', 'XLY'
    ]
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

def get_recommended_training_set(category='MIXED_OPTIMAL', limit=50):
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