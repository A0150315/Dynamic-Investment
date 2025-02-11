import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np  # 确保导入了 numpy
import logging

from main import calculate_market_indicators, load_config, calculate_trend_score, calculate_risk_score, should_pause_investment, get_market_signal  # 导入所有需要的函数


# 配置日志记录, 添加时间戳
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_dynamic_investment_strategy(data, config, i, total_assets):
    """
    计算当前时间步的投资权重。
    Args:
        data: 完整的历史数据 (DataFrame)。
        config: 配置字典。
        i: 当前时间步的索引。
        total_assets: 当前时间步开始时的总资产。

    Returns:
        float: 当前时间步的投资权重。
    """

    # 1. 获取当前时间步的数据切片
    #   - 为了计算指标，我们需要足够多的历史数据，所以要从 i 往前取足够多的数据
    max_sma_period = max(config.get("sma_periods", [20, 50, 200]))
    start_index = max(0, i - max_sma_period + 1)  # 确保 start_index 不小于 0
    data_slice = data.iloc[start_index:i+1]
   

    # 2. 计算指标
    indicators = calculate_market_indicators(data_slice, periods=config.get("sma_periods", [20, 50, 200]), rsi_period=config.get("rsi_period", 14))

    # 3. 获取当前价格 (收盘价)
    current_price = data['Close'].iloc[i]

    # 4. 计算投资金额 (与 calculate_dynamic_investment 逻辑类似)
    trend_score = calculate_trend_score(current_price, indicators,
                                       sma_weights=config.get('sma_weights', {'sma_200': 10, 'sma_50': 5, 'sma_20': 5}),
                                       macd_weight=config.get('macd_weight', 10),
                                       rsi_weights=config.get('rsi_weights', {'normal': 10, 'overbought': -10, 'oversold': 10}))

    risk_score = calculate_risk_score(current_price, data_slice, indicators)

    trend_adjustment = (trend_score - 50) / 100
    risk_adjustment = (50 - risk_score) / 100
    base_adjustment = 1 + trend_adjustment + risk_adjustment
    investment_amount = config['base_investment'] * base_adjustment

    if config['risk_control'] and risk_score > 75:
        investment_amount *= 0.5
    if config['risk_control'] and should_pause_investment(current_price, indicators, config['pause_threshold']):
        investment_amount = 0

    upper_limit = config['base_investment'] * config['upper_limit_factor']
    lower_limit = config['base_investment'] * config['lower_limit_factor']
    investment_amount = max(min(investment_amount, upper_limit), lower_limit)

    # 5. 计算权重
    weight = investment_amount / total_assets if total_assets > 0 else 0.0

    return weight


class DynamicInvestmentStrategy(bt.Strategy):
    """
    定义 bt 的策略类。
    """
    params = (('config', None),)  # 使用 params 来接收 config

    def __init__(self):
        super(DynamicInvestmentStrategy, self).__init__()
        if self.params.config is None:
             raise ValueError("配置未提供给策略")
        self.config = self.params.config
        self.total_assets = self.config['initial_capital']  # 初始化总资产
        self.shares = 0  # 初始化持仓数量
        self.first_trade = True #是否首次交易

    def next(self):
        """
        在每个时间步执行的逻辑。
        """

        # 确保有足够的数据来计算指标 (至少要有最长 SMA 周期的数据)
        max_sma_period = max(self.config.get("sma_periods", [200]))
        if len(self.data) < max_sma_period:
            return
        # 构建 DataFrame
        data_df = pd.DataFrame({
            'Open': self.data.open.get(ago=0, size=len(self.data)),
            'High': self.data.high.get(ago=0, size=len(self.data)),
            'Low': self.data.low.get(ago=0, size=len(self.data)),
            'Close': self.data.close.get(ago=0, size=len(self.data)),
            'Volume': self.data.volume.get(ago=0, size=len(self.data)),
        }, index=self.datas[0].datetime.get(ago=0, size=len(self.data)))
        #data_df.index = pd.to_datetime(data_df.index, unit='D') #删除这行
        data_df.index = [bt.num2date(x) for x in data_df.index] # 使用 bt.num2date

        #如果是首次交易, 且有收盘数据
        if self.first_trade and len(self.data.close) > 0:
            weight = calculate_dynamic_investment_strategy(data_df, self.config, len(self.data)-1, self.total_assets)
            #全仓买入
            self.order_target_percent(target=weight)
            #更新持仓和总资产
            price = self.data.close[0]  # 获取当前价格
            self.shares = self.total_assets * weight / price  # 计算买入的份额，这里做了简化，没有考虑手续费
            self.total_assets = self.total_assets * (1-weight) + self.shares * price
            self.first_trade = False #首次交易完成
            logging.info(f"{self.datetime.date()}: 首笔投资 - 权重 {weight:.4f}, 价格 {price:.2f}, 持仓 {self.shares:.2f}, 总资产 {self.total_assets:.2f}")
            return

        #非首次交易
        # 获取当前价格
        price = self.data.close[0]
        #计算当前总资产 = 现金 + 股票市值 （这里做了简化，假设没有其他费用）
        self.total_assets = self.broker.get_cash() + self.shares * price

        weight = calculate_dynamic_investment_strategy(data_df, self.config, len(self.data)-1, self.total_assets)

        # bt 会根据权重自动进行交易
        self.order_target_percent(target=weight)

        # 更新持仓数量和总资产
        if weight > 0: #买入
          #获取交易后的持仓
          new_shares = self.getposition(self.data).size  # 获取当前持仓数量 (bt 会自动更新)
          #计算由于交易新增的持仓
          delta_shares = new_shares - self.shares
          #更新总资产： 扣除买入股票的金额 =  原来的现金 - 新增股票的金额
          self.total_assets -= delta_shares * price

        #更新持仓
        self.shares = self.getposition(self.data).size

        logging.info(f"{self.datetime.date()}: 权重 {weight:.4f}, 价格 {price:.2f}, 持仓 {self.shares:.2f}, 总资产 {self.total_assets:.2f}")


def run_backtest(config):
    """
    运行回测。
    """
    # 获取历史数据
    ticker = config['ticker']
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 回测5年数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    print(f"获取 {ticker} 的历史数据 ({start_date} - {end_date})...")
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"获取到 {len(data)} 条数据。")
    if data.empty:
        raise ValueError("获取数据失败")

    # 将价格数据转换为 bt 需要的格式 (必须只有 close 列, 列名是ticker)
    bt_data = data[['Close']].copy()
    bt_data.columns = [ticker]

    bt_data = bt.feeds.PandasData(dataname=bt_data)

    # 创建策略, 传入 config
    strategy = DynamicInvestmentStrategy

    # 创建回测任务
    cerebro = bt.Cerebro()  # 创建 Cerebro 引擎
    cerebro.addstrategy(strategy, config=config)  # 添加策略，并传入 config
    cerebro.adddata(bt_data) #添加数据
    cerebro.broker.setcash(config['initial_capital']) #设置初始资金

    # 设置佣金（可选）
    cerebro.broker.setcommission(commission=0.001)  # 设置 0.1% 的佣金

    # 运行回测
    results = cerebro.run()

    # 获取回测结果 (bt 默认返回一个 list, 其中包含策略实例)
    strategy_instance = results[0]
    print(strategy_instance.analyzers)
    return strategy_instance.analyzers.getbyname('pyfolio') # 获取 PyFolio 分析器返回的结果


if __name__ == "__main__":
    # 加载配置
    config = load_config("config.json")
    # 设置初始资金
    config['initial_capital'] = 10000

    # 运行回测
    result = run_backtest(config)

     # 查看回测结果（使用 PyFolio）
    if result:
        returns, positions, transactions, gross_lev = result.get_pf_items()

        print("回测结果:")
        print("\n收益:")
        print(returns.tail())

        print("\n持仓:")
        print(positions.tail())

        print("\n交易:")
        print(transactions.tail())
    else:
        print("回测失败。")