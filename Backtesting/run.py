import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from backtesting import Backtest

# 导入你的策略
from long_only_strategy import LongOnlyStrategy
from improved_long_only_strategy import ImprovedLongOnlyStrategy
from stop_loss_strategy import StopLossStrategy
from multiIndicator_strategy import MultiIndicatorStrategy
from strategy_test import StrategyTest

def get_latest_data(ticker, lookback_days=100):
    """获取足够的历史数据用于计算指标"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        multi_level_index=False
    )
    
    return data

def get_live_signal(data, strategy_class):
    """基于历史数据运行策略并获取当前信号"""
    bt = Backtest(data, strategy_class, cash=10000, commission=.002)
    stats = bt.run()
    
    # 获取最后的指标值和价格
    current_price = data['Close'].iloc[-1]
    
    # 这里假设所有策略都有至少一个技术指标
    if hasattr(stats['_strategy'], '_indicators') and len(stats['_strategy']._indicators) > 0:
        current_indicator = stats['_strategy']._indicators[0][-1]
    else:
        current_indicator = None
        
    # 获取当前持仓状态
    current_position = 0
    if not stats['_trades'].empty:
        last_trade = stats['_trades'].iloc[-1]
        if pd.isna(last_trade['ExitTime']):  # 仍在持仓
            current_position = last_trade['Size']
    
    # 根据策略生成信号
    signal = "UNKNOWN"
    
    # 这是一个通用逻辑，你可能需要根据特定策略调整
    if strategy_class.__name__ == "LongOnlyStrategy" or strategy_class.__name__ == "ImprovedLongOnlyStrategy":
        if current_indicator and current_price > current_indicator:
            signal = "BUY" if current_position <= 0 else "HOLD"
        else:
            signal = "SELL" if current_position > 0 else "WAIT"
    elif strategy_class.__name__ == "StopLossStrategy":
        # 为止损策略添加特定逻辑
        # 这里是简化示例
        if current_indicator and current_price > current_indicator:
            signal = "BUY" if current_position <= 0 else "HOLD"
        else:
            signal = "SELL" if current_position > 0 else "WAIT"
    elif strategy_class.__name__ == "MultiIndicatorStrategy":
        # 为多指标策略添加特定逻辑
        # 你需要访问策略特有的指标
        if current_indicator and current_price > current_indicator:
            signal = "BUY" if current_position <= 0 else "HOLD"
        else:
            signal = "SELL" if current_position > 0 else "WAIT"
    
    return {
        'price': current_price,
        'indicator': current_indicator,
        'position': current_position,
        'signal': signal
    }

def live_trading_dashboard(ticker, strategy_classes):
    """为给定的股票和多个策略生成实盘交易信号"""
    data = get_latest_data(ticker)
    
    print(f"\n实盘交易分析 - {ticker} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据范围: {data.index[0].date()} 至 {data.index[-1].date()}")
    
    for strategy_class in strategy_classes:
        result = get_live_signal(data, strategy_class)
        
        print(f"\n策略: {strategy_class.__name__}")
        print(f"当前价格: {result['price']:.2f}")
        if result['indicator']:
            print(f"主要指标值: {result['indicator']:.2f}")
        print(f"当前信号: {result['signal']}")
        print(f"建议操作: {get_action_description(result['signal'])}")

def get_action_description(signal):
    """将信号转换为可执行的操作描述"""
    if signal == "BUY":
        return "买入新仓位"
    elif signal == "SELL":
        return "卖出所有持仓"
    elif signal == "HOLD":
        return "持有当前仓位"
    elif signal == "WAIT":
        return "等待入场机会"
    else:
        return "需要进一步分析"

if __name__ == "__main__":
    # 你可以选择一个或多个想要实盘操作的策略
    strategies = [LongOnlyStrategy, ImprovedLongOnlyStrategy, StopLossStrategy, MultiIndicatorStrategy, StrategyTest]
    
    # 分析一只或多只股票
    tickers = ["INTC"]
    for ticker in tickers:
        live_trading_dashboard(ticker, strategies)