import yfinance as yf
from backtesting import Backtest
import pandas as pd
from datetime import date, timedelta
import os
import argparse

# from long_only_strategy import LongOnlyStrategy
# from improved_long_only_strategy import ImprovedLongOnlyStrategy
# from stop_loss_strategy import StopLossStrategy
# from multiIndicator_strategy import MultiIndicatorStrategy
# from strategy_test import StrategyTest
# from mean_reversion_with_rsi import MeanReversionWithRSI
from ml_strategy import MLStrategy
from model_trainer import train_master_model
# 导入股票分类模块
from stock_categories import get_recommended_training_set

data_map = {}

sum_map = {}

def out_put_result(stats, data):
    if not stats["_trades"].empty:
        last_trade = stats["_trades"].iloc[-1]

        current_price = data["Close"].iloc[-1]
        current_ma20 = stats["_strategy"]._indicators[0][-1]

        if current_price > current_ma20:
            suggestion = "BUY" if not pd.isna(last_trade["ExitTime"]) else "HOLD"
        else:
            suggestion = "SELL" if pd.isna(last_trade["ExitTime"]) else "WAIT"
    else:
        current_price = data["Close"].iloc[-1]
        current_ma20 = stats["_strategy"]._indicators[0][-1]
        suggestion = "BUY" if current_price > current_ma20 else "WAIT"
        
    # 如果是ML策略，打印我们自己记录的交易统计
    if hasattr(stats["_strategy"].__class__, 'get_trade_stats') and stats["_strategy"].__class__.ticker:
        print("\n我们的交易记录统计:")
        try:
            trade_stats = stats["_strategy"].__class__.get_trade_stats(stats["_strategy"].__class__.ticker)
            print(f"总交易次数: {trade_stats['total_trades']}")
            print(f"胜率: {trade_stats['win_rate']*100:.2f}%")
            print(f"平均收益率: {trade_stats['avg_return']:.2f}%")
            print(f"最佳交易: {trade_stats['best_trade']:.2f}%")
            print(f"最差交易: {trade_stats['worst_trade']:.2f}%")
            print(f"当前持仓数量: {trade_stats['open_positions']}")
        except Exception as e:
            print(f"获取交易统计时出错: {e}")


def main(ticker):
    if ticker in data_map:
        data = data_map[ticker]
    else:
        # start_date = "2008-01-01"
        today = date.today()
        start_date = today - timedelta(days=365 * 5)
        data = yf.download(
            ticker, 
            start=start_date, 
            end=today, 
            interval="1d", 
            auto_adjust=True,
            multi_level_index=False
        )
        
        # 确保数据列是一维的
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if hasattr(data[col], 'values') and data[col].values.ndim > 1:
                data[col] = data[col].values.flatten()
                
        data_map[ticker] = data

    # print(data.head())
    # print(data.isna().sum())
    # print(f"数据起止日期: {data.index[0]} 至 {data.index[-1]}")

    # 创建输出文件夹（如果不存在）
    output_dir = "backtest_results"
    os.makedirs(output_dir, exist_ok=True)

    for strategy in [
        # LongOnlyStrategy,
        MLStrategy,
        # ImprovedLongOnlyStrategy,
        # StopLossStrategy,
        # MultiIndicatorStrategy,
        # MeanReversionWithRSI,
        # StrategyTest,
    ]:
        # 设置使用主模型的标志
        if strategy == MLStrategy:
            MLStrategy.use_master_model = True
        else:
            MLStrategy.use_master_model = False
            
        print(f"\n############################回测策略: {strategy.__name__} - {ticker}")
        strategy.ticker = ticker
        bt = Backtest(data, strategy, cash=1000, commission=0.0015)
        stats = bt.run()

        # 保存HTML文件，文件名包含ticker和策略名称
        html_filename = f"{output_dir}/{ticker}_{strategy.__name__}.html"
        bt.plot(filename=html_filename)
        print(f"回测结果已保存至：{html_filename}")
        
        out_put_result(stats, data)
        result = (
            stats["_equity_curve"]["Equity"].iloc[-1]
            - stats["_equity_curve"]["Equity"].iloc[0]
        )
        
        # 更新结果汇总
        strategy_key = f"{strategy.__name__}"
        if strategy_key in sum_map:
            sum_map[strategy_key] += result
        else:
            sum_map[strategy_key] = result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="股票回测程序")
    parser.add_argument("--train-model", action="store_true", help="是否先训练主模型")
    parser.add_argument("--years", type=int, default=10, help="训练数据的年数")
    parser.add_argument("--dataset", type=str, default="MIXED_OPTIMAL", 
                      help="使用推荐的训练数据集类型: US_LARGE_CAP, US_TECH, CHINA, INDICES, ETF, MIXED_OPTIMAL")
    args = parser.parse_args()
    
    # 允许自定义股票列表或使用推荐数据集
    custom_stocks = {
        "CHINA": [
            "JD",
            "NTES",
            "PDD",
            "BILI",
            "TCEHY",
            "BABA",
        ],
        "US": [
            "MMM",
            "INTC",
            "AMD",
            "MCD",
            "AAPL",
            "TSLA",
            "TSM",
            "GOOG",
            "META",
            "QQQ",
            "SPY",
            "MSFT",
            "AMZN",
            "NVDA",
            "QCOM",
        ]
    }
    
    # 根据参数选择数据集
    if args.dataset and args.dataset.upper() in ["MY","US_LARGE_CAP", "US_TECH", "CHINA", "INDICES", "ETF", "MIXED_OPTIMAL"]:
        # 使用推荐的训练数据集
        print(f"使用推荐的 {args.dataset} 数据集进行训练")
        test_stocks = get_recommended_training_set(args.dataset.upper())
    else:
        # 使用自定义股票集
        print("使用自定义股票列表进行训练")
        test_stocks = custom_stocks["MY"]
    
    # 如果需要训练主模型
    if args.train_model:
        print("开始训练主模型...")
        model = train_master_model(years=args.years, category=args.dataset.upper())
        if model is not None:
            print("主模型训练成功，将用于回测")
            MLStrategy.master_model = model
        else:
            print("主模型训练失败")

    sum_map = {}
    for ticker in test_stocks:
        print(f"\n回测股票: {ticker}")
        main(ticker)
    
    print(f"\n\n\n各策略回测结果汇总:")
    for key in sum_map:
        print(f"{key} : {sum_map[key]}")
