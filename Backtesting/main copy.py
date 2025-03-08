import yfinance as yf
from backtesting import Backtest
import pandas as pd
from datetime import date, timedelta
import os
import argparse

from long_only_strategy import LongOnlyStrategy
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


def main(ticker, use_master_model=False):
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
        if strategy == MLStrategy and use_master_model:
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
    use_master_model = True
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
    if args.dataset and args.dataset.upper() in ["US_LARGE_CAP", "US_TECH", "CHINA", "INDICES", "ETF", "MIXED_OPTIMAL"]:
        # 使用推荐的训练数据集
        print(f"使用推荐的 {args.dataset} 数据集进行训练")
        training_stocks = get_recommended_training_set(args.dataset.upper())
        # 测试使用自定义的中国股票
        test_stocks = custom_stocks["CHINA"]
    else:
        # 使用自定义股票集
        print("使用自定义股票列表进行训练")
        training_stocks = custom_stocks["CHINA"] + custom_stocks["US"]
        test_stocks = custom_stocks["CHINA"]
    
    # 如果需要训练主模型
    if args.train_model:
        use_master_model = False
        print("开始训练主模型...")
        print(f"训练股票: {training_stocks}")
        model = train_master_model(training_stocks, years=args.years)
        if model is not None:
            print("主模型训练成功，将用于回测")
            MLStrategy.master_model = model
            use_master_model = True
        else:
            print("主模型训练失败")
            use_master_model = False
    else:
        use_master_model = False
        
    # 如果是使用主模型，尝试加载
    if use_master_model and MLStrategy.master_model is None:
        # 检查模型文件是否存在
        if os.path.exists(MLStrategy.master_model_path):
            try:
                import joblib
                print(f"尝试从 {MLStrategy.master_model_path} 加载模型...")
                MLStrategy.master_model = joblib.load(MLStrategy.master_model_path)
                print(f"已从文件加载主模型: {MLStrategy.master_model_path}")
                
                # 测试模型是否可用于预测
                try:
                    # 检查模型需要多少特征
                    feature_count = 32  # 默认特征数
                    
                    # 如果模型有feature_names_in_属性，获取确切的特征数
                    if hasattr(MLStrategy.master_model, 'feature_names_in_'):
                        feature_count = len(MLStrategy.master_model.feature_names_in_)
                        print(f"模型需要 {feature_count} 个特征")
                        
                    # 如果是Pipeline，可能需要检查其中的特定组件
                    elif hasattr(MLStrategy.master_model, 'steps'):
                        for name, step in MLStrategy.master_model.steps:
                            if hasattr(step, 'feature_names_in_'):
                                feature_count = len(step.feature_names_in_)
                                print(f"模型组件 {name} 需要 {feature_count} 个特征")
                                break
                    
                    # 创建一个匹配的测试数据集
                    import numpy as np
                    test_X = np.random.rand(1, feature_count)
                    
                    # 检查是否有特征名信息文件
                    features_file = MLStrategy.master_model_path.replace('.pkl', '_features.txt')
                    feature_names = []
                    
                    if os.path.exists(features_file):
                        with open(features_file, 'r') as f:
                            feature_names = [line.strip() for line in f.readlines()]
                        print(f"从文件加载了 {len(feature_names)} 个特征名")
                        
                        if len(feature_names) == feature_count:
                            # 创建带有特征名的测试数据
                            test_df = pd.DataFrame([np.random.random(feature_count)], 
                                                 columns=feature_names)
                            _ = MLStrategy.master_model.predict_proba(test_df)
                        else:
                            print(f"特征名数量 ({len(feature_names)}) 与模型期望的特征数 ({feature_count}) 不匹配")
                            _ = MLStrategy.master_model.predict_proba(test_X)
                    else:
                        # 使用简单的numpy数组
                        _ = MLStrategy.master_model.predict_proba(test_X)
                    
                    print("模型测试成功，可以正常预测")
                except Exception as e:
                    print(f"模型测试失败，无法预测: {e}")
                    use_master_model = False
                    MLStrategy.master_model = None
            except Exception as e:
                print(f"加载主模型失败: {e}")
                use_master_model = False
        else:
            print(f"主模型文件不存在: {MLStrategy.master_model_path}")
            use_master_model = False

    sum_map = {}
    for ticker in test_stocks:
        print(f"\n回测股票: {ticker}")
        main(ticker, use_master_model=use_master_model)
    
    print(f"\n\n\n各策略回测结果汇总:")
    for key in sum_map:
        print(f"{key} : {sum_map[key]}")
