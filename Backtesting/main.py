import logging
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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

today = date.today()
handler = logging.FileHandler(f'{today.strftime("%Y-%m-%d")}.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def out_put_result(data):
    """
    根据回测结果和MLStrategy.trade_records，输出第二天的投资建议
    
    Args:
        data: 回测使用的价格数据
    
    Returns:
        None，直接打印投资建议
    """
    ticker = MLStrategy.ticker
    if not ticker or ticker not in MLStrategy.trade_records:
        print("未找到交易记录，无法提供建议")
        return
    
    # 获取ticker的交易记录
    record = MLStrategy.trade_records[ticker]
    last_action = record.get('last_action')
    last_action_time = record.get('last_action_time')
    last_action_price = record.get('last_action_price')
    # 获取最后一天的日期和价格
    last_date = data.index[-1]
    last_price = data.Close[-1]
    
    strategy_instance = None
    # 获取最后一天的预测
    if hasattr(MLStrategy, 'instance') and MLStrategy.instance:
        strategy_instance = MLStrategy.instance
        if hasattr(strategy_instance, 'predictions') and len(strategy_instance.predictions) > 0:
            last_prediction = strategy_instance.predictions[-1]
            dynamic_threshold = strategy_instance.dynamic_threshold
        else:
            print("未找到预测数据，无法提供建议")
            return
    else:
        # 尝试从模型重新获取预测
        try:
            # 创建临时策略实例进行预测
            temp_strategy = MLStrategy()
            strategy_instance = temp_strategy
            temp_strategy.ticker = ticker
            temp_strategy.data = data
            temp_strategy.init()
            temp_strategy._train_model()
            
            # 获取最后一天的预测
            last_prediction = temp_strategy.predictions[-1]
            dynamic_threshold = temp_strategy.dynamic_threshold
        except Exception as e:
            print(f"尝试获取预测时出错: {e}")
            return
    
    # 根据预测值和阈值确定投资建议
    next_date = last_date + pd.Timedelta(days=1)
    # 跳过周末
    while next_date.weekday() > 4:  # 5,6 是周六日
        next_date = next_date + pd.Timedelta(days=1)
    
    print("\n" + "="*50)
    print(f"当前日期: {last_date.strftime('%Y-%m-%d')} (星期{last_date.weekday()+1})")
    print(f"下一交易日: {next_date.strftime('%Y-%m-%d')} (星期{next_date.weekday()+1})")
    logging.info(f"最新收盘价: {last_price:.2f}")
    print(f"预测值: {last_prediction:.4f}, 阈值: {dynamic_threshold:.2f}")
    logging.info(f"最后一次操作: {last_action},时间: {last_action_time},价格: {last_action_price}")
    
    print("\n【投资建议】")
    logging.info(f"{ticker} {last_date.strftime('%Y-%m-%d')}")
    if last_prediction > dynamic_threshold:
        logging.info(f"预测上涨")
        buy_size = strategy_instance.calculate_position_size(last_prediction, last_price, strategy_instance.equity)
        if buy_size > 0:
            logging.info(f"买入：{buy_size}股")
            
        add_size = strategy_instance.calculate_position_size(last_prediction, last_price, strategy_instance.equity*0.3)
        if add_size > 0:
            logging.info(f"加仓：{add_size}股")

        if last_action == 'buy':
            if last_price > dynamic_threshold+0.05 and add_size > 0:
                logging.info(f"建议：当前价格高于阈值，可以加仓")
                logging.info(f"加仓：{add_size}股")
            else:
                # 已经买入，可以继续持有
                logging.info(f"建议: 继续持有 (WAIT)")
            logging.info(f"原因: 预测值 {last_prediction:.4f} > 阈值 {dynamic_threshold:.2f}，预计市场将继续上涨")
        elif buy_size > 0:
            # 还没买入或已经卖出
            logging.info(f"建议: 买入 (BUY)")
            logging.info(f"买入：{buy_size}股")
            logging.info(f"原因: 预测值 {last_prediction:.4f} > 阈值 {dynamic_threshold:.2f}，预计市场将上涨")
        else:
            logging.info(f"建议: 观望 (WAIT)")
            logging.info(f"原因: 预测值 {last_prediction:.4f} > 阈值 {dynamic_threshold:.2f}，预计市场将上涨，当前无持仓，无需操作")
    elif last_prediction < (1 - dynamic_threshold):
        logging.info(f"预测下跌，有就卖")
        if last_action == 'sell' or not last_action:
            # 已经卖出或从未买入
            logging.info(f"建议: 观望 (WAIT)")
            logging.info(f"原因: 预测值 {last_prediction:.4f} < 阈值 {1-dynamic_threshold:.2f}，预计市场将下跌，当前无持仓，无需操作")
        else:
            # 当前持有
            logging.info(f"建议: 卖出 (SELL)")
            logging.info(f"原因: 预测值 {last_prediction:.4f} < 阈值 {1-dynamic_threshold:.2f}，预计市场将下跌")
    else:
        logging.info(f"止损：{strategy_instance.stop_loss_pct},卖出")
        logging.info(f"止盈：{strategy_instance.take_profit_pct}，卖一半")
        # 观望
        logging.info(f"建议: 观望 (WAIT)")
        logging.info(f"原因: 预测值 {last_prediction:.4f} 在不确定区间 [{1-dynamic_threshold:.2f}, {dynamic_threshold:.2f}]，建议持观望态度")
    
    logging.info("="*50)
    
    # 输出持仓状态
    if 'open_trades' in record and record['open_trades']:
        total_shares = sum(size for _, _, size in record['open_trades'])
        total_cost = sum(price * size for _, price, size in record['open_trades'])
        avg_price = total_cost / total_shares if total_shares > 0 else 0
        
        # 计算当前盈亏
        if total_shares > 0:
            profit_pct = (last_price / avg_price - 1) * 100
            print(f"当前持仓: {total_shares:.0f} 股, 平均买入价: {avg_price:.2f}, "
                  f"当前盈亏: {profit_pct:.2f}%")


def main(ticker, end_date=None):
    if ticker in data_map:
        data = data_map[ticker]
    else:
        # start_date = "2008-01-01"
        today = date.today() if end_date is None else end_date.date()
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
        bt = Backtest(data, strategy, cash=2000, commission=0.0015)
        stats = bt.run()

        # 保存HTML文件，文件名包含ticker和策略名称
        html_filename = f"{output_dir}/{ticker}_{strategy.__name__}.html"
        # bt.plot(filename=html_filename)
        print(f"回测结果已保存至：{html_filename}")
        
        out_put_result(data)
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
    parser.add_argument("--ticker", type=str, help="股票代码，例如：AAPL")
    parser.add_argument("--date", type=str, help="指定日期进行分析 (格式: YYYY-MM-DD)，默认为今天")
    args = parser.parse_args()

    # 可选: 训练主模型
    if args.train_model:
        # 使用推荐的训练集
        if args.dataset in ["US_LARGE_CAP", "US_TECH", "CHINA", "INDICES", "ETF", "MIXED_OPTIMAL"]:
            training_tickers = get_recommended_training_set(args.dataset, years=args.years)
        else:
            # 使用默认混合数据集
            training_tickers = get_recommended_training_set("MIXED_OPTIMAL", years=args.years)
            
        print(f"使用 {len(training_tickers)} 只股票训练主模型...")
        train_master_model(training_tickers, years=args.years)

    # 如果提供了特定的股票代码，则使用它
    if args.ticker:
        tickers_to_test = [args.ticker]
    elif args.dataset is not None:
        tickers_to_test = get_recommended_training_set(args.dataset.upper())

    # 如果提供了特定日期，则设置结束日期
    if args.date:
        try:
            end_date = pd.to_datetime(args.date)
            print(f"分析截止到 {end_date.strftime('%Y-%m-%d')} 的数据")
        except:
            end_date = pd.to_datetime('today')
            print(f"日期格式无效，使用今天 {end_date.strftime('%Y-%m-%d')} 作为结束日期")
    else:
        end_date = pd.to_datetime('today')
        print(f"使用今天 {end_date.strftime('%Y-%m-%d')} 作为结束日期")

    for ticker in tickers_to_test:
        try:
            print(f"\n处理股票: {ticker}")
            main(ticker, end_date)
        except Exception as e:
            print(f"处理 {ticker} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 打印总结果
    print("\n================= 总结 =================")
    for strategy, total_return in sum_map.items():
        print(f"{strategy}: 总收益 = {total_return:.2f}")
    print("=========================================")

    import time
    time.sleep(5)  # 等待日志写入完成

    try:
        from log_mailer import send_log_by_email
        log_file_path = f"{today.strftime('%Y-%m-%d')}.log"
        send_log_by_email(log_file_path, delete_after=True)
    except Exception as e:
        print(f"发送日志邮件时出错: {e}")
