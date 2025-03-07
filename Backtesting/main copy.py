import yfinance as yf
from backtesting import Backtest
import pandas as pd
from datetime import date, timedelta

from long_only_strategy import LongOnlyStrategy
from improved_long_only_strategy import ImprovedLongOnlyStrategy
from stop_loss_strategy import StopLossStrategy
from multiIndicator_strategy import MultiIndicatorStrategy
from strategy_test import StrategyTest
from mean_reversion_with_rsi import MeanReversionWithRSI

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

        # print(f"\n最后一次交易详情:")
        # print(f"- 买入时间: {last_trade['EntryTime']}")
        # print(f"- 买入价格: {last_trade['EntryPrice']:.2f}")
        # print(f"- 买入数量: {abs(last_trade['Size'])} 股")

        # if pd.isna(last_trade['ExitTime']):
        #     print(f"- 当前状态: 仍在持仓")
        #     print(f"- 当前价格: {current_price:.2f}")
        #     print(f"- 当前收益: {(current_price - last_trade['EntryPrice']) * abs(last_trade['Size']):.2f} " +
        #         f"({(current_price / last_trade['EntryPrice'] - 1) * 100:.2f}%)")
        # else:
        #     print(f"- 卖出时间: {last_trade['ExitTime']}")
        #     print(f"- 卖出价格: {last_trade['ExitPrice']:.2f}")
        #     print(f"- 交易收益: {last_trade['PnL']:.2f} ({last_trade['ReturnPct']:.2%})")

        # print(f"- 持仓时长: {last_trade['Duration']}")
    else:
        current_price = data["Close"].iloc[-1]
        current_ma20 = stats["_strategy"]._indicators[0][-1]
        suggestion = "BUY" if current_price > current_ma20 else "WAIT"
        # print(f"尚无交易记录")

    # print(f"\n当前市场分析:")
    # print(f"- 最新收盘价: {data['Close'].iloc[-1]:.2f}")
    # print(f"- 20日均线值: {stats['_strategy']._indicators[0][-1]:.2f}")
    # print(
    #     f"- 价格相对均线: {'高于均线' if data['Close'].iloc[-1] > stats['_strategy']._indicators[0][-1] else '低于均线'}"
    # )
    # print(f"\n今日操作建议: {suggestion}")

    # print(f"\n回测统计结果:")
    # print(f"- 起始资金: ${stats['_equity_curve']['Equity'].iloc[0]:.2f}")
    # print(f"- 最终资金: ${stats['_equity_curve']['Equity'].iloc[-1]:.2f}")
    # print(
    #     f"- 总收益率: {(stats['_equity_curve']['Equity'].iloc[-1] / stats['_equity_curve']['Equity'].iloc[0] - 1) * 100:.2f}%"
    # )
    # print(f"- 最大回撤: {stats['Max. Drawdown [%]']:.2f}%")
    # print(f"- 交易次数: {stats['# Trades']}")
    # print(f"- 胜率: {stats['Win Rate [%]']:.2f}%")


def main(ticker):
    if ticker in data_map:
        data = data_map[ticker]
    else:
        # start_date = "2008-01-01"
        today = date.today()
        start_date = today - timedelta(days=365 * 5)
        data = yf.download(
            ticker, start=start_date, end=today, interval="1d", multi_level_index=False
        )
        # data = yf.download(
        #     ticker,
        #     start=start_date,
        #     end=end_date,
        #     interval="1d",
        #     multi_level_index=False,
        # )
        data_map[ticker] = data

    # print(data.head())
    # print(data.isna().sum())
    # print(f"数据起止日期: {data.index[0]} 至 {data.index[-1]}")

    for strategy in [
        LongOnlyStrategy,
        # ImprovedLongOnlyStrategy,
        # StopLossStrategy,
        # MultiIndicatorStrategy,
        # MeanReversionWithRSI,
        # StrategyTest,
    ]:
        # print(f"\n\n############################回测策略: {strategy.__name__}")
        strategy.ticker = ticker
        bt = Backtest(data, strategy, cash=1000, commission=0.0015)
        stats = bt.run()
        # bt.plot()
        out_put_result(stats, data)
        result = (
            stats["_equity_curve"]["Equity"].iloc[-1]
            - stats["_equity_curve"]["Equity"].iloc[0]
        )
        if strategy.__name__ in sum_map:
            sum_map[strategy.__name__] += result
        else:
            sum_map[strategy.__name__] = result


if __name__ == "__main__":
    list1 = [
        "INTC",
        "WBA",
        "KHC",
        "M",
        "AAL",
        "NCLH",
        "PARA",
        "SLB",
        "BIIB",
        "NTES",
    ]
    list2 = [
        "AMD",
        "MCD",
        "AAPL",
        "TSLA",
        "TSM",
        "GOOG",
        "META",
        "QQQ",
        "MCD",
        "MSFT",
        "AMZN",
        "NVDA",
        "QCOM",
        "BABA",
    ]

    # for ticker in list1:
    #     # print(f"\n回测股票: {ticker}")
    #     main(ticker)

    # for key in sum_map:
    #     print(f"{key} : {sum_map[key]}")

    # sum_map = {}
    # for ticker in list2:
    #     # print(f"\n回测股票: {ticker}")
    #     main(ticker)
    # print(f"\n\n\n")
    # for key in sum_map:
    #     print(f"{key} : {sum_map[key]}")

    sum_map = {}
    for ticker in list1 + list2:
        print(f"\n回测股票: {ticker}")
        main(ticker)
    print(f"\n\n\n")
    for key in sum_map:
        print(f"{key} : {sum_map[key]}")
