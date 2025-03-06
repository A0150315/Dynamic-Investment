from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import talib


class MeanReversionWithRSI(Strategy):
    # 定义可调参数
    n_short = 10  # 短期均线
    n_long = 50  # 长期均线
    rsi_period = 14
    rsi_upper = 70
    rsi_lower = 30
    stop_loss = 0.02  # 2%止损
    take_profit = 0.05  # 5%止盈
    position_size_pct = 0.1  # 每次使用10%的可用资金

    def init(self):
        # 使用内置的SMA函数计算短期和长期均线
        self.sma_short = self.I(SMA, self.data.Close, self.n_short)
        self.sma_long = self.I(SMA, self.data.Close, self.n_long)

        # 计算RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)

    def next(self):
        # 确保指标已初始化
        if len(self.data.Close) < self.n_long or not self.rsi[-1]:
            return

        # 当前价格
        price = self.data.Close[-1]

        # 如果已有持仓，检查止损和止盈
        if self.position:
            # 获取入场价格 - 使用持仓成本计算
            if self.position.is_long:
                # 对于多头持仓，使用pl和pl_pct计算入场价格
                entry_price = price / (1 + self.position.pl_pct)
            else:
                # 对于空头持仓，使用pl和pl_pct计算入场价格
                entry_price = price / (1 - self.position.pl_pct)
                
            if price <= entry_price * (1 - self.stop_loss):  # 止损
                self.position.close()
            elif price >= entry_price * (1 + self.take_profit):  # 止盈
                self.position.close()
            return

        available_cash = self.equity * self.position_size_pct  # 可用于交易的资金
        size = int(available_cash / price)  # 计算可买入的股数（取整）

        # 买入条件：短期均线上穿长期均线，且RSI不过超买
        if crossover(self.sma_short, self.sma_long) and self.rsi[-1] < self.rsi_upper:
            if size > 0:  # 确保有足够资金买入至少1股
                self.buy(size=size)

        # 卖出条件：短期均线下穿长期均线，且RSI不过超卖，仅平仓
        elif crossover(self.sma_long, self.sma_short) and self.rsi[-1] > self.rsi_lower and self.position:
            self.position.close()  # 平仓所有持仓