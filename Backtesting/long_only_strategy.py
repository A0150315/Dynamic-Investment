from backtesting import Strategy
from backtesting.test import SMA
from datetime import date, timedelta


class LongOnlyStrategy(Strategy):
    def init(self):
        self.ma20 = self.I(SMA, self.data.Close, 20)

        self.max_risk_per_trade = 0.2

        # 添加字典用于记录入场价格
        self.entry_prices = {}

    def next(self):
        # print("*"*10)
        size = 0
        price = 0
        signal = "WAIT"
        available_cash = self.equity * self.max_risk_per_trade

        # position_status = "有持仓" if self.position else "无持仓"
        # print(f"日期:{self.data.index[-1]}, 收盘价:{self.data.Close[-1]:.2f}, MA20:{self.ma20[-1]:.2f}, {position_status}")
        if self.position:
            entry_price = self.entry_prices.get(self.ticker, 0)
            # print(f"持仓数量: {self.position.size}, 持仓成本: {entry_price:.2f}, 当前价格: {self.data.Close[-1]:.2f}")

        if self.data.Close[-1] > self.ma20[-1] and not self.position:
            price = self.data.Close[-1]
            size = int(available_cash / price)

            if size > 0:
                self.buy(size=size)
                # 记录入场价格
                self.entry_prices[self.ticker] = price
                signal = "BUY"
                # print(f"日期:{self.data.index[-1]}，买入信号: {size}股，每股价格 {price:.2f}，总金额 {size * price:.2f}")

        elif self.data.Close[-1] < self.ma20[-1] and self.position:
            self.position.close()
            # 清除入场价格记录
            if self.ticker in self.entry_prices:
                del self.entry_prices[self.ticker]
            signal = "SELL ALL"
            # print(f"日期:{self.data.index[-1]}，卖出信号: 价格 {self.data.Close[-1]:.2f} < MA20 {self.ma20[-1]:.2f}, 当前总金额 {self.equity:.2f}")

        today = date.today() - timedelta(days=1)
        last_data_date = self.data.index[-1].date()
        is_latest_data = today == last_data_date

        if is_latest_data:
            print(
                f"日期:{self.data.index[-1]}，信号: {signal}，总金额 {size * price:.2f}"
            )
            # echo csv
            with open("investment_results.csv", "a") as f:
                f.write(
                    f"{self.data.index[-1]},"
                    f"{self.ticker},"
                    f"{size * price:.2f},"
                    f"{price},"
                    f"{signal},"
                    f"{size},"
                )
