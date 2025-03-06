from backtesting import Strategy
from backtesting.test import SMA

class LongOnlyStrategy(Strategy):
    def init(self):
        self.ma20 = self.I(SMA, self.data.Close, 20)
        
        self.max_risk_per_trade = 0.2
    
    def next(self):
        available_cash = self.equity * self.max_risk_per_trade
        
        # position_status = "有持仓" if self.position else "无持仓"
        # print(f"日期:{self.data.index[-1]}, 收盘价:{self.data.Close[-1]:.2f}, MA20:{self.ma20[-1]:.2f}, {position_status}")
        
        if self.data.Close[-1] > self.ma20[-1] and not self.position:
            price = self.data.Close[-1]
            size = int(available_cash / price)
            
            if size > 0:
                self.buy(size=size)
                # print(f"日期:{self.data.index[-1]}，买入信号: {size}股，每股价格 {price:.2f}，总金额 {size * price:.2f}")
        
        elif self.data.Close[-1] < self.ma20[-1] and self.position:
            self.position.close()
            # print(f"日期:{self.data.index[-1]}，卖出信号: 价格 {self.data.Close[-1]:.2f} < MA20 {self.ma20[-1]:.2f}")