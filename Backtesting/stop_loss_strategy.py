from backtesting import Strategy
from backtesting.test import SMA

class StopLossStrategy(Strategy):
    def init(self):
        self.ma20 = self.I(SMA, self.data.Close, 20)
        self.max_risk_per_trade = 0.2
        
        # 止损和止盈参数
        self.stop_loss_pct = 0.05  # 5%止损
        self.take_profit_pct = 0.10  # 10%止盈
        
        # 记录买入价格
        self.entry_price = 0
    
    def next(self):
        available_cash = self.equity * self.max_risk_per_trade
        
        # 基本买入策略不变
        if self.data.Close[-1] > self.ma20[-1] and not self.position:
            price = self.data.Close[-1]
            size = int(available_cash / price)
            
            if size > 0:
                # print(f"日期:{self.data.index[-1]}，买入信号")
                self.buy(size=size)
                self.entry_price = price
        
        # 基于均线的卖出信号
        elif self.data.Close[-1] < self.ma20[-1] and self.position:
            self.position.close()
            # print(f"日期:{self.data.index[-1]}，卖出信号")
        
        # 止损策略
        elif self.position and (self.data.Close[-1] / self.entry_price - 1) <= -self.stop_loss_pct:
            self.position.close()
            # print(f"日期:{self.data.index[-1]}，卖出信号")
            
        # 止盈策略
        elif self.position and (self.data.Close[-1] / self.entry_price - 1) >= self.take_profit_pct:
            self.position.close()
            # print(f"日期:{self.data.index[-1]}，卖出信号")