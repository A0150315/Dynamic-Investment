from backtesting import Strategy
from backtesting.test import SMA
import pandas as pd


class MultiIndicatorStrategy(Strategy):
    def init(self):
        # 均线
        self.ma20 = self.I(SMA, self.data.Close, 20)
        self.ma50 = self.I(SMA, self.data.Close, 50)
        
        # 添加RSI指标
        self.rsi = self.I(self.compute_rsi, self.data.Close, 14)
        
        # 添加成交量变化检测
        self.volume_ma = self.I(SMA, self.data.Volume, 20)
        
        self.max_risk_per_trade = 0.2
        self.entry_price = 0
    
    def compute_rsi(self, price, window):
        delta = pd.Series(price).diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def next(self):
        if not self.position:
            # 买入条件:
            # 1. 价格高于20日均线
            # 2. 20日均线高于50日均线（中期上升趋势）
            # 3. RSI低于70（非超买）
            # 4. 成交量高于平均成交量（放量上涨）
            if (self.data.Close[-1] > self.ma20[-1] and
                self.ma20[-1] > self.ma50[-1] and
                self.rsi[-1] < 70 and
                self.data.Volume[-1] > self.volume_ma[-1]):
                
                price = self.data.Close[-1]
                size = int(self.equity * self.max_risk_per_trade / price)
                
                if size > 0:
                    self.buy(size=size)
                    self.entry_price = price
                    # print(f"日期:{self.data.index[-1]}，买入信号: 多指标确认")
        
        elif self.position:
            # 卖出条件:
            # 1. 价格低于20日均线且RSI高于30（不是超卖）
            # 或者
            # 2. RSI超过80（超买区域）
            if ((self.data.Close[-1] < self.ma20[-1] and self.rsi[-1] > 30) or
                self.rsi[-1] > 80):
                
                self.position.close()
                # print(f"日期:{self.data.index[-1]}，卖出信号: {'价格跌破均线' if self.data.Close[-1] < self.ma20[-1] else 'RSI超买'}")