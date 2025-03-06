from backtesting import Strategy
from backtesting.test import SMA
import pandas as pd


class ImprovedLongOnlyStrategy(Strategy):
    def init(self):
        # 保留原有20日均线
        self.ma20 = self.I(SMA, self.data.Close, 20)
        # 增加50日均线作为长期趋势判断
        self.ma50 = self.I(SMA, self.data.Close, 50)
        
        # 增加波动率指标
        self.atr = self.I(lambda x: pd.Series(x).rolling(14).std(), self.data.Close)
        
        self.max_risk_per_trade = 0.2
        # 设置最小持仓天数，避免频繁交易
        self.min_holding_days = 10
        self.days_since_entry = 0

    def next(self):
        # 打印当前日期

        if self.position:
            self.days_since_entry += 1
            
        available_cash = self.equity * self.max_risk_per_trade
        # position_status = "有持仓" if self.position else "无持仓"
        
        # 市场趋势判断
        # trend = "上升趋势" if self.ma20[-1] > self.ma50[-1] else "下降趋势"
        
        # 价格相对于均线的偏离程度
        deviation = (self.data.Close[-1] / self.ma20[-1] - 1) * 100
        
        # print(f"日期:{self.data.index[-1]}, 价格:{self.data.Close[-1]:.2f}, MA20:{self.ma20[-1]:.2f}, {position_status}, {trend}")
        
        # 买入条件：价格高于MA20且MA20上穿MA50（趋势转强）且没有持仓
        if (self.data.Close[-1] > self.ma20[-1] and 
            self.ma20[-1] > self.ma50[-1] and 
            self.ma20[-2] <= self.ma50[-2] and 
            not self.position):
            
            price = self.data.Close[-1]
            size = int(available_cash / price)
            
            if size > 0:
                self.buy(size=size)
                self.days_since_entry = 0
                # print(f"日期:{self.data.index[-1]}，买入信号: {size}股，每股价格 {price:.2f}，趋势转强")
        
        # 卖出条件：价格低于MA20且已持仓超过最小持仓天数，或者价格偏离均线过大（超过5%）
        elif ((self.data.Close[-1] < self.ma20[-1] and 
              self.position and 
              self.days_since_entry > self.min_holding_days) or
              (self.position and deviation < -5)):
            
            self.position.close()
            # print(f"日期:{self.data.index[-1]}，卖出信号: 价格 {self.data.Close[-1]:.2f} < MA20 {self.ma20[-1]:.2f} 或偏离过大")