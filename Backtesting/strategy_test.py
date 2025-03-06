from backtesting import Strategy
import pandas as pd
import numpy as np

class StrategyTest(Strategy):
    """
    基于Strategy002的策略，原作者: Gerald Lonlas
    github: https://github.com/freqtrade/freqtrade-strategies
    
    已转换为backtesting.py格式
    """
    
    # 最小收益目标 (不直接适用于backtesting.py，但可以在逻辑中使用)
    # minimal_roi = {"60": 0.01, "30": 0.03, "20": 0.04, "0": 0.05}
    
    # 止损设置 (将在策略逻辑中实现)
    stoploss_percent = 0.10
    
    def init(self):
        # 计算关键指标
        self.close = self.data.Close
        self.high = self.data.High
        self.low = self.data.Low
        self.open = self.data.Open
        self.volume = self.data.Volume
        
        # 计算RSI
        self.rsi = self.I(self.compute_rsi, self.close, 14)
        
        # 计算Fisher RSI转换
        self.fisher_rsi = self.I(self.compute_fisher_rsi, self.rsi)
        
        # 计算随机指标
        self.slowk = self.I(self.compute_stoch, self.high, self.low, self.close)
        
        # 计算布林带
        self.bb_lower = self.I(self.compute_bb_lower, self.close, self.high, self.low)
        
        # 计算抛物线SAR
        self.sar = self.I(self.compute_sar, self.high, self.low)
        
        # 计算锤子线形态
        self.hammer = self.I(self.compute_hammer, self.open, self.high, self.low, self.close)
        
        # 交易参数
        self.entry_price = 0
        self.entry_time = None
        self.trailing_stop = False
        
    def compute_rsi(self, price, window):
        # 计算RSI
        delta = pd.Series(price).diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def compute_fisher_rsi(self, rsi):
        # 计算Fisher RSI转换
        fisher_rsi = 0.1 * (rsi - 50)
        return (np.exp(2 * fisher_rsi) - 1) / (np.exp(2 * fisher_rsi) + 1)
    
    def compute_stoch(self, high, low, close, k_period=14, d_period=3):
        # 计算随机指标
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        highest_high = high_series.rolling(window=k_period).max()
        lowest_low = low_series.rolling(window=k_period).min()
        
        # 计算%K
        k = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
        
        # 计算%D (通常是%K的3日移动平均)
        d = k.rolling(window=d_period).mean()
        
        return k
    
    def compute_bb_lower(self, close, high, low, window=20, stds=2):
        # 计算布林带下轨
        typical_price = (high + low + close) / 3
        tp_series = pd.Series(typical_price)
        
        # 移动平均
        ma = tp_series.rolling(window=window).mean()
        
        # 标准差
        std = tp_series.rolling(window=window).std()
        
        # 下轨
        lower = ma - (std * stds)
        
        return lower
    
    def compute_sar(self, high, low, acceleration=0.02, maximum=0.2):
        # 简化版SAR计算
        # 注意：这是一个简化实现，不如talib的SAR准确
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        
        # 使用pandas计算一个简单的趋势指标作为SAR替代
        # 在实际应用中，最好使用talib的SAR实现
        trend = high_series.rolling(window=5).mean() - low_series.rolling(window=10).mean()
        
        # 我们这里简化为高点和低点的一个结合计算
        # 真实SAR计算比这复杂得多
        sar = high_series.rolling(window=5).max() - (high_series - low_series).rolling(window=5).mean() * 0.5
        
        return sar
    
    def compute_hammer(self, open_price, high, low, close):
        """
        计算锤子线形态，修复返回numpy数组而非单个值的问题
        """
        # 将输入转换为pandas Series以便于向量化处理
        open_series = pd.Series(open_price)
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        # 计算实体大小
        body = abs(close_series - open_series)
        # 计算高低点范围
        range_hl = high_series - low_series
        
        # 避免除零，替换零值为极小值
        range_hl = range_hl.replace(0, 0.0001)
        
        # 计算实体占总范围的比例
        body_to_range = body / range_hl
        
        # 计算下影线长度
        lower_shadow = pd.Series(index=close_series.index, dtype=float)
        # 阳线
        mask_bullish = close_series > open_series
        lower_shadow[mask_bullish] = open_series[mask_bullish] - low_series[mask_bullish]
        # 阴线
        mask_bearish = close_series <= open_series
        lower_shadow[mask_bearish] = close_series[mask_bearish] - low_series[mask_bearish]
        
        # 计算上影线长度
        upper_shadow = pd.Series(index=close_series.index, dtype=float)
        # 阳线
        upper_shadow[mask_bullish] = high_series[mask_bullish] - close_series[mask_bullish]
        # 阴线
        upper_shadow[mask_bearish] = high_series[mask_bearish] - open_series[mask_bearish]
        
        # 定义锤子线条件
        hammer_condition = ((body_to_range < 0.3) &  # 小实体
                            (lower_shadow > (2 * body)) &  # 长下影线
                            (upper_shadow < (0.1 * range_hl)))  # 几乎没有上影线
        
        # 创建结果数组
        result = hammer_condition.astype(float) * 100
        
        # 返回numpy数组
        return result.values
    
    def next(self):
        # 检查是否需要止损
        if self.position and self.position.is_long:
            # 计算当前亏损百分比
            current_loss_pct = (self.data.Close[-1] / self.entry_price - 1)
            
            # 如果亏损超过止损线，平仓
            if current_loss_pct < -self.stoploss_percent:
                self.position.close()
                print(f"触发止损: 亏损 {current_loss_pct:.2%}")
                return
        
        # 买入信号
        if not self.position:
            # 买入条件:
            # 1. RSI低于30（超卖）
            # 2. 慢随机指标低于20（超卖）
            # 3. 价格低于布林带下轨（超卖）
            # 4. 出现锤子线形态（潜在反转）
            if (self.rsi[-1] < 30 and
                self.slowk[-1] < 20 and
                self.bb_lower[-1] > self.close[-1] and
                self.hammer[-1] == 100):
                
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.entry_time = self.data.index[-1]
                print(f"买入信号: RSI={self.rsi[-1]:.2f}, SLOWK={self.slowk[-1]:.2f}, 锤子线形态确认")
        
        # 卖出信号
        elif self.position.is_long:
            # 卖出条件:
            # 1. SAR高于收盘价（趋势可能反转）且
            # 2. Fisher RSI大于0.3（不再超卖）
            if (self.sar[-1] > self.close[-1] and
                self.fisher_rsi[-1] > 0.3):
                
                self.position.close()
                print(f"卖出信号: SAR={self.sar[-1]:.2f} > 价格={self.close[-1]:.2f}, Fisher RSI={self.fisher_rsi[-1]:.2f}")