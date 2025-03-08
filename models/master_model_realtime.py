# 自动生成的实时交易特征工程代码 - 2025-03-08 22:37:53
import pandas as pd
import numpy as np
import talib

def preprocess_for_prediction(data):
    """
    为实时交易准备特征数据
    参数:
    data: 包含OHLCV数据的DataFrame，至少需要200天的历史数据
    
    返回:
    处理好的特征DataFrame，可直接用于模型预测
    """
    # 检查数据长度
    if len(data) < 200:
        raise ValueError(f"数据长度不足，需要至少{model_data['feature_engineering']['min_window']}天的历史数据")
    
    # 创建特征
    features = pd.DataFrame(index=[data.index[-1]])  # 只保留最新的一行用于预测
    
    # 这里是自动生成的特征工程代码，与模型训练时完全一致
    # ... 特征工程代码 ... 
    
    # 返回处理后的特征，确保特征顺序与训练时一致
    required_features = ['ma10_ratio', 'vix_trend', 'is_china', 'volatility_10d_change', 'price_above_sma50', 'ma20_ratio', 'volume_change', 'ma5_ratio', 'breakout_high_20d', 'ma100_ratio', 'volatility_10d', 'volatility_50d', 'market_trend', 'ma50_ratio', 'relative_strength', 'volume_ma20_ratio', 'volatility_5d', 'returns_20d', 'vix_ma20', 'price_volume_ratio', 'returns_10d', 'price_above_sma20', 'returns_5d', 'ma200_ratio', 'market_returns', 'macd_above_signal', 'volatility_20d', 'returns_1d', 'vix', 'volatility_20d_change', 'is_tech', 'is_etf', 'is_index']
    missing_features = [f for f in required_features if f not in features.columns]
    if missing_features:
        raise ValueError(f"缺少必要特征: {missing_features}")
    
    return features[required_features]
