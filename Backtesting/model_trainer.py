import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import datetime as dt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import talib as ta
import joblib
import os
from sklearn.impute import SimpleImputer
import time  # 导入time模块用于延迟
import logging  # 添加日志支持
# 导入股票分类模块
from stock_categories import get_recommended_training_set, get_sector_features

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_model_training.log"),
        logging.StreamHandler()
    ]
)

def SMA(values, window):
    """简单移动平均"""
    return pd.Series(values).rolling(window).mean()

def RSI(values, window):
    """相对强弱指标"""
    # 转换输入为pandas Series以统一处理
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    
    delta = values.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    # 避免除以零的情况 - 确保用安全的方式处理
    # 创建一个掩码找出所有avg_loss为0的位置
    zero_mask = avg_loss == 0
    
    # 初始化RSI序列
    rsi = pd.Series(index=values.index)
    
    # 对于avg_loss不为0的情况，正常计算
    non_zero_mask = ~zero_mask
    rs = avg_gain[non_zero_mask] / avg_loss[non_zero_mask]
    rsi[non_zero_mask] = 100 - (100 / (1 + rs))
    
    # 对于avg_loss为0但avg_gain不为0的情况，RSI = 100
    gain_mask = (avg_gain > 0) & zero_mask
    rsi[gain_mask] = 100
    
    # 对于avg_loss和avg_gain都为0的情况，RSI = 50
    neutral_mask = (avg_gain == 0) & zero_mask
    rsi[neutral_mask] = 50
    
    return rsi

def create_features(data_window, prediction_days=5):
    """
    从价格数据创建特征
    """
    if len(data_window) < 250:  # 确保数据足够计算指标
        print(f"警告: 数据长度不足 ({len(data_window)}), 无法生成所有特征")
        return pd.DataFrame()
        
    features = pd.DataFrame(index=data_window.index)
    
    # 确保数据为numpy float64类型，TA-Lib需要这种类型
    # 修复维度问题：确保所有数据都是一维数组
    try:
        close = np.array(data_window.Close, dtype=np.float64)
        if close.ndim > 1:
            close = close.flatten()
            
        high = np.array(data_window.High, dtype=np.float64)
        if high.ndim > 1:
            high = high.flatten()
            
        low = np.array(data_window.Low, dtype=np.float64)
        if low.ndim > 1:
            low = low.flatten()
            
        volume = np.array(data_window.Volume, dtype=np.float64)
        if volume.ndim > 1:
            volume = volume.flatten()
            
        open_price = np.array(data_window.Open, dtype=np.float64)
        if open_price.ndim > 1:
            open_price = open_price.flatten()
    except Exception as e:
        print(f"处理数据格式时出错: {e}")
        print(f"数据形状: Close={data_window.Close.shape if hasattr(data_window.Close, 'shape') else 'unknown'}")
        return pd.DataFrame()
    
    # 基础价格特征
    features['returns_1d'] = pd.Series(data_window.Close).pct_change()
    features['returns_5d'] = pd.Series(data_window.Close).pct_change(5)
    features['returns_10d'] = pd.Series(data_window.Close).pct_change(10)
    features['returns_20d'] = pd.Series(data_window.Close).pct_change(20)
    
    # 移动平均特征
    for window in [5, 10, 20, 50, 100, 200]:
        ma = pd.Series(data_window.Close).rolling(window, min_periods=1).mean()
        # 使用安全除法避免除以0
        features[f'ma{window}_ratio'] = np.where(ma > 0, data_window.Close / ma, 1)
    
    # 高级特征 - 移动平均交叉信号
    try:
        # 添加SMA交叉特征
        sma5 = pd.Series(data_window.Close).rolling(5, min_periods=1).mean()
        sma20 = pd.Series(data_window.Close).rolling(20, min_periods=1).mean()
        sma50 = pd.Series(data_window.Close).rolling(50, min_periods=1).mean()
        
        # SMA交叉信号 (1=金叉, -1=死叉, 0=无交叉)
        features['sma5_cross_sma20'] = ((sma5 > sma20) & (sma5.shift(1) <= sma20.shift(1))).astype(int) - \
                                     ((sma5 < sma20) & (sma5.shift(1) >= sma20.shift(1))).astype(int)
        features['sma20_cross_sma50'] = ((sma20 > sma50) & (sma20.shift(1) <= sma50.shift(1))).astype(int) - \
                                      ((sma20 < sma50) & (sma20.shift(1) >= sma50.shift(1))).astype(int)
                                      
        # 价格相对移动平均位置
        features['price_above_sma20'] = (data_window.Close > sma20).astype(int)
        features['price_above_sma50'] = (data_window.Close > sma50).astype(int)
    except Exception as e:
        print(f"添加移动平均交叉特征时出错: {e}")
        
    # 波动率特征
    for window in [5, 10, 20, 50]:
        features[f'volatility_{window}d'] = features['returns_1d'].rolling(window, min_periods=1).std()
    
    # 添加波动率变化率
    for window in [10, 20]:
        features[f'volatility_{window}d_change'] = features[f'volatility_{window}d'].pct_change(5)
    
    # 成交量特征
    features['volume_change'] = pd.Series(data_window.Volume).pct_change()
    features['volume_ma20_ratio'] = data_window.Volume / pd.Series(data_window.Volume).rolling(20, min_periods=1).mean()
    
    # 价格与成交量关系特征
    # 增加1避免除以0
    features['price_volume_ratio'] = data_window.Close / (data_window.Volume + 1)
    
    # 添加突破特征
    try:
        # 价格突破特征
        for window in [20, 50]:
            # 计算n天高点和低点
            rolling_high = pd.Series(data_window.High).rolling(window).max()
            rolling_low = pd.Series(data_window.Low).rolling(window).min()
            
            # 突破信号
            features[f'breakout_high_{window}d'] = (data_window.Close > rolling_high.shift(1)).astype(int)
            features[f'breakout_low_{window}d'] = (data_window.Close < rolling_low.shift(1)).astype(int)
    except Exception as e:
        print(f"添加突破特征时出错: {e}")
    
    # 添加技术指标
    try:
        # 动量指标
        features['rsi_14'] = pd.Series(ta.RSI(close, timeperiod=14))
        features['cci_20'] = pd.Series(ta.CCI(high, low, close, timeperiod=20))
        features['adx_14'] = pd.Series(ta.ADX(high, low, close, timeperiod=14))
        
        # 添加指标拐点特征
        features['rsi_trend'] = (features['rsi_14'] > features['rsi_14'].shift(1)).astype(int)
        features['adx_trend'] = (features['adx_14'] > features['adx_14'].shift(1)).astype(int)
        
        # 超买超卖指标
        features['rsi_overbought'] = (features['rsi_14'] > 70).astype(int)
        features['rsi_oversold'] = (features['rsi_14'] < 30).astype(int)
        
        # 趋势指标
        macd, macdsignal, macdhist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        features['macd'] = pd.Series(macd)
        features['macdsignal'] = pd.Series(macdsignal)
        features['macdhist'] = pd.Series(macdhist)
        
        # MACD柱状图变化
        features['macd_hist_change'] = pd.Series(macdhist).diff()
        features['macd_above_signal'] = (macd > macdsignal).astype(int)
        
        # 布林带
        upperband, middleband, lowerband = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        # 安全计算比率，避免除以0和无穷大值
        features['bb_upper_ratio'] = pd.Series(np.where(upperband > 0, close / upperband, 1))
        features['bb_lower_ratio'] = pd.Series(np.where(lowerband > 0, close / lowerband, 1))
        features['bb_width'] = pd.Series(np.where(middleband > 0, (upperband - lowerband) / middleband, 0))
        
        # 布林带突破
        features['price_above_bb_upper'] = (data_window.Close > upperband).astype(int)
        features['price_below_bb_lower'] = (data_window.Close < lowerband).astype(int)
        
        # 价格模式
        features['hammer'] = pd.Series(ta.CDLHAMMER(open=open_price, high=high, low=low, close=close))
        features['engulfing'] = pd.Series(ta.CDLENGULFING(open=open_price, high=high, low=low, close=close))
        features['doji'] = pd.Series(ta.CDLDOJI(open=open_price, high=high, low=low, close=close))
        
        # 交易量指标
        features['obv'] = pd.Series(ta.OBV(close, volume))
        features['ad'] = pd.Series(ta.AD(high, low, close, volume))
        
    except Exception as e:
        print(f"计算TA-Lib指标出错: {e}")
        # 回退到基本RSI计算
        features['rsi_14'] = RSI(data_window.Close, 14)
    
    # 目标变量：未来n天是否上涨
    # 改进：根据股票波动性调整上涨阈值，并考虑回撤
    # 计算未来n天的价格序列
    future_prices = pd.Series(data_window.Close).shift(-prediction_days)
    # 未来n天内的最低价格，用于计算最大回撤
    future_min_prices = pd.Series(data_window.Close)
    for i in range(1, prediction_days+1):
        future_min_prices = pd.concat([future_min_prices, pd.Series(data_window.Close).shift(-i)], axis=1).min(axis=1)
    
    # 计算股票的历史波动率
    hist_volatility = features['returns_1d'].rolling(20).std().mean() * np.sqrt(252)  # 年化波动率
    
    # 根据波动率调整上涨阈值，波动越大，要求的回报越高
    if hist_volatility < 0.2:  # 低波动性股票
        up_threshold = 0.01
    elif hist_volatility < 0.4:  # 中等波动性股票
        up_threshold = 0.015
    else:  # 高波动性股票
        up_threshold = 0.02
    
    # 计算预期收益率和最大回撤
    expected_return = (future_prices / data_window.Close - 1)
    max_drawdown = (future_min_prices / data_window.Close - 1).abs()
    
    # 定义目标变量：当预期收益超过阈值且回撤可接受时为1
    features['target'] = ((expected_return > up_threshold) & (max_drawdown < 2*up_threshold)).astype(int)
    
    # 验证和清理特征数据，确保没有无穷大值和NaN
    for col in features.columns:
        if col != 'target':  # 不处理目标变量
            # 检查无穷大值
            inf_mask = np.isinf(features[col])
            if inf_mask.sum() > 0:
                print(f"特征 {col} 中有 {inf_mask.sum()} 个无穷大值，将替换为0")
                features.loc[inf_mask, col] = 0
            
            # 处理NaN值
            if features[col].isna().sum() > 0:
                features[col] = features[col].fillna(0)
    
    return features

def train_master_model( years=10, save_path="models/master_model.pkl", category=None):
    """
    训练一个适用于多支股票的主模型
    
    参数:
    ticker_list: 股票代码列表
    years: 训练数据的年数
    save_path: 模型保存路径
    
    返回:
    训练好的模型
    """
    ticker_list = get_recommended_training_set(category)
    logging.info("开始训练主模型...")
    logging.info(f"使用的股票: {ticker_list}")
    
    all_features = pd.DataFrame()
    today = date.today()
    start_date = today - timedelta(days=365 * years)
    
    # 确保models目录存在
    os.makedirs("models", exist_ok=True)
    
    for ticker_idx, ticker in enumerate(ticker_list):
        logging.info(f"处理股票 {ticker} ({ticker_idx+1}/{len(ticker_list)}) 的数据...")
        
        # 最大重试次数
        max_retries = 3
        retry_delay = 2  # 秒
        
        for retry in range(max_retries):
            try:
                # 获取数据 - 显式设置auto_adjust和multi_level_index参数
                logging.info(f"下载 {ticker} 的历史数据 (从 {start_date} 到 {today})...")
                data = yf.download(
                    ticker, 
                    start=start_date, 
                    end=today, 
                    interval="1d", 
                    progress=False,
                    auto_adjust=True,  # 明确设置
                    multi_level_index=False  # 避免多级索引
                )
                
                if len(data) < 252:  # 至少需要一年的数据
                    logging.warning(f"股票 {ticker} 数据不足 ({len(data)} 行)，跳过")
                    break
                
                logging.info(f"成功获取 {len(data)} 行 {ticker} 的数据")
                
                # 确保数据列是一维的
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if hasattr(data[col], 'values') and data[col].values.ndim > 1:
                        data[col] = data[col].values.flatten()
                    
                # 创建特征
                logging.info(f"为 {ticker} 创建特征...")
                features = create_features(data)
                
                if len(features) == 0:
                    logging.warning(f"无法为股票 {ticker} 生成特征，跳过")
                    break
                    
                # 添加股票标识符和市场特征
                features['ticker'] = ticker
                
                # 添加一些市场相关特征（可选）
                try:
                    # 获取市场指数数据作为参考
                    if '^' not in ticker:  # 非指数
                        logging.info("获取市场指数数据...")
                        market_data = yf.download(
                            '^GSPC', 
                            start=start_date, 
                            end=today, 
                            interval="1d", 
                            progress=False,
                            auto_adjust=True,
                            multi_level_index=False
                        )
                        
                        if hasattr(market_data.Close, 'values') and market_data.Close.values.ndim > 1:
                            market_data.Close = market_data.Close.values.flatten()
                            
                        market_returns = market_data['Close'].pct_change()
                        
                        # 对齐日期 - 使用更稳健的方法
                        # 使用reindex而非简单的intersection，确保所有交易日都有数据
                        features['market_returns'] = market_returns.reindex(features.index)
                        # 使用前向填充处理缺失值，比0填充更合理
                        features['market_returns'] = features['market_returns'].ffill().fillna(0)  
                        features['relative_strength'] = features['returns_1d'] - features['market_returns']
                        
                        # 添加市场趋势指标
                        market_ma50 = market_data['Close'].rolling(50).mean()
                        market_ma200 = market_data['Close'].rolling(200).mean()
                        market_trend = (market_ma50 > market_ma200).astype(int)  # 1=牛市, 0=熊市
                        features['market_trend'] = market_trend.reindex(features.index).ffill().fillna(0)
                        
                        # 添加VIX指数作为市场恐慌指标
                        try:
                            vix_data = yf.download(
                                '^VIX', 
                                start=start_date, 
                                end=today, 
                                interval="1d", 
                                progress=False
                            )
                            # 确保VIX数据正确对齐
                            vix_close = vix_data['Close']
                            # 先将VIX数据重索引到特征索引
                            vix_close_aligned = vix_close.reindex(features.index).ffill()
                            features['vix'] = vix_close_aligned
                            
                            # 计算MA20并对齐
                            vix_ma20 = vix_close.rolling(20).mean()
                            vix_ma20_aligned = vix_ma20.reindex(features.index).ffill()
                            features['vix_ma20'] = vix_ma20_aligned
                            
                            # 使用对齐后的数据进行比较
                            features['vix_trend'] = (vix_close_aligned > vix_ma20_aligned).astype(int)
                        except Exception as e:
                            logging.warning(f"无法添加VIX数据: {str(e)}")
                            # 添加错误详情以便调试
                            if 'align' in str(e):
                                logging.debug("索引对齐错误。VIX索引示例: %s, 特征索引示例: %s", 
                                            str(vix_close.index[:3]) if 'vix_close' in locals() else "未知",
                                            str(features.index[:3]))
                except Exception as e:
                    logging.warning(f"无法添加市场特征: {str(e)}")
                
                # 添加股票分类标签
                # 为不同类型的股票添加标签，允许模型学习股票类别特定模式
                # 使用股票分类模块获取特征
                sector_features = get_sector_features(ticker)
                for feature_name, value in sector_features.items():
                    features[feature_name] = value
                
                # 合并到主数据集，先给每个股票加上唯一标识，避免日期重复问题
                # 使用MultiIndex来区分不同股票相同日期的数据
                features['ticker_temp'] = ticker
                features = features.reset_index()
                features = features.set_index(['Date', 'ticker_temp'])
                
                # 合并时保留原始index结构
                if len(all_features) == 0:
                    all_features = features
                else:
                    all_features = pd.concat([all_features, features])
                
                logging.info(f"已添加 {len(features)} 行来自 {ticker} 的数据")
                
                # 记录一些关于索引类型的信息，用于调试
                if ticker_idx == 0:
                    logging.info(f"特征数据索引类型: {type(features.index)}")
                
                # 成功获取数据，跳出重试循环
                break
                
            except Exception as e:
                logging.error(f"处理股票 {ticker} 时出错 (尝试 {retry+1}/{max_retries}): {str(e)}")
                if retry < max_retries - 1:
                    logging.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    logging.error(f"已达到最大重试次数，跳过股票 {ticker}")
        
        # 每处理5个股票后暂停一下，避免API限制
        if ticker_idx % 5 == 4:
            logging.info("暂停2秒，避免达到API限制...")
            time.sleep(2)
    
    # 检查是否有足够的数据
    if len(all_features) < 1000:
        logging.warning(f"警告: 总数据量不足 ({len(all_features)} 行)，模型可能表现不佳")
    else:
        logging.info(f"总共收集了 {len(all_features)} 行数据")
    
    # 确保索引是日期类型，可能的话按日期排序
    if not isinstance(all_features.index, pd.DatetimeIndex):
        logging.warning("合并后的数据索引不是日期类型，这可能影响时间序列分析")
        logging.info(f"索引类型: {type(all_features.index)}")
        logging.info(f"索引示例: {all_features.index[:5]}")
        
        # 检查是否使用了MultiIndex
        if isinstance(all_features.index, pd.MultiIndex):
            logging.info("检测到MultiIndex，提取日期部分和股票标识...")
            # 提取日期部分
            date_part = all_features.index.get_level_values(0)
            ticker_part = all_features.index.get_level_values(1)
            logging.info(f"日期范围: {date_part.min()} 到 {date_part.max()}")
            
            # 在后续处理中，我们将考虑股票维度
            ticker_unique = ticker_part.unique()
            logging.info(f"不同股票数量: {len(ticker_unique)}")
    else:
        # 按日期排序
        logging.info("按时间顺序对数据排序")
        all_features = all_features.sort_index()
    
    # 转换特征为数值类型，确保模型训练时没有问题
    for col in all_features.columns:
        if col not in ['ticker', 'ticker_temp', 'Date'] and col != 'target':
            try:
                all_features[col] = pd.to_numeric(all_features[col], errors='coerce')
            except Exception as e:
                logging.warning(f"无法将特征 {col} 转换为数值: {str(e)}")
    
    # 删除ticket列，准备训练
    if 'ticker' in all_features.columns:
        all_features = all_features.drop('ticker', axis=1)
    
    # 检查所有列是否还有ticker_temp列
    extra_cols = [col for col in all_features.columns if col not in ['target'] and 'temp' in col]
    if extra_cols:
        logging.info(f"删除临时列: {extra_cols}")
        all_features = all_features.drop(extra_cols, axis=1)
    
    # 分离特征和目标变量
    X = all_features.drop('target', axis=1)
    y = all_features['target']
    
    print(f"特征数量: {X.shape[1]}")
    print(f"目标变量分布: 0 (不上涨): {(y==0).sum()}, 1 (上涨): {(y==1).sum()}")
    
    # 检查并处理NaN值
    if X.isna().sum().sum() > 0:
        print(f"数据中存在NaN值，将进行处理。NaN值总数: {X.isna().sum().sum()}")
        print("各特征NaN值数量:")
        nan_counts = X.isna().sum()
        for col in nan_counts[nan_counts > 0].index:
            print(f"  - {col}: {nan_counts[col]}")
        
        # 填充NaN值
        X = X.fillna(X.mean())
        
        # 检查是否还有NaN值（如果某列全是NaN，mean()会返回NaN）
        if X.isna().sum().sum() > 0:
            print("仍有NaN值，用0填充")
            X = X.fillna(0)
    
    # 检查并处理无穷大值和极大值
    inf_count = np.isinf(X).sum().sum()
    if inf_count > 0:
        print(f"数据中存在{inf_count}个无穷大值，将进行处理")
        # 将无穷大替换为列的均值或0（如果均值计算失败）
        for col in X.columns:
            mask = np.isinf(X[col])
            if mask.sum() > 0:
                print(f"  - 特征 {col} 中有 {mask.sum()} 个无穷大值")
                # 获取有限值的均值
                finite_mean = X.loc[~np.isinf(X[col]), col].mean()
                if np.isnan(finite_mean):
                    X.loc[mask, col] = 0
                else:
                    X.loc[mask, col] = finite_mean
    
    # 检查极大值和极小值
    # 对于每一列，将超出95%分位数3倍的值视为极端值
    for col in X.columns:
        q95 = X[col].quantile(0.95)
        q05 = X[col].quantile(0.05)
        upper_bound = q95 + 3 * (q95 - q05)
        lower_bound = q05 - 3 * (q95 - q05)
        
        upper_mask = X[col] > upper_bound
        lower_mask = X[col] < lower_bound
        
        if upper_mask.sum() > 0 or lower_mask.sum() > 0:
            print(f"特征 {col} 中有 {upper_mask.sum()} 个上限极端值和 {lower_mask.sum()} 个下限极端值")
            # 用上下界值替换极端值
            X.loc[upper_mask, col] = upper_bound
            X.loc[lower_mask, col] = lower_bound
    
    # 分割训练集和验证集 - 改进：考虑股票类型，防止数据泄露
    print("使用时间序列划分方法，考虑股票类型...")
    
    # 1. 确定分割日期 - 使用时间索引的话，取倒数20%的日期
    if isinstance(all_features.index, pd.DatetimeIndex):
        # 单一索引情况
        unique_dates = all_features.index.unique()
        split_date = unique_dates[int(len(unique_dates) * 0.8)]
        train_mask = all_features.index < split_date
        val_mask = all_features.index >= split_date
    elif isinstance(all_features.index, pd.MultiIndex):
        # 多级索引情况
        unique_dates = all_features.index.get_level_values(0).unique()
        split_date = unique_dates[int(len(unique_dates) * 0.8)]
        train_mask = all_features.index.get_level_values(0) < split_date
        val_mask = all_features.index.get_level_values(0) >= split_date
    else:
        # 回退到基于位置的划分
        split_idx = int(len(X) * 0.8)
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[:split_idx] = True
        val_mask = ~train_mask
    
    # 2. 按照股票类型进行分层，确保每种类型的股票都有合理的训练/验证比例
    stock_types = []
    if 'is_tech' in X.columns:
        stock_types.extend(['is_tech', 'is_china', 'is_etf', 'is_index'])
        
    # 分割数据
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    
    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
    
    # 检查不同类型股票的分布
    if stock_types:
        for st in stock_types:
            if st in X.columns:
                train_count = X_train[st].sum()
                val_count = X_val[st].sum()
                print(f"{st} 在训练集中: {train_count} 行, 在验证集中: {val_count} 行")
    
    # 特征选择（可选）- 减少特征数量，提高模型鲁棒性
    print("执行特征选择...")
    
    # 1. 检查并处理特征共线性
    print("检查特征共线性...")
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.9)]
    
    if high_corr_cols:
        print(f"发现 {len(high_corr_cols)} 个高相关性特征 (>0.9)")
        print(f"示例: {high_corr_cols[:5] if len(high_corr_cols) > 5 else high_corr_cols}")
        
        # 保留这些特征，但在特征选择阶段考虑这一点
        # 不直接删除，因为某些可能具有独特信息
    
    # 2. 使用SelectFromModel进行特征选择，但使用更稳健的方法
    # 考虑多种选择标准
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import RFE
    
    # 2.1. 使用几种不同的方法选择特征
    importance_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='mean')
    importance_selector.fit(X_train, y_train)
    importance_features = X_train.columns[importance_selector.get_support()]
    
    # 2.2. 使用RFE（递归特征消除）作为补充
    rfe_selector = RFE(ExtraTreesClassifier(n_estimators=100, random_state=42), 
                       n_features_to_select=min(30, X_train.shape[1] // 2),
                       step=5)
    rfe_selector.fit(X_train, y_train)
    rfe_features = X_train.columns[rfe_selector.support_]
    
    # 2.3. 结合两种方法，取并集
    selected_features = list(set(importance_features) | set(rfe_features))
    print(f"从 {X_train.shape[1]} 个特征中选择了 {len(selected_features)} 个")
    
    # 2.4. 显示重要特征信息，包括两种方法
    print("重要特征 (基于RandomForest):")
    feature_importances = list(zip(X_train.columns, importance_selector.estimator_.feature_importances_))
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(feature_importances[:10]):
        print(f"  {i+1}. {feature}: {importance:.4f} {'(已选择)' if feature in selected_features else ''}")
    
    print("\n重要特征 (基于RFE):")
    for i, feature in enumerate(rfe_features[:10]):
        print(f"  {i+1}. {feature} {'(已选择)' if feature in selected_features else ''}")
    
    # 3. 确保股票类型特征被保留 (如果有的话)
    for st in stock_types:
        if st in X_train.columns and st not in selected_features:
            selected_features.append(st)
            print(f"保留特征: {st} (股票类型特征)")
    
    # 更新数据集，只保留选择的特征
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    
    # 计算时间衰减权重 - 让最近的数据有更高的权重
    logging.info("计算时间衰减权重...")
    # 使用更健壮的方法检测索引类型并提取日期
    try:
        if isinstance(X_train.index, pd.DatetimeIndex):
            # 单一的DatetimeIndex情况
            max_date = X_train.index.max()
            # 直接创建一个Series，避免使用Index对象
            days_diff = pd.Series([(max_date - d).days for d in X_train.index], index=X_train.index)
            
            # 指数衰减函数: weight = exp(-days_diff / half_life)
            half_life = 365 * 2  # 半衰期为2年
            sample_weights = np.exp(-days_diff / half_life)
            
            # 归一化权重使其和为样本数量
            sample_weights = sample_weights / sample_weights.mean()
            logging.info(f"使用时间衰减权重: 最大权重={sample_weights.max():.2f}, 最小权重={sample_weights.min():.2f}")
        elif isinstance(X_train.index, pd.MultiIndex):
            # MultiIndex情况 - 提取日期部分
            logging.info("检测到MultiIndex，提取日期部分计算权重...")
            date_part = X_train.index.get_level_values(0)
            max_date = date_part.max()
            
            if isinstance(max_date, (pd.Timestamp, dt.datetime)):
                # 计算日期差
                days_diff = np.array([(max_date - d).days if isinstance(d, (pd.Timestamp, dt.datetime)) else 0 
                                     for d in date_part])
                
                # 指数衰减函数
                half_life = 365 * 2  # 半衰期为2年
                sample_weights = np.exp(-days_diff / half_life)
                
                # 归一化权重
                sample_weights = sample_weights / sample_weights.mean()
                logging.info(f"使用MultiIndex时间衰减权重: 最大权重={sample_weights.max():.2f}, 最小权重={sample_weights.min():.2f}")
            else:
                logging.warning(f"索引的日期部分不是日期类型 ({type(max_date)}), 不使用时间衰减权重")
                sample_weights = None
        else:
            logging.info(f"索引不是日期类型 ({type(X_train.index)}), 不使用时间衰减权重")
            sample_weights = None
    except Exception as e:
        logging.warning(f"计算时间衰减权重时出错: {str(e)}")
        logging.warning("将使用均等权重")
        sample_weights = None
    
    # 训练几个不同的模型
    models = {
        'RandomForest': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # 添加缺失值处理
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=200, max_depth=8, 
                                               min_samples_leaf=10, random_state=42, 
                                               class_weight='balanced'))
        ]),
        'GradientBoosting': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # 添加缺失值处理
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(n_estimators=150, max_depth=5, 
                                                   learning_rate=0.05, random_state=42,
                                                   subsample=0.8))  # 添加子采样以减少过拟合
        ]),
        'ExtraTrees': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', ExtraTreesClassifier(n_estimators=200, max_depth=10,
                                              min_samples_leaf=10, random_state=42,
                                              class_weight='balanced'))
        ])
    }
    
    # 添加评估指标
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    best_model_name = None
    best_f1 = 0.0  # 使用F1分数作为主要评估指标，更平衡地考虑精确率和召回率
    best_model = None
    
    model_results = []  # 存储所有模型的结果，用于比较
    
    for name, model in models.items():
        try:
            logging.info(f"训练模型: {name}...")
            # 使用样本权重（如果有）
            if sample_weights is not None and isinstance(X_train.index, pd.DatetimeIndex) and isinstance(sample_weights, (pd.Series, np.ndarray)):
                try:
                    model.fit(X_train, y_train, classifier__sample_weight=sample_weights)
                except Exception as e:
                    logging.warning(f"使用带权重训练失败: {str(e)}，尝试不带权重训练")
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            # 预测和评估
            y_pred = model.predict(X_val)
            
            # 计算各种评估指标
            accuracy = (y_pred == y_val).mean()
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            # 如果模型支持predict_proba，计算AUC
            auc = 0.5  # 默认值
            if hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X_val)[:,1]
                    auc = roc_auc_score(y_val, y_prob)
                except Exception as e:
                    logging.warning(f"计算AUC时出错: {str(e)}")
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_val, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # 计算上涨和下跌的预测准确率
            up_accuracy = (y_pred[y_val==1] == 1).mean() if (y_val==1).sum() > 0 else 0
            down_accuracy = (y_pred[y_val==0] == 0).mean() if (y_val==0).sum() > 0 else 0
            
            # 输出详细评估结果
            print(f"\n{name} 模型评估:")
            print(f"验证集准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}, AUC: {auc:.4f}")
            print(f"上涨预测准确率: {up_accuracy:.4f}, 下跌预测准确率: {down_accuracy:.4f}")
            print(f"混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            
            # 存储结果
            model_results.append({
                'name': name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'up_accuracy': up_accuracy,
                'down_accuracy': down_accuracy
            })
            
            # 使用F1分数作为主要指标选择最佳模型
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_model = model
        except Exception as e:
            print(f"{name} 训练失败: {e}")
    
    if best_model is None:
        print("所有模型训练失败，无法返回模型")
        return None
    
    # 打印所有模型的比较结果
    print("\n所有模型比较:")
    for res in model_results:
        print(f"{res['name']}: F1={res['f1']:.4f}, Accuracy={res['accuracy']:.4f}, AUC={res['auc']:.4f}")
    
    print(f"\n最佳模型: {best_model_name}, F1分数: {best_f1:.4f}")
    
    # 保存模型
    try:
        # 获取模型实际使用的特征
        actual_feature_names = X_train.columns.tolist()  # 使用选择后的特征列表
        
        # 打印特征信息以便调试
        logging.info(f"模型使用的特征总数: {len(actual_feature_names)}")
        logging.info("前10个特征:")
        for i, feature in enumerate(actual_feature_names[:10]):
            logging.info(f"  {i+1}. {feature}")
        
        # 创建含有更多元数据的模型对象
        model_data = {
            'model': best_model,
            'features': actual_feature_names,
            'training_date': date.today().isoformat(),
            'date_range': {
                'start': start_date.isoformat(),
                'end': today.isoformat()
            },
            'tickers': ticker_list,
            'performance': {
                'accuracy': model_results[0]['accuracy'] if model_results else 0,
                'f1': best_f1,
                'precision': next((r['precision'] for r in model_results if r['name'] == best_model_name), 0),
                'recall': next((r['recall'] for r in model_results if r['name'] == best_model_name), 0),
                'model_name': best_model_name
            },
            'model_type': best_model_name,
            'is_realtime_ready': True,  # 标记是否可用于实时交易
            'feature_engineering': {
                'selected_features': len(actual_feature_names),
                'original_features': X.shape[1],
                'requires_previous_day': True,  # 标记是否需要前一天的数据
                'min_window': 200,  # 计算特征所需的最小历史窗口
            },
            'version': '2.0',  # 版本号，便于追踪模型变更
            'training_params': {
                'years': years,
                'stocks_count': len(ticker_list),
                'total_records': len(X),
            }
        }
        
        # 保存模型
        joblib.dump(model_data, save_path)
        logging.info(f"模型已保存到 {save_path}")
        
        # 保存历史版本，便于回溯
        history_path = save_path.replace('.pkl', f'_{category}.pkl')
        joblib.dump(model_data, history_path)
        logging.info(f"模型历史版本已保存: {history_path}")
        
        # 保存详细的特征信息
        features_file = save_path.replace('.pkl', '_features.txt')
        with open(features_file, 'w') as f:
            for feature in actual_feature_names:
                f.write(f"{feature}\n")
        logging.info(f"特征列信息已保存到 {features_file}")
        
        # 保存带行列名的数据样本，便于测试
        sample_file = save_path.replace('.pkl', '_sample.csv')
        sample_data = X_train.iloc[:1].copy()
        sample_data.to_csv(sample_file)
        logging.info(f"样本数据已保存到 {sample_file}")
        
    except Exception as e:
        logging.error(f"保存模型时出错: {str(e)}")
    
    return best_model

def create_update_schedule():
    """创建模型定期更新计划配置文件"""
    config = {
        "model_update_frequency": "weekly",  # weekly, monthly, quarterly
        "update_day": "Monday",  # 周几更新
        "update_hour": 1,  # 凌晨1点
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", 
                   "QQQ", "SPY", "BABA", "JD", "BIDU", "PDD"],
        "years_of_data": 5,
        "save_path": "models/master_model.pkl",
        "keep_versions": 5,  # 保留多少个历史版本
        "auto_deploy": True,  # 是否自动部署新模型
        "notify_email": "",  # 可选的通知邮箱
        "evaluation_threshold": {  # 模型性能阈值，低于此阈值将不自动部署
            "f1_min": 0.55,
            "accuracy_min": 0.55
        }
    }
    
    # 保存配置
    import json
    with open("model_update_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    logging.info("已创建模型更新计划配置文件: model_update_config.json")
    
    # 创建示例cron作业
    cron_job = """
# 每周一凌晨1点更新模型
0 1 * * 1 cd /path/to/project && python -m Backtesting.model_trainer
"""
    with open("model_update_cron.txt", "w") as f:
        f.write(cron_job.strip())
    
    logging.info("已创建示例cron作业: model_update_cron.txt")
    return config

if __name__ == "__main__":
    # 测试训练函数
    start_time = time.time()
    logging.info("=== 开始股票模型训练 ===")
    
    # 检查是否有命令行参数
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--create-schedule":
        create_update_schedule()
        sys.exit(0)
    
    test_tickers = ["MMM",
        "INTC",
        "AMD",
        "MCD",
        "AAPL",
        "TSLA",
        "TSM",
        "GOOG",
        "META",
        "QQQ",
        "SPY",
        "MSFT",
        "AMZN",
        "NVDA",
        "QCOM","BIDU",
        "JD",
        "NTES",
        "PDD",
        "BILI",
        "TCEHY",
        "BABA",]
    
    try:
        model = train_master_model(test_tickers, years=5)
        if model:
            logging.info("模型训练成功完成!")
        else:
            logging.error("模型训练失败!")
    except Exception as e:
        logging.error(f"训练过程中发生错误: {str(e)}")
    
    # 计算总运行时间
    total_time = time.time() - start_time
    logging.info(f"总运行时间: {total_time/60:.2f} 分钟")
    logging.info("=== 训练过程结束 ===") 