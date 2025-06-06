import numpy as np
import pandas as pd
from backtesting import Strategy
import talib as ta  # 添加Ta-Lib库用于计算技术指标
import joblib
import os
import logging
import traceback

from model_trainer import train_master_model
from stock_categories import get_stock_type, get_sector_features
from data_service import DataService

logger = logging.getLogger(__name__)


class MLStrategy(Strategy):
    """
    基于机器学习的交易策略
    使用随机森林分类器预测未来价格走势
    """

    # 定义策略参数
    window = 20  # 特征窗口大小
    n_train_days = 252 * 3  # 训练数据量（约3年）
    prediction_threshold = 0.55  # 降低预测置信度阈值，使更容易产生交易信号
    prediction_days = 5  # 预测未来n天的价格走势

    # 风险管理参数
    stop_loss_pct = 0.05  # 止损比例
    take_profit_pct = 0.15  # 止盈比例
    max_position_pct = 0.9  # 最大仓位比例
    position_scaling = 4.0  # 仓位缩放因子

    ticker = None  # 股票代码（用于记录）

    # 添加类变量来存储主模型
    master_model = None
    master_model_path = "models/master_model.pkl"

    # 存储当前实例，方便外部访问
    instance = None

    def init(self):
        """
        策略初始化
        """
        self.data_service = DataService()

        # 获取整个回测期间的日期范围
        if self.data is not None and not self.data.index.empty:
            self.global_start_date = self.data.index.min().date()
            self.global_end_date = self.data.index.max().date()

            logger.info(
                "Pre-fetching market (^GSPC) and VIX (^VIX) data for the entire backtest period..."
            )
            # Use the data service to get data
            self._market_data_full = self.data_service.get_market_index_data(
                self.global_start_date, self.global_end_date
            )
            self._vix_data_full = self.data_service.get_vix_data(
                self.global_start_date, self.global_end_date
            )
            logger.info("Market and VIX data pre-fetching complete.")
        else:
            logger.warning(
                "self.data.index is empty or None in init. Cannot pre-fetch market/VIX data."
            )
            self.global_start_date = None
            self.global_end_date = None
            self._market_data_full = None
            self._vix_data_full = None

        # 为随机数生成器设置固定种子，确保结果可重复
        np.random.seed(42)

        # 初始化策略，生成指标，准备机器学习模型
        self.trade_results = []
        self._prev_entry_price = None
        self._prev_position = False

        # 计算基本技术指标作为特征
        self.ma20 = self.I(self.SMA, self.data.Close, 20)
        self.ma50 = self.I(self.SMA, self.data.Close, 50)
        self.ma200 = self.I(self.SMA, self.data.Close, 200)

        # 计算更多技术指标作为特征
        self.rsi = self.I(self.RSI, self.data.Close, 14)
        self.volume_ma = self.I(self.SMA, self.data.Volume, 20)

        # 初始化模型和预测结果存储
        self.model = None
        self.predictions = np.ones(len(self.data.Close)) * 0.5  # 默认为0.5
        self.trained = False

        # 交易表现跟踪
        self.dynamic_threshold = self.prediction_threshold  # 动态调整的阈值
        self.market_regime = None  # 市场状态：bull, bear, neutral

        # 交易记录
        self.instance_trade_record = {
            "open_trades": [],
            "closed_trades": [],
            "last_action": None,
            "last_action_price": None,
            "last_action_time": None,
            "trade_count": 0,
            "action_log": [],
        }

        # 在策略初始化时训练模型
        self._train_model()

    @staticmethod
    def SMA(values, window):
        """简单移动平均"""
        return pd.Series(values).rolling(window).mean()

    @staticmethod
    def RSI(values, window):
        """相对强弱指标"""
        delta = pd.Series(values).diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        # 避免除以零的情况
        avg_loss = avg_loss.replace(0, 1e-10)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _create_features(self, data_window):
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
            print(
                f"数据形状: Close={data_window.Close.shape if hasattr(data_window.Close, 'shape') else 'unknown'}"
            )
            return pd.DataFrame()

        # 基础价格特征
        features["returns_1d"] = pd.Series(data_window.Close).pct_change()
        features["returns_5d"] = pd.Series(data_window.Close).pct_change(5)
        features["returns_10d"] = pd.Series(data_window.Close).pct_change(10)
        features["returns_20d"] = pd.Series(data_window.Close).pct_change(20)

        # 移动平均特征
        for window in [5, 10, 20, 50, 100, 200]:
            ma = pd.Series(data_window.Close).rolling(window, min_periods=1).mean()
            features[f"ma{window}_ratio"] = data_window.Close / ma

        # 高级特征 - 移动平均交叉信号
        try:
            # 添加SMA交叉特征
            sma5 = pd.Series(data_window.Close).rolling(5, min_periods=1).mean()
            sma20 = pd.Series(data_window.Close).rolling(20, min_periods=1).mean()
            sma50 = pd.Series(data_window.Close).rolling(50, min_periods=1).mean()

            # SMA交叉信号 (1=金叉, -1=死叉, 0=无交叉)
            features["sma5_cross_sma20"] = (
                (sma5 > sma20) & (sma5.shift(1) <= sma20.shift(1))
            ).astype(int) - ((sma5 < sma20) & (sma5.shift(1) >= sma20.shift(1))).astype(
                int
            )
            features["sma20_cross_sma50"] = (
                (sma20 > sma50) & (sma20.shift(1) <= sma50.shift(1))
            ).astype(int) - (
                (sma20 < sma50) & (sma20.shift(1) >= sma50.shift(1))
            ).astype(
                int
            )

            # 价格相对移动平均位置
            features["price_above_sma20"] = (data_window.Close > sma20).astype(int)
            features["price_above_sma50"] = (data_window.Close > sma50).astype(int)
        except Exception as e:
            print(f"添加移动平均交叉特征时出错: {e}")

        # 波动率特征
        for window in [5, 10, 20, 50]:
            features[f"volatility_{window}d"] = (
                features["returns_1d"].rolling(window, min_periods=1).std()
            )

        # 添加波动率变化率
        for window in [10, 20]:
            features[f"volatility_{window}d_change"] = features[
                f"volatility_{window}d"
            ].pct_change(5)

        # 成交量特征
        features["volume_change"] = pd.Series(data_window.Volume).pct_change()
        features["volume_ma20_ratio"] = (
            data_window.Volume
            / pd.Series(data_window.Volume).rolling(20, min_periods=1).mean()
        )

        # 价格与成交量关系特征
        features["price_volume_ratio"] = data_window.Close / (
            data_window.Volume + 1
        )  # 避免除零

        # 添加突破特征
        try:
            # 价格突破特征
            for window in [20, 50]:
                # 计算n天高点和低点
                rolling_high = pd.Series(data_window.High).rolling(window).max()
                rolling_low = pd.Series(data_window.Low).rolling(window).min()

                # 突破信号
                features[f"breakout_high_{window}d"] = (
                    data_window.Close > rolling_high.shift(1)
                ).astype(int)
                features[f"breakout_low_{window}d"] = (
                    data_window.Close < rolling_low.shift(1)
                ).astype(int)
        except Exception as e:
            print(f"添加突破特征时出错: {e}")

        # 添加技术指标
        try:
            # 动量指标
            features["rsi_14"] = pd.Series(ta.RSI(close, timeperiod=14))
            features["cci_20"] = pd.Series(ta.CCI(high, low, close, timeperiod=20))
            features["adx_14"] = pd.Series(ta.ADX(high, low, close, timeperiod=14))

            # 添加指标拐点特征
            features["rsi_trend"] = (
                features["rsi_14"] > features["rsi_14"].shift(1)
            ).astype(int)
            features["adx_trend"] = (
                features["adx_14"] > features["adx_14"].shift(1)
            ).astype(int)

            # 超买超卖指标
            features["rsi_overbought"] = (features["rsi_14"] > 70).astype(int)
            features["rsi_oversold"] = (features["rsi_14"] < 30).astype(int)

            # 趋势指标
            macd, macdsignal, macdhist = ta.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            features["macd"] = pd.Series(macd)
            features["macdsignal"] = pd.Series(macdsignal)
            features["macdhist"] = pd.Series(macdhist)

            # MACD柱状图变化
            features["macd_hist_change"] = pd.Series(macdhist).diff()
            features["macd_above_signal"] = (macd > macdsignal).astype(int)

            # 布林带
            upperband, middleband, lowerband = ta.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            features["bb_upper_ratio"] = pd.Series(close / upperband)
            features["bb_lower_ratio"] = pd.Series(close / lowerband)
            features["bb_width"] = pd.Series((upperband - lowerband) / middleband)

            # 布林带突破
            features["price_above_bb_upper"] = (data_window.Close > upperband).astype(
                int
            )
            features["price_below_bb_lower"] = (data_window.Close < lowerband).astype(
                int
            )

            # 价格模式
            features["hammer"] = pd.Series(
                ta.CDLHAMMER(open=open_price, high=high, low=low, close=close)
            )
            features["engulfing"] = pd.Series(
                ta.CDLENGULFING(open=open_price, high=high, low=low, close=close)
            )
            features["doji"] = pd.Series(
                ta.CDLDOJI(open=open_price, high=high, low=low, close=close)
            )

            # 交易量指标
            features["obv"] = pd.Series(ta.OBV(close, volume))
            features["ad"] = pd.Series(ta.AD(high, low, close, volume))

        except Exception as e:
            print(f"计算TA-Lib指标出错: {e}")
            # 回退到基本RSI计算
            features["rsi_14"] = self.RSI(data_window.Close, 14)

        # ===添加市场相关特征===
        # 这是解决特征不匹配问题的关键部分
        # 添加与市场指数相关的特征，确保与训练数据特征一致
        try:
            # --- Use pre-fetched Market Data ---
            if self._market_data_full is not None and not self._market_data_full.empty:
                # Calculate indicators from the full dataset once
                # Check if these calculations were already done and cached, if performance is an issue
                # For simplicity now, we recalculate based on the full data
                market_close = self._market_data_full["Close"]
                market_ma50 = market_close.rolling(50).mean()
                market_ma200 = market_close.rolling(200).mean()
                market_trend = (market_ma50 > market_ma200).astype(int)
                market_returns = market_close.pct_change()

                # Align the calculated series to the current features DataFrame index
                features["market_trend"] = market_trend.reindex(
                    features.index, method="ffill"
                ).fillna(0)
                features["market_returns"] = market_returns.reindex(
                    features.index, method="ffill"
                ).fillna(0)

                # Calculate relative strength using the aligned market returns
                if "returns_1d" in features.columns:
                    # Ensure alignment before subtraction, fillna just in case
                    aligned_market_returns = features["market_returns"].fillna(0)
                    aligned_stock_returns = features["returns_1d"].fillna(0)
                    features["relative_strength"] = (
                        aligned_stock_returns - aligned_market_returns
                    )
                else:
                    features["relative_strength"] = (
                        0  # Should not happen if returns_1d is calculated
                    )
            else:
                logger.warning(
                    "Pre-fetched market data (_market_data_full) is missing or empty. Using default values for market features."
                )
                features["market_trend"] = 0
                features["market_returns"] = (
                    0  # Or use stock returns as approximation? Be consistent.
                )
                features["relative_strength"] = 0

            # --- Use pre-fetched VIX Data ---
            if self._vix_data_full is not None and not self._vix_data_full.empty:
                # Calculate indicators from the full dataset once
                vix_close = self._vix_data_full["Close"]
                vix_ma20 = vix_close.rolling(20).mean()
                # Calculate means once, maybe store them in init if needed frequently
                vix_mean = vix_close.mean()
                vix_ma20_mean = vix_ma20.mean()

                # Align VIX data and indicators
                features["vix"] = vix_close.reindex(
                    features.index, method="ffill"
                ).fillna(
                    vix_mean if not pd.isna(vix_mean) else 20
                )  # Use calculated mean or default 20
                features["vix_ma20"] = vix_ma20.reindex(
                    features.index, method="ffill"
                ).fillna(
                    vix_ma20_mean if not pd.isna(vix_ma20_mean) else 20
                )  # Use calculated mean or default 20
                # Calculate trend based on aligned values
                features["vix_trend"] = (features["vix"] > features["vix_ma20"]).astype(
                    int
                )
            else:
                logger.warning(
                    "Pre-fetched VIX data (_vix_data_full) is missing or empty. Using default values for VIX features."
                )
                features["vix"] = 20
                features["vix_ma20"] = 20
                features["vix_trend"] = 0

        except Exception as e:
            logger.error(
                f"Error processing pre-fetched market/VIX data in _create_features: {e}",
                exc_info=True,
            )
            features["market_trend"] = 0
            features["market_returns"] = 0
            features["relative_strength"] = 0
            features["vix"] = 20
            features["vix_ma20"] = 20
            features["vix_trend"] = 0

        # 添加股票类型特征
        if hasattr(MLStrategy, "ticker") and MLStrategy.ticker is not None:
            ticker = MLStrategy.ticker

            # 使用stock_categories模块中的get_sector_features函数获取股票类型特征
            sector_features = get_sector_features(ticker)

            # 将获取的特征添加到features中
            for feature_name, value in sector_features.items():
                features[feature_name] = value
        else:
            # 如果没有ticker信息，全部设为0
            features["is_tech"] = 0
            features["is_china"] = 0
            features["is_etf"] = 0
            features["is_index"] = 0
            features["is_finance"] = 0
            features["is_healthcare"] = 0
            features["is_consumer"] = 0

        # 目标变量：未来n天是否上涨
        prediction_days = self.prediction_days
        future_prices = pd.Series(data_window.Close).shift(-prediction_days)
        features["target"] = (future_prices > data_window.Close).astype(int)

        # 填充NaN值 - 修复废弃警告
        features = features.ffill().bfill().fillna(0)

        # 添加验证，检查是否有无穷值
        if np.isinf(features.values).any():
            print("警告: 特征数据包含无穷值，将替换为0")
            features = features.replace([np.inf, -np.inf], 0)

        # 检查可能的异常值
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_mean = features[col].mean()
            col_std = features[col].std()
            if col_std > 0:
                # 检查是否有极端异常值 (超过均值±10个标准差)
                extreme_values = features[abs(features[col] - col_mean) > 10 * col_std]
                if len(extreme_values) > 0:
                    print(f"警告: 特征 '{col}' 包含极端值，可能影响模型表现")

        return features

    def _train_model(self):
        """
        加载预训练模型
        """
        # if MLStrategy.master_model is not None:
        #     # 检查是否为字典格式
        #     if isinstance(MLStrategy.master_model, dict) and 'model' in MLStrategy.master_model:
        #         print("使用已加载的主模型（字典格式）")
        #         self.model = MLStrategy.master_model['model']  # 使用'model'键中的实际模型
        #         self.model_features = MLStrategy.master_model['features']
        #     else:
        #         print("使用已加载的主模型（普通格式）")
        #         self.model = MLStrategy.master_model
        #     self.trained = True
        # else:
        # 尝试从文件加载
        try:
            category = get_stock_type(self.ticker).upper()
            model_path = MLStrategy.master_model_path.replace(
                ".pkl", f"_{category}.pkl"
            )

            try:
                # 尝试加载主模型
                if os.path.exists(model_path):
                    print(f"尝试从文件加载主模型: {model_path}")
                    model_data = joblib.load(model_path)

                    # 检查模型格式
                    # 打印模型数据类型，帮助调试
                    print(f"模型数据类型: {type(model_data)}")

                    # 如果是字典并包含features和model，则为新格式
                    if (
                        isinstance(model_data, dict)
                        and "features" in model_data
                        and "model" in model_data
                    ):
                        print(f"加载新格式的模型（含元数据）")
                        MLStrategy.master_model = model_data  # 保存整个字典
                        self.model_features = model_data["features"]
                        print(
                            f"模型训练日期: {model_data.get('training_date', '未知')}"
                        )
                        print(f"模型使用的特征数: {len(self.model_features)}")
                        self.model = model_data[
                            "model"
                        ]  # 保存实际的模型对象而非整个字典
                    else:
                        # 旧格式，直接使用
                        print(f"加载旧格式的模型（无元数据）")
                        # 打印模型键（如果是字典）
                        if isinstance(model_data, dict):
                            print(f"模型键: {list(model_data.keys())}")
                        MLStrategy.master_model = model_data
                        self.model_features = None  # 旧模型没有保存特征列表
                        self.model = model_data

                    print(f"从文件加载主模型: {model_path}")
                    self.trained = True
                else:
                    print(f"主模型文件不存在: {model_path}，请先训练主模型")
                    MLStrategy.master_model = train_master_model(
                        years=10, category=category
                    )
                    if (
                        isinstance(MLStrategy.master_model, dict)
                        and "features" in MLStrategy.master_model
                    ):
                        self.model_features = MLStrategy.master_model["features"]
                        self.model = MLStrategy.master_model["model"]  # 确保使用model键
                    else:
                        self.model_features = None
                        self.model = MLStrategy.master_model  # 旧格式
            except Exception as e:
                print(f"加载主模型时出错: {e}，请先训练主模型")
                print(f"错误详情: {traceback.format_exc()}")
                self.model_features = None
                self.model = None
        except Exception as e:
            print(f"加载主模型时出错: {e}，请先训练主模型")
            self.model_features = None

        print("开始为所有数据生成预测...")

        all_data = pd.DataFrame(
            {
                "Open": self.data.Open,
                "High": self.data.High,
                "Low": self.data.Low,
                "Close": self.data.Close,
                "Volume": self.data.Volume,
            },
            index=self.data.index,
        )

        all_features = self._create_features(all_data)
        if len(all_features) > 0 and self.model is not None:
            print(
                f"特征数据长度: {len(all_features)}, 交易日数据长度: {len(self.data)}"
            )

            # 确保特征与模型期望匹配
            all_features = self._ensure_model_features(all_features)

            X_all = all_features.drop("target", axis=1)
            self.predictions = np.ones(len(self.data.Close)) * 0.5  # 默认为0.5

            match_count = 0
            print(f"开始预测...")
            # 找到特征数据在原始数据中的位置
            for i, idx in enumerate(X_all.index):
                try:
                    if idx in self.data.index:
                        data_idx = self.data.index.get_loc(idx)
                        if data_idx < len(self.predictions):
                            try:
                                # 确保数据没有NaN值
                                row_data = X_all.iloc[i : i + 1].copy()
                                if row_data.isna().any().any():
                                    row_data = row_data.fillna(0)

                                # 使用预训练模型时，确保特征列匹配
                                # 检查是否有模型特征信息
                                model_features = getattr(self, "model_features", None)
                                if model_features is not None:
                                    # 创建一个新的DataFrame，包含所有模型需要的特征
                                    new_row_data = pd.DataFrame(
                                        0, index=row_data.index, columns=model_features
                                    )

                                    # 计算缺失的特征
                                    missing_features = [
                                        f
                                        for f in model_features
                                        if f not in row_data.columns
                                    ]

                                    # 将现有特征复制到新DataFrame
                                    for col in row_data.columns:
                                        if col in model_features:
                                            new_row_data[col] = row_data[col]

                                    # 使用新的DataFrame替代原始数据
                                    row_data = new_row_data

                                    # 打印警告消息如果缺失大量特征
                                    if len(missing_features) > 10 and i == 0:
                                        percentage = (
                                            len(missing_features)
                                            / len(model_features)
                                            * 100
                                        )
                                        print(
                                            f"警告: 缺失 {len(missing_features)}/{len(model_features)} ({percentage:.1f}%) 个特征，可能导致预测不准确"
                                        )
                                        print(f"前5个缺失特征: {missing_features[:5]}")

                                # 使用模型预测上涨概率
                                # 检查是否使用新格式模型(字典)
                                if hasattr(self.model, "predict_proba"):
                                    try:
                                        prob = self.model.predict_proba(row_data)[0][1]
                                        self.predictions[data_idx] = prob
                                        match_count += 1
                                    except Exception as e:
                                        logging.warning(
                                            f"调用predict_proba出错: {e}，回退到predict方法"
                                        )
                                        # 回退到直接预测
                                        pred = self.model.predict(row_data)[0]
                                        self.predictions[data_idx] = float(pred)
                                        match_count += 1
                                else:
                                    # 使用predict方法作为替代
                                    pred = self.model.predict(row_data)[0]
                                    # 将二元预测（0或1）转换为概率值
                                    self.predictions[data_idx] = float(pred)
                            except Exception as e:
                                print(f"预测第 {i} 行出错: {e}")
                                # 使用默认预测值
                                self.predictions[data_idx] = 0.5
                                continue
                except Exception as e:
                    print(f"处理索引 {idx} 时出错: {e}")
                    continue

            print(f"总共匹配了 {match_count}/{len(X_all)} 行数据进行预测")
            print(
                f"预测值分布: 最小值={self.predictions.min():.4f}, 最大值={self.predictions.max():.4f}, 平均值={self.predictions.mean():.4f}"
            )
        else:
            if len(all_features) == 0:
                print("无法生成特征数据，使用默认预测值")
            if self.model is None:
                print("模型未加载，使用默认预测值")

        self.trained = True
        print(
            f"模型训练完成，总数据: {len(self.data.Close)}，特征数据: {len(all_features)}"
        )

    def _ensure_model_features(self, features_df):
        """
        确保特征数据包含模型所需的所有特征列
        """
        # 尝试从模型中获取所需特征列表
        required_features = None

        # 1. 如果模型是字典，直接从字典中获取
        if isinstance(self.model, dict) and "features" in self.model:
            required_features = self.model["features"]
            actual_model = self.model["model"]
        else:
            actual_model = self.model

            # 2. 检查是否有相关的特征文件
            feature_file = MLStrategy.master_model_path.replace(".pkl", "_features.txt")
            if os.path.exists(feature_file):
                try:
                    with open(feature_file, "r") as f:
                        required_features = [line.strip() for line in f.readlines()]
                except:
                    pass

            # 3. 从模型对象中获取
            if required_features is None:
                if hasattr(actual_model, "feature_names_in_"):
                    required_features = actual_model.feature_names_in_
                elif hasattr(actual_model, "steps"):
                    # 尝试从Pipeline中找到相关步骤
                    for _, step in actual_model.steps:
                        if hasattr(step, "feature_names_in_"):
                            required_features = step.feature_names_in_
                            break

        # 如果没有找到必要的特征列表，返回原始特征
        if required_features is None:
            print("警告: 无法确定模型所需特征，使用所有可用特征")
            return features_df

        # 当前特征列表
        current_features = features_df.columns.tolist()

        # 检查特征匹配情况
        common_features = [f for f in required_features if f in current_features]
        missing_features = [f for f in required_features if f not in current_features]
        extra_features = [
            f for f in current_features if f not in required_features and f != "target"
        ]

        print(
            f"特征匹配情况: 模型需要{len(required_features)}个特征，"
            f"匹配{len(common_features)}个，缺失{len(missing_features)}个，额外{len(extra_features)}个"
        )

        if missing_features:
            print(
                f"缺失的特征 (将使用默认值): {missing_features[:3]}"
                + (
                    f"... 等{len(missing_features)}个"
                    if len(missing_features) > 3
                    else ""
                )
            )

            # 为缺失的特征添加默认值
            for feature in missing_features:
                if "is_" in feature:
                    # 股票类型特征默认为0
                    features_df[feature] = 0
                elif "vix" in feature:
                    # VIX相关特征默认为20（长期平均）
                    features_df[feature] = 20
                elif "market_trend" in feature:
                    # 市场趋势默认为0
                    features_df[feature] = 0
                elif feature == "market_returns":
                    # 市场回报，可以用returns_1d和0的平均值作为默认值
                    if "returns_1d" in features_df.columns:
                        features_df[feature] = (
                            features_df["returns_1d"] * 0.8
                        )  # 市场回报通常小于个股回报
                    else:
                        features_df[feature] = 0
                elif feature == "relative_strength":
                    # 相对强度，如果有returns_1d和market_returns，可以直接计算
                    if (
                        "returns_1d" in features_df.columns
                        and "market_returns" in features_df.columns
                    ):
                        features_df[feature] = (
                            features_df["returns_1d"] - features_df["market_returns"]
                        )
                    elif "returns_1d" in features_df.columns:
                        features_df[feature] = features_df[
                            "returns_1d"
                        ]  # 假设市场回报为0
                    else:
                        features_df[feature] = 0
                else:
                    # 其他特征默认为0
                    features_df[feature] = 0

        # 确保无穷大值和NaN值被替换
        features_df = features_df.replace([np.inf, -np.inf], 0).fillna(0)

        # 关键修改: 只保留模型需要的特征，丢弃多余特征
        if len(required_features) > 0:
            # 确保保留target列（如果存在）
            if "target" in features_df.columns and "target" not in required_features:
                required_features_with_target = list(required_features) + ["target"]
                result_df = features_df[required_features_with_target]
            else:
                result_df = features_df[required_features]

            print(
                f"已过滤特征: 从{len(features_df.columns)}个特征中保留了{len(result_df.columns)}个必要特征"
            )
            return result_df

        return features_df

    # 定义仓位大小计算函数
    def calculate_position_size(self, prediction, current_price, equity):
        """
        根据预测概率和当前资产计算仓位大小
        """
        # 防御性检查
        if equity <= 0 or current_price <= 0:
            return 0

        # 信心基础比例 - 预测概率越高，使用的资金比例越大
        confidence = min(1.0, max(0, prediction - 0.5))  # 将预测值转换为0-0.5的置信度
        base_ratio = confidence * self.position_scaling
        base_ratio = min(1.0, max(0, base_ratio))  # 再次确保在0-1之间

        # 计算目标资金比例
        target_equity_ratio = base_ratio * self.max_position_pct

        # 计算实际金额和股数
        position_value = equity * target_equity_ratio
        shares = int(position_value / current_price)

        # 确保不超过可用资金
        if shares * current_price > equity:
            shares = int(equity / current_price)

        return max(0, shares)  # 确保不会返回负数

    def next(self):
        """
        交易逻辑：根据机器学习模型的预测进行交易
        """
        if not self.trained:
            return

        # 获取当前位置
        current_idx = len(self.data) - 1

        # 只在有足够的历史数据后交易
        if current_idx < max(self.window, 200):
            return

        # 防御性检查：确保当前索引有效
        if current_idx >= len(self.data.Close):
            print(f"警告: 当前索引 {current_idx} 超出了数据范围 {len(self.data.Close)}")
            return

        try:
            current_price = self.data.Close[-1]
            current_prediction = self.predictions[current_idx]

            # 防御性检查：确保预测值有效
            if np.isnan(current_prediction):
                print(f"警告: 当前预测值为NaN，使用默认值0.5")
                current_prediction = 0.5
        except Exception as e:
            print(f"获取当前价格或预测值时出错: {e}")
            return

        # 动态调整预测阈值
        try:
            self._adjust_threshold()
        except Exception as e:
            print(f"调整阈值时出错: {e}")
            self.dynamic_threshold = self.prediction_threshold  # 使用默认阈值

        # 跟踪交易结果
        # 如果上一个柱有持仓，但现在没有，说明刚刚平仓，记录结果
        try:
            if (
                current_idx > 0
                and hasattr(self, "_prev_position")
                and self._prev_position
                and not self.position
            ):
                # 计算交易收益率
                if (
                    hasattr(self, "_prev_entry_price")
                    and self._prev_entry_price is not None
                    and self._prev_entry_price > 0
                ):
                    # 使用前一柱的价格作为退出价格
                    exit_price = self.data.Close[-2]
                    trade_return = (exit_price / self._prev_entry_price - 1) * 100

                    # 打印详细的调试信息
                    print(
                        f"交易详情 - 入场价: {self._prev_entry_price:.2f}, 出场价: {exit_price:.2f}, 收益率: {trade_return:.2f}%"
                    )

                    # 确保trade_results已经初始化
                    if not hasattr(self, "trade_results"):
                        self.trade_results = []

                    self.trade_results.append(trade_return)

                    # 计算并打印胜率
                    win_count = sum(1 for r in self.trade_results if r > 0)
                    total_trades = len(self.trade_results)
                    win_rate = win_count / max(1, total_trades)

                    print(
                        f"交易结束: 收益率 {trade_return:.2f}%, 累计交易: {total_trades}, "
                        f"胜利: {win_count}/{total_trades}, 胜率: {win_rate:.2f}"
                    )
                else:
                    print(
                        f"警告: 无法计算收益率，入场价未设置或无效: {getattr(self, '_prev_entry_price', 'N/A')}"
                    )
        except Exception as e:
            print(f"记录交易结果时出错: {e}")

        # 记录当前持仓状态，用于下一个柱的交易结果跟踪
        try:
            self._prev_position = bool(self.position)
            if self.position:
                # 使用我们自己记录的交易数据获取入场价格
                if hasattr(self, "my_open_trades") and self.my_open_trades:
                    # 计算加权平均买入价格
                    total_cost = sum(
                        price * size for _, price, size in self.my_open_trades
                    )
                    total_shares = sum(size for _, _, size in self.my_open_trades)

                    if total_shares > 0:
                        self._prev_entry_price = total_cost / total_shares
                        print(
                            f"从my_open_trades记录中获取加权平均入场价格: {self._prev_entry_price:.2f}"
                        )
                    else:
                        # 如果没有有效的交易记录，使用当前价格
                        self._prev_entry_price = current_price
                        print(
                            f"my_open_trades记录无效，使用当前价格作为入场价格: {self._prev_entry_price:.2f}"
                        )
                else:
                    # 如果没有my_open_trades记录，使用当前价格
                    self._prev_entry_price = current_price
                    print(
                        f"无交易记录，使用当前价格作为入场价格: {self._prev_entry_price:.2f}"
                    )
        except Exception as e:
            print(f"记录持仓状态时出错: {e}")
            # 确保 _prev_position 至少被设置为 False
            if not hasattr(self, "_prev_position"):
                self._prev_position = False

        # 输出当前预测值和市场状态
        if current_idx % 20 == 0:  # 每隔20个交易日输出一次
            print(
                f"日期: {self.data.index[current_idx]}, 价格: {current_price:.2f}, "
                f"预测概率: {current_prediction:.4f}, 阈值: {self.dynamic_threshold:.2f}, "
                f"市场状态: {self.market_regime}"
            )

            # 显示持仓信息
            if self.position:
                # 获取我们自己记录的持仓信息
                if hasattr(self, "my_open_trades") and self.my_open_trades:
                    total_shares = sum(size for _, _, size in self.my_open_trades)
                    total_cost = sum(
                        price * size for _, price, size in self.my_open_trades
                    )
                    avg_price = total_cost / total_shares if total_shares > 0 else 0

                    position_return = self.get_position_return()
                    print(
                        f"当前持仓: {total_shares:.2f} 股, 平均买入价: {avg_price:.2f}, "
                        f"当前收益: {position_return*100:.2f}%"
                    )
                else:
                    # 退化方案
                    position_return = self.get_position_return()
                    print(
                        f"当前持仓: {self.position.size} 股, "
                        f"当前收益: {position_return*100:.2f}%"
                    )

        # === 简化交易逻辑以增加交易频率 ===

        # 直接根据预测值和阈值做决策，减少其他条件限制
        if current_prediction > self.dynamic_threshold:
            # 预测上涨，有买入信号
            if not self.position:
                # 没有持仓，开仓
                position_size = self.calculate_position_size(
                    current_prediction, current_price, self.equity
                )
                if position_size > 0:
                    self.safe_buy(size=position_size)
                    print(
                        f"买入: {self.data.index[current_idx]}, 价格: {current_price:.2f}, "
                        f"预测: {current_prediction:.4f}, 阈值: {self.dynamic_threshold:.2f}, 买入: {position_size}股"
                    )
            elif current_prediction > self.dynamic_threshold + 0.05:
                # 已有仓位但预测非常强烈，考虑加仓
                additional_size = self.calculate_position_size(
                    current_prediction, current_price, self.equity * 0.3
                )  # 只使用剩余资金的30%
                if additional_size > 0:
                    self.safe_buy(size=additional_size)
                    print(
                        f"加仓: {self.data.index[current_idx]}, 价格: {current_price:.2f}, "
                        f"预测: {current_prediction:.4f}, 阈值: {self.dynamic_threshold:.2f}, 加仓: {additional_size}股"
                    )

        elif current_prediction < (1 - self.dynamic_threshold):
            # 预测下跌，有卖出信号
            if self.position:
                sell_size = self.position.size
                self.safe_sell()  # 卖出全部
                print(
                    f"卖出: {self.data.index[current_idx]}, 价格: {current_price:.2f}, "
                    f"预测: {current_prediction:.4f}, 阈值: {1-self.dynamic_threshold:.2f}, 卖出: {sell_size}股"
                )

        # 检查止损条件 (简化为只在非卖出信号时检查)
        elif self.position:
            position_return = self.get_position_return()

            # 止损条件
            if position_return <= -self.stop_loss_pct:
                sell_size = self.position.size
                self.safe_sell()
                print(
                    f"止损: {self.data.index[current_idx]}, 价格: {current_price:.2f}, "
                    f"亏损: {position_return:.2%}, 卖出: {sell_size}股"
                )

            # 止盈条件
            elif position_return >= self.take_profit_pct:
                sell_size = max(1, self.position.size // 2)  # 只卖出一半
                self.safe_sell(size=sell_size)
                print(
                    f"止盈: {self.data.index[current_idx]}, 价格: {current_price:.2f}, "
                    f"收益: {position_return:.2%}, 卖出: {sell_size}股"
                )

    def _detect_market_regime(self):
        """
        检测当前市场状态
        返回: 'bull'(牛市), 'bear'(熊市), 'neutral'(震荡市)
        """
        # 获取当前价格与移动平均的关系
        curr_idx = len(self.data) - 1
        if curr_idx < 200:  # 需要足够的数据
            return "neutral"

        try:
            price = self.data.Close[curr_idx]
            ma20 = self.ma20[curr_idx]
            ma50 = self.ma50[curr_idx]
            ma200 = self.ma200[curr_idx]

            # 检查NaN值
            if np.isnan(ma20) or np.isnan(ma50) or np.isnan(ma200):
                print(f"警告: 移动平均值包含NaN，返回中性市场状态")
                return "neutral"

            # 计算移动平均线的斜率（趋势强度）
            if curr_idx >= 20:
                ma20_prev = self.ma20[curr_idx - 20]
                if np.isnan(ma20_prev):
                    ma20_slope = 0
                else:
                    ma20_slope = (ma20 - ma20_prev) / 20
            else:
                ma20_slope = 0

            if curr_idx >= 50:
                ma50_prev = self.ma50[curr_idx - 50]
                if np.isnan(ma50_prev):
                    ma50_slope = 0
                else:
                    ma50_slope = (ma50 - ma50_prev) / 50
            else:
                ma50_slope = 0

            # 牛市定义: 价格在所有均线之上，且短期均线向上
            if price > ma20 > ma50 > ma200 and ma20_slope > 0 and ma50_slope > 0:
                return "bull"

            # 熊市定义: 价格在所有均线之下，且短期均线向下
            elif price < ma20 < ma50 < ma200 and ma20_slope < 0 and ma50_slope < 0:
                return "bear"

            # 其他情况为震荡市
            return "neutral"
        except Exception as e:
            print(f"检测市场状态时出错: {e}，返回中性市场状态")
            return "neutral"

    def _calculate_volatility(self):
        """
        计算当前市场的相对波动率
        返回值为相对于历史平均波动的比率，1表示正常波动，>1表示高波动，<1表示低波动
        """
        # 计算最近20天的波动率
        recent_prices = self.data.Close[-20:]
        recent_returns = np.diff(recent_prices) / recent_prices[:-1]
        recent_volatility = np.std(recent_returns)

        # 计算过去一年(252个交易日)的平均波动率
        historical_prices = self.data.Close[-252:]
        historical_returns = np.diff(historical_prices) / historical_prices[:-1]

        # 使用滚动窗口计算历史上的20天波动率序列
        rolling_std = []
        for i in range(len(historical_returns) - 19):
            window_returns = historical_returns[i : i + 20]
            rolling_std.append(np.std(window_returns))

        # 计算历史波动率的平均值
        avg_historical_volatility = np.mean(rolling_std)

        # 计算当前波动率相对于历史平均的比率
        relative_volatility = (
            recent_volatility / avg_historical_volatility
            if avg_historical_volatility > 0
            else 1.0
        )

        return relative_volatility

    def _adjust_threshold(self):
        """
        根据市场状态和历史交易表现动态调整预测阈值
        """
        # 检测市场状态
        current_regime = self._detect_market_regime()
        self.market_regime = current_regime

        # 基础阈值
        base_threshold = self.prediction_threshold

        try:
            # 根据市场状态调整
            if current_regime == "bull":
                # 牛市降低买入门槛
                regime_adjustment = -0.07  # 增大降低幅度
            elif current_regime == "bear":
                # 熊市提高买入门槛，但保持较低以确保有交易
                regime_adjustment = 0.02  # 减小提高幅度
            else:
                try:
                    # 根据震荡程度调整
                    volatility = self._calculate_volatility()  # 相对波动率
                    regime_adjustment = (
                        -0.03 + (volatility - 1) * 0.01
                    )  # 高波动时提高阈值
                except Exception as e:
                    print(f"计算波动率时出错: {e}")
                    regime_adjustment = -0.03

        except Exception as e:
            print(f"计算市场状态时出错: {e}")
            regime_adjustment = 0

        # 根据最近交易结果调整（如果有）
        performance_adjustment = 0
        # 更复杂的表现评估
        if len(self.trade_results) >= 10:  # 使用更多的历史记录
            # 考虑胜率和收益率两个因素
            recent_trades = self.trade_results[-10:]
            win_rate = sum(1 for r in recent_trades if r > 0) / len(recent_trades)
            avg_profit = sum(recent_trades) / len(recent_trades)

            # 胜率调整
            win_rate_adj = 0
            if win_rate < 0.4:
                win_rate_adj = 0.03
            elif win_rate > 0.7:  # 提高标准
                win_rate_adj = -0.03

            # 盈利能力调整
            profit_adj = 0
            if avg_profit < -0.01:  # 平均亏损
                profit_adj = 0.02
            elif avg_profit > 0.02:  # 平均盈利较好
                profit_adj = -0.02

            performance_adjustment = win_rate_adj + profit_adj
        # 添加市场波动率因素
        vix_value = (
            self.data.VIX[-1] if hasattr(self.data, "VIX") else 20
        )  # 默认值或使用实际VIX
        vix_adjustment = 0

        if vix_value > 30:  # 高波动环境
            vix_adjustment = 0.03  # 高波动提高阈值，减少交易频率
        elif vix_value < 15:  # 低波动环境
            vix_adjustment = -0.02  # 低波动降低阈值，增加交易机会

        # 计算调整后的阈值
        adjusted_threshold = (
            base_threshold + regime_adjustment + performance_adjustment + vix_adjustment
        )
        # 动态设定阈值范围限制
        if current_regime == "bull":
            min_threshold = 0.48  # 牛市可以降低最小阈值
            max_threshold = 0.58  # 也降低最大阈值
        elif current_regime == "bear":
            min_threshold = 0.52  # 熊市提高最小阈值
            max_threshold = 0.65  # 也提高最大阈值
        else:
            min_threshold = 0.50  # 震荡市使用标准范围
            max_threshold = 0.60

        self.dynamic_threshold = max(
            min_threshold, min(max_threshold, adjusted_threshold)
        )

        return self.dynamic_threshold

    def get_position_return(self):
        """
        安全获取当前持仓收益率
        返回: 当前持仓的收益率(浮点数), 如果没有持仓返回0
        """
        if not self.position:
            return 0

        current_price = self.data.Close[-1]

        # 优先使用我们自己记录的交易数据
        if hasattr(self, "my_open_trades") and self.my_open_trades:
            # 计算加权平均买入价格
            total_cost = sum(price * size for _, price, size in self.my_open_trades)
            total_shares = sum(size for _, _, size in self.my_open_trades)

            if total_shares > 0:
                avg_buy_price = total_cost / total_shares
                return current_price / avg_buy_price - 1
            else:
                print("警告: my_open_trades中没有有效的交易记录，总股数为0")

        # 如果没有我们自己的记录，尝试使用系统属性
        if hasattr(self.position, "pl_pct"):
            return self.position.pl_pct

        # 尝试获取入场价格
        entry_price = None

        if hasattr(self.position, "entry_price"):
            entry_price = self.position.entry_price
        elif hasattr(self.position, "open_price"):
            entry_price = self.position.open_price
        elif hasattr(self, "_prev_entry_price"):
            entry_price = self._prev_entry_price

        # 如果无法获取入场价格，使用合理的默认值
        if entry_price is None or entry_price <= 0:
            print("警告: 无法获取有效的入场价格，假设从当前价格的95%开始")
            # 使用当前价格的95%作为一个合理的默认值，表示小幅盈利
            entry_price = current_price * 0.95

        # 计算收益率
        return current_price / entry_price - 1

    def get_position_value(self):
        """
        计算当前持仓的市场价值
        返回: 当前持仓的市场价值(持仓数量 * 当前价格)
        """
        if not self.position:
            return 0

        # 使用当前价格
        current_price = self.data.Close[-1]

        # 首先尝试我们自己的记录
        if hasattr(self, "my_open_trades") and self.my_open_trades:
            # 计算总持仓数量
            total_shares = sum(size for _, _, size in self.my_open_trades)
            return total_shares * current_price

        # 退化方案：使用position.size
        return self.position.size * current_price

    def safe_buy(self, size=None):
        """
        安全地执行买入操作，带有错误处理
        size: 买入数量，如果为None则使用默认值(全部资金)
        返回: 成功返回True，失败返回False
        """
        try:
            # 检查size参数
            if size is not None and size <= 0:
                print(f"警告: 买入数量 {size} 无效，不执行买入")
                return False

            # 检查价格
            current_price = self.data.Close[-1]
            if current_price <= 0:
                print(f"警告: 当前价格 {current_price} 无效，不执行买入")
                return False

            # 检查可用资金
            if size is None and self.equity <= 0:
                print(f"警告: 账户资金 {self.equity} 不足，不执行买入")
                return False

            # 执行买入
            price_before = current_price  # 交易前价格
            self.buy(size=size)

            # 计算实际买入的数量（可能与请求的size不同）
            actual_size = size
            if actual_size is None and self.position:
                actual_size = self.position.size

            # 记录交易
            if actual_size and actual_size > 0:
                current_time = self.data.index[-1]

                # 记录到实例变量
                if not hasattr(self, "my_open_trades"):
                    self.my_open_trades = []

                trade_record = (current_time, price_before, actual_size)
                self.my_open_trades.append(trade_record)

                self.instance_trade_record["open_trades"].append(trade_record)
                self.instance_trade_record["last_action"] = "buy"
                self.instance_trade_record["last_action_price"] = price_before
                self.instance_trade_record["last_action_time"] = current_time
                self.instance_trade_record["trade_count"] += 1
                self.instance_trade_record["action_log"].append(
                    ("buy", current_time, price_before, actual_size)
                )

                print(
                    f"记录买入交易: 时间={current_time}, 价格={price_before:.2f}, 数量={actual_size}"
                )
                return True
            else:
                print("警告: 买入后未获取到有效的仓位大小")
                return False
        except Exception as e:
            print(f"买入操作失败: {e}")
            return False

    def safe_sell(self, size=None):
        """
        安全地执行卖出操作，带有错误处理
        size: 卖出数量，如果为None则卖出全部
        返回: 成功返回True，失败返回False
        """
        try:
            if self.position:
                # 确定卖出数量
                actual_size = size if size is not None else self.position.size
                current_size = self.position.size

                # 执行卖出前获取价格和时间
                price_before = self.data.Close[-1]
                current_time = self.data.index[-1]

                # 获取我们自己记录的交易数据
                my_trades = getattr(self, "my_open_trades", [])

                # 实际执行卖出 - 修复部分平仓逻辑
                if actual_size >= current_size:
                    # 全部平仓
                    self.position.close()
                    remaining_size = 0
                else:
                    # 记录是多头还是空头
                    is_long = (
                        self.position.is_long
                        if hasattr(self.position, "is_long")
                        else True
                    )

                    # 计算保留的仓位大小
                    remaining_size = current_size - actual_size

                    # 关闭全部仓位
                    self.position.close()

                    # 重新开一个小的仓位
                    if remaining_size > 0:
                        try:
                            if is_long:
                                self.buy(size=remaining_size)
                            else:
                                self.sell(size=remaining_size)
                            print(f"部分平仓后重新开仓: {remaining_size} 股")
                        except Exception as e:
                            print(f"重新开仓失败: {e}")
                            remaining_size = 0

                # 更新交易记录
                if my_trades:
                    # 计算平均买入价格
                    total_cost = sum(price * size for _, price, size in my_trades)
                    total_shares = sum(size for _, _, size in my_trades)

                    if total_shares <= 0:
                        print(
                            "警告: 交易记录中的总持仓数量为0或负数，使用当前价格作为平均买入价格"
                        )
                        avg_buy_price = price_before
                    else:
                        avg_buy_price = total_cost / total_shares

                    # 计算收益率
                    profit_pct = (price_before / avg_buy_price - 1) * 100

                    # 记录平仓交易
                    closed_trade = (
                        my_trades[0][0],
                        avg_buy_price,
                        actual_size,
                        current_time,
                        price_before,
                        profit_pct,
                    )

                    # 更新我的交易记录
                    if remaining_size == 0:
                        # 全部平仓，清空open_trades
                        self.my_open_trades = []
                    else:
                        # 部分平仓，更精确地管理
                        # 创建一个队列，按FIFO顺序处理交易
                        new_trades = []
                        remaining = actual_size

                        for trade_time, trade_price, trade_size in my_trades:
                            if remaining <= 0:
                                # 已经平掉了足够的份额，剩下的保持不变
                                new_trades.append((trade_time, trade_price, trade_size))
                            elif remaining >= trade_size:
                                # 这笔交易全部平掉
                                remaining -= trade_size
                            else:
                                # 这笔交易部分平掉
                                new_trades.append(
                                    (trade_time, trade_price, trade_size - remaining)
                                )
                                remaining = 0

                        self.my_open_trades = new_trades

                    self.instance_trade_record["open_trades"] = (
                        self.my_open_trades.copy()
                    )
                    self.instance_trade_record["closed_trades"].append(closed_trade)
                    self.instance_trade_record["last_action"] = "sell"
                    self.instance_trade_record["last_action_time"] = current_time
                    self.instance_trade_record["last_action_price"] = price_before
                    self.instance_trade_record["action_log"].append(
                        ("sell", current_time, price_before, actual_size)
                    )

                    print(
                        f"记录卖出交易: 时间={current_time}, 价格={price_before:.2f}, "
                        f"数量={actual_size}, 收益率={profit_pct:.2f}%"
                    )

                return True
            return False
        except Exception as e:
            print(f"卖出操作失败: {e}")
            return False
