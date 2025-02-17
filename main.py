import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import json

# 配置日志记录, 添加时间戳
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path="config.json"):
    """从JSON文件加载配置"""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            # 配置参数验证
            if config["base_investment"] <= 0:
                raise ValueError("base_investment 必须大于0")
            if config["upper_limit_factor"] <= config["lower_limit_factor"]:
                raise ValueError("upper_limit_factor 必须大于 lower_limit_factor")
            if not (0 <= config.get("trend_threshold_buy", 60) <= 100):
                raise ValueError("trend_threshold_buy 必须在 [0, 100] 范围内")
            if not (0 <= config.get("trend_threshold_strong_buy", 70) <= 100):
                raise ValueError("trend_threshold_strong_buy 必须在 [0, 100] 范围内")
            if not (0 <= config.get("risk_threshold_low", 30) <= 100):
                raise ValueError("risk_threshold_low 必须在 [0, 100] 范围内")
            if not (0 <= config.get("risk_threshold_high", 70) <= 100):
                raise ValueError("risk_threshold_high 必须在 [0, 100] 范围内")
            return config
    except FileNotFoundError:
        logging.error(f"配置文件 '{config_path}' 未找到")
        raise
    except json.JSONDecodeError:
        logging.error(f"配置文件 '{config_path}' 格式错误")
        raise
    except ValueError as e:
        logging.error(f"配置参数错误: {e}")
        raise


def calculate_market_indicators(
    data: pd.DataFrame, periods: list = [20, 50, 200], rsi_period: int = 14
) -> dict:
    """计算市场技术指标"""
    missing_values = data.isnull().sum()
    if missing_values.any():
        logging.warning(f"数据中存在缺失值:\n{missing_values}")
        data.fillna(method="ffill", inplace=True)
        data.fillna(method="bfill", inplace=True)


    close = data["Close"]

    indicators = {}
    # 优化SMA计算
    for period in periods:
        indicators[f"sma_{period}"] = close.rolling(window=period).mean()

    # 计算RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(
        alpha=1 / rsi_period, min_periods=rsi_period, adjust=False
    ).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()

    rs = avg_gain / avg_loss
    indicators["rsi"] = 100 - (100 / (1 + rs))

    # 计算MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    indicators["macd"] = exp1 - exp2
    indicators["macd"] = indicators["macd"].fillna(0)  # 避免后续 NaN 影响
    indicators["signal"] = indicators["macd"].ewm(span=9, adjust=False).mean()

    return indicators


def calculate_trend_score(
    current_price: float,
    indicators: dict,
    sma_weights: dict,
    macd_weight: float,
    rsi_weights: dict,
    sma_deviation_weight: float,
) -> float:
    """计算市场趋势得分"""
    score = 50

    # Use .item() to get scalar values for calculation
    latest_sma_200 = indicators.get("sma_200", pd.Series([0.0])).iloc[-1].item()
    latest_sma_50 = indicators.get("sma_50", pd.Series([0.0])).iloc[-1].item()
    latest_sma_20 = indicators.get("sma_20", pd.Series([0.0])).iloc[-1].item()
    latest_rsi = indicators.get("rsi", pd.Series([50.0])).iloc[-1].item()
    latest_macd = indicators.get("macd", pd.Series([0.0])).iloc[-1].item()
    latest_signal = indicators.get("signal", pd.Series([0.0])).iloc[-1].item()

    deviation = np.tanh((current_price - latest_sma_200) / max(latest_sma_200, 1e-3))

    if deviation > 0:
        score -= deviation * sma_deviation_weight  # 高于均线，趋势可能过热，扣分
    else:
        score += abs(deviation) * sma_deviation_weight  # 低于均线，可能是低估机会，加分

    # 价格与均线的偏离程度
    if latest_sma_50 > 0:
        score -= (
            (current_price - latest_sma_50) / latest_sma_50
        ) * sma_deviation_weight
    if latest_sma_20 > 0:
        score -= (
            (current_price - latest_sma_20) / latest_sma_20
        ) * sma_deviation_weight

    # from scipy.stats import linregress
    # def get_slope(series, window=5):
    #     x = np.arange(window)
    #     y = series.iloc[-window:].values
    #     slope = linregress(x, y).slope
    #     return slope * 100

    macd_slope = (
        indicators["macd"]
        .diff()
        .ewm(span=5, adjust=False)
        .mean()
        .fillna(0)
        .iloc[-1]
        .item()
    )

    if macd_slope > 0:
        score += macd_weight * 0.5  # MACD 上升趋势，加分
    else:
        score -= macd_weight * 0.5  # MACD 下降趋势，减分

    sma_50_slope = indicators["sma_50"].diff().rolling(window=3).mean().iloc[-1].item()
    if sma_50_slope > 0:
        score += sma_weights.get("sma_50_up", 0)

    sma_20_slope = indicators["sma_20"].diff().rolling(window=3).mean().iloc[-1].item()
    if sma_20_slope > 0:
        score += sma_weights.get("sma_20_up", 0)

    # MACD死叉，降低趋势得分
    if latest_macd < latest_signal:
        score -= macd_weight

    # RSI
    if latest_rsi >= 80:
        score -= 15  # 过热市场可能需要更大扣分
    elif latest_rsi <= 20:
        score += 15  # 超卖市场可能需要更大加分
    elif 40 < latest_rsi < 60:
        score += rsi_weights.get("normal", 0)
    elif latest_rsi >= 70:
        score += rsi_weights.get("overbought", 0)
    elif latest_rsi <= 30:
        score += rsi_weights.get("oversold", 0)

    return min(max(score, 0), 100)


def calculate_risk_score(
    current_price: float,
    data: pd.DataFrame,
    indicators: dict,
    volatility_weight: float,
    sma_deviation_weight: float,
) -> float:
    """计算风险分数"""
    score = 50

    returns = data["Close"].pct_change()
    if returns.empty or len(returns) < 20:
        volatility = 0.0
    else:
        volatility = float(
            returns.rolling(window=20).std().dropna().iloc[-1] * np.sqrt(252)
        )

    latest_sma_50 = float(indicators.get("sma_50", pd.Series([0.0])).iloc[-1])

    max_volatility_impact = 30
    median_volatility = returns.rolling(window=100).std().median() * np.sqrt(252)
    median_volatility = median_volatility.item()
    volatility_adjustment = (volatility - median_volatility) * volatility_weight * 100
    score += np.minimum(
        max_volatility_impact, np.maximum(-max_volatility_impact, volatility_adjustment)
    )


    # 价格负向偏离SMA50，增加风险得分
    if latest_sma_50 > 0:  # avoid ZeroDivisionError
        deviation = min(
            max((latest_sma_50 - current_price) / (latest_sma_50 + 1e-6), -0.5), 0.5
        )
        score += deviation * sma_deviation_weight

    return min(max(score, 0), 100)


def should_pause_investment(
    current_price: float,
    trend_score: float,
    indicators: dict,
    pause_threshold: float = 1.15,
) -> bool:
    """判断是否应该暂停投资"""
    # Use .item() to get scalar values for comparison
    rsi_value = indicators.get("rsi", pd.Series([50.0])).iloc[-1].item()
    sma_50_value = indicators.get("sma_50", pd.Series([0.0])).iloc[-1].item()
    macd_below_signal = (
        indicators["macd"].iloc[-1].item() < indicators["signal"].iloc[-1].item()
    )
    trend_weak = trend_score < 40  # 额外条件
    return (
        (rsi_value > 80 and macd_below_signal)
        or (rsi_value < 20 and macd_below_signal and trend_weak)
        or (current_price > sma_50_value * pause_threshold)
    )


def get_market_signal(
    trend_score: float,
    risk_score: float,
    trend_threshold_strong_buy: int,
    trend_threshold_buy: int,
    risk_threshold_low: int,
    risk_threshold_high: int,
) -> str:
    """获取市场信号"""
    neutral_range = max(5, 10 - (risk_score / 10)) 
    if trend_score > trend_threshold_strong_buy and risk_score < risk_threshold_low:
        return "强烈买入"
    elif trend_score > trend_threshold_buy and risk_score < risk_threshold_high:
        return "买入"
    elif trend_score < (100 - trend_threshold_buy) and risk_score > risk_threshold_high:
        return "卖出 || 观望"
    elif abs(trend_score - 50) < neutral_range and abs(risk_score - 50) < neutral_range:
        return "持平"
    else:
        return "谨慎"


def get_technical_summary(indicators: dict) -> dict:
    """获取技术指标摘要"""
    return {
        "macd_signal": (
            "看多"
            if float(indicators.get("macd", pd.Series([0.0])).iloc[-1].item())
            > float(indicators.get("signal", pd.Series([0.0])).iloc[-1].item())
            else "看空"
        ),  # Use .item()
        "rsi": float(
            indicators.get("rsi", pd.Series([50.0])).iloc[-1].item()
        ),  # Use .item()
        "price_position": (
            "强势"
            if float(indicators.get("sma_20", pd.Series([0.0])).iloc[-1].item())
            > float(indicators.get("sma_50", pd.Series([0.0])).iloc[-1].item())
            else "弱势"
        ),  # Use .item()
    }


def calculate_dynamic_investment(config: dict) -> dict:
    """增强版动态定投计算器"""
    required_keys = [
        "ticker",
        "base_investment",
        "period",
        "adjustment_factor",
        "upper_limit_factor",
        "lower_limit_factor",
        "risk_control",
        "pause_threshold",
    ]
    if not all(key in config for key in required_keys):
        raise ValueError(f"配置文件缺少必要的键: {', '.join(required_keys)}")

    ticker = config["ticker"]
    base_investment = config["base_investment"]
    # ... 其他参数 ...

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3 * 365)
        data = yf.download(ticker, start=start_date, end=end_date)
        if len(data) == 0:
            raise ValueError(f"无法获取{ticker}的数据，请检查股票代码是否正确")

        indicators = calculate_market_indicators(
            data,
            periods=config.get("sma_periods", [20, 50, 200]),
            rsi_period=config.get("rsi_period", 14),
        )

        current_price = data["Close"].iloc[-1].item()  # Use .item()
        volume = data["Volume"].iloc[-1].item()  # Use .item()
        avg_volume = (
            data["Volume"].rolling(window=20).mean().iloc[-1].item()
        )  # Use .item()

        trend_score = calculate_trend_score(
            current_price,
            indicators,
            sma_weights=config.get(
                "sma_weights",
                {
                    "sma_200": 10,
                    "sma_50": 5,
                    "sma_20": 5,
                    "sma_20_up": 5,
                    "sma_50_up": 5,
                },
            ),
            macd_weight=config.get("macd_weight", 10),
            rsi_weights=config.get(
                "rsi_weights", {"normal": 10, "overbought": -10, "oversold": 10}
            ),
            sma_deviation_weight=config.get("sma_deviation_weight", 50),
        )

        risk_score = calculate_risk_score(
            current_price,
            data,
            indicators,
            volatility_weight=config.get("volatility_weight", 2),
            sma_deviation_weight=config.get("sma_deviation_weight_risk", 100),
        )

        # trend_adjustment = np.clip(
        #     np.sign(trend_score - 50) * (abs(trend_score - 50) / 50) ** 1.1 * 0.2, -0.3, 0.3
        # )
        # risk_adjustment = np.clip(
        #     np.sign(50 - risk_score) * (abs(50 - risk_score) / 50) ** 1.1 * 0.2, -0.3, 0.3
        # )
        sensitivity = 0.5 * (risk_score / 100)  # 风险越高，调整越保守
        trend_adjustment = np.tanh((trend_score - 50) / (10 - sensitivity * 8)) * 0.4
        risk_adjustment = np.tanh((risk_score - 50) / 5) * 0.3

        base_adjustment = 1 + trend_adjustment - risk_adjustment
        investment_amount = base_investment * base_adjustment

        if config["risk_control"] and risk_score > 75:
            investment_amount *= 0.5
        if config["risk_control"] and should_pause_investment(
            current_price, trend_score, indicators, config["pause_threshold"]
        ):
            investment_amount = 0

        upper_limit = base_investment * config["upper_limit_factor"]
        lower_limit = base_investment * config["lower_limit_factor"]
        investment_amount = max(min(investment_amount, upper_limit), lower_limit)

        return {
            "investment_amount": round(investment_amount, 2),
            "current_price": round(current_price, 2),
            "trend_score": round(trend_score, 2),
            "risk_score": round(risk_score, 2),
            "market_signal": get_market_signal(
                trend_score,
                risk_score,
                config.get("trend_threshold_strong_buy", 70),
                config.get("trend_threshold_buy", 60),
                config.get("risk_threshold_low", 30),
                config.get("risk_threshold_high", 70),
            ),
            "volume_ratio": round(volume / avg_volume, 2),
            "technical_indicators": get_technical_summary(indicators),
        }

    except (IOError, OSError) as e:
        logging.error(f"网络错误，无法获取 {ticker} 的数据: {e}")
        raise
    except ValueError as e:
        logging.error(f"数据错误，{ticker}: {e}")
        raise
    except KeyError as e:
        logging.error(
            f"数据中缺少必要的列: {e}, 请检查yfinance返回的数据是否包含Close, Volume等"
        )
        raise
    except Exception as e:
        logging.exception(f"计算 {ticker} 定投金额时发生未知错误: {e}")
        raise


if __name__ == "__main__":
    # try:
        config = load_config("config.json")

        for name in [
            "VOO",
            "QQQ",
            "XLK",
            "IGV",
            "NTES",
            "BIDU",
            "INTC",
            "AAPL",
            "TSLA",
            "MSFT",
            "AMZN",
            "GOOGL",
            "NVDA",
            "PLTR",
            "SONY",
            "QCOM",
            "TSM",
            "SPOT",
            "KO",
            "MCD",
            "TCEHY",
            "NFLX",
            "META",
            "AMZN",
        ]:
            config["ticker"] = name
            result = calculate_dynamic_investment(config)

            print(f"\n当前标的: {config['ticker']}{result['market_signal']}")
            print("\n=== 投资建议 ===")
            print(f"基准定投金额: {config['base_investment']:.2f}")
            print(f"建议投资金额: {result['investment_amount']:.2f}")
            print(f"当前价格: {result['current_price']:.2f}")
            print(f"市场信号: {result['market_signal']}")
            print(f"趋势得分: {result['trend_score']}")
            print(f"风险得分: {result['risk_score']}")
            print(f"成交量比: {result['volume_ratio']}")

            print("\n=== 技术指标 ===")
            print(f"MACD信号: {result['technical_indicators']['macd_signal']}")
            print(f"RSI: {result['technical_indicators']['rsi']:.2f}")
            print(f"价格位置: {result['technical_indicators']['price_position']}")

            with open("investment_results.csv", "a") as f:
                f.write(
                    # 2025/2/11 12:32
                    f"{datetime.now().strftime('%Y/%m/%d %H:%M')},"
                    f"{config['ticker']},"
                    f"{result['investment_amount']:.2f},"
                    f"{result['current_price']:.2f},"
                    f"{result['market_signal']},"
                    f"{result['trend_score']:.2f},"
                    f"{result['risk_score']:.2f},"
                    f"{result['volume_ratio']:.2f},"
                    f"{result['technical_indicators']['macd_signal']},"
                    f"{result['technical_indicators']['rsi']:.2f},"
                    f"{result['technical_indicators']['price_position']}\n"
                )

    # except Exception as e:
    #     print(f"程序执行出错: {str(e)}")
