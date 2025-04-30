import yfinance as yf
import pandas as pd
from datetime import date, timedelta, datetime
import logging
import time

# 配置日志 (可以共享主日志配置，或单独配置)
logger = logging.getLogger(__name__)  # 使用模块名作为 logger 名称
# 如果希望它使用 main.py 中配置的 logger，可以不做额外配置
# 如果需要独立配置，取消下面的注释
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler() # 输出到控制台
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)


class DataService:
    """
    负责获取和缓存金融数据 (股票、指数、VIX) 的服务类。
    使用内存缓存来减少重复的 API 调用。
    """

    def __init__(self, cache_ttl_seconds=3600):
        """
        初始化 DataService。

        Args:
            cache_ttl_seconds (int): 内存缓存的生存时间（秒）。默认为 1 小时。
                                     设为 0 或 None 可禁用基于时间的缓存，
                                     但仍会缓存同一实例生命周期内完全相同的请求。
        """
        self._cache = {}  # 缓存结构: {(symbol, start_str, end_str): (timestamp, data)}
        self.cache_ttl_seconds = cache_ttl_seconds
        logger.info(f"DataService initialized. Cache TTL: {cache_ttl_seconds} seconds.")

    def _get_cache_key(self, symbol, start_date, end_date):
        """生成缓存键"""
        # 规范化日期为 ISO 格式字符串以确保一致性
        start_str = (
            start_date.isoformat()
            if isinstance(start_date, (date, datetime))
            else str(start_date)
        )
        end_str = (
            end_date.isoformat()
            if isinstance(end_date, (date, datetime))
            else str(end_date)
        )
        return (symbol.upper(), start_str, end_str)

    def _is_cache_valid(self, cache_entry):
        """检查缓存条目是否仍然有效"""
        if self.cache_ttl_seconds is None or self.cache_ttl_seconds <= 0:
            return True  # 如果禁用 TTL，则缓存永不因时间过期（但在实例生命周期内）

        timestamp, _ = cache_entry
        now = time.time()
        return (now - timestamp) < self.cache_ttl_seconds

    def get_symbol_data(
        self, symbol, start_date, end_date, interval="1d", max_retries=3, retry_delay=2
    ):
        """
        获取指定代码（股票、指数、ETF等）的历史数据。

        Args:
            symbol (str): 股票或指数代码 (e.g., "AAPL", "^GSPC").
            start_date (date or str): 开始日期 (YYYY-MM-DD).
            end_date (date or str): 结束日期 (YYYY-MM-DD).
            interval (str): 数据间隔 ('1d', '1wk', '1mo', etc.). 默认为 '1d'.
            max_retries (int): 下载失败时的最大重试次数。
            retry_delay (int): 重试前的初始延迟（秒）。

        Returns:
            pd.DataFrame or None: 包含 OHLCV 数据的 DataFrame，如果失败则返回 None。
                                  DataFrame 索引是日期时间对象。
        """
        # 规范化日期类型
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        # yfinance 的 end date 是不包含的，所以加一天
        end_date_yf = end_date + timedelta(days=1)

        cache_key = self._get_cache_key(symbol, start_date, end_date)

        # 检查缓存
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.debug(f"Cache hit for {symbol} ({start_date} to {end_date})")
                return cache_entry[1].copy()  # 返回副本以防外部修改缓存
            else:
                logger.debug(f"Cache expired for {symbol} ({start_date} to {end_date})")
                del self._cache[cache_key]  # 删除过期条目

        logger.debug(
            f"Cache miss for {symbol} ({start_date} to {end_date}). Fetching from yfinance..."
        )

        # 从 yfinance 下载数据
        current_retry = 0
        data = None
        while current_retry < max_retries:
            try:
                data = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date_yf,  # 使用调整后的结束日期
                    interval=interval,
                    progress=False,
                    auto_adjust=True,  # 推荐使用，自动调整拆分和股息
                    ignore_tz=True,  # 忽略时区信息，简化处理
                    multi_level_index=False
                )

                if data is None or data.empty:
                    logger.warning(
                        f"No data returned for {symbol} from {start_date} to {end_date}."
                    )
                    # 缓存空结果，避免重复尝试无效代码
                    self._cache[cache_key] = (time.time(), pd.DataFrame())
                    return pd.DataFrame()

                # 基本的数据清洗和验证
                # 确保索引是 DatetimeIndex
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)

                # 移除完全是 NaN 的行 (yfinance 有时会返回奇怪的空行)
                data.dropna(axis=0, how="all", inplace=True)

                # 检查必要的列是否存在
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                if not all(col in data.columns for col in required_cols):
                    logger.error(
                        f"Downloaded data for {symbol} is missing required columns."
                    )
                    raise ValueError("Missing required columns")  # 触发重试或失败

                # 确保数据类型正确
                for col in required_cols:
                    data[col] = pd.to_numeric(data[col], errors="coerce")
                data.dropna(
                    subset=required_cols, inplace=True
                )  # 删除转换后产生的 NaN 行

                logger.info(
                    f"Successfully downloaded {len(data)} rows for {symbol} ({start_date} to {end_date})"
                )
                # 存入缓存
                self._cache[cache_key] = (time.time(), data.copy())
                return data.copy()  # 返回副本

            except Exception as e:
                current_retry += 1
                logger.warning(
                    f"Error downloading {symbol} (Attempt {current_retry}/{max_retries}): {e}"
                )
                if current_retry < max_retries:
                    time.sleep(retry_delay * (2 ** (current_retry - 1)))  # 指数退避
                else:
                    logger.error(
                        f"Failed to download {symbol} after {max_retries} attempts."
                    )
                    # 缓存失败信息 (None)
                    self._cache[cache_key] = (time.time(), None)
                    return None

        # 如果循环结束仍未成功 (理论上不会到这里，因为 return 在循环内)
        return None

    def get_stock_data(self, ticker, start_date, end_date, interval="1d"):
        """获取单个股票的数据 (对 get_symbol_data 的封装)"""
        return self.get_symbol_data(ticker, start_date, end_date, interval=interval)

    def get_market_index_data(
        self, start_date, end_date, index_symbol="^GSPC", interval="1d"
    ):
        """获取市场指数数据 (默认为 S&P 500)"""
        return self.get_symbol_data(
            index_symbol, start_date, end_date, interval=interval
        )

    def get_vix_data(self, start_date, end_date, interval="1d"):
        """获取 VIX 指数数据"""
        # VIX 通常不需要 auto_adjust，因为它不是股票
        # 但为了接口统一性，这里仍然调用通用方法，yf 会处理
        return self.get_symbol_data("^VIX", start_date, end_date, interval=interval)


# --- 使用示例 ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 创建服务实例
    data_service = DataService(cache_ttl_seconds=5)  # 短缓存用于测试

    # 定义日期范围
    today = date.today()
    start_dt = today - timedelta(days=365 * 1)  # 获取一年数据

    # 1. 获取股票数据
    print("\n--- 获取 AAPL 数据 ---")
    aapl_data = data_service.get_stock_data("AAPL", start_dt, today)
    if aapl_data is not None:
        print(
            f"获取到 AAPL 数据 {aapl_data.shape[0]} 行, 从 {aapl_data.index.min()} 到 {aapl_data.index.max()}"
        )
        # print(aapl_data.head())
    else:
        print("获取 AAPL 数据失败")

    # 2. 再次获取 AAPL 数据 (应命中缓存)
    print("\n--- 再次获取 AAPL 数据 ---")
    aapl_data_cached = data_service.get_stock_data("AAPL", start_dt, today)
    if aapl_data_cached is not None:
        print(f"从缓存获取到 AAPL 数据 {aapl_data_cached.shape[0]} 行")
    else:
        print("从缓存获取 AAPL 数据失败")

    # 3. 获取市场指数数据
    print("\n--- 获取 ^GSPC 数据 ---")
    gspc_data = data_service.get_market_index_data(start_dt, today)
    if gspc_data is not None:
        print(
            f"获取到 ^GSPC 数据 {gspc_data.shape[0]} 行, 从 {gspc_data.index.min()} 到 {gspc_data.index.max()}"
        )
    else:
        print("获取 ^GSPC 数据失败")

    # 4. 获取 VIX 数据
    print("\n--- 获取 ^VIX 数据 ---")
    vix_data = data_service.get_vix_data(start_dt, today)
    if vix_data is not None:
        print(
            f"获取到 ^VIX 数据 {vix_data.shape[0]} 行, 从 {vix_data.index.min()} 到 {vix_data.index.max()}"
        )
    else:
        print("获取 ^VIX 数据失败")

    # 5. 测试无效代码
    print("\n--- 获取无效代码数据 ---")
    invalid_data = data_service.get_stock_data("INVALIDTICKERXYZ", start_dt, today)
    if invalid_data is not None and invalid_data.empty:
        print("获取无效代码数据按预期返回空 DataFrame")
    elif invalid_data is None:
        print("获取无效代码数据失败 (None)")
    else:
        print(f"获取无效代码数据时发生意外情况: {type(invalid_data)}")

    # 6. 测试缓存过期 (如果 cache_ttl_seconds 较短)
    print("\n--- 测试缓存过期 ---")
    print(f"等待 {data_service.cache_ttl_seconds + 5} 秒...")
    time.sleep(data_service.cache_ttl_seconds + 5)
    print("再次获取 AAPL 数据...")
    aapl_data_expired = data_service.get_stock_data("AAPL", start_dt, today)
    if aapl_data_expired is not None:
        print(f"再次获取到 AAPL 数据 {aapl_data_expired.shape[0]} 行 (应为重新下载)")
