# API 参考

本 API 文档涵盖了 AKQuant 的核心类和方法。

## 1. 高级入口 (High-Level API)

### `akquant.run_backtest`

最常用的回测入口函数，封装了引擎的初始化和配置过程。

```python
def run_backtest(
    data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame], List[Bar]]] = None,
    strategy: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None] = None,
    symbol: Union[str, List[str]] = "BENCHMARK",
    initial_cash: Optional[float] = None,
    commission_rate: Optional[float] = None,
    stamp_tax_rate: float = 0.0,
    transfer_fee_rate: float = 0.0,
    min_commission: float = 0.0,
    execution_mode: Union[ExecutionMode, str] = ExecutionMode.NextOpen,
    timezone: Optional[str] = None,
    t_plus_one: bool = False,
    initialize: Optional[Callable[[Any], None]] = None,
    context: Optional[Dict[str, Any]] = None,
    history_depth: Optional[int] = None,
    warmup_period: int = 0,
    lot_size: Union[int, Dict[str, int], None] = None,
    show_progress: Optional[bool] = None,
    start_time: Optional[Union[str, Any]] = None,
    end_time: Optional[Union[str, Any]] = None,
    config: Optional[BacktestConfig] = None,
    instruments_config: Optional[Union[List[InstrumentConfig], Dict[str, InstrumentConfig]]] = None,
    custom_matchers: Optional[Dict[AssetType, Any]] = None,
    **kwargs: Any,
) -> BacktestResult
```

**关键参数:**

*   `data`: 回测数据。支持单个 DataFrame，或 `{symbol: DataFrame}` 字典。
*   `strategy`: 策略类或实例。也支持传入 `on_bar` 函数（函数式编程风格）。
*   `symbol`: 标的代码或代码列表。
*   `initial_cash`: 初始资金 (默认 1,000,000.0)。
*   `execution_mode`: 执行模式。
    *   `ExecutionMode.NextOpen`: 下一 Bar 开盘价成交 (默认)。
    *   `ExecutionMode.CurrentClose`: 当前 Bar 收盘价成交。
*   `t_plus_one`: 是否启用 T+1 交易规则 (默认 False)。如果启用，将强制使用中国市场模型。
*   `warmup_period`: 策略预热期。指定需要预加载的历史数据长度（Bar 数量），用于计算指标。
*   `start_time` / `end_time`: 回测开始/结束时间。
*   `config`: `BacktestConfig` 配置对象，用于集中管理配置。
*   `instruments_config`: 标的配置。用于设置期货/期权等非股票资产的参数（如乘数、保证金）。
    *   接收 `List[InstrumentConfig]` 或 `{symbol: InstrumentConfig}`。

### `akquant.BacktestConfig`

用于集中配置回测参数的数据类。

```python
@dataclass
class BacktestConfig:
    strategy_config: StrategyConfig
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    instruments: Optional[List[str]] = None
    instruments_config: Optional[Union[List[InstrumentConfig], Dict[str, InstrumentConfig]]] = None
    benchmark: Optional[str] = None
    timezone: str = "Asia/Shanghai"
    show_progress: bool = True
    history_depth: int = 0
```

### `akquant.StrategyConfig`

策略层面的配置，包含资金、费率和风控。

```python
@dataclass
class StrategyConfig:
    initial_cash: float = 100000.0
    commission_rate: float = 0.0
    stamp_tax_rate: float = 0.0
    transfer_fee_rate: float = 0.0
    min_commission: float = 0.0
    enable_fractional_shares: bool = False
    risk: Optional[RiskConfig] = None
```

### `akquant.InstrumentConfig`

用于配置单个标的属性的数据类。

```python
@dataclass
class InstrumentConfig:
    symbol: str
    asset_type: str = "STOCK"  # "STOCK", "FUTURES", "FUND", "OPTION"
    multiplier: float = 1.0    # 合约乘数
    margin_ratio: float = 1.0  # 保证金率 (0.1 表示 10% 保证金)
    tick_size: float = 0.01    # 最小变动价位
    lot_size: int = 1          # 最小交易单位
    # 期权相关
    option_type: Optional[str] = None  # "CALL" 或 "PUT"
    strike_price: Optional[float] = None
    expiry_date: Optional[str] = None  # YYYY-MM-DD
```

## 2. 策略开发 (Strategy)

### `akquant.Strategy`

策略基类。用户应继承此类并重写回调方法。

**回调方法:**

*   `on_start()`: 策略启动时触发。用于订阅 (`subscribe`) 和注册指标。
*   `on_bar(bar: Bar)`: K 线闭合时触发。
*   `on_tick(tick: Tick)`: Tick 到达时触发。
*   `on_timer(payload: str)`: 定时器触发。
*   `on_stop()`: 策略停止时触发。
*   `on_train_signal(context)`: 滚动训练信号触发 (ML 模式)。

**属性与快捷访问:**

*   `self.symbol`: 当前正在处理的标的代码。
*   `self.close`, `self.open`, `self.high`, `self.low`, `self.volume`: 当前 Bar/Tick 的价格和成交量。
*   `self.position`: 当前标的持仓对象 (`Position`)，包含 `size` 和 `available` 属性。
*   `self.now`: 当前回测时间 (`pd.Timestamp`)。

**交易方法:**

*   `buy(symbol, quantity, price=None, ...)`: 买入。不指定 `price` 则为市价单。
*   `sell(symbol, quantity, price=None, ...)`: 卖出。
*   `short(symbol, quantity, price=None, ...)`: 卖空。
*   `cover(symbol, quantity, price=None, ...)`: 平空。
*   `stop_buy(symbol, trigger_price, quantity, ...)`: 止损买入 (Stop Market)。当价格突破 `trigger_price` 时触发市价买单。
*   `stop_sell(symbol, trigger_price, quantity, ...)`: 止损卖出 (Stop Market)。当价格跌破 `trigger_price` 时触发市价卖单。
*   `order_target_value(target_value, symbol, price=None)`: 调整至目标持仓市值。
*   `order_target_percent(target_percent, symbol, price=None)`: 调整至目标账户占比。
*   `close_position(symbol)`: 平仓指定标的。
*   `cancel_order(order_id: str)`: 撤销指定订单。
*   `cancel_all_orders(symbol)`: 取消指定标的的所有挂单。如果不指定 `symbol`，则取消所有挂单。

**数据与工具:**

*   `get_history(count, symbol, field="close") -> np.ndarray`: 获取历史数据数组 (Zero-Copy)。
*   `get_history_df(count, symbol) -> pd.DataFrame`: 获取历史数据 DataFrame (OHLCV)。
*   `get_position(symbol) -> float`: 获取当前持仓量。
*   `get_cash() -> float`: 获取当前可用资金。
*   `get_account() -> Dict[str, float]`: 获取账户详情快照。包含 `cash` (可用资金), `equity` (总权益), `market_value` (持仓市值), 以及 `frozen_cash` 和 `margin` (预留字段，暂为0)。
*   `get_order(order_id) -> Order`: 获取指定订单详情。
*   `get_open_orders(symbol) -> List[Order]`: 获取当前未完成订单列表。
*   `subscribe(instrument_id: str)`: 订阅行情。对于多标的回测或实盘，必须显式订阅才能接收 `on_tick`/`on_bar` 回调。
*   `log(msg: str, level: int)`: 输出带时间戳的日志。
*   `schedule(trigger_time, payload)`: 注册单次定时任务。
*   `add_daily_timer(time_str, payload)`: 注册每日定时任务。

**机器学习支持:**

*   `set_rolling_window(train_window, step)`: 设置滚动训练窗口。
*   `get_rolling_data(length, symbol)`: 获取滚动训练数据 (X, y)。
*   `prepare_features(df, mode)`: (需重写) 特征工程与标签生成。

### `akquant.Bar`

K 线数据对象。

*   `timestamp`: Unix 时间戳 (纳秒)。
*   `open`, `high`, `low`, `close`, `volume`: OHLCV 数据。
*   `symbol`: 标的代码。

## 3. 核心引擎 (Core)

### `akquant.Engine`

回测引擎的主入口 (通常通过 `run_backtest` 隐式使用)。

**配置方法:**

*   `set_timezone(offset: int)`: 设置时区偏移。
*   `use_simulated_execution()` / `use_realtime_execution()`: 设置执行环境。
*   `set_execution_mode(mode)`: 设置撮合模式。
*   `set_history_depth(depth)`: 设置历史数据缓存长度。

**市场与费率配置:**

*   `use_simple_market()`: 启用简单市场。
*   `use_china_market()`: 启用中国市场。
*   `set_stock_fee_rules(commission, stamp_tax, transfer_fee, min_commission)`: 设置费率。

## 4. 交易对象 (Trading Objects)

### `akquant.Order`

*   `id`: 订单 ID。
*   `symbol`: 标的代码。
*   `side`: `OrderSide.Buy` / `OrderSide.Sell`。
*   `order_type`: `OrderType.Market` / `OrderType.Limit` 等。
*   `status`: `OrderStatus.New` / `Filled` / `Cancelled` 等。
*   `quantity` / `filled_quantity`: 委托/成交数量。
*   `average_filled_price`: 成交均价。

### `akquant.Instrument`

合约定义。

```python
Instrument(
    symbol="AAPL",
    asset_type=AssetType.Stock,
    multiplier=1.0,
    margin_ratio=1.0,
    tick_size=0.01,
    option_type=None,
    strike_price=None,
    expiry_date=None
)
```

## 5. 投资组合与风控 (Portfolio & Risk)

### `akquant.RiskConfig`

风控配置。

```python
@dataclass
class RiskConfig:
    active: bool = True
    safety_margin: float = 0.0001
    max_order_size: Optional[float] = None
    max_order_value: Optional[float] = None
    max_position_size: Optional[float] = None
    restricted_list: Optional[List[str]] = None
```

## 6. 结果分析 (Analysis)

### `akquant.BacktestResult`

回测结果对象。

**属性:**

*   `metrics_df`: 绩效指标表格。
*   `trades_df`: 交易记录表格。
*   `orders_df`: 委托记录表格。
*   `positions_df`: 每日持仓详情。
*   `equity_curve`: 权益曲线。
*   `cash_curve`: 现金曲线。
