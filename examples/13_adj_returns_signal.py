import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, Strategy

dates = pd.date_range("2025-01-01", periods=60, freq="D", tz="Asia/Shanghai")
close = np.linspace(10, 12, len(dates)) + np.sin(np.linspace(0, 6, len(dates)))
factor = np.ones(len(dates))
factor[30:] = 0.5
adj_close = close * factor
high = close * 1.01
low = close * 0.99
open_ = close * 0.995
vol = np.random.randint(1000, 2000, size=len(dates))
df = pd.DataFrame(
    {
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
        "adj_close": adj_close,
    }
)
df["symbol"] = "TEST"


class AdjSignal(Strategy):
    """
    使用后复权价格计算信号的策略示例.

    Demonstrate using adjusted close for signals while executing with real prices.
    """

    warmup_period = 5

    def on_bar(self, bar: Bar) -> None:
        """
        K线闭合回调.

        :param bar: 当前K线数据
        """
        x = self.get_history(2, bar.symbol, "adj_close")
        if x is None or len(x) < 2:
            return
        r = x[-1] / x[-2] - 1.0
        pos = self.get_position(bar.symbol)
        if pos == 0 and r > 0:
            self.buy(bar.symbol, 100)
        elif pos > 0 and r < 0:
            self.close_position(bar.symbol)


result = aq.run_backtest(
    data=df,
    strategy=AdjSignal,
    initial_cash=100000.0,
    symbol="TEST",
)
print(result)
