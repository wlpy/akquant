use super::market_data::extract_decimal;
use super::types::{AssetType, OptionType, SettlementType};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockInstrument {
    pub symbol: String,
    pub lot_size: Decimal,
    pub tick_size: Decimal,
    // Add other stock-specific fields if needed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundInstrument {
    pub symbol: String,
    pub lot_size: Decimal,
    pub tick_size: Decimal,
    // Add other fund-specific fields if needed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuturesInstrument {
    pub symbol: String,
    pub multiplier: Decimal,
    pub margin_ratio: Decimal,
    pub tick_size: Decimal,
    pub expiry_date: Option<u32>,
    pub settlement_type: Option<SettlementType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionInstrument {
    pub symbol: String,
    pub multiplier: Decimal,
    pub tick_size: Decimal,
    pub option_type: OptionType,
    pub strike_price: Decimal,
    pub expiry_date: u32,
    pub underlying_symbol: String,
    pub settlement_type: Option<SettlementType>,
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// 交易标的
///
/// :ivar symbol: 代码
/// :ivar asset_type: 资产类型
/// :ivar multiplier: 合约乘数
/// :ivar margin_ratio: 保证金比率
/// :ivar tick_size: 最小变动价位
pub struct Instrument {
    #[pyo3(get)]
    pub asset_type: AssetType,
    pub inner: InstrumentEnum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstrumentEnum {
    Stock(StockInstrument),
    Fund(FundInstrument),
    Futures(FuturesInstrument),
    Option(OptionInstrument),
}

#[gen_stub_pymethods]
#[pymethods]
impl Instrument {
    /// 创建交易标的
    ///
    /// :param symbol: 代码
    /// :param asset_type: 资产类型
    /// :param multiplier: 合约乘数
    /// :param margin_ratio: 保证金比率
    /// :param tick_size: 最小变动价位
    /// :param option_type: 期权类型 (可选)
    /// :param strike_price: 行权价 (可选)
    /// :param expiry_date: 到期日 (可选)
    /// :param lot_size: 最小交易单位 (可选, 默认为1)
    /// :param underlying_symbol: 标的代码 (可选)
    /// :param settlement_type: 结算方式 (可选)
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (symbol, asset_type, multiplier=None, margin_ratio=None, tick_size=None, option_type=None, strike_price=None, expiry_date=None, lot_size=None, underlying_symbol=None, settlement_type=None))]
    pub fn new(
        symbol: String,
        asset_type: AssetType,
        multiplier: Option<&Bound<'_, PyAny>>,
        margin_ratio: Option<&Bound<'_, PyAny>>,
        tick_size: Option<&Bound<'_, PyAny>>,
        option_type: Option<OptionType>,
        strike_price: Option<&Bound<'_, PyAny>>,
        expiry_date: Option<u32>,
        lot_size: Option<&Bound<'_, PyAny>>,
        underlying_symbol: Option<String>,
        settlement_type: Option<SettlementType>,
    ) -> PyResult<Self> {
        let mult = if let Some(m) = multiplier {
            extract_decimal(m)?
        } else {
            Decimal::from(1)
        };

        let margin = if let Some(m) = margin_ratio {
            extract_decimal(m)?
        } else {
            Decimal::from(1) // Default 100% margin (no leverage)
        };

        let tick = if let Some(t) = tick_size {
            extract_decimal(t)?
        } else {
            Decimal::new(1, 2) // Default 0.01
        };

        let strike = if let Some(s) = strike_price {
            Some(extract_decimal(s)?)
        } else {
            None
        };

        let lot = if let Some(l) = lot_size {
            extract_decimal(l)?
        } else {
            Decimal::from(1)
        };

        let inner = match asset_type {
            AssetType::Stock => InstrumentEnum::Stock(StockInstrument {
                symbol: symbol.clone(),
                lot_size: lot,
                tick_size: tick,
            }),
            AssetType::Fund => InstrumentEnum::Fund(FundInstrument {
                symbol: symbol.clone(),
                lot_size: lot,
                tick_size: tick,
            }),
            AssetType::Futures => InstrumentEnum::Futures(FuturesInstrument {
                symbol: symbol.clone(),
                multiplier: mult,
                margin_ratio: margin,
                tick_size: tick,
                expiry_date,
                settlement_type,
            }),
            AssetType::Option => {
                // Ensure required fields for Option are present
                let opt_type = option_type.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Option type required for Option")
                })?;
                let strike_val = strike.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Strike price required for Option")
                })?;
                let expiry_val = expiry_date.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Expiry date required for Option")
                })?;
                let underlying_val = underlying_symbol.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Underlying symbol required for Option")
                })?;

                InstrumentEnum::Option(OptionInstrument {
                    symbol: symbol.clone(),
                    multiplier: mult,
                    tick_size: tick,
                    option_type: opt_type,
                    strike_price: strike_val,
                    expiry_date: expiry_val,
                    underlying_symbol: underlying_val,
                    settlement_type,
                })
            }
        };

        Ok(Instrument { asset_type, inner })
    }

    #[getter]
    fn get_symbol(&self) -> String {
        match &self.inner {
            InstrumentEnum::Stock(s) => s.symbol.clone(),
            InstrumentEnum::Fund(f) => f.symbol.clone(),
            InstrumentEnum::Futures(f) => f.symbol.clone(),
            InstrumentEnum::Option(o) => o.symbol.clone(),
        }
    }

    #[getter]
    /// 获取合约乘数.
    /// :return: 合约乘数
    fn get_multiplier(&self) -> f64 {
        match &self.inner {
            InstrumentEnum::Futures(f) => f.multiplier.to_f64().unwrap_or_default(),
            InstrumentEnum::Option(o) => o.multiplier.to_f64().unwrap_or_default(),
            _ => 1.0,
        }
    }

    #[getter]
    /// 获取保证金比率.
    /// :return: 保证金比率
    fn get_margin_ratio(&self) -> f64 {
        match &self.inner {
            InstrumentEnum::Futures(f) => f.margin_ratio.to_f64().unwrap_or_default(),
            _ => 1.0,
        }
    }

    #[getter]
    /// 获取最小变动价位.
    /// :return: 最小变动价位
    fn get_tick_size(&self) -> f64 {
        match &self.inner {
            InstrumentEnum::Stock(s) => s.tick_size.to_f64().unwrap_or_default(),
            InstrumentEnum::Fund(f) => f.tick_size.to_f64().unwrap_or_default(),
            InstrumentEnum::Futures(f) => f.tick_size.to_f64().unwrap_or_default(),
            InstrumentEnum::Option(o) => o.tick_size.to_f64().unwrap_or_default(),
        }
    }

    #[getter]
    /// 获取最小交易单位.
    /// :return: 最小交易单位
    fn get_lot_size(&self) -> f64 {
        match &self.inner {
            InstrumentEnum::Stock(s) => s.lot_size.to_f64().unwrap_or_default(),
            InstrumentEnum::Fund(f) => f.lot_size.to_f64().unwrap_or_default(),
            InstrumentEnum::Futures(_) => 1.0,
            InstrumentEnum::Option(_) => 1.0,
        }
    }

    #[setter]
    /// 设置最小交易单位.
    /// :param value: 最小交易单位
    fn set_lot_size(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let val = extract_decimal(value)?;
        match &mut self.inner {
            InstrumentEnum::Stock(s) => s.lot_size = val,
            InstrumentEnum::Fund(f) => f.lot_size = val,
            _ => {} // Ignore for others or raise error?
        }
        Ok(())
    }
}

// Add public accessors for internal Rust usage to avoid breaking changes everywhere immediately
impl Instrument {
    pub fn symbol(&self) -> &str {
        match &self.inner {
            InstrumentEnum::Stock(s) => &s.symbol,
            InstrumentEnum::Fund(f) => &f.symbol,
            InstrumentEnum::Futures(f) => &f.symbol,
            InstrumentEnum::Option(o) => &o.symbol,
        }
    }

    pub fn multiplier(&self) -> Decimal {
        match &self.inner {
            InstrumentEnum::Futures(f) => f.multiplier,
            InstrumentEnum::Option(o) => o.multiplier,
            _ => Decimal::ONE,
        }
    }

    pub fn margin_ratio(&self) -> Decimal {
        match &self.inner {
            InstrumentEnum::Futures(f) => f.margin_ratio,
            _ => Decimal::ONE,
        }
    }

    pub fn lot_size(&self) -> Decimal {
        match &self.inner {
            InstrumentEnum::Stock(s) => s.lot_size,
            InstrumentEnum::Fund(f) => f.lot_size,
            _ => Decimal::ONE,
        }
    }

    pub fn tick_size(&self) -> Decimal {
        match &self.inner {
            InstrumentEnum::Stock(s) => s.tick_size,
            InstrumentEnum::Fund(f) => f.tick_size,
            InstrumentEnum::Futures(f) => f.tick_size,
            InstrumentEnum::Option(o) => o.tick_size,
        }
    }

    pub fn expiry_date(&self) -> Option<u32> {
        match &self.inner {
            InstrumentEnum::Futures(f) => f.expiry_date,
            InstrumentEnum::Option(o) => Some(o.expiry_date),
            _ => None,
        }
    }

    pub fn underlying_symbol(&self) -> Option<&String> {
        match &self.inner {
            InstrumentEnum::Option(o) => Some(&o.underlying_symbol),
            _ => None,
        }
    }

    pub fn strike_price(&self) -> Option<Decimal> {
        match &self.inner {
            InstrumentEnum::Option(o) => Some(o.strike_price),
            _ => None,
        }
    }

    pub fn option_type(&self) -> Option<OptionType> {
        match &self.inner {
            InstrumentEnum::Option(o) => Some(o.option_type),
            _ => None,
        }
    }
}
