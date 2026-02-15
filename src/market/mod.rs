pub mod fund;
pub mod futures;
pub mod manager;
pub mod option;
pub mod stock;

use crate::model::{AssetType, Instrument, OrderSide, TradingSession};
use chrono::NaiveTime;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum MarketConfig {
    China(ChinaMarketConfig),
    Simple(SimpleMarketConfig),
}

impl Default for MarketConfig {
    fn default() -> Self {
        MarketConfig::China(ChinaMarketConfig::default())
    }
}

impl MarketConfig {
    pub fn create_model(&self) -> Box<dyn MarketModel> {
        match self {
            MarketConfig::China(c) => Box::new(ChinaMarket::from_config(c.clone())),
            MarketConfig::Simple(c) => Box::new(SimpleMarket::from_config(c.clone())),
        }
    }
}

/// 简单市场配置
#[derive(Clone, Debug)]
pub struct SimpleMarketConfig {
    pub commission_rate: Decimal,
    pub stamp_tax: Decimal,
    pub transfer_fee: Decimal,
    pub min_commission: Decimal,
}

impl Default for SimpleMarketConfig {
    fn default() -> Self {
        Self {
            commission_rate: Decimal::from_str("0.0003").unwrap(),
            stamp_tax: Decimal::ZERO,
            transfer_fee: Decimal::ZERO,
            min_commission: Decimal::ZERO,
        }
    }
}

/// 简单市场模型 (如加密货币/外汇)
pub struct SimpleMarket {
    pub config: SimpleMarketConfig,
}

impl SimpleMarket {
    pub fn from_config(config: SimpleMarketConfig) -> Self {
        Self { config }
    }
}

impl MarketModel for SimpleMarket {
    fn get_session_status(&self, _time: NaiveTime) -> TradingSession {
        TradingSession::Continuous
    }

    fn calculate_commission(
        &self,
        instrument: &Instrument,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
    ) -> Decimal {
        let turnover = price * quantity * instrument.multiplier();
        let mut commission = turnover * self.config.commission_rate;
        if commission < self.config.min_commission {
            commission = self.config.min_commission;
        }
        let tax = if side == OrderSide::Sell {
            turnover * self.config.stamp_tax
        } else {
            Decimal::ZERO
        };
        let transfer = turnover * self.config.transfer_fee;
        commission + tax + transfer
    }

    fn update_available_position(
        &self,
        available_positions: &mut HashMap<String, Decimal>,
        instrument: &Instrument,
        quantity: Decimal,
        side: OrderSide,
    ) {
        let symbol = instrument.symbol();
        match side {
            OrderSide::Buy => {
                available_positions.entry(symbol.to_string()).or_insert(Decimal::ZERO);
                if let Some(pos) = available_positions.get_mut(symbol) {
                    *pos += quantity;
                }
            }
            OrderSide::Sell => {
                if let Some(pos) = available_positions.get_mut(symbol) {
                    *pos -= quantity;
                }
            }
        }
    }

    fn on_day_close(
        &self,
        _positions: &HashMap<String, Decimal>,
        _available_positions: &mut HashMap<String, Decimal>,
        _instruments: &HashMap<String, Instrument>,
    ) {}
}

#[derive(Clone, Debug)]
pub struct SessionRange {
    pub start: NaiveTime,
    pub end: NaiveTime,
    pub session: TradingSession,
}

#[derive(Clone, Debug)]
pub struct ChinaMarketConfig {
    pub stock: stock::StockConfig,
    pub futures: futures::FuturesConfig,
    pub fund: fund::FundConfig,
    pub option: option::OptionConfig,
    pub sessions: Vec<SessionRange>,
}

fn default_sessions() -> Vec<SessionRange> {
    let t_9_15 = NaiveTime::from_hms_opt(9, 15, 0).unwrap();
    let t_9_25 = NaiveTime::from_hms_opt(9, 25, 0).unwrap();
    let t_9_30 = NaiveTime::from_hms_opt(9, 30, 0).unwrap();
    let t_11_30 = NaiveTime::from_hms_opt(11, 30, 0).unwrap();
    let t_13_00 = NaiveTime::from_hms_opt(13, 0, 0).unwrap();
    let t_14_57 = NaiveTime::from_hms_opt(14, 57, 0).unwrap();
    let t_15_00 = NaiveTime::from_hms_opt(15, 0, 1).unwrap();
    vec![
        SessionRange { start: t_9_15, end: t_9_25, session: TradingSession::CallAuction },
        SessionRange { start: t_9_25, end: t_9_30, session: TradingSession::PreOpen },
        SessionRange { start: t_9_30, end: t_11_30, session: TradingSession::Continuous },
        SessionRange { start: t_11_30, end: t_13_00, session: TradingSession::Break },
        SessionRange { start: t_13_00, end: t_14_57, session: TradingSession::Continuous },
        SessionRange { start: t_14_57, end: t_15_00, session: TradingSession::CallAuction },
    ]
}

impl Default for ChinaMarketConfig {
    fn default() -> Self {
        Self {
            stock: stock::StockConfig::default(),
            futures: futures::FuturesConfig::default(),
            fund: fund::FundConfig::default(),
            option: option::OptionConfig::default(),
            sessions: default_sessions(),
        }
    }
}

pub trait MarketModel: Send + Sync {
    fn calculate_commission(
        &self,
        instrument: &Instrument,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
    ) -> Decimal;

    fn update_available_position(
        &self,
        available_positions: &mut HashMap<String, Decimal>,
        instrument: &Instrument,
        quantity: Decimal,
        side: OrderSide,
    );

    fn on_day_close(
        &self,
        positions: &HashMap<String, Decimal>,
        available_positions: &mut HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
    );

    fn get_session_status(&self, time: NaiveTime) -> TradingSession;
}

pub struct ChinaMarket {
    pub config: ChinaMarketConfig,
}

impl ChinaMarket {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self { config: ChinaMarketConfig::default() }
    }
    pub fn from_config(config: ChinaMarketConfig) -> Self {
        Self { config }
    }
}

impl MarketModel for ChinaMarket {
    fn get_session_status(&self, time: NaiveTime) -> TradingSession {
        for range in &self.config.sessions {
            if time >= range.start && time < range.end {
                return range.session;
            }
        }
        TradingSession::Closed
    }

    fn calculate_commission(
        &self,
        instrument: &Instrument,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
    ) -> Decimal {
        match instrument.asset_type {
            AssetType::Stock => stock::calculate_commission(
                &self.config.stock, instrument, side, price, quantity, instrument.multiplier(),
            ),
            AssetType::Futures => futures::calculate_commission(
                &self.config.futures, instrument, side, price, quantity, instrument.multiplier(),
            ),
            AssetType::Fund => fund::calculate_commission(
                &self.config.fund, instrument, side, price, quantity, instrument.multiplier(),
            ),
            AssetType::Option => option::calculate_commission(
                &self.config.option, instrument, side, price, quantity, instrument.multiplier(),
            ),
        }
    }

    fn update_available_position(
        &self,
        available_positions: &mut HashMap<String, Decimal>,
        instrument: &Instrument,
        quantity: Decimal,
        side: OrderSide,
    ) {
        let symbol = &instrument.symbol();
        match instrument.asset_type {
            AssetType::Stock => {
                stock::update_available_position(
                    &self.config.stock, available_positions, symbol, quantity, side
                );
            }
            AssetType::Fund => {
                fund::update_available_position(
                    &self.config.fund, available_positions, symbol, quantity, side
                );
            }
            AssetType::Futures => {
                futures::update_available_position(
                    &self.config.futures, available_positions, symbol, quantity, side
                );
            }
             AssetType::Option => {
                option::update_available_position(
                    &self.config.option, available_positions, symbol, quantity, side
                );
            }
        }
    }

    fn on_day_close(
        &self,
        positions: &HashMap<String, Decimal>,
        available_positions: &mut HashMap<String, Decimal>,
        instruments: &HashMap<String, Instrument>,
    ) {
        for (symbol, quantity) in positions {
            let is_t_plus_one_asset = if let Some(instr) = instruments.get(symbol) {
                matches!(instr.asset_type, AssetType::Stock | AssetType::Fund)
            } else {
                false
            };

            // 使用各自配置中的 T+1 设置
            let should_settle = if let Some(instr) = instruments.get(symbol) {
                match instr.asset_type {
                    AssetType::Stock => self.config.stock.t_plus_one,
                    AssetType::Fund => self.config.fund.t_plus_one,
                    _ => false,
                }
            } else {
                false
            };

            if is_t_plus_one_asset && should_settle {
                available_positions.insert(symbol.clone(), *quantity);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveTime;

    #[test]
    fn test_china_market_session() {
        let market = ChinaMarket::new();

        // 9:15 -> CallAuction
        assert_eq!(
            market.get_session_status(NaiveTime::from_hms_opt(9, 15, 0).unwrap()),
            TradingSession::CallAuction
        );
        // 9:30 -> Continuous
        assert_eq!(
            market.get_session_status(NaiveTime::from_hms_opt(9, 30, 0).unwrap()),
            TradingSession::Continuous
        );
        // 12:00 -> Break
        assert_eq!(
            market.get_session_status(NaiveTime::from_hms_opt(12, 0, 0).unwrap()),
            TradingSession::Break
        );
        // 18:00 -> Closed
        assert_eq!(
            market.get_session_status(NaiveTime::from_hms_opt(18, 0, 0).unwrap()),
            TradingSession::Closed
        );
    }
}
