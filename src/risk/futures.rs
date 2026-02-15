use crate::error::AkQuantError;
use crate::model::{Instrument, Order};
use crate::portfolio::Portfolio;
use rust_decimal::Decimal;
use std::collections::HashMap;

use super::rule::RiskRule;
use super::RiskConfig;

/// Placeholder for specific futures margin rule
/// Currently generic CashMarginRule covers basic margin checks.
#[derive(Debug, Clone)]
pub struct FuturesMarginRule;

impl RiskRule for FuturesMarginRule {
    fn name(&self) -> &'static str {
        "FuturesMarginRule"
    }

    fn check(
        &self,
        _order: &Order,
        _portfolio: &Portfolio,
        _instrument: &Instrument,
        _instruments: &HashMap<String, Instrument>,
        _active_orders: &[Order],
        _current_prices: &HashMap<String, Decimal>,
        _config: &RiskConfig,
    ) -> Result<(), AkQuantError> {
        // TODO: Implement advanced futures margin logic (e.g. maintenance margin vs initial margin)
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn RiskRule> {
        Box::new(self.clone())
    }
}
