use crate::error::AkQuantError;
use crate::model::{Instrument, Order};
use crate::portfolio::Portfolio;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::fmt::Debug;

use super::RiskConfig;

/// Trait for risk check rules
pub trait RiskRule: Send + Sync + Debug {
    /// Check if the order passes the risk rule
    fn check(
        &self,
        order: &Order,
        portfolio: &Portfolio,
        instrument: &Instrument,
        instruments: &HashMap<String, Instrument>,
        active_orders: &[Order],
        current_prices: &HashMap<String, Decimal>,
        config: &RiskConfig,
    ) -> Result<(), AkQuantError>;

    /// Get the name of the rule
    fn name(&self) -> &'static str;

    /// Clone the rule
    fn clone_box(&self) -> Box<dyn RiskRule>;
}

impl Clone for Box<dyn RiskRule> {
    fn clone(&self) -> Box<dyn RiskRule> {
        self.clone_box()
    }
}
