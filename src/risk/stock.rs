use crate::error::AkQuantError;
use crate::model::{Instrument, Order, OrderSide};
use crate::portfolio::Portfolio;
use rust_decimal::Decimal;
use std::collections::HashMap;

use super::rule::RiskRule;
use super::RiskConfig;

/// Check available position for selling (T+1 rule usually implied by available_positions)
#[derive(Debug, Clone)]
pub struct StockAvailablePositionRule;

impl RiskRule for StockAvailablePositionRule {
    fn name(&self) -> &'static str {
        "StockAvailablePositionRule"
    }

    fn check(
        &self,
        order: &Order,
        portfolio: &Portfolio,
        _instrument: &Instrument,
        _instruments: &HashMap<String, Instrument>,
        active_orders: &[Order],
        _current_prices: &HashMap<String, Decimal>,
        _config: &RiskConfig,
    ) -> Result<(), AkQuantError> {
        if order.side == OrderSide::Sell {
            let available = portfolio
                .available_positions
                .get(&order.symbol)
                .cloned()
                .unwrap_or(Decimal::ZERO);

            let pending_sell: Decimal = active_orders
                .iter()
                .filter(|o| o.symbol == order.symbol && o.side == OrderSide::Sell)
                .map(|o| o.quantity - o.filled_quantity)
                .sum();

            if available - pending_sell < order.quantity {
                return Err(AkQuantError::OrderError(format!(
                    "Risk: Insufficient available position for {}. Available: {}, Pending Sell: {}, Required: {}",
                    order.symbol, available, pending_sell, order.quantity
                )));
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn RiskRule> {
        Box::new(self.clone())
    }
}
