use crate::model::Instrument;
use rust_decimal::Decimal;

/// Trait for calculating margin requirements for a position.
pub trait MarginCalculator: Send + Sync {
    /// Calculate the margin required for a position.
    ///
    /// # Arguments
    /// * `quantity` - The position quantity (positive for long, negative for short).
    /// * `price` - The current price of the instrument.
    /// * `instrument` - The instrument details.
    /// * `underlying_price` - The price of the underlying asset (optional, for derivatives).
    ///
    /// # Returns
    /// The required margin amount (always non-negative).
    fn calculate_margin(
        &self,
        quantity: Decimal,
        price: Decimal,
        instrument: &Instrument,
        underlying_price: Option<Decimal>,
    ) -> Decimal;
}

/// Default calculator for linear instruments (Stocks, Funds).
/// Margin = Quantity * Price * Multiplier * MarginRatio
#[derive(Debug, Clone, Copy, Default)]
pub struct LinearMarginCalculator;

impl MarginCalculator for LinearMarginCalculator {
    fn calculate_margin(
        &self,
        quantity: Decimal,
        price: Decimal,
        instrument: &Instrument,
        _underlying_price: Option<Decimal>,
    ) -> Decimal {
        let multiplier = instrument.multiplier();
        let margin_ratio = instrument.margin_ratio();
        (quantity * price * multiplier).abs() * margin_ratio
    }
}

/// Calculator for Futures.
/// Same as Linear, but conceptually distinct for potential future complexity (e.g. spread margin).
#[derive(Debug, Clone, Copy, Default)]
pub struct FuturesMarginCalculator;

impl MarginCalculator for FuturesMarginCalculator {
    fn calculate_margin(
        &self,
        quantity: Decimal,
        price: Decimal,
        instrument: &Instrument,
        _underlying_price: Option<Decimal>,
    ) -> Decimal {
        let multiplier = instrument.multiplier();
        let margin_ratio = instrument.margin_ratio();
        (quantity * price * multiplier).abs() * margin_ratio
    }
}

/// Calculator for Options.
/// Handles Long (usually 0 margin) and Short (complex margin) positions.
#[derive(Debug, Clone, Copy, Default)]
pub struct OptionMarginCalculator;

impl MarginCalculator for OptionMarginCalculator {
    fn calculate_margin(
        &self,
        quantity: Decimal,
        price: Decimal,
        instrument: &Instrument,
        underlying_price: Option<Decimal>,
    ) -> Decimal {
        use rust_decimal::prelude::*;

        if quantity > Decimal::ZERO {
            // Long Option: No maintenance margin (premium paid upfront)
            // But some brokers might require full value if not paid?
            // Standard practice is 0 for maintenance margin if fully paid.
            Decimal::ZERO
        } else {
            // Short Option: Complex margin
            // Simplified Model: (OptionPrice + UnderlyingPrice * MarginRatio) * Multiplier * Abs(Qty)
            // If underlying price not available, fallback to OptionPrice * (1 + MarginRatio)
            let abs_qty = quantity.abs();
            let underlying_price = underlying_price.unwrap_or(Decimal::ZERO);
            let multiplier = instrument.multiplier();
            let margin_ratio = instrument.margin_ratio();

            let margin_per_unit = if underlying_price > Decimal::ZERO {
                price + (underlying_price * margin_ratio)
            } else {
                // Fallback: assume margin ratio applies to option value itself (e.g. 100% + extra)
                price * (Decimal::ONE + margin_ratio)
            };

            margin_per_unit * multiplier * abs_qty
        }
    }
}
