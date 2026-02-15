use crate::event::Event;
use crate::execution::slippage::SlippageModel;
use crate::model::{ExecutionMode, Instrument, Order};
use rust_decimal::Decimal;

/// 撮合器接口
pub trait ExecutionMatcher: Send + Sync {
    fn match_order(
        &self,
        order: &mut Order,
        event: &Event,
        instrument: &Instrument,
        execution_mode: ExecutionMode,
        slippage: &dyn SlippageModel,
        volume_limit_pct: Decimal,
        bar_index: usize,
    ) -> Option<Event>;
}
