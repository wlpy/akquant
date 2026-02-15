use crate::event::Event;
use crate::execution::ExecutionClient;
use crate::model::{Order, TradingSession};

/// 实盘执行器 (Realtime Execution Client)
/// 对接外部交易接口 (如 CTP/Broker API)
pub struct RealtimeExecutionClient;

impl RealtimeExecutionClient {
    pub fn new() -> Self {
        RealtimeExecutionClient
    }
}

impl Default for RealtimeExecutionClient {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionClient for RealtimeExecutionClient {
    fn is_live(&self) -> bool {
        true
    }

    fn on_order(&mut self, _order: Order) {
        // In real impl, send to broker API
    }

    fn on_cancel(&mut self, _order_id: &str) {
        // In real impl, send cancel to broker API
    }

    fn on_event(
        &mut self,
        _event: &Event,
        _instruments: &std::collections::HashMap<String, crate::model::Instrument>,
        _portfolio: &crate::portfolio::Portfolio,
        _last_prices: &std::collections::HashMap<String, rust_decimal::Decimal>,
        // _risk_manager: &crate::risk::RiskManager,
        _market_model: &dyn crate::market::MarketModel,
        _execution_mode: crate::model::ExecutionMode,
        _bar_index: usize,
        _session: TradingSession,
    ) -> Vec<Event> {
        // In realtime, this might check for interaction with broker
        Vec::new()
    }
}
