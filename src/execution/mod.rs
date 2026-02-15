pub mod common;
pub mod futures;
pub mod matcher;
pub mod option;
pub mod realtime;
pub mod simulated;
pub mod slippage;
pub mod stock;

pub use common::CommonMatcher;
pub use matcher::ExecutionMatcher;
pub use realtime::RealtimeExecutionClient;
pub use simulated::SimulatedExecutionClient;
pub use slippage::{SlippageModel, FixedSlippage, PercentSlippage, ZeroSlippage};

use crate::event::Event;
use crate::model::{Order, TradingSession};

/// 交易执行接口 (Execution Client Trait)
pub trait ExecutionClient: Send + Sync {
    /// 接收新订单
    fn on_order(&mut self, order: Order);

    /// 取消订单请求
    fn on_cancel(&mut self, order_id: &str);

    /// 处理市场事件并返回执行报告
    fn on_event(
        &mut self,
        event: &Event,
        instruments: &std::collections::HashMap<String, crate::model::Instrument>,
        portfolio: &crate::portfolio::Portfolio,
        last_prices: &std::collections::HashMap<String, rust_decimal::Decimal>,
        // risk_manager: &crate::risk::RiskManager, // Removed: Risk checks are done before Execution
        market_model: &dyn crate::market::MarketModel,
        execution_mode: crate::model::ExecutionMode,
        bar_index: usize,
        session: TradingSession,
    ) -> Vec<Event>;

    /// 设置滑点模型 (仅回测有效)
    fn set_slippage_model(&mut self, _model: Box<dyn SlippageModel>) {}

    /// 设置成交量限制 (仅回测有效)
    fn set_volume_limit(&mut self, _limit: f64) {}

    /// 是否为实盘模式
    #[allow(dead_code)]
    fn is_live(&self) -> bool {
        false
    }
}
