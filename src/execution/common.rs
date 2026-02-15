use crate::event::Event;
use crate::execution::slippage::SlippageModel;
use crate::model::{
    ExecutionMode, Instrument, Order, OrderSide, OrderStatus, OrderType, TimeInForce, Trade,
};
use rust_decimal::Decimal;
use uuid::Uuid;

/// 通用撮合逻辑
pub struct CommonMatcher;

impl CommonMatcher {
    pub fn match_order(
        order: &mut Order,
        event: &Event,
        instrument: &Instrument,
        execution_mode: ExecutionMode,
        slippage: &dyn SlippageModel,
        volume_limit_pct: Decimal,
        bar_index: usize,
    ) -> Option<Event> {
        // 0. 检查最小交易单位 (Lot Size)
        // 仅针对买入订单，且必须存在标的定义
        if order.side == OrderSide::Buy {
            if order.quantity % instrument.lot_size() != Decimal::ZERO {
                order.status = OrderStatus::Rejected;
                order.reject_reason = format!(
                    "Quantity {} is not a multiple of lot size {}",
                    order.quantity,
                    instrument.lot_size()
                );
                match event {
                    Event::Bar(b) => order.updated_at = b.timestamp,
                    Event::Tick(t) => order.updated_at = t.timestamp,
                    _ => {}
                }
                return Some(Event::ExecutionReport(order.clone(), None));
            }
        }

        match event {
            Event::Bar(bar) => {
                if order.symbol != bar.symbol {
                    return None;
                }

                // 1. 检查是否触发止损/止盈
                if let Some(trigger_price) = order.trigger_price {
                    let triggered = match order.side {
                        OrderSide::Buy => bar.high >= trigger_price, // 价格突破触发
                        OrderSide::Sell => bar.low <= trigger_price, // 价格跌破触发
                    };

                    if !triggered {
                        return None; // 未触发，跳过
                    }

                    // 触发后，清除 trigger_price，并根据类型转换为市价或限价单
                    order.trigger_price = None;
                    match order.order_type {
                        OrderType::StopMarket => order.order_type = OrderType::Market,
                        OrderType::StopLimit => order.order_type = OrderType::Limit,
                        _ => {}
                    }
                }

                // 2. 撮合逻辑
                let mut execute_price: Option<Decimal> = None;

                match order.order_type {
                    OrderType::Market | OrderType::StopMarket => {
                        // 市价单
                        execute_price = match execution_mode {
                            ExecutionMode::NextOpen => Some(bar.open),
                            ExecutionMode::CurrentClose => Some(bar.close),
                            ExecutionMode::NextAverage => {
                                Some((bar.open + bar.high + bar.low + bar.close) / Decimal::from(4))
                            }
                            ExecutionMode::NextHighLowMid => {
                                Some((bar.high + bar.low) / Decimal::from(2))
                            }
                        };
                    }
                    OrderType::Limit | OrderType::StopLimit => {
                        // 限价单
                        if let Some(limit_price) = order.price {
                            let avg_price =
                                (bar.open + bar.high + bar.low + bar.close) / Decimal::from(4);
                            let mid_price = (bar.high + bar.low) / Decimal::from(2);
                            match order.side {
                                OrderSide::Buy => {
                                    // 买单：最低价 <= 限价
                                    if bar.low <= limit_price {
                                        match execution_mode {
                                            ExecutionMode::NextAverage => {
                                                execute_price = Some(limit_price.min(avg_price));
                                            }
                                            ExecutionMode::NextHighLowMid => {
                                                execute_price = Some(limit_price.min(mid_price));
                                            }
                                            _ => {
                                                execute_price = Some(limit_price.min(bar.open));
                                                if execute_price.unwrap() > limit_price {
                                                    execute_price = Some(limit_price);
                                                }
                                            }
                                        }
                                    }
                                }
                                OrderSide::Sell => {
                                    // 卖单：最高价 >= 限价
                                    if bar.high >= limit_price {
                                        match execution_mode {
                                            ExecutionMode::NextAverage => {
                                                execute_price = Some(limit_price.max(avg_price));
                                            }
                                            ExecutionMode::NextHighLowMid => {
                                                execute_price = Some(limit_price.max(mid_price));
                                            }
                                            _ => {
                                                execute_price = Some(limit_price.max(bar.open));
                                                if execute_price.unwrap() < limit_price {
                                                    execute_price = Some(limit_price);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some(price) = execute_price {
                    // Apply Slippage
                    let final_price = slippage.calculate_price(price, order.quantity, order.side);

                    // Check Volume Limit
                    let max_qty = if volume_limit_pct > Decimal::ZERO {
                        bar.volume * volume_limit_pct
                    } else {
                        Decimal::MAX
                    };

                    let trade_qty = (order.quantity - order.filled_quantity).min(max_qty);

                    if trade_qty > Decimal::ZERO {
                        order.status = OrderStatus::Filled;
                        order.updated_at = bar.timestamp;
                        // Check if partial fill
                        if trade_qty < order.quantity - order.filled_quantity {
                            order.status = OrderStatus::Submitted;
                        }

                        order.filled_quantity += trade_qty;

                        // Update weighted average price
                        let current_filled = order.filled_quantity;
                        let prev_filled = current_filled - trade_qty;
                        let prev_avg = order.average_filled_price.unwrap_or(Decimal::ZERO);
                        let new_avg =
                            (prev_avg * prev_filled + final_price * trade_qty) / current_filled;
                        order.average_filled_price = Some(new_avg);

                        let trade = Trade {
                            id: Uuid::new_v4().to_string(),
                            order_id: order.id.clone(),
                            symbol: order.symbol.clone(),
                            side: order.side,
                            quantity: trade_qty,
                            price: final_price,
                            commission: Decimal::ZERO,
                            timestamp: bar.timestamp,
                            bar_index,
                        };
                        return Some(Event::ExecutionReport(order.clone(), Some(trade)));
                    }
                } else if order.time_in_force == TimeInForce::IOC
                    || order.time_in_force == TimeInForce::FOK
                {
                    // IOC/FOK 未能立即成交则取消
                    order.status = OrderStatus::Cancelled;
                    order.updated_at = bar.timestamp;
                    return Some(Event::ExecutionReport(order.clone(), None));
                }
            }
            Event::Tick(tick) => {
                if order.symbol != tick.symbol {
                    return None;
                }

                // 1. 检查是否触发止损/止盈
                if let Some(trigger_price) = order.trigger_price {
                    let triggered = match order.side {
                        OrderSide::Buy => tick.price >= trigger_price,
                        OrderSide::Sell => tick.price <= trigger_price,
                    };
                    if !triggered {
                        return None;
                    }

                    order.trigger_price = None;
                    match order.order_type {
                        OrderType::StopMarket => order.order_type = OrderType::Market,
                        OrderType::StopLimit => order.order_type = OrderType::Limit,
                        _ => {}
                    }
                }

                let mut execute_price: Option<Decimal> = None;
                match order.order_type {
                    OrderType::Market | OrderType::StopMarket => {
                        execute_price = Some(tick.price);
                    }
                    OrderType::Limit | OrderType::StopLimit => {
                        if let Some(limit_price) = order.price {
                            match order.side {
                                OrderSide::Buy => {
                                    if tick.price <= limit_price {
                                        execute_price = Some(limit_price);
                                    }
                                }
                                OrderSide::Sell => {
                                    if tick.price >= limit_price {
                                        execute_price = Some(limit_price);
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some(price) = execute_price {
                    // Apply Slippage
                    let final_price = slippage.calculate_price(price, order.quantity, order.side);

                    // Check Volume Limit
                    let max_qty = if volume_limit_pct > Decimal::ZERO {
                        tick.volume * volume_limit_pct
                    } else {
                        Decimal::MAX
                    };

                    let trade_qty = (order.quantity - order.filled_quantity).min(max_qty);

                    if trade_qty > Decimal::ZERO {
                        order.status = OrderStatus::Filled;
                        order.updated_at = tick.timestamp;
                        order.filled_quantity += trade_qty;

                        if order.filled_quantity < order.quantity {
                            order.status = OrderStatus::Submitted;
                        }

                        let current_filled = order.filled_quantity;
                        let prev_filled = current_filled - trade_qty;
                        let prev_avg = order.average_filled_price.unwrap_or(Decimal::ZERO);
                        let new_avg =
                            (prev_avg * prev_filled + final_price * trade_qty) / current_filled;
                        order.average_filled_price = Some(new_avg);

                        let trade = Trade {
                            id: Uuid::new_v4().to_string(),
                            order_id: order.id.clone(),
                            symbol: order.symbol.clone(),
                            side: order.side,
                            quantity: trade_qty,
                            price: final_price,
                            commission: Decimal::ZERO,
                            timestamp: tick.timestamp,
                            bar_index,
                        };
                        return Some(Event::ExecutionReport(order.clone(), Some(trade)));
                    }
                } else if order.time_in_force == TimeInForce::IOC
                    || order.time_in_force == TimeInForce::FOK
                {
                    order.status = OrderStatus::Cancelled;
                    order.updated_at = tick.timestamp;
                    return Some(Event::ExecutionReport(order.clone(), None));
                }
            }
            _ => {}
        }
        None
    }
}
