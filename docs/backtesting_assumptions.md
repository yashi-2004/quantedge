# QuantEdge Backtesting Assumptions & Anti-Leakage Safeguards

## 1. Microstructure & Execution Model

A persistent source of erroneous results in quantitative research is theoretical backtesting that assumes instantaneous, frictionless execution at current closing prices. QuantEdge enforces realistic microstructure mechanics:

### 1.1 Signal-to-Execution Delay
- **Signal Timestamp ($t$)**: Portfolio weights $W_t^*$ are generated using data available through market close of session $t$.
- **Execution Timestamp ($t+1$)**: Orders are simulated to execute at session $t+1$ (execution lag = 1) or $t+2$.
- **Anti-Leakage Guarantee**: A strategy cannot achieve returns using prices that occurred prior to or simultaneous with the signal decision.

### 1.2 Proportional Transaction Cost & Slippage Model
When rebalancing from held drifted weights $w_{t-1}^+$ to target weights $w_t^*$, turnover is calculated as:

$$\Delta w_t = \sum_{i=1}^N |w_{t, i}^* - w_{t-1, i}^+|$$

Total friction is debited directly from portfolio capital prior to compounding returns:

$$\text{Cost}_t = V_{t-1} \cdot \Delta w_t \cdot \left( \frac{\text{Fee}_{\text{bps}} + \text{Slippage}_{\text{bps}}}{10,000} \right)$$

- Default Fee: 10 bps (0.0010) per trade.
- Default Slippage: 5 bps (0.0005) linear market impact.
- Total Round-Trip Friction: 15–30 bps.

### 1.3 Weight Drift Dynamics
Between rebalancing intervals, asset weights naturally drift with market price changes:

$$w_{t, i}^+ = \frac{w_{t, i}^* (1 + R_{t, i})}{\sum_{j=1}^N w_{t, j}^* (1 + R_{t, j})}$$

QuantEdge tracks exact drifted weights, preventing synthetic zero-cost rebalancing.

---

## 2. Performance Metrics Equations

1. **CAGR (Compound Annual Growth Rate)**:
   $$\text{CAGR} = \left( \frac{V_T}{V_0} \right)^{\frac{252}{N}} - 1$$
2. **Sharpe Ratio**:
   $$\text{Sharpe} = \frac{\text{CAGR} - r_f}{\sigma_{\text{ann}}}$$
3. **Sortino Ratio**:
   $$\text{Sortino} = \frac{\text{CAGR} - r_f}{\sqrt{252} \cdot \sqrt{\frac{1}{N}\sum \min(0, r_t - r_f/252)^2}}$$
4. **Maximum Drawdown**:
   $$\text{MDD} = \min_{t \in [0, T]} \left( \frac{V_t - \max_{s \le t} V_s}{\max_{s \le t} V_s} \right)$$
5. **Calmar Ratio**:
   $$\text{Calmar} = \frac{\text{CAGR}}{|\text{MDD}|}$$
6. **Information Ratio**:
   $$\text{IR} = \frac{\text{CAGR}_{\text{strategy}} - \text{CAGR}_{\text{benchmark}}}{\sqrt{252} \cdot \text{std}(R_{\text{strat}} - R_{\text{bench}})}$$
