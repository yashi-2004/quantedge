# QuantEdge Mathematical Model Documentation

## 1. Econometric Models

### 1.1 ARIMA(p, d, q) Process
An Autoregressive Integrated Moving Average process on a differenced series $w_t = \Delta^d y_t$ is specified as:

$$w_t = c + \sum_{i=1}^p \phi_i w_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t, \quad \epsilon_t \sim \text{WN}(0, \sigma^2)$$

- **Stationarity Condition**: Roots of the autoregressive polynomial $1 - \sum \phi_i z^i = 0$ must lie strictly outside the complex unit circle ($|z| > 1$).
- **Order Selection**: Selected by minimizing the Akaike Information Criterion:
  $$\text{AIC} = 2k - 2\ln(\hat{L})$$

### 1.2 GARCH(1,1) Volatility Process (Bollerslev 1986)
Financial return innovations $\epsilon_t = r_t - \mu$ with conditional variance $\sigma_t^2$:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

- **Persistence Condition**: $\alpha + \beta < 1$. High persistence ($\approx 0.95 - 0.99$) reflects long memory of volatility shocks.
- **Shock Half-Life**:
  $$\tau_{1/2} = \frac{\ln(0.5)}{\ln(\alpha + \beta)}$$
- **Unconditional Long-Run Volatility**:
  $$\sigma_L = \sqrt{\frac{\omega}{1 - \alpha - \beta} \cdot 252}$$

### 1.3 Vector Autoregression (VAR)
A multivariate system of $k$ endogenous financial series:

$$Y_t = c + \sum_{i=1}^p A_i Y_{t-i} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \Sigma_\epsilon)$$

- **Stability**: Roots of $\det(I_k - A_1 z - \dots - A_p z^p) = 0$ must lie outside unit circle.
- **Orthogonalized IRF**: Cholesky factorization $\Sigma_\epsilon = P P^T$ yields uncorrelated structural innovations.

### 1.4 Kalman Filter (Linear Gaussian State Space)
- **Observation Equation**: $y_t = H_t \theta_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$
- **State Transition Equation**: $\theta_t = F \theta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$
- **Predict Step**:
  $$\hat{\theta}_{t|t-1} = F \hat{\theta}_{t-1|t-1}, \quad P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$
- **Update Step**:
  $$K_t = P_{t|t-1} H_t^T (H_t P_{t|t-1} H_t^T + R)^{-1}$$
  $$\hat{\theta}_{t|t} = \hat{\theta}_{t|t-1} + K_t (y_t - H_t \hat{\theta}_{t|t-1})$$
  $$P_{t|t} = (I - K_t H_t) P_{t|t-1}$$

---

## 2. Classical Machine Learning Models

1. **Ridge Regression**: Minimizes $\|y - Xw\|_2^2 + \alpha \|w\|_2^2$. Prevents multicollinearity instability across correlated indicators.
2. **Lasso Regression**: Minimizes $\|y - Xw\|_2^2 + \alpha \|w\|_1$. Enforces sparsity for automatic feature selection.
3. **ElasticNet**: Convex combination of $L_1$ and $L_2$ penalties, grouping correlated predictors.
4. **Random Forest & Gradient Boosting**: Ensembles of decision trees capturing non-linear threshold effects.
5. **XGBoost / HistGradientBoosting**: Optimized second-order Taylor expansion tree-boosting with explicit depth and regularization control.

---

## 3. Deep Learning Architectures (PyTorch CPU)

1. **LSTM (Long Short-Term Memory)**: Retains long-term state via input, forget, and output gating mechanisms:
   $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
   $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
   $$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
   $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o), \quad h_t = o_t \odot \tanh(c_t)$$
2. **TCN (Temporal Convolutional Network)**: Receptive field dilated exponentially ($d \in \{1, 2, 4\}$) with strictly causal padding ($padding = (k-1) \cdot d$) preventing future leakage.
3. **TimeSeriesTransformer**: Multi-head self-attention with fixed sinusoidal positional encoding:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

---

## 4. Portfolio Optimization Frameworks

### 4.1 Markowitz Mean-Variance
$$\max_w \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}} \quad \text{s.t.} \quad \sum w_i = 1, \quad 0 \le w_i \le w_{\max}$$

### 4.2 Risk Parity (Equal Risk Contribution)
Each asset contributes equally to total portfolio volatility $\sigma_p$:
$$\text{RC}_i = w_i \frac{(\Sigma w)_i}{\sigma_p} = \frac{1}{N} \sigma_p$$
Solved via SLSQP minimizing $\sum_{i,j} \left( w_i (\Sigma w)_i - w_j (\Sigma w)_j \right)^2$.

### 4.3 Black-Litterman Model
Combines reverse-engineered market equilibrium returns $\Pi = \lambda \Sigma w_{\text{mkt}}$ with views $P E[R] = Q + \epsilon$:
$$E[R] = \left[ (\tau \Sigma)^{-1} + P^T \Omega^{-1} P \right]^{-1} \left[ (\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q \right]$$

### 4.4 Rockafellar-Uryasev CVaR Linear Program
$$\min_{w, \gamma, u} \gamma + \frac{1}{T(1-\alpha)} \sum_{t=1}^T u_t \quad \text{s.t.} \quad u_t \ge -r_t^T w - \gamma, \quad u_t \ge 0, \quad \sum w_i = 1$$
