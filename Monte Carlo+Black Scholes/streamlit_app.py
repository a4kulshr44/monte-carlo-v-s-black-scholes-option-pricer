import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px




def monte_carlo_option_pricing(option_type, S, K, sigma, r, T, num_simulations):
    dt = T / 252  # Assuming 252 trading days in a year
    nudt = (r - 0.5 * sigma**2) * dt
    sidt = sigma * np.sqrt(dt)
    
    Z = np.random.standard_normal((num_simulations, int(252*T)))
    S_T = S * np.exp(np.cumsum(nudt + sidt * Z, axis=1))
    
    if option_type == "Call":
        payoffs = np.maximum(S_T[:, -1] - K, 0)
    else:  # Put option
        payoffs = np.maximum(K - S_T[:, -1], 0)
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price, S_T

# Page configuration
st.set_page_config(
    page_title="Option Pricing Models",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")

# Custom CSS (keep the existing CSS)
st.markdown("""
<style>
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 12px;
    width: auto;
    margin: 0 auto;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.metric-call {
    background-color: #e6f3ff;
    color: #0066cc;
}

.metric-put {
    background-color: #fff0e6;
    color: #cc6600;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: bold;
    margin: 0;
}

.metric-label {
    font-size: 1.2rem;
    margin-bottom: 4px;
}

.stButton > button {
    width: 100%;
}

.sidebar .stButton > button {
    background-color: #4CAF50;
    color: white;
}

</style>
""", unsafe_allow_html=True)

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        d1 = (log(self.current_price / self.strike) + 
              (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (
                  self.volatility * sqrt(self.time_to_maturity)
              )
        d2 = d1 - self.volatility * sqrt(self.time_to_maturity)

        self.call_price = self.current_price * norm.cdf(d1) - (
            self.strike * exp(-(self.interest_rate * self.time_to_maturity)) * norm.cdf(d2)
        )
        self.put_price = (
            self.strike * exp(-(self.interest_rate * self.time_to_maturity)) * norm.cdf(-d2)
        ) - self.current_price * norm.cdf(-d1)

        # Greeks
        self.call_delta = norm.cdf(d1)
        self.put_delta = -norm.cdf(-d1)
        self.gamma = norm.pdf(d1) / (self.current_price * self.volatility * sqrt(self.time_to_maturity))
        self.vega = self.current_price * norm.pdf(d1) * sqrt(self.time_to_maturity)
        self.call_theta = -(self.current_price * norm.pdf(d1) * self.volatility / (2 * sqrt(self.time_to_maturity))) - \
                          self.interest_rate * self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        self.put_theta = -(self.current_price * norm.pdf(d1) * self.volatility / (2 * sqrt(self.time_to_maturity))) + \
                         self.interest_rate * self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)
        self.call_rho = self.strike * self.time_to_maturity * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        self.put_rho = -self.strike * self.time_to_maturity * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)

        return self.call_price, self.put_price

def plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, option_type, purchase_price):
    pnl = np.zeros((len(vol_range), len(spot_range)))
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_model.current_price = spot
            bs_model.volatility = vol
            call_price, put_price = bs_model.calculate_prices()
            if option_type == 'call':
                pnl[i, j] = call_price - purchase_price
            else:
                pnl[i, j] = put_price - purchase_price
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pnl, xticklabels=spot_range.round(2), yticklabels=vol_range.round(2), 
                cmap='RdYlGn', center=0, ax=ax)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Volatility')
    ax.set_title(f'{option_type.capitalize()} Option PnL')
    return fig

# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Option Pricing Models")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/akshat-kulshreshtha-9314421a2/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Akshat Kulshreshtha`</a>', unsafe_allow_html=True)

    model = st.radio("Select Model", ["Black-Scholes", "Monte Carlo"])

    current_price = st.number_input("Current Asset Price", value=100.00, min_value=0.01, step=0.01)
    strike = st.number_input("Strike Price", value=100.00, min_value=0.01, step=0.01)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0, min_value=0.01, step=0.01)
    volatility = st.number_input("Volatility (σ)", value=0.20, min_value=0.01, step=0.01)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05, min_value=0.00, step=0.01)
    
    if model == "Monte Carlo":
        num_simulations = st.number_input("Number of Simulations", value=1000, min_value=100, step=100)

    call_purchase_price = st.number_input("Call Purchase Price", value=0.02, min_value=0.00, step=0.01)
    put_purchase_price = st.number_input("Put Purchase Price", value=0.02, min_value=0.00, step=0.01)

    st.markdown("---")
    st.subheader("Heatmap Parameters")
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=max(current_price*0.8, 0.01), step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=max(current_price*1.2, 0.02), step=0.01)
    vol_min = st.slider('Min Volatility', min_value=0.01, max_value=1.0, value=max(volatility*0.5, 0.01), step=0.01)
    vol_max = st.slider('Max Volatility', min_value=0.01, max_value=1.0, value=min(volatility*1.5, 1.0), step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 20)
    vol_range = np.linspace(vol_min, vol_max, 20)

# Main Page for Output Display
st.title("Option Pricing Models")

# Calculate Call and Put values
if model == "Black-Scholes":
    bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
    call_price, put_price = bs_model.calculate_prices()
else:
    call_price, _ = monte_carlo_option_pricing("Call", current_price, strike, volatility, interest_rate, time_to_maturity, num_simulations)
    put_price, _ = monte_carlo_option_pricing("Put", current_price, strike, volatility, interest_rate, time_to_maturity, num_simulations)

# Display input parameters
st.subheader("Input Parameters")
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
    "Call Purchase Price": [call_purchase_price],
    "Put Purchase Price": [put_purchase_price],
}
if model == "Monte Carlo":
    input_data["Number of Simulations"] = [num_simulations]
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Display Call and Put Values
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

if model == "Black-Scholes":
    # Display Greeks
    st.subheader("Option Greeks")
    greeks_data = {
        "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
        "Call": [bs_model.call_delta, bs_model.gamma, bs_model.vega, bs_model.call_theta, bs_model.call_rho],
        "Put": [bs_model.put_delta, bs_model.gamma, bs_model.vega, bs_model.put_theta, bs_model.put_rho]
    }
    greeks_df = pd.DataFrame(greeks_data)

    # Apply formatting only to numeric columns
    st.table(greeks_df.style.format({
        "Call": "{:.4f}",
        "Put": "{:.4f}"
    }))

    st.title("Options PnL - Interactive Heatmap")
    st.info("Explore how option PnL changes with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, while maintaining a constant 'Strike Price'.")

    # Interactive Heatmaps for Call and Put Options PnL
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Call Option PnL Heatmap")
        heatmap_fig_call = plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, 'call', call_purchase_price)
        st.pyplot(heatmap_fig_call)

    with col2:
        st.subheader("Put Option PnL Heatmap")
        heatmap_fig_put = plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, 'put', put_purchase_price)
        st.pyplot(heatmap_fig_put)

else:
    st.subheader("Monte Carlo Simulation Price Paths")
    _, price_paths_call = monte_carlo_option_pricing("Call", current_price, strike, volatility, interest_rate, time_to_maturity, num_simulations)
    
    # Combine the simulated price paths into a single DataFrame
    df = pd.DataFrame(price_paths_call).T
    # Rename columns
    df.columns = [f"Path {i+1}" for i in range(num_simulations)]

    # Create an interactive plot using Plotly Express
    fig = px.line(df, labels={"index": "Time Steps", "value": "Stock Price"}, title="Simulated Price Paths")
    st.plotly_chart(fig, use_container_width=True)

# Add explanations
st.markdown("""
## Comparative Analysis: Black-Scholes vs. Monte Carlo Simulation

1. **Accuracy**:
   - Black-Scholes: Provides exact solutions for European options under specific assumptions.
   - Monte Carlo: Can be more accurate for complex options or when Black-Scholes assumptions are violated.

2. **Flexibility**:
   - Black-Scholes: Limited to European options with specific assumptions.
   - Monte Carlo: Can handle a wide variety of option types and underlying asset behaviors.

3. **Computation Time**:
   - Black-Scholes: Very fast, provides instant results.
   - Monte Carlo: Can be time-consuming, especially for a large number of simulations.

4. **Assumptions**:
   - Black-Scholes: Assumes constant volatility, no dividends, and log-normal distribution of returns.
   - Monte Carlo: Can incorporate more realistic assumptions like changing volatility or dividend payments.

5. **Visualization**:
   - Black-Scholes: Provides a single price and Greeks.
   - Monte Carlo: Allows visualization of potential price paths, giving insight into the range of possible outcomes.

6. **When to use Black-Scholes**:
   - For quick pricing of European options
   - When the underlying asset follows Black-Scholes assumptions closely
   - When you need Greeks (sensitivity measures)

7. **When to use Monte Carlo**:
   - For complex options (e.g., Asian options, barrier options)
   - When the underlying asset doesn't follow Black-Scholes assumptions
   - When you need to model specific price path dependencies
   - To get a distribution of possible option values

In practice, both methods are valuable and are often used in conjunction. Black-Scholes provides quick estimates and Greeks, while Monte Carlo can provide more detailed insights and handle more complex scenarios.
""")

def monte_carlo_option_pricing(option_type, S, K, sigma, r, T, num_simulations):
    dt = T / 252  # Assuming 252 trading days in a year
    nudt = (r - 0.5 * sigma**2) * dt
    sidt = sigma * np.sqrt(dt)
    
    Z = np.random.standard_normal((num_simulations, int(252*T)))
    S_T = S * np.exp(np.cumsum(nudt + sidt * Z, axis=1))
    
    if option_type == "Call":
        payoffs = np.maximum(S_T[:, -1] - K, 0)
    else:  # Put option
        payoffs = np.maximum(K - S_T[:, -1], 0)
    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price, S_T
