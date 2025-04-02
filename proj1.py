import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def tumor_dystrophin_model(y, t, params):
    """Defines the system of ODEs for tumor-dystrophin interaction."""
    T, D, S = y
    r, K, f, k, g, h, S0, j = params['r'], params['K'], params['f'], \
                               params['k'], params['g'], params['h'], \
                               params['S0'], params['j']

    dTdt = r * T * (1 - T / K) - f * D * T
    dDdt = -k * D + g * S * D
    dSdt = h * (S0 - S) - j * D * S
    return [dTdt, dDdt, dSdt]


# Streamlit App Setup
st.set_page_config(layout="wide")
st.title('Interactive Tumor–Dystrophin Interaction ODE Model')
st.markdown("""
Visualize the dynamics of tumor size (T), dystrophin level (D),
and an impact factor (S) based on the provided ODE system.
Use the sliders in the sidebar to adjust parameters and initial conditions,
and observe the changes in the plots.
""")

st.sidebar.header('Model Parameters')

# --- Create Sliders for Parameters with specific defaults and ranges ---
# Adjust min/max/step based on typical values and desired exploration range

# r (Default: 0.1) - Tumor growth rate
r = st.sidebar.slider('r (Tumor growth rate)',
                      min_value=0.0,
                      max_value=1.0,  # Adjust max as needed
                      value=0.21,
                      step=0.01)

# K (Default: 10000) - Carrying capacity
K = st.sidebar.slider('K (Carrying capacity)',
                      min_value=1000.0,
                      max_value=50000.0, # Allow significant variation
                      value=10000.0,
                      step=500.0) # Larger step for large range

# f (Default: 0.01) - Dystrophin effect on T
f = st.sidebar.slider('f (Dystrophin effect on T)',
                      min_value=0.0,
                      max_value=0.1, # Range around the small default
                      value=0.015,
                      step=0.001)

# k (Default: 0.5) - Dystrophin degradation
k = st.sidebar.slider('k (Dystrophin degradation)',
                      min_value=0.0,
                      max_value=2.0, # Range around 0.5
                      value=1.4,
                      step=0.05)

# g (Default: 0.1) - S effect on D
g = st.sidebar.slider('g (S effect on D)',
                      min_value=0.0,
                      max_value=0.5, # Range around 0.1
                      value=0.41,
                      step=0.01)

# h (Default: 0.2) - S recovery rate
h = st.sidebar.slider('h (S recovery rate)',
                      min_value=0.0,
                      max_value=1.0, # Range around 0.2
                      value=0.2,
                      step=0.02) # Adjusted step

# S0_param (Default: 15.0) - Baseline S level (Parameter S0)
S0_param = st.sidebar.slider('S0 (Baseline S level)',
                             min_value=1.0,
                             max_value=50.0, # Range around 15
                             value=8.0,
                             step=0.5)

# j (Default: 0.5) - D effect on S
j = st.sidebar.slider('j (D effect on S)',
                      min_value=0.0,
                      max_value=2.0, # Range around 0.5
                      value=0.1,
                      step=0.05)

parameters = {'r': r, 'K': K, 'f': f, 'k': k, 'g': g, 'h': h, 'S0': S0_param, 'j': j}

st.sidebar.header('Initial Conditions')
# Sliders/Inputs for Initial Conditions
T0 = st.sidebar.number_input('Initial Tumor Size (T0)', 0.0, K, 1.0, 1.0) # Use K as max reasonable starting T
D0 = st.sidebar.number_input('Initial Dystrophin Level (D0)', 0.0, 50.0, 2.0, 1.0)
S0_init = st.sidebar.number_input('Initial Impact Factor (S0_init)', 0.0, S0_param * 1.5, S0_param, 1.0) # Can start at S0 or elsewhere

initial_conditions = [T0, D0, S0_init]

st.sidebar.header('Simulation Settings')
t_max = st.sidebar.slider('Simulation Time (t_max)', 10.0, 1000.0, 10.0, 10.0)
num_points = 500 # Keep fixed or make adjustable if needed (st.sidebar....)
t = np.linspace(0, t_max, num_points) # Time vector

# --------------------------------------------------
# 3. Solve the ODE System (Runs automatically on slider change)
# --------------------------------------------------
solution = odeint(tumor_dystrophin_model, initial_conditions, t, args=(parameters,))
T_sol = solution[:, 0]
D_sol = solution[:, 1]
S_sol = solution[:, 2]


# Visualize the Results
st.subheader('Time Series Plot')
fig1, ax1 = plt.subplots(figsize=(10, 6)) # Create Matplotlib figure and axes

# Plot Tumor Size (T)
ax1.plot(t, T_sol, label='Tumor Size (T)', color='red', linewidth=2)
# Plot Dystrophin Level (D)
ax1.plot(t, D_sol, label='Dystrophin Level (D)', color='blue', linewidth=2)
# Plot Impact Factor (S)
ax1.plot(t, S_sol, label='Impact Factor (S)', color='green', linewidth=2)

# Add plot details
ax1.set_title('Model Dynamics Over Time', fontsize=14)
ax1.set_xlabel('Time (t)', fontsize=12)
ax1.set_ylabel('Population / Level', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(bottom=0) # Ensure y-axis starts at 0
fig1.tight_layout()

st.pyplot(fig1)


# --- Optional: Phase Portrait ---
st.subheader('Phase Portrait (Tumor vs Dystrophin)')
fig2, ax2 = plt.subplots(figsize=(7, 6)) # Create a second figure/axes

ax2.plot(T_sol, D_sol, color='purple')
ax2.set_title('Phase Portrait: Tumor (T) vs Dystrophin (D)')
ax2.set_xlabel('Tumor Size (T)')
ax2.set_ylabel('Dystrophin Level (D)')
# Add arrows or starting point marker for clarity (optional)
ax2.plot(T_sol[0], D_sol[0], 'go', markersize=8, label='Start') # Green circle at start
ax2.plot(T_sol[-1], D_sol[-1], 'ro', markersize=8, label='End') # Red circle at end
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)
fig2.tight_layout()

# Display the second plot
st.pyplot(fig2)

# --- Display Current Parameters (Optional) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Current Values")
st.sidebar.write("**Parameters:**")
st.sidebar.json(parameters)
st.sidebar.write("**Initial Conditions:**")
st.sidebar.write(f"T0 = {T0}, D0 = {D0}, S0_init = {S0_init}")