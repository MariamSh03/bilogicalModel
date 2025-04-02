import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# 1. Define the ODE System Model
def tumor_dystrophin_model(t, y, params):
    """
    Defines the ODE system for tumor-dystrophin interaction.
    y = [T, D, S]
    params = [r, K, f, k, g, h, S0, j]
    """
    T, D, S = y
    r, K, f, k, g, h, S0, j = params

    # Avoid potential numerical issues if T, D, or S become slightly negative
    T = max(T, 0)
    D = max(D, 0)
    S = max(S, 0)

    dTdt = r * T * (1 - T / K) - f * D * T
    dDdt = -k * D + g * S * D
    dSdt = h * (S0 - S) - j * D * S

    # Prevent division by zero or large numbers if K is small or T is huge
    if K <= 0: dTdt = -f * D * T # Simplified if no carrying capacity
    elif T > 2*K : dTdt = -r*T*(T/K) - f*D*T # Faster decay if way over K

    return [dTdt, dDdt, dSdt]

# 2. Define Parameters and Boundary Conditions
# Parameters from the presentation table (slide 9)
params = {
    'r': 0.04,    # 1/day
    'K': 1e9,     # Cells
    'f': 0.01,    # 1/(Arbitrary units * day)
    'k': 0.1,     # 1/day
    'g': 0.05,    # 1/(Arbitrary units * day)
    'h': 0.1,     # 1/day
    'S0_param': 100,  # Arbitrary units (Max level in healthy tissue)
    'j': 0.01     # 1/(Arbitrary units * day)
}
param_list = [params['r'], params['K'], params['f'], params['k'],
              params['g'], params['h'], params['S0_param'], params['j']]

# Known Initial Conditions
T_initial = 1e3    # Cells
D_initial = 10     # Arbitrary units

# Final Time and Target Final Condition
T_f = 100          # days
T_final_target = 5e8 # Target tumor size at T_f (example value)

# Time span for integration
t_span = [0, T_f]
t_eval = np.linspace(t_span[0], t_span[1], 200) # Points for evaluation/plotting

# 3. Define the Objective Function for the Shooting Method
def objective_function(S_guess, T_init, D_init, t_span_obj, params_list, T_target):
    """
    Solves the IVP with a guess for S(0) and returns the error
    between the computed T(T_f) and the target T_final.
    """
    if S_guess < 0: # Constrain S(0) to be non-negative if physically required
        return np.inf # Return large error for invalid guesses

    y0 = [T_init, D_init, S_guess]
    # Use a robust solver, increase tolerances if needed
    sol = solve_ivp(tumor_dystrophin_model, t_span_obj, y0,
                    args=(params_list,), dense_output=True, method='LSODA', # LSODA is often good for stiff problems
                    rtol=1e-6, atol=1e-8) # Adjust tolerances if necessary

    if not sol.success or sol.status != 0: # Check status explicitly
        # Handle integration failure, maybe return a large error
        print(f"Warning: Integration failed or did not complete successfully for S(0) = {S_guess:.4f}, Status: {sol.status}, Message: {sol.message}")
        # Return a large value that maintains the sign relative to the target, if possible
        # If T_target is large, return a very large positive number, otherwise very large negative?
        # Or just return np.inf might be safer for the root finder.
        return np.inf

    # Get the computed value of T at the final time T_f
    T_computed_final = sol.sol(t_span_obj[1])[0] # Index 0 corresponds to T

    # Check for NaN or Inf results
    if not np.isfinite(T_computed_final):
        print(f"Warning: Computed T({t_span_obj[1]}) is not finite for S(0) = {S_guess:.4f}")
        return np.inf

    error = T_computed_final - T_target
    # Uncomment the line below for detailed debugging during bracket finding:
    # print(f"Trying S(0)={S_guess:<8.4f}, T({T_f})={T_computed_final:.4e}, Error={error:.4e}")
    return error

# --- Modify the Bracket Testing Loop ---
print("\n--- Testing objective function AND PLOTTING T(t) ---")
test_S_values = np.linspace(0.1, 150, 8) # Reduce points for plotting clarity
errors = {}
possible_bracket = []
last_s_val = None
last_error = None

plt.figure(figsize=(10, 6)) # Create a figure for the test plots

for s_val in test_S_values:
    print(f"--- Testing S(0) = {s_val:.2f} ---")
    y0_test = [T_initial, D_initial, s_val]
    try:
        sol_test = solve_ivp(tumor_dystrophin_model, t_span, y0_test,
                             args=(param_list,), dense_output=True, t_eval=t_eval,
                             method='LSODA', rtol=1e-6, atol=1e-8)

        if sol_test.success and sol_test.status == 0:
            T_test_sol = sol_test.y[0]
            T_computed_final = T_test_sol[-1]
            error = T_computed_final - T_final_target
            errors[s_val] = error
            print(f"  S(0) = {s_val:<6.2f} -> Error = {error:.3e} (T(Tf) = {T_computed_final:.3e})")

            # Plot this trajectory
            plt.plot(sol_test.t, T_test_sol, label=f'S(0)={s_val:.1f}')

            if last_error is not None and error * last_error < 0:
                possible_bracket = [last_s_val, s_val]
                print(f"  >>> Potential bracket found: [{last_s_val:.2f}, {s_val:.2f}] <<<")
            last_s_val = s_val
            last_error = error
        else:
            print(f"  Integration failed for S(0) = {s_val:.2f}. Status: {sol_test.status}")
            last_error = None
            last_s_val = None

    except Exception as e:
        print(f"  Failed to compute for S(0) = {s_val:<6.2f}: {e}")
        last_error = None
        last_s_val = None

# Finalize the test plot
plt.axhline(T_final_target, color='k', linestyle='--', label=f'Target T({T_f})')
plt.xlabel("Time (days)")
plt.ylabel("Tumor Size T(t)")
plt.title("Tumor Growth Trajectories for Different S(0) Guesses")
plt.yscale('log') # Use log scale to see small values
plt.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
plt.show()

print("--- End of bracket testing ---")
# --- End of Modified Loop ---