# Tumor-Dystrophin ODE Model: Technical Documentation

This document provides technical details about the numerical methods used to solve the Tumor-Dystrophin interaction Ordinary Differential Equation (ODE) model and explains the interpretation of the generated visualizations.

## 1. ODE Solving Method

The Python code utilizes the `scipy.integrate.odeint` function to numerically solve the system of ODEs.

*   **Underlying Algorithm:** `odeint` acts as a wrapper for the **LSODA** (Livermore Solver for Ordinary Differential Equations with Automatic method switching) algorithm, which originates from the well-established FORTRAN library ODEPACK.
*   **Adaptive Step-Size:** LSODA employs an adaptive step-size strategy. It automatically adjusts the size of the time steps during integration – taking larger steps when the solution changes slowly and smaller steps when it changes rapidly. This optimizes for both computational efficiency and numerical accuracy.
*   **Handling Stiffness:** A key feature of LSODA is its ability to handle both **non-stiff** and **stiff** ODE systems.
    *   *Non-stiff* problems are typically solved using Adams methods (explicit predictor-corrector methods).
    *   *Stiff* problems, where different components of the solution change at vastly different rates, require implicit methods like Backward Differentiation Formulas (BDF) for numerical stability.
*   **Automatic Switching:** LSODA automatically detects the stiffness of the problem at different points in the integration interval and switches between Adams (for non-stiff regions) and BDF (for stiff regions) methods accordingly. This makes `odeint` a robust and versatile solver suitable for a wide range of ODE problems without requiring the user to diagnose stiffness beforehand.

## 2. Rationale for Solving as a System

The model involves three state variables:
*   `T(t)`: Tumor size at time `t`
*   `D(t)`: Dystrophin expression level at time `t`
*   `S(t)`: Impact factor (related to age/staging) at time `t`

It is essential to solve these equations as a **system** because the variables are **interdependent** or **coupled**:

*   The rate of change of tumor size (`dT/dt`) is influenced by the current dystrophin level (`D(t)`).
*   The rate of change of dystrophin (`dD/dt`) is influenced by the current impact factor (`S(t)`).
*   The rate of change of the impact factor (`dS/dt`) is influenced by the current dystrophin level (`D(t)`).

Because the evolution of each variable depends on the current state of the *other* variables, they cannot be solved independently. The numerical solver calculates the changes in `T`, `D`, and `S` simultaneously at each small time step, correctly capturing the dynamic interactions and feedback loops defined by the model equations.

## 3. Graph Interpretation

The visualizations generated by the code provide insight into the model's behaviour:

### 3.1. Time Series Plot

*   **Axes:**
    *   X-axis: Represents **Time (t)** progressing from the start to the end of the simulation (`t_max`).
    *   Y-axis: Represents the **Level or Size** of each state variable (`T`, `D`, `S`).
*   **Representation:** Each colored line tracks the value of one specific variable (`T(t)`, `D(t)`, or `S(t)`) as it evolves over the simulation time.
*   **Insights:** This plot reveals the **temporal dynamics** of the system. You can observe:
    *   Whether the tumor grows, shrinks, or stabilizes.
    *   How dystrophin levels change in response to the tumor and impact factor.
    *   How the impact factor itself evolves.
    *   Whether the system reaches a steady state (equilibrium) where the values no longer change significantly.
    *   Potential oscillatory behaviour.

### 3.2. Phase Portrait (e.g., T vs D)

*   **Axes:**
    *   X-axis: Represents the value of one state variable (e.g., **Tumor Size T**).
    *   Y-axis: Represents the value of another state variable (e.g., **Dystrophin Level D**).
*   **Representation:** The curve traces the **trajectory** of the system's state in the plane defined by the two plotted variables. Time is an *implicit* parameter determining the path along the curve. Each point on the curve corresponds to the simultaneous values of the two variables at a specific moment in time. The start (e.g., green circle) and end (e.g., red circle) points indicate the direction of evolution.
*   **Insights:** Phase portraits help visualize the **relationship between variables** and the overall system behaviour:
    *   Do trajectories converge towards a single point (indicating a stable equilibrium)?
    *   Do trajectories follow a closed loop (indicating oscillations or limit cycles)?
    *   How does the level of one variable typically change as the other variable changes?
    *   Are there different possible long-term outcomes depending on the starting conditions (multiple attractors)?

By analyzing both the time series and phase portraits under different parameter sets, one can gain a deeper understanding of the model's predictions and the influence of various biological factors (represented by the parameters) on the tumor-dystrophin interaction.