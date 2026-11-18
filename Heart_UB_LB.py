import numpy as np
from scipy.integrate import solve_ivp

# Parameters:
R_UB = 0.125  # Resistance to/from Upper Body (mmHg·min/ml)
R_LB = 0.144  # Resistance to/from Lower Body (mmHg·min/ml)
C_UB = 0.9  # Compliance UB (ml/mmHg)
C_LB = 1.4  # Compliance LB (ml/mmHg)
P_venous = 5  # Venous return pressure (mmHg)

# Heart model parameters:
E_max = 2.0  # mmHg/ml
E_min = 0.06
V0 = 5  # ml
R_vent = 0.01  # mmHg·min/ml
V_initial = 15  # ml

# Cardiac cycle timing:
T_sys = 0.15  # systolic duration (approx 35% of T_cycle)
T_offset = 0.05  # delay after cycle start
T_cycle = 0.43  # Duration of one cardiac cycle in minutes (~0.43 min = ~26 sec = 140 bpm fetal)


# Define activation function A(t) based on Pennati's Figure 2:
def A_t(t):
    t_mod = t % T_cycle
    if T_offset <= t_mod <= T_offset + T_sys:
        tau = (t_mod - T_offset) / T_sys
        return np.sin(np.pi * tau) ** 2
    else:
        return 0


# Elastance function:
def E_t(t):
    return E_min + (E_max - E_min) * A_t(t)


# Ventricle volume and its derivative (simple assumed waveform):
def V_t(t):
    return 15 + 5 * np.sin(2 * np.pi * (t % T_cycle) / T_cycle)


def dV_dt(t):
    return 5 * 2 * np.pi / T_cycle * np.cos(2 * np.pi * (t % T_cycle) / T_cycle)


# Heart pressure using A(t):
def P_heart(t):
    return E_t(t) * (V_t(t) - V0) + R_vent * dV_dt(t)


# ODE system:
def circulation_model(t, y):
    P_UB, P_LB = y  # Pressures in UB and LB compartments
    P_h = P_heart(t)

    # Flows:
    Q_H_to_UB = (P_h - P_UB) / R_UB
    Q_UB_to_H = (P_UB - P_venous) / R_UB

    Q_H_to_LB = (P_h - P_LB) / R_LB
    Q_LB_to_H = (P_LB - P_venous) / R_LB

    # Pressure changes due to flow in/out:
    dP_UB_dt = (Q_H_to_UB - Q_UB_to_H) / C_UB
    dP_LB_dt = (Q_H_to_LB - Q_LB_to_H) / C_LB

    return [dP_UB_dt, dP_LB_dt]


# Time span for simulation:
t_span = (0, 2)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
# Initial pressures for UB and LB:
P_UB0 = 20
P_LB0 = 20

# Solve system
solution = solve_ivp(circulation_model, t_span, [P_UB0, P_LB0], t_eval=t_eval)
P_heart_vals = [P_heart(t) for t in t_eval]
A_vals = [A_t(t) for t in t_eval]
