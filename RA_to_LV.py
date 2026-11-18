from Heart_UB_LB import *

# Cardiovascular parameters:
C_RA = 5.0  # Right atrium compliance (ml/mmHg)
C_LV = 1.0  # Left ventricle compliance (ml/mmHg)
R_RA_LV = 0.01  # Resistance from RA to LV (mmHg·min/ml)
P_venous = 5  # Venous pressure (mmHg)


# ODE system:
def heart_model(t, y):
    P_RA, V_LV = y
    E = E_t(t)
    dV_LV_dt = (P_RA - (E * (V_LV - V0))) / R_RA_LV
    P_LV = E * (V_LV - V0) + R_vent * dV_LV_dt

    Q_venous = (P_venous - P_RA) / 0.05  # Venous inflow
    Q_RA_LV = (P_RA - P_LV) / R_RA_LV  # RA to LV flow

    dP_RA_dt = (Q_venous - Q_RA_LV) / C_RA

    return [dP_RA_dt, dV_LV_dt]


# Initial conditions:
P_RA0 = 5  # mmHg
V_LV0 = 10  # ml

# Solve ODE:
sol = solve_ivp(heart_model, t_span, [P_RA0, V_LV0], t_eval=t_eval)
P_RA_vals = sol.y[0]
V_LV_vals = sol.y[1]
P_LV_vals = [E_t(t) * (V - V0) + R_vent * ((P_RA_vals[i] - (E_t(t) * (V - V0))) / R_RA_LV)
             for i, (t, V) in enumerate(zip(t_eval, V_LV_vals))]
