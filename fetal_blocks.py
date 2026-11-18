# Blocks to build the pressure-volume network model based on the 23-compartment fetal circulation system
# from Pennati et al. (1997)


import numpy as np

# ======================
# Global parameters
# ======================
U0 = 40  # mmHg
E_dia = 0.3  # mmHg/ml
E_sys = 3.0  # mmHg/ml
R_vent = 0.08  # mmHg·s/ml
V0_vent = 6  # ml
beta = 2
T_cycle = 60.0 / 135.0  # s, 135 bpm
T_sys = (0.15 / 0.43) * T_cycle  # keep systolic fraction
T_offset = (0.05 / 0.43) * T_cycle  # keep activation offset fraction
T_atria = 0.12  # Atrial contraction duration (s) - Adjust as needed
AT_advance = 0.07  # Atrium advance（s）to the ventricle

# ======================
# Ventricular Activation function
# ======================
def A_t(t):
    t_mod = t % T_cycle
    if T_offset <= t_mod <= T_offset + T_sys:
        tau = (t_mod - T_offset) / T_sys
        return np.sin(np.pi * tau) ** 2
    return 0

# ======================
# Atrial Activation function
# ======================
def A_atria(t):
    t_mod = t % T_cycle
    start = (T_offset - AT_advance) % T_cycle
    end = (start + T_atria) % T_cycle

    if start <= end:
        active = (start <= t_mod <= end)
        tau = (t_mod - start) / T_atria if active else None
    else:
        # [start, T_cycle) ∪ [0, end]
        active = (t_mod >= start) or (t_mod <= end)
        tau = ((t_mod - start) % T_cycle) / T_atria if active else None

    if active:
        return np.sin(np.pi * tau) ** 2
    return 0.0


# ======================
# Base classes
# ======================
class Ventricle: #Ventricular element class
    def __init__(self, U0=U0, E_dia=E_dia, E_sys=E_sys, R=R_vent, V0=V0_vent, A_func=A_t):
        self.is_ventricle = True
        self.U0 = U0
        self.E_dia = E_dia
        self.E_sys = E_sys
        self.R = R
        self.V0 = V0
        self.A_func = A_func

    def U(self, t):
        return self.U0 * self.A_func(t)

    def E(self, t):
        return self.E_dia + self.E_sys * self.A_func(t)

    def pressure(self, t, V, dVdt):
        return self.U(t) + self.E(t) * (V - self.V0) + self.R * dVdt

    def dV_dt(self, Q_in, Q_out):
        return Q_in - Q_out


class Passive:
    def __init__(self, C, V0=0):
        self.C = C
        self.V0 = V0
        self.is_ventricle = False

    def pressure(self, V, t=None):
        return (V - self.V0) / self.C

    def dV_dt(self, Q_in, Q_out):
        return Q_in - Q_out


# ======================
# Compartments
# ======================
class UpperBody(Passive):
    def __init__(self):
        super().__init__(C=0.85, V0=18)


class RightAtrium(Passive):
    # Partial passive still inheriting Passive class for easier coding
    def __init__(self):
        super().__init__(C=2.0, V0=0)
        self.Ua0 = 3.0

    def pressure(self, V, t=None):
        Ua = self.Ua0 * A_atria(t) if t is not None else 0.0
        return Ua + (V - self.V0) / self.C


class RightVentricle(Ventricle):
    def __init__(self):
        super().__init__()


class LeftAtrium(Passive):
    def __init__(self):
        super().__init__(C=1.0, V0=0)  # Table 1
        self.Ua0 = 3.0  # Table 1

    def pressure(self, V, t=None):
        Ua = self.Ua0 * A_atria(t) if t is not None else 0.0
        return Ua + (V - self.V0) / self.C


class LeftVentricle(Ventricle):
    def __init__(self):
        super().__init__()


class Placenta(Passive):
    def __init__(self):
        super().__init__(C=1.5, V0=0)


class PulmonaryArtery1(Passive):
    def __init__(self):
        super().__init__(C=0.08, V0=0)


class PulmonaryArtery2(Passive):
    def __init__(self):
        super().__init__(C=0.08, V0=0)


class Lung(Passive):
    def __init__(self):
        super().__init__(C=0.4, V0=0)  # Table 2


class Brain(Passive):
    def __init__(self):
        super().__init__(C=0.3, V0=0)  # Table 2


class CerebralArteries(Passive):
    def __init__(self):
        super().__init__(C=0.1, V0=0)  # Table 2


class AscendingAorta(Passive):
    def __init__(self):
        super().__init__(C=0.05, V0=0)  # Table 2


class AorticArch(Passive):
    def __init__(self):
        super().__init__(C=0.08, V0=0)  # AO1 Table 2


class ThoracicDescendingAorta(Passive):
    def __init__(self):
        super().__init__(C=0.07, V0=0)  # AO2 Table 2


class AbdominalDescendingAorta(Passive):
    def __init__(self):
        super().__init__(C=0.04, V0=0)  # AO3 Table 2


class FemoralBifurcation(Passive):
    def __init__(self):
        super().__init__(C=0.05, V0=0)  # AO4 Table 2


class SuperiorVenaCava(Passive):
    def __init__(self):
        super().__init__(C=1.0, V0=0)  # Table 2


class InferiorVenaCava(Passive):
    def __init__(self):
        super().__init__(C=0.6, V0=0)  # Table 2


class Kidney(Passive):
    def __init__(self):
        super().__init__(C=0.02, V0=0)  # Table 2


class Intestine(Passive):
    def __init__(self):
        super().__init__(C=0.25, V0=0)  # Table 2


class Leg(Passive):
    def __init__(self):
        super().__init__(C=4.0, V0=0)  # Table 2


class Hepatic(Passive):
    def __init__(self):
        super().__init__(C=3.0, V0=0)  # Table 2


class UmbilicalVein(Passive):
    def __init__(self):
        super().__init__(C=0.3, V0=0)  # Table 2


# --- Helpers: valves & pure resistors ---------------------------------
def valve_flow(P_up, P_down, R_open, R_closed, *, name="", allow_backflow=False):
    """Standard valve: opens only if P_up > P_down. Otherwise closed.
    If allow_backflow=True, backflow uses R_closed (very large)."""
    if P_up > P_down:
        R = R_open
        Q = (P_up - P_down) / R
    else:
        if allow_backflow:
            Q = (P_up - P_down) / R_closed  # small backflow through huge resistance
        else:
            Q = 0.0  # hard stop
    return Q


def resistive_flow(P_up, P_down, R):
    """Pure resistor (no valve): Ohmic law."""
    return (P_up - P_down) / R
