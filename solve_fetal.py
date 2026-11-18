# Solves the Pennati model in an arbitrary geometry

from scipy.integrate import solve_ivp
from fetal_blocks import *

# --- Build model compartments ---
#these are defined in fetal_blocks by a compliance and an initial volume
compartments = {
    'UB': UpperBody(),
    'RA': RightAtrium(),
    'RV': RightVentricle(),
    'LA': LeftAtrium(),
    'LV': LeftVentricle(),
    'PLAC': Placenta(),
    'PA1': PulmonaryArtery1(),  # main trunk
    'PA2': PulmonaryArtery2(),  # distal branches
    'LUNG': Lung(),
    'BR': Brain(),
    'CA': CerebralArteries(),
    'AA': AscendingAorta(),
    'AO1': AorticArch(),
    'AO2': ThoracicDescendingAorta(),
    'AO3': AbdominalDescendingAorta(),
    'AO4': FemoralBifurcation(),
    'SVC': SuperiorVenaCava(),
    'IVC': InferiorVenaCava(),
    'KID': Kidney(),
    'INTE': Intestine(),
    'LEG': Leg(),
    'HE': Hepatic(),
    'UV': UmbilicalVein(),
}
order = ['UB', 'RA', 'RV', 'LA', 'LV', 'PLAC', 'PA1', 'PA2', 'LUNG', 'BR',
         'CA', 'AA', 'AO1', 'AO2', 'AO3', 'AO4', 'SVC', 'IVC', 'KID', 'INTE',
         'LEG', 'HE', 'UV']
idx = {name: i for i, name in enumerate(order)}

# --- Compartment connections: from, to, R, K, L, beta, one_way ---
connections = [
    ('LA', 'LV', 0.02, 0.002, 0.0016, 2.0, False),
    ('RA', 'RV', 0.01, 0.002, 0.0016, 2.0, False),
    ('RV', 'PA1', 0.01, 0.001, 0.0, 2.0, False),
    ('LV', 'AA', 0.01, 0.001, 0.0, 2.0, False),
    ('AA', 'AO1', 0.12, 0.0, 0.002, 2.0, False),
    ('AO1', 'AO2', 0.4, 0.0, 0.0, 2.0, False),
    ('AO2', 'AO3', 0.04, 0.0, 0.0, 2.0, False),
    ('AO3', 'AO4', 0.06, 0.0, 0.0, 2.0, False),
    ('PA1', 'PA2', 0.07, 0.0, 0.002, 2.0, False),
    ('PA2', 'LUNG', 13.5, 0.0, 0.0, 2.0, False),
    ('AO1', 'CA', 0.3, 0.0, 0.08, 2.0, False),
    ('CA', 'BR', 3.0, 0.0, 0.0, 2.0, False),
    ('BR', 'SVC', 8.5, 0.0, 0.0, 2.0, False),
    ('AO1', 'UB', 8.0, 0.0, 0.0, 2.0, False),
    ('UB', 'SVC', 4.9, 0.0, 0.0, 2.0, False),
    ('AO3', 'HE', 81.0, 0.0, 0.0, 2.0, False),
    ('AO3', 'INTE', 34.0, 0.0, 0.0, 2.0, False),
    ('INTE', 'HE', 7.0, 0.0, 0.0, 2.0, False),
    ('AO3', 'KID', 3.5, 0.0, 0.0, 2.0, False),
    ('KID', 'IVC', 14.0, 0.0, 0.0, 2.0, False),
    ('AO4', 'PLAC', 3.9, 0.0, 0.0, 2.0, False),
    ('PLAC', 'UV', 3.4, 0.0, 0.0, 2.0, False),
    ('AO4', 'LEG', 3.5, 0.0, 0.0, 2.0, False),
    ('LEG', 'IVC', 0.6, 0.0, 0.0, 2.0, False),
    ('UV', 'HE', 0.5, 0.0, 0.0, 2.0, False),
    ('HE', 'IVC', 0.16, 0.0, 0.0, 2.0, False),
    ('SVC', 'RA', 0.2, 0.0, 0.0, 2.0, False),
    ('IVC', 'RA', 0.12, 0.0, 0.0, 2.0, False),
    ('LUNG', 'LA', 2.0, 0.0, 0.0, 2.0, False),
    # Fetal shunts (one‑way): FO, DA, DV
    ('IVC', 'LA', 0.001, 0.4, 0.0, 0.625, True),  # foramen ovale
    ('PA2', 'AO2', 0.01, 0.009, 0.006, 2.0, True),  # ductus arteriosus
    ('UV', 'IVC', 1.3, 0.26, 0.0, 2.0, True),  # ductus venosus
]

# Valve-gated links (open only if Pup > Pdown):
VALVES = {('RA', 'RV'), ('LA', 'LV'), ('RV', 'PA1'), ('LV', 'AA')}
EPS = 1e-6


# --- Initial volumes inferred from reference pressures ---
def init_volume_from_pref(comp, p_ref: float, t0: float = 0.0):
    # Ventricles: p = U(t)+E(t)*(V-V0) → V = V0 + (p_ref - U(t0))/E(t0)
    if hasattr(comp, "U") and hasattr(comp, "E"):
        U0 = comp.U(t0)
        E0 = max(comp.E(t0), 1e-9)
        return comp.V0 + (p_ref - U0) / E0
    # Passive compliant chambers: V = V0 + C * p_ref
    if hasattr(comp, "C"):
        return comp.V0 + comp.C * p_ref
    # Fallback:
    return comp.V0


def build_V0_init(comps: dict, p_refs: dict, t0: float = 0.0):
    return {name: init_volume_from_pref(comp, p_refs.get(name, 0.0), t0)
            for name, comp in comps.items()}


p_ref = {
    'UB': 19.3, 'RA': 2.1, 'RV': 21.8, 'LA': 3.0, 'LV': 21.9,
    'PA1': 44.0, 'PA2': 43.1, 'LUNG': 11.0,
    'BR': 32.4, 'CA': 42.1,
    'AA': 44.4, 'AO1': 43.1, 'AO2': 41.4, 'AO3': 40.8, 'AO4': 40.3,
    'SVC': 4.7, 'IVC': 4.3,
    'KID': 32.1, 'INTE': 11.1, 'LEG': 10.2, 'HE': 5.1,
    'UV': 6.7, 'PLAC': 22.4,
}
V0_init = build_V0_init(compartments, p_ref)


# --- Helpers ---
def split_links(conns):
    inertial, resistive = [], []
    for (f, t, R, K, L, beta, one_way) in conns:
        if L and L > 0.0:
            inertial.append((f, t, R, K, L, beta, one_way))
        else:
            resistive.append((f, t, R, K, 0.0, beta, one_way))
    return resistive, inertial


def flow_from_dp(delta_p, R, K, beta):
    # Steady relation: R Q + K |Q|^beta = Δp
    if K == 0.0:
        return delta_p / R
    if beta == 2.0:
        dp = abs(delta_p)
        s = 1.0 if delta_p >= 0.0 else -1.0
        disc = R * R + 4.0 * K * dp
        q_abs = (-R + np.sqrt(max(disc, 0.0))) / (2.0 * K)
        return s * q_abs
    # Newton solve for general beta:
    q = delta_p / (R + 1e-12)
    for _ in range(20):
        sign_q = 1.0 if q >= 0.0 else -1.0
        absq = max(abs(q), EPS)
        f = R * q + K * (absq ** beta) * sign_q - delta_p
        df = R + K * beta * (absq ** (beta - 1.0))
        step = f / df
        q -= step
        if abs(step) < 1e-12:
            break
    return q


# Volume floor for passive chambers during pressure evaluation (does not change state).
# Adds a stiff penalty near the floor to avoid non-physical negative pressure cascades.
def apply_volume_floor_for_pressure(V, comps, floor_frac=0.2, floor_abs=1e-6, stiff=1e6):
    Veff = V.copy()
    penalty = np.zeros_like(V)
    for i, c in enumerate(comps):
        if getattr(c, "is_ventricle", False):
            continue
        v_floor = max(c.V0 * floor_frac, c.V0 + floor_abs)
        if Veff[i] < v_floor:
            deficit = v_floor - Veff[i]
            Veff[i] = v_floor
            penalty[i] = stiff * deficit
    return Veff, penalty


def build_p0_alpha(t, V, comps):
    # p0: elastic/active part; alpha: viscous part for ventricles
    n = len(comps)
    p0 = np.zeros(n)
    alpha = np.zeros(n)
    Veff, penalty = apply_volume_floor_for_pressure(V, comps)

    for i, c in enumerate(comps):
        if getattr(c, "is_ventricle", False):
            p0[i] = c.U(t) + c.E(t) * (V[i] - c.V0)
            alpha[i] = c.R
        else:
            p0[i] = c.pressure(Veff[i], t) + penalty[i]
            alpha[i] = 0.0
    return p0, alpha


# --- ODE RHS with single-entry flow bookkeeping and safeguards ---
def rhs(t, y, comps, connections, idx, valves, valve_eps):
    resistive, inertial = split_links(connections)
    n = len(comps)
    V = y[:n]
    m = len(inertial)
    Qstate = y[n:n + m] if m else np.empty(0)

    p0, alpha = build_p0_alpha(t, V, comps)

    # Linear system A dV = b from resistive links (after linearisation):
    A = np.eye(n)
    b = np.zeros(n)

    # Mass ledger: single entry per link
    def book_mass(Aname, Bname, Q):
        i, j = idx[Aname], idx[Bname]
        b[i] -= Q
        b[j] += Q

    # Resistive links:
    for (f, tname, R, K, L0, beta_link, one_way) in resistive:
        i, j = idx[f], idx[tname]

        # Gate valves and one‑way shunts by p0:
        if (f, tname) in VALVES and p0[i] <= p0[j] + EPS:
            continue
        if one_way and p0[i] <= p0[j] + EPS:
            continue

        dp0 = p0[i] - p0[j]
        Q0 = flow_from_dp(dp0, R, K, beta_link)

        # Mass balance: record once
        book_mass(f, tname, Q0)

        # dQ/d(Δp) at Q0:
        if K == 0.0:
            S = 1.0 / R
        else:
            if beta_link == 2.0:
                S = 1.0 / (R + 2.0 * K * abs(Q0))
            else:
                absq = max(abs(Q0), EPS)
                S = 1.0 / (R + K * beta_link * (absq ** (beta_link - 1.0)))

        # Assemble A (linearisation of p = p0 + alpha dV):
        A[i, i] += S * alpha[i]
        A[i, j] += -S * alpha[j]
        A[j, j] += S * alpha[j]
        A[j, i] += -S * alpha[i]

    # Inertial links:
    dQ = np.zeros(m) if m else np.empty(0)
    for k, (f, tname, R, K, L, beta_link, one_way) in enumerate(inertial):
        i, j = idx[f], idx[tname]

        closed = False
        if (f, tname) in VALVES and p0[i] <= p0[j] + EPS:
            closed = True
        if one_way and p0[i] <= p0[j] + EPS:
            closed = True

        Qk = Qstate[k]
        if closed:
            # Fast exponential decay when closed:
            dQ[k] = -300.0 * Qk
            continue

        # Mass balance by the current inertial flow:
        book_mass(f, tname, Qk)

        # Use p0 to compute dp (decouples from unknown dV in this step):
        dp = p0[i] - p0[j]
        if K == 0.0 or beta_link == 2.0:
            nonlin = K * abs(Qk) * Qk
        else:
            nonlin = K * (abs(Qk) ** beta_link) * (1.0 if Qk >= 0 else -1.0)
        dQ[k] = (dp - R * Qk - nonlin) / L

    # Solve volumes:
    dV = np.linalg.solve(A, b)

    # Safeguards:
    mass_residual = float(np.sum(dV))
    if not np.isfinite(mass_residual):
        raise RuntimeError("Non-finite mass residual in dV/dt")
    if abs(mass_residual) > 1e-6:
        print(f"[WARN] mass residual = {mass_residual:.3e} at t={t:.4f}")

    for nm, ii in idx.items():
        if abs(dV[ii]) > 5e3:
            print(f"[HOT] {nm}: dV/dt={dV[ii]:.3e} at t={t:.4f}")

    for nm, ii in idx.items():
        Vi = V[ii]
        if Vi < -50.0 or Vi > 1e4:
            raise RuntimeError(f"Volume blow-up in {nm}: V={Vi:.3e} at t={t:.4f}")
        if Vi < -1.0:
            print(f"[CLIP] {nm}: V slightly negative ({Vi:.3f}) at t={t:.4f}")

    return np.concatenate([dV, dQ]) if m else dV


# --- Integrate ---
comps = [compartments[name] for name in order]
resistive, inertial = split_links(connections)
n, m = len(comps), len(inertial)

V0 = np.array([V0_init[name] for name in order], dtype=float)
Q0 = np.zeros(m, dtype=float)
_y0 = np.concatenate([V0, Q0])

T_START, T_END = 0.0, 7.0
sol = solve_ivp(
    fun=lambda tt, yy: rhs(tt, yy, comps, connections, idx, VALVES, EPS),
    t_span=(T_START, T_END),
    t_eval = np.linspace(T_START, T_END, 3000),
    y0=_y0,
    method="BDF",
    rtol=2e-5, atol=5e-8,  # moderate tolerances
    #max_step=5e-5,  # small early step to limit overshoot
)

if not sol.success:
    print("status:", sol.status, "msg:", sol.message)

# --- Resample inside the solved window ---

#Y_full = sol.sol(t_eval) if sol.sol is not None else np.vstack([
#    np.interp(t_eval, sol.t, sol.y[i, :]) for i in range(sol.y.shape[0])
#])
#mask = (t_eval >= sol.t[0]) & (t_eval <= sol.t[-1])
t_plot = sol.t#t_eval[mask]
Y = sol.y#Y_full[:, mask]

# --- Pressure post‑processing ---
def compute_pressures(t, yfull):
    V = yfull[:n]
    p0, alpha = build_p0_alpha(t, V, comps)
    dy = rhs(t, yfull, comps, connections, idx, VALVES, EPS)
    dV = dy[:n]
    return p0 + alpha * dV


pressures = {name: np.zeros_like(t_plot) for name in order}
for k, tk in enumerate(t_plot):
    yk = Y[:, k]
    pk = compute_pressures(tk, yk)
    for name, i in idx.items():
        pressures[name][k] = pk[i]

volumes = {name: Y[idx[name], :] for name in order}

# print("t_end =", sol.t[-1])
# print("status =", sol.status, "message =", sol.message)
# for nm in ['RA', 'RV', 'LA', 'LV', 'AA', 'PA1']:
#     v0 = volumes[nm][0];
#     p0 = pressures[nm][0]
#     v1 = volumes[nm][-1];
#     p1 = pressures[nm][-1]
#     print(f"{nm}: V(0)={v0:.3f}, P(0)={p0:.3f} | V(end)={v1:.3f}, P(end)={p1:.3f}")
