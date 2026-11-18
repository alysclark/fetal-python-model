# Visualisation of the model

import matplotlib.pyplot as plt

from RA_to_LV import *
from solve_fetal import t_plot, pressures, volumes


# UB/LB heart circulation plot:
def plot_heart_ub_lb(t_eval, UB_pressure, LB_pressure, heart_pressure, A_vals, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(t_eval, UB_pressure, label='Upper Body Pressure (UB)')
    plt.plot(t_eval, LB_pressure, label='Lower Body Pressure (LB)')
    plt.plot(t_eval, heart_pressure, label='Heart Pressure', linestyle='--', color='black')
    plt.plot(t_eval, A_vals, label='A(t) Activation Function', linestyle='dotted', color='magenta')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pressure (mmHg) / Activation (unitless)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)


# RA to LV plot:
def plot_ra_to_lv(t_eval, P_RA_vals, P_LV_vals, V_LV_vals, A_vals, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(t_eval, P_RA_vals, label='Right Atrium Pressure (RA)')
    plt.plot(t_eval, P_LV_vals, label='Left Ventricle Pressure (LV)')
    plt.plot(t_eval, V_LV_vals, label='LV Volume (ml)', linestyle='--')
    plt.plot(t_eval, A_vals, label='A(t)', linestyle='dotted', color='magenta')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Pressure (mmHg) / Volume (ml)')
    plt.title("Right Atrium and Left Ventricle Circulatory Model")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)


# plot_heart_ub_lb(
#     t_eval=t_eval,
#     UB_pressure=solution.y[0],
#     LB_pressure=solution.y[1],
#     heart_pressure=P_heart_vals,
#     A_vals=A_vals,
#     save_path='Heart_UB_LB.png'
# )

# plot_ra_to_lv(
#     t_eval=t_eval,
#     P_RA_vals=P_RA_vals,
#     P_LV_vals=P_LV_vals,
#     V_LV_vals=V_LV_vals,
#     A_vals=[A_t(t) for t in t_eval],
#     save_path='RA_to_LV.png'
# )


def plot_basic_blocks(results, save_path=None):
    t = results['t_eval']
    plt.figure(figsize=(12, 6))
    plt.plot(t, results['P_UB'], label='UB Pressure')
    plt.plot(t, results['P_LB'], label='LB Pressure')
    plt.plot(t, results['P_RA'], label='RA Pressure')
    plt.plot(t, results['P_LV'], label='LV Pressure')
    # plt.plot(t, results['V_LV'], label='LV Volume', linestyle='--')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pressure (mmHg)')
    plt.title('Pennati Core Block Simulation: UB, LB, RA, LV')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)


# plot_basic_blocks(results, save_path='core_blocks_P_only.png')


# With placenta now
def plot_with_PL(pressures, save_path=None):
    plt.figure(figsize=(12, 6))
    for name in ['UB', 'LB', 'RA', 'RV', 'LA', 'LV', 'PL']:
        plt.plot(sol.t, pressures[name], label=f'{name} pressure')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.title('Pennati Simulation: UB, LB, RA, RV, LA, LV, PL')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)


# plot_with_PL(pressures, save_path='with_PL.png')

# Plots for the complete Pennati model:
GROUPS = [
    ["RA", "RV", "LA", "LV"],
    ["PA1", "PA2", "LUNG", "UB"],
    ["BR", "CA"],
    ["AA", "AO1", "AO2", "AO3", "AO4"],
    ["SVC", "IVC"],
    ["KID", "INTE", "LEG", "HE"],
    ["UV", "PLAC"],
]


def plot_group_series(t, series_dict, names, ylabel, title, save_path):
    """Basic plot function for one group"""
    plt.figure(figsize=(12, 6))
    for name in names:
        if name in series_dict:
            plt.plot(t, series_dict[name], label=name)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_group_pressures(names, t=None, save_path="pressures.png"):
    """Plot pressures vs time for one group"""
    if t is None:
        t = t_plot
    title = f"Pressures vs Time ({', '.join(names)})"
    plot_group_series(t, pressures, names, "Pressure (mmHg)", title, save_path)


def plot_group_volumes(names, t=None, save_path="volumes.png"):
    """Plot volumes vs time for one group"""
    if t is None:
        t = t_plot
    title = f"Volumes vs Time ({', '.join(names)})"
    plot_group_series(t, volumes, names, "Volume (ml)", title, save_path)


def plot_all_groups(t=None):
    """Loop over all groups and save two plots for each"""
    if t is None:
        t = t_plot
    for i, names in enumerate(GROUPS, start=1):
        plot_group_pressures(names, t, save_path=f"group_{i:02d}_pressures.png")
        plot_group_volumes(names, t, save_path=f"group_{i:02d}_volumes.png")


# Generate all 14 plots:
plot_all_groups()


# KID pressure vs volume:
def plot_kid_pressure_volume(t=None, save_path="KID_pressure_volume.png"):
    if t is None:
        t = t_plot
    p = pressures["KID"]
    v = volumes["KID"]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    p_line, = ax1.plot(t, p, label="KID Pressure (mmHg)", color="tab:blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pressure (mmHg)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    v_line, = ax2.plot(t, v, label="KID Volume (ml)", linestyle="--", color="tab:orange")
    ax2.set_ylabel("Volume (ml)")

    # Choose non-proportional y-limits so curves separate:
    pmin, pmax = float(p.min()), float(p.max())
    vmin, vmax = float(v.min()), float(v.max())
    ax1.set_ylim(pmin - 0.05 * (pmax - pmin), pmax + 0.05 * (pmax - pmin))  # tight for pressure
    vspan = (vmax - vmin)
    ax2.set_ylim(vmin - 0.40 * vspan, vmax + 0.40 * vspan)  # stretch volume axis to break overlay

    # Combined legend:
    ax1.legend([p_line, v_line], [l.get_label() for l in (p_line, v_line)], loc="best")
    plt.title("KID: Pressure and Volume vs Time")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


plot_kid_pressure_volume(save_path="KID_pressure_volume.png")
