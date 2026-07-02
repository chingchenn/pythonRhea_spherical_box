#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:07:28 2026

@author: chingchen
"""
import os
import re
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

COLORS = [
    '#CD5C5C',
    '#35838D',
    '#849DAB',
    '#414F67',
    '#97795D',
    '#7B9E87',
    '#9B6B8A',
    '#4E6E8E',
    '#C47F3E',
    '#5C7A5C',
    '#8B6F6F',
    '#4A7C7C',
    '#7A6B9B',
    '#6B8E7A',
]

### built-in defaults (copied from the C #define's), used for any .ini key

DEFAULTS = {
    'type-linear': 2,                                  
    'type-nonlinear': 2,                              
    'model': 'UWYL_LADD_UCUT',
    'representative-Pas': 1.0e20,
    'min': 1.0e-2,
    'max': 1.0e4,
    'upper-mantle-scaling': 4.0e3,
    'upper-mantle-arrhenius-activation-energy': 17.5,
    'upper-mantle-arrhenius-activation-volume': 0.0,
    'lower-mantle-scaling': 4.0e5,
    'lower-mantle-arrhenius-activation-energy': 17.5,
    'lower-mantle-arrhenius-activation-volume': 0.0,
    'stress-exponent': 3.0,
    'yield-strength': None,                            
    'neutral': 0.5,                                  
    'temperature-difference-K': 1400.0,
    'temperature-min-K': 300.0,
    'gas-constant-J-Kmol': 8.314,
    'earth-radius': 6.371e6,
    'mantle-depth': 6.6e5,                              # m; UM/LM interface depth (not domain extent)
    'lower-upper-mantle-interface-depth': -1,
}

kappa = 1e-6                    

sec_per_myr = 1e6 * 365.25 * 24 * 3600.0

use_ladd_add = True             
override_type_linear = None     

strain_rate = 1e-15            
age_myr = 50.0                 
full_mantle_depth_km = 2890.0   


# ==========================================
### read an options.ini into a flat dict, e.g. ini['Viscosity']['stress-exponent']
# ==========================================
def read_ini(path):
    sections = {}
    cur = None
    with open(path) as f:
        for raw in f:
            line = raw.split('#', 1)[0].rstrip()
            if not line.strip():
                continue
            m = re.match(r'\s*\[(.+?)\]', line)
            if m:
                cur = m.group(1)
                sections.setdefault(cur, {})
                continue
            if cur is None or '=' not in line:
                continue
            key, val = line.split('=', 1)
            sections[cur][key.strip()] = val.strip().rstrip(';').strip()
    return sections


def get_hsc_temp_nd(depth_km, age_myr, temp_neutral):
    age_s = age_myr * sec_per_myr
    depth_m = depth_km * 1000.0
    return temp_neutral * erf(depth_m / (2.0 * np.sqrt(kappa * age_s)))


# ==========================================
### viscosity at one (T, depth) point, following rhea_viscosity.c
# ==========================================
def get_viscosity(T, depth_km, p, strain_rate):
    sr_dim = strain_rate * (p['earth_radius']**2 / kappa)
    is_um = depth_km < p['interface_depth_km']
    h_nd = depth_km * 1000.0 / p['earth_radius']

    n = p['stress_exponent'] if (is_um and p['stress_exponent'] >= 1.0) else 1.0
    srw_exp = 1.0 / n

    A = p['um_scaling'] if is_um else p['lm_scaling']
    E = (p['um_E'] if is_um else p['lm_E']) * srw_exp
    Va = (p['um_Va'] if is_um else p['lm_Va']) * srw_exp

    # --- linear component, rhea_viscosity_linear_comp() type-linear = 0..4 ---
    if p['type_linear'] == 0:                              # CONST
        eta_lin = 1.0                                       # RHEA_VISCOSITY_NEUTRAL_VALUE
    elif p['type_linear'] == 1:                             # TEMPREVERSE
        eta_lin = 1.0 - T
    elif p['type_linear'] == 2:                             # ARRHENIUS
        eta_lin = np.exp(E * (p['temp_neutral'] - T))
    elif p['type_linear'] == 3:                              # ARRHENIUS_TEMPINVERSE
        T_K = p['T_min_K'] + p['T_diff_K'] * T
        T_K_neutral = p['T_min_K'] + p['T_diff_K'] * p['temp_neutral']
        eta_lin = np.exp(E * (1.0 / T_K - 1.0 / T_K_neutral))
    elif p['type_linear'] == 4:                              # ARRHENIUS_FULL
        T_K = p['T_min_K'] + p['T_diff_K'] * T
        eta_lin = np.exp((E + Va * h_nd) / T_K)
    else:
        raise ValueError(f"type-linear={p['type_linear']} not recognized (0-4)")

    # --- scaling (LADD models subtract visc_min from the raw scaling) ---
    scaling = A**srw_exp
    if use_ladd_add and p['visc_min'] < scaling:
        scaling -= p['visc_min']
    eta_lin *= scaling

    # --- strain-rate weakening (upper mantle only) ---
    if is_um and p['has_srw']:
        eta_srw = eta_lin * pow(sr_dim, srw_exp - 1.0)
    else:
        eta_srw = eta_lin
    eta_srw = min(eta_srw, p['visc_max'])                   # (U) upper cut

    # --- yielding (upper mantle only) ---
    eta_yield = p['yield_strength'] / (2.0 * sr_dim) if (is_um and p['has_yld']) else np.inf
    if is_um and p['has_yld'] and 2.0 * eta_srw * sr_dim > p['yield_strength']:
        viscosity = eta_yield
    else:
        viscosity = eta_srw

    # --- lower bound ---
    if use_ladd_add:
        viscosity = viscosity + p['visc_min']               # (L) LADD: add
    else:
        viscosity = max(viscosity, p['visc_min'])            # (L) UCUT/LCUT: clip

    eta_ref = p['eta_ref']
    return eta_srw * eta_ref, eta_yield * eta_ref, viscosity * eta_ref


def resolve_params(V, D, T):
    """V, D, T are flat dicts (Viscosity/Domain/Temperature-style keys).
    Any missing key falls back to DEFAULTS."""
    def g(section, key):
        return section[key] if key in section else DEFAULTS[key]

    mantle_depth_km = float(g(D, 'mantle-depth')) / 1e3
    interface_depth_m = float(g(D, 'lower-upper-mantle-interface-depth'))
    interface_depth_km = mantle_depth_km if interface_depth_m < 0 else interface_depth_m / 1e3

    type_nonlinear = int(g(V, 'type-nonlinear'))
    stress_exponent = float(g(V, 'stress-exponent'))
    yield_strength = g(V, 'yield-strength')
    yield_strength = float(yield_strength) if yield_strength is not None else None

    return {
        'earth_radius': float(g(D, 'earth-radius')),
        'interface_depth_km': interface_depth_km,
        'type_linear': override_type_linear if override_type_linear is not None else int(g(V, 'type-linear')),
        'temp_neutral': float(g(T, 'neutral')),
        'T_min_K': float(g(T, 'temperature-min-K')),
        'T_diff_K': float(g(T, 'temperature-difference-K')),
        'visc_min': float(g(V, 'min')),
        'visc_max': float(g(V, 'max')),
        'eta_ref': float(g(V, 'representative-Pas')),
        'um_scaling': float(g(V, 'upper-mantle-scaling')),
        'um_E': float(g(V, 'upper-mantle-arrhenius-activation-energy')),
        'um_Va': float(g(V, 'upper-mantle-arrhenius-activation-volume')),
        'lm_scaling': float(g(V, 'lower-mantle-scaling')),
        'lm_E': float(g(V, 'lower-mantle-arrhenius-activation-energy')),
        'lm_Va': float(g(V, 'lower-mantle-arrhenius-activation-volume')),
        'stress_exponent': stress_exponent,
        'yield_strength': yield_strength,
        'has_srw': type_nonlinear in (0, 2) and stress_exponent >= 1.0,
        'has_yld': type_nonlinear in (1, 2) and (yield_strength is not None and yield_strength > 0),
    }


def load_params(ini_path):
    ini = read_ini(ini_path)
    return resolve_params(ini.get('Viscosity', {}), ini.get('Domain', {}), ini.get('Temperature', {}))


def manual_params(overrides):
    """overrides: flat dict using the same key names as DEFAULTS, e.g.
    {'upper-mantle-scaling': 1e6, 'stress-exponent': 3.0, 'yield-strength': 5e7}.
    No ini file needed -- unset keys fall back to DEFAULTS. Since Viscosity/
    Domain/Temperature key names never collide, the same dict can stand in
    for all three sections."""
    return resolve_params(overrides, overrides, overrides)


### models to compare: from .ini files, and/or manually typed-in parameters
ini_dir = "/Users/chingchen/pCloud Drive/Rhea_snapshot/anvil/scratch/26summer"
run_ids = ['s040101', 's040102', 's040121', 's040122'] 
models = [(load_params(os.path.join(ini_dir, f'NC_{r}_options.ini')), r) for r in run_ids]

# manual parameter sets -
manual_models = [
    (manual_params({
        'type-linear': 4,                                  
        'type-nonlinear': 4, 
        'upper-mantle-scaling': 3.8e-7,
        'upper-mantle-arrhenius-activation-energy': 6.4e4,
        'upper-mantle-arrhenius-activation-volume': 0.0,
        'lower-mantle-scaling': 4.0e5,
        'lower-mantle-arrhenius-activation-energy': 17.5,
        'lower-mantle-arrhenius-activation-volume': 0.0,
        'stress-exponent': 3.0,
        'yield-strength': 5.28e7,
    }), 'manual_test1'),
    (manual_params({
        'type-linear': 2,
        'type-nonlinear': 2, 
        'upper-mantle-scaling': 1e8,
        'upper-mantle-arrhenius-activation-energy': 36,
        'upper-mantle-arrhenius-activation-volume': 0.0,
        'lower-mantle-scaling': 4.0e5,
        'lower-mantle-arrhenius-activation-energy': 17.5,
        'lower-mantle-arrhenius-activation-volume': 0.0,
        'stress-exponent': 3.0,
        'yield-strength': 5.28e7,
    }), 'manual_test2'),
]

models = models + manual_models

fontsize = 15
bwidth = 2
depths = np.linspace(0, full_mantle_depth_km, 500)

fig, ax = plt.subplots(1, 1, figsize=(8, 10))

for i, (p, label) in enumerate(models):
    R_gas = DEFAULTS['gas-constant-J-Kmol']
    if p['type_linear'] == 2:                     # E is nondim -> real E = E * R * dT
        um_E_kJmol = p['um_E'] * R_gas * p['T_diff_K'] / 1e3
        lm_E_kJmol = p['lm_E'] * R_gas * p['T_diff_K'] / 1e3
    elif p['type_linear'] == 4:                    # E is already E/R in Kelvin -> real E = E * R
        um_E_kJmol = p['um_E'] * R_gas / 1e3
        lm_E_kJmol = p['lm_E'] * R_gas / 1e3
    else:
        um_E_kJmol = lm_E_kJmol = np.nan

    yield_MPa = np.nan
    if p['yield_strength'] is not None:
        yield_Pa = p['yield_strength'] * p['eta_ref'] * kappa / p['earth_radius']**2
        yield_MPa = yield_Pa / 1e6

    print('------------------------------')
    print(label)
    print(f'  upper mantle activation energy = {um_E_kJmol:.3f} kJ/mol')
    print(f'  lower mantle activation energy = {lm_E_kJmol:.3f} kJ/mol')
    print(f'  yield stress                   = {yield_MPa:.3f} MPa')

    T_nd = get_hsc_temp_nd(depths, age_myr, p['temp_neutral'])

    srw_list, yield_list, final_list = [], [], []
    for d, t in zip(depths, T_nd):
        s, y, f = get_viscosity(t, d, p, strain_rate)
        srw_list.append(s)
        yield_list.append(y)
        final_list.append(f)

    c_final = COLORS[i % len(COLORS)]
    ax.semilogx(final_list, depths, '-', color=c_final, lw=2.5, label=label)

    avg_um = np.mean(np.array(final_list)[(depths > 200) & (depths < 600)])
    avg_lm = np.mean(np.array(final_list)[(depths > 1200) & (depths < 2600)])
    print(f'  upper mantle viscosity         = {avg_um:.4e} Pa s')
    print(f'  lower mantle viscosity         = {avg_lm:.4e} Pa s')
    print(f'  ratio UM/LM                    = {avg_um / avg_lm:.3e}')

ax.grid()
ax.axhline(660, color='gray', linestyle=':', label='660 km')
ax.set_xlim(1e18, 1e24)
ax.set_ylim(full_mantle_depth_km, 0)
ax.set_xlabel('Viscosity (Pa s)', fontsize=fontsize)
ax.set_ylabel('Depth (km)', fontsize=fontsize)
ax.tick_params(axis='both', labelsize=fontsize, width=bwidth, length=8,
                direction='in', top=True, right=True)
for spine in ax.spines.values():
    spine.set_linewidth(bwidth)
ax.legend(fontsize=fontsize - 2, loc='lower left')
fig.tight_layout()

