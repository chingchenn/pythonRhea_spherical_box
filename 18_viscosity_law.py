#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 11:28:54 2026

@author: chingchen
"""

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

### parameters setup
eta_ref = 1.0e20          # Pa·s 
b = 6371e3              
kappa = 1e-6             
R_air = 8.314
t_scale = b**2 / kappa   
fontsize=15

tm = 1400.0
ts = 0.0
temp_neutral=1

delta_T = 1400 # K
sec_per_myr = 1e6 * 365.25 * 24 * 3600.0
strain_rate = 1e-15

# input parameters
input_upper_mantle_scaling = 1.0e7  # 5e6
input_upper_mantle_arrhenius_activation_energy = 36
input_lower_mantle_scaling = 90 #
input_lower_mantle_arrhenius_activation_energy = 17.2

input_stress_exponent = 3.0
input_yield_strength  = 5.28e7

### visccosity calculation


real_upper_mantle_arrhenius_activation_energy = input_upper_mantle_arrhenius_activation_energy * R_air * delta_T 

real_lower_mantle_arrhenius_activation_energy = input_lower_mantle_arrhenius_activation_energy * R_air * delta_T 
real_yield_strength = input_yield_strength * eta_ref * kappa / b**2


print('------------------------------')
print('upper mantle activation energy')
print(f'{real_upper_mantle_arrhenius_activation_energy:e}', 'J/mol')
print('------------------------------')
print('lower mantle activation energy',)
print(f'{real_lower_mantle_arrhenius_activation_energy:e}', 'J/mol')
print('------------------------------')
print('yield stress',)
print(f'{real_yield_strength:e}', 'Pa')
print('------------------------------')
def get_rhea_viscosity_v2(T, depth_km):
    sr_dim = strain_rate * (6371000**2 / 1e-6) 
    # print(sr_dim)
    is_um = depth_km < 660
    n = input_stress_exponent if is_um else 1.0
    

    A_eff = pow(input_upper_mantle_scaling if is_um else input_lower_mantle_scaling, 1.0/n)
    E_eff = (input_upper_mantle_arrhenius_activation_energy if is_um else input_lower_mantle_arrhenius_activation_energy) / n
    


    eta_lin = A_eff * np.exp(E_eff * (temp_neutral - T)) 
    eta_srw = eta_lin * pow(sr_dim, (1.0/n) - 1.0)
    eta_srw = np.minimum(eta_srw, 1e4)
    eta_yield = input_yield_strength/ (2.0 * sr_dim)
    viscosity = np.minimum(eta_srw, eta_yield)
    
    viscosity = np.clip(viscosity, 1.0e-2, 1.0e+4)
    
    return eta_srw * eta_ref, eta_yield*eta_ref, viscosity*eta_ref



def get_hsc_temp_nd(depth_km, age_myr):
    age_s = age_myr * sec_per_myr
    depth_m = depth_km * 1000.0
    T_phys = ts + (tm - ts) * erf(depth_m / (2.0 * np.sqrt(kappa * age_s)))
    
    return T_phys / tm



fig, ax = plt.subplots(1, 1, figsize=(8, 10))
depths = np.linspace(0, 2890, 500)
T1_nd = get_hsc_temp_nd(depths, 80.0)  # 80 Ma

srw_list = []
yield_list = []
final_list = []

for d, t in zip(depths, T1_nd):
    s, y, f = get_rhea_viscosity_v2(t, d)
    srw_list.append(s)
    yield_list.append(y)
    final_list.append(f)

ax.semilogx(srw_list, depths, 'r--', alpha=0.6, label='$\eta_{srw}$ (Non-linear Flow)')
ax.semilogx(yield_list, depths, 'b--', alpha=0.6, label='$\eta_{yield}$ (Plastic Limit)')
ax.semilogx(final_list, depths, 'k-', linewidth=2.5, label='Final Viscosity (Selected)')
ax.grid()
ax.axhline(660, color='gray', linestyle=':', label='660km Discontinuity')
ax.set_xlim(1e18, 1e24)
ax.set_ylim(2890,0)
ax.tick_params(axis='both', labelsize=fontsize,width=2, length=8, direction='in',top=True, right=True)

for spine in ax.spines.values():
    spine.set_linewidth(2)

ax2 = ax.twiny()
T_phys = T1_nd * tm 
ax2.plot(T_phys, depths, 'g-', linewidth=2, label='Temperature (K)', alpha=0.8)


ax2.set_xlabel('Temperature (K)', fontsize=fontsize, color='green')
ax2.set_xlim(0, 1500) 
ax2.tick_params(axis='x', labelsize=fontsize, labelcolor='green', width=2, length=8, direction='in')
ax.legend(fontsize=15,loc='lower left', )

avg_um = np.mean(np.array(final_list)[(depths > 200) & (depths < 600)])
avg_lm = np.mean(np.array(final_list)[(depths > 1200) & (depths < 2600)])
print('upper mantle viscosity = ', f'{avg_um:.4e}')
print('lower mantle viscosity = ', f'{avg_lm:.4e}')
print('------------------------------')
print('ratio UM/LM')
print(f'{avg_um/avg_lm:.3e}')
# print('------------------------------')
