#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 22:50:49 2026

@author: chingchen
"""



from function_rhea_analysis import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

model_bottom = 800.0
model_number = 'NC_s050048'
r_min = 1.0 - model_bottom/6371.0   # upper mantle 


mesh = pv.read(f'{PATH_MODEL}/26summer/{model_number}_input.pvtu')


# 
y_mid, th_mid, th_min, th_max, r_min, r_max, mantle_depth_km, half_width_deg = get_mesh_midplane(mesh)

cloud, TT, RR, X_km, Z_km = build_sampling_cloud(
    th_mid, 
    th_min,   
    th_max,   
    r_min, r_max,
    nx=1001, nz=1001, y_mid=y_mid)

# 
T = sample_scalar_field(mesh, cloud, 'viscosity', nz=1001, nx=1001)
#T = sample_scalar_field(mesh, cloud, 'temperature', nz=1001, nx=1001)


fig, ax = plt.subplots(figsize=(8, 5))

#im = ax.pcolormesh(X_km, Z_km, T , shading='auto', cmap='jet',vmin=0, vmax=1400,alpha=0.6)
im = ax.pcolormesh(X_km, Z_km, np.log10(T) , shading='auto', cmap='jet_r',vmin=18, vmax=24,alpha=0.6)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
fig.colorbar(im, cax=cax, label='temperature')
ax.set_ylim(model_bottom,0)
# ax.set_xlim(0,2000)
ax.set_aspect('equal')
ax.set_xlabel("Distance along surface (km)")
ax.set_ylabel("Depth (km)")
ax.set_title(f'{model_number}')

