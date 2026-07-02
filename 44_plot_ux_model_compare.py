#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:49:02 2026

@author: chingchen
"""

import numpy as np
import matplotlib.pyplot as plt
import function_rhea_analysis as fra
import pyvista as pv


PATH = '/Users/chingchen/Desktop/Rhea/rhea_model/26spring/'
PATH = '/Users/chingchen/Desktop/Rhea/rhea_model/26summer/'
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
fig, axes = plt.subplots(1, 1, figsize=(12, 5),
                              sharex=True,
                              gridspec_kw={'hspace': 0.08})


model_list = ['NC_s040122','NC_s040121','NC_s040102','NC_s040101']
for oo, model_number in enumerate(model_list):

    mesh_sec = pv.read(f'{PATH}{model_number}_solution_secondary.pvtu')
    mesh_sec = pv.read(f'{PATH}{model_number}_solution_secondary.pvtu')
    y_mid, th_mid, th_min, th_max, r_min, r_max, _, _ = fra.get_mesh_midplane(mesh_sec)
    
    x_km, vt_surf, vr_surf = fra.get_surface_velocity(
        model_number, th_min, th_max, path=PATH, scale_mm_yr=True)
    
    x_km_rel = x_km - th_mid * fra.R_EARTH / 1000.0  
    
    
    print(f"{'='*50}")
    print(f"Model: {model_number}")
    print(f"  Vt (horizontal): max={np.nanmax(np.abs(vt_surf)):.2f}  "
          f"mean={np.nanmean(np.abs(vt_surf)):.2f} mm/yr")
    print(f"  Vr (vertical):   max={np.nanmax(np.abs(vr_surf)):.2f}  "
          f"mean={np.nanmean(np.abs(vr_surf)):.2f} mm/yr")
    
    
    # ── plot ─────────────────────────────────────────────────────────────────────
    
    ax0 = axes
    ax0.plot(x_km_rel, vt_surf, color=COLORS[oo], lw=1.8, label = f'{model_number}')
    ax0.set_ylabel('$V_t$ (mm/yr)', fontsize=11)
    ax0.legend(loc='upper right', fontsize=10)
    ax0.grid(alpha=0.3)
    ax0.set_title(f'{model_number}  —  surface velocity comparison', fontsize=12)

ax0.set_xlim(x_km_rel.min(), x_km_rel.max())
