#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:45:08 2026

@author: chingchen

Row 1 (top)    : dynamic topography (raw + smoothed)
Row 2 (middle) : surface horizontal velocity u_x (= V_t, tangential)
Row 3 (bottom) : viscosity field (pcolormesh) + velocity quiver
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d
import function_rhea_analysis as fra


# ── parameters ───────────────────────────────────────────────────────────────
model_number = 'NC_s040121'
nx, nz       = 1001, 1001
r_min, r_max = 0.6, 1.0
vmin, vmax   = 1e18, 1e24
scale        = 20
depth        = 660
step_x = step_z = 40
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
# ── paths ────────────────────────────────────────────────────────────────────
PATH        = '/Users/chingchen/Desktop/Rhea/rhea_model/26summer/'
PATH_OUTPUT = '/Users/chingchen/Desktop/Rhea/rhea_process_input/'

# ── mesh ─────────────────────────────────────────────────────────────────────
mesh_sec = pv.read(f'{PATH}{model_number}_solution_secondary.pvtu')  # viscosity
mesh_pri = pv.read(f'{PATH}{model_number}_solution_primary.pvtu')    # velocity

y_mid, th_mid, th_min, th_max, r_min, r_max, mantle_depth_km, half_width_deg = \
    fra.get_mesh_midplane(mesh_sec)

cloud, TT, RR, X_km, Z_km = fra.build_sampling_cloud(
    th_mid, th_min, th_max, r_min, r_max, nx=nx, nz=nz, y_mid=y_mid)

# ── viscosity field ──────────────────────────────────────────────────────────
F = fra.sample_scalar_field(mesh_sec, cloud, 'viscosity', nz, nx)

# ── velocity field (for quiver) ──────────────────────────────────────────────
Vt, Vr = fra.sample_velocity_field(mesh_pri, cloud, TT, nz, nx)

# ── surface u_x (= V_t, horizontal/tangential) ───────────────────────────────
x_ux_km, vt_surf, vr_surf = fra.get_surface_velocity(
    model_number, th_min, th_max, path=PATH, scale_mm_yr=True)
x_ux_km_rel = x_ux_km - th_mid * fra.R_EARTH / 1000.0

# ── dynamic topography (+ gravity-based smoothing, as in 22) ────────────────
fra.compute_dynamic_topography(model_number, path=PATH, savepath=PATH_OUTPUT, save=True)
x_topo_km, h_km = fra.load_topography(
    model_number, savepath=PATH_OUTPUT, theta_min=th_min, theta_max=th_max, th_mid=th_mid)

xg, hg, g_mgal = fra.topo_to_gravity_fft(
    x_topo_km, h_km, delta_rho=1670.0, z_obs_km=0.0, dx_km=2.0, smooth_sigma=0)

mask = (xg > -10000) & (xg < -800)   # ← reference window, change as needed
h_ref = np.mean(hg[mask])
hg_shift = gaussian_filter1d(hg - h_ref, sigma=5)

# ── plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
gs  = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2.4], width_ratios=[39, 1],
                        hspace=0.08, wspace=0.002)

ax_topo = fig.add_subplot(gs[0, 0])
ax_ux   = fig.add_subplot(gs[1, 0], sharex=ax_topo)
ax      = fig.add_subplot(gs[2, 0], sharex=ax_topo)
ax_cb   = fig.add_subplot(gs[:, 1])

# ── row 1: topography ─────────────────────────────────────────────────────
ax_topo.set_title(model_number)
ax_topo.plot(x_topo_km, h_km, lw=2, color=COLORS[2], label='dyn_topo')
ax_topo.plot(xg, hg_shift, lw=1.5, color=COLORS[0], linestyle='dashed', label='smoothed')
ax_topo.axhline(0, color='gray', lw=0.8, linestyle='--')
ax_topo.set_ylabel("Topo (km)", fontsize=10)
ax_topo.grid(alpha=0.3)
ax_topo.legend(fontsize=9)
ax_topo.tick_params(axis='x', labelbottom=False)

# ── row 2: u_x (surface horizontal velocity) ─────────────────────────────
ax_ux.plot(x_ux_km_rel, vt_surf, lw=2, color=COLORS[4], label='$u_x$ ($V_t$, horizontal)')
ax_ux.axhline(0, color='gray', lw=0.8, linestyle='--')
ax_ux.set_ylabel("$u_x$\n(mm/yr)", fontsize=10)
ax_ux.grid(alpha=0.3)
ax_ux.legend(fontsize=9)
ax_ux.tick_params(axis='x', labelbottom=False)

# ── row 3: viscosity field + velocity quiver ─────────────────────────────
im = ax.pcolormesh(X_km, Z_km, F, shading='auto', cmap='jet_r',
                    norm=LogNorm(vmin=vmin, vmax=vmax), alpha=0.6)

Xs = X_km[::step_z, ::step_x]
Zs = Z_km[::step_z, ::step_x]
Us = Vt[::step_z, ::step_x]
Ws = (-Vr)[::step_z, ::step_x]
mask_q = np.isfinite(Us) & np.isfinite(Ws)
ax.quiver(Xs[mask_q], Zs[mask_q], Us[mask_q], Ws[mask_q],
          angles='xy', scale_units='xy', scale=scale)

ax.set_ylim(depth, 0)
ax.set_xlim(X_km.min(), X_km.max())
ax.set_xlabel("Distance along surface (km)")
ax.set_ylabel("Depth (km)")
ax.set_aspect('equal', adjustable='box')

# ── colorbar (viscosity, shared right column) ────────────────────────────
cb = fig.colorbar(im, cax=ax_cb)
cb.set_label("Viscosity (Pa·s)", fontsize=11)

# ── align x-limits and layout ─────────────────
ax_topo.set_xlim(X_km.min(), X_km.max())
ax_ux.set_xlim(X_km.min(), X_km.max())

fig.canvas.draw()
pos_ax   = ax.get_position()
pos_topo = ax_topo.get_position()
pos_ux   = ax_ux.get_position()

ax_topo.set_position([pos_ax.x0, pos_topo.y0, pos_ax.width, pos_topo.height])
ax_ux.set_position([pos_ax.x0, pos_ux.y0, pos_ax.width, pos_ux.height])

# secondary top axis in degrees, on the topmost panel
km_to_deg = lambda x: x / (fra.R_EARTH / 1000.0) * (180 / np.pi)
deg_to_km = lambda d: d * (fra.R_EARTH / 1000.0) * (np.pi / 180)
ax_topo_top = ax_topo.secondary_xaxis('top', functions=(km_to_deg, deg_to_km))
ax_topo_top.set_xlabel("Distance (°)", fontsize=10)

pos_cb = ax_cb.get_position()
ax_cb.set_position([pos_ax.x0 + pos_ax.width + 0.01,
                     pos_ax.y0,
                     pos_cb.width,
                     pos_ax.height + pos_ux.height + pos_topo.height + 0.08])

# plt.savefig(f'{PATH_OUTPUT}{model_number}_topo_ux_viscosity.png', dpi=150, bbox_inches='tight')
# plt.show()