#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:57:02 2026

@author: chingchen
"""


import time
start = time.perf_counter()


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.interpolate import interp1d
model = 'NC_060030'
model_vol = 'NC_060030'
slab_profile='100_slab_distance'

model = 'IB_s040020'
model_vol = 'NC_060030'
slab_profile='039_slab_distance'
fontsize=20

# --------------------- parameters ---------------------
R_model_km = 6371.0          
mantle_depth_km = 660.0
subx = 128
subz = 32
factor = 0.125
theta_half = np.pi/8 * (subx/subz) * factor   # = 0.1963 rad = 11.25°

theta_min = -theta_half   # -11.25°
theta_max = +theta_half   # +11.25°


trench_s_km = 1200.0



slab_age_myr = 140.0       # trench age
op_age_myr   = 50.0      # overriding plate age

r_nd_top    = 1.0
r_nd_bottom = 1.0 - mantle_depth_km / R_model_km

# domain length
domain_total_km = 2 * theta_half * R_model_km
print(f"Domain total: {domain_total_km:.1f} km")  # 應該是 2500 km

kappa = 1e-6
tm = 1400.0
ts = 0.0


# ---------------------mesh for temperature ---------------------
nr, nth = 1001, 1001
r_nd = np.linspace(r_nd_bottom, 1.0, nr)     
theta = np.linspace(theta_min, theta_max, nth)
TH, RND = np.meshgrid(theta, r_nd) 

# spherical to Carti
S_km = (TH - theta_min) * R_model_km
Depth_km = (1.0 - RND) * R_model_km
Depth_m  = Depth_km * 1000.0

# slab 
cutoff_depth_km = 440   # = 440 km
mask_um = Depth_km <= cutoff_depth_km 

# --------------------- surface age ---------------------
v_cm_yr = 2.0
sec_per_myr = 1e6 * 365.25 * 24 * 3600.0
v_m_s = (v_cm_yr / 100.0) / (365.25 * 24 * 3600.0)

age_surf_myr = np.zeros_like(S_km)
age_surf_myr[S_km <  trench_s_km] = slab_age_myr
age_surf_myr[S_km >= trench_s_km] = op_age_myr
AGE_s = age_surf_myr * sec_per_myr


T = ts + (tm - ts) * erf(Depth_m / (2.0 * np.sqrt(kappa * AGE_s)))





# --------------------- slab temperature  ---------------------
slab_data = np.loadtxt(f'/Users/chingchen/Desktop/Rhea/rhea_process_input/{slab_profile}.txt')


slab_x_raw = slab_data[:, 0] # km, from trench
slab_z_raw = np.abs(slab_data[:, 1]) # km depth positive



if slab_z_raw[0] > 0:
    slope = (slab_x_raw[1] - slab_x_raw[0]) / (slab_z_raw[1] - slab_z_raw[0])
    x_at_z0 = slab_x_raw[0] - slope * slab_z_raw[0]
    slab_x_km = np.insert(slab_x_raw, 0, x_at_z0)
    slab_z_km = np.insert(slab_z_raw, 0, 0.0)
else:
    slab_x_km = slab_x_raw
    slab_z_km = slab_z_raw



# polyline (m)
slab_x = slab_x_km * 1000.0
slab_z = slab_z_km * 1000.0

# slab arc to subducting time (s)
dx = np.diff(slab_x)
dz = np.diff(slab_z)
ds = np.sqrt(dx*dx + dz*dz)
cum_s = np.concatenate(([0.0], np.cumsum(ds)))
t_sub = cum_s / v_m_s
f_t_sub = interp1d(slab_z, t_sub, bounds_error=False, fill_value=np.nan)

# trench age 
age_at_trench_s = slab_age_myr * sec_per_myr 

# x_local = (S_km - trench_s_km) * 1000 (m)
# z = Depth_m (m)
x_local = (S_km - trench_s_km) * 1000.0
z_local = Depth_m

# polyline segments
x0, z0 = slab_x[:-1], slab_z[:-1]
x1, z1 = slab_x[1:],  slab_z[1:]
seg_dx = x1 - x0
seg_dz = z1 - z0
seg_len2 = seg_dx*seg_dx + seg_dz*seg_dz + 1e-30
seg_len = np.sqrt(seg_len2)

T_slab = T.copy()

for i in range(nr):
    zp = z_local[i, :]
    if np.all(~mask_um[i, :]):
        continue

    xp = x_local[i, :]

    #  x_local >= 0
    active = (xp >= 0.0) & mask_um[i, :]
    if not np.any(active):
        continue

    best_d = np.full_like(xp, np.inf, dtype=float)
    best_xproj = np.zeros_like(xp, dtype=float)
    best_zproj = np.zeros_like(xp, dtype=float)
    best_tx = np.zeros_like(xp, dtype=float)
    best_tz = np.zeros_like(xp, dtype=float)

    for j in range(len(x0)):
        t = ((xp - x0[j]) * seg_dx[j] + (zp - z0[j]) * seg_dz[j]) / seg_len2[j]
        t = np.clip(t, 0.0, 1.0)
        xproj = x0[j] + t * seg_dx[j]
        zproj = z0[j] + t * seg_dz[j]
        d = np.sqrt((xp - xproj)**2 + (zp - zproj)**2)

        tx = seg_dx[j] / seg_len[j]
        tz = seg_dz[j] / seg_len[j]

        m = d < best_d
        best_d[m] = d[m]
        best_xproj[m] = xproj[m]
        best_zproj[m] = zproj[m]
        best_tx[m] = tx
        best_tz[m] = tz

    # normal vector (pointing into the slab interior)
    nx1, nz1 = -best_tz, best_tx
    nx2, nz2 =  best_tz, -best_tx

    rx = xp - best_xproj
    rz = zp - best_zproj

    # slab interior
    use1 = nx1 < nx2
    nx  = np.where(use1, nx1, nx2)
    nzv = np.where(use1, nz1, nz2)
    
    dperp = rx * nx + rz * nzv
    dperp = np.maximum(dperp, 0.0)

    # subducting slab age (sinking velocity)
    t_total = age_at_trench_s + f_t_sub(best_zproj)

    # thermal thickness consistent with slab age (instead of fixed 70 km)
    thermal_thickness = 2.32 * np.sqrt(kappa * t_total)   # m, per point
    inside = active & (dperp > 0.0) & (dperp <= thermal_thickness)

    temp_slab = ts + (tm - ts) * erf(dperp / (2.0 * np.sqrt(kappa * t_total)))

    T_slab[i, inside] = np.minimum(T_slab[i, inside], temp_slab[inside])
    
print("T_slab min:", T_slab.min())
print("T_slab max:", T_slab.max())
print("difference (T - T_slab) max:", np.abs(T - T_slab).max())
inside_count = np.sum(T_slab < T)
print("pixels modified by slab:", inside_count)
T = T_slab
from scipy.ndimage import gaussian_filter

T_final = gaussian_filter(T, sigma=3)

## ---------- weak zone -------------
z_slab_wz_km = np.linspace(0.0, 200.0, 500)  # depth from surface (km)

f_x_interp = interp1d(slab_z_km, slab_x_km, kind='linear', fill_value=np.nan)
x_slab_wz_km = f_x_interp(z_slab_wz_km)  # km, from trench
s_slab_wz_km = trench_s_km + x_slab_wz_km


theta_slab_wz = theta_min + (s_slab_wz_km / R_model_km)
r_nd_slab_wz  = 1.0 - (z_slab_wz_km / R_model_km)

x_nd_slab = r_nd_slab_wz * np.sin(theta_slab_wz)
z_nd_slab = r_nd_slab_wz * np.cos(theta_slab_wz)

y_levels = np.linspace(-0.00613588, 0.00613588, 50)
x_slab_3 = np.tile(x_nd_slab, len(y_levels))
z_slab_3 = np.tile(z_nd_slab, len(y_levels))
y_slab_3 = np.repeat(y_levels, len(x_nd_slab))

points_slab = np.column_stack([x_slab_3, y_slab_3, z_slab_3])
labels_slab = np.ones(points_slab.shape[0], dtype=int)  # slab = 1


print(len(points_slab))


##
## ---------- plotting -------------
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 12))

pcm = ax.pcolormesh(TH, RND, T_final, shading='auto', cmap='RdYlBu_r',vmin=0.0, vmax=1400.0)

ax.set_theta_zero_location("W")   # left
ax.set_theta_direction(-1)        # clockwise
ax.set_thetamin(-45)
ax.set_thetamax(45)

ax.set_rlim(0.0, 1.0)
ax.set_anchor('C')
ax.set_aspect('equal', adjustable='box')
theta_line = np.linspace(np.deg2rad(-45), np.deg2rad(45), 200)
ax.plot(theta_line, np.full_like(theta_line, 1.0), 'k', lw=1.5)
ax.plot(theta_line, np.full_like(theta_line, r_nd_bottom),'k', lw=1.5)

ax.set_theta_zero_location("N")

cax = fig.add_axes([0.92, 0.12, 0.035, 0.76])
cbar = fig.colorbar(pcm, cax=cax)
cbar.set_label("Temperature", fontsize=14)
cbar.ax.tick_params(labelsize=14)


ax.plot(theta_slab_wz,r_nd_slab_wz, linestyle='--', color='k', linewidth=1.5, label='Weak slab top')
ax.tick_params(axis='both', which='major', labelsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize)
plt.show()

from scipy.interpolate import RegularGridInterpolator

# 1D axes
r_nd_1d = r_nd
theta_1d = theta
interp_T = RegularGridInterpolator((r_nd_1d, theta_1d), T_final, method='linear', bounds_error=False, fill_value=np.nan)

coords = np.loadtxt(f'/Users/chingchen/Desktop/Rhea/rhea_process_input/{model_vol}_coordinates_vol.txt')
x, y, z = coords.T
r_nd_q = np.sqrt(x*x + y*y+ z*z)
theta_q = np.arctan2(y, x)   

mask_in = ((r_nd_q >= r_nd_1d.min()) & (r_nd_q <= r_nd_1d.max()) & (theta_q >= theta_1d.min()) & (theta_q <= theta_1d.max()))
Tq = np.full(coords.shape[0], np.nan)
Tq[mask_in] = interp_T(np.column_stack([r_nd_q[mask_in], theta_q[mask_in]]))
final_temp_column = Tq / 1400.0

points_all = points_slab
labels_all = labels_slab

print("slab weakzone points:", len(labels_slab))
np.savetxt(f'/Users/chingchen/Desktop/Rhea/rhea_process_input/{model}_weakzone_pointcloud.txt',points_all, fmt='%.6f')
np.savetxt(f'/Users/chingchen/Desktop/Rhea/rhea_process_input/{model}_weakzone_labels.txt',labels_all.astype(int), fmt='%d')
np.savetxt(f'/Users/chingchen/Desktop/Rhea/rhea_process_input/{model}_temperature.txt',final_temp_column, fmt='%.4f')
print(f"y 範圍: {y.min():.6f} ~ {y.max():.6f}")
## ---------- plotting -------------

X_flat_km = (TH - theta_min) * R_model_km 

Z_depth_km = (1.0 - RND) * R_model_km
x_weak_flat = (theta_slab_wz - theta_min) * R_model_km
z_weak_flat = (1.0 - r_nd_slab_wz) * R_model_km


fig, ax = plt.subplots(figsize=(10, 10)) 
pcm = ax.pcolormesh(X_flat_km, Z_depth_km,T_final,shading='auto', cmap='RdYlBu_r', vmin=0.0, vmax=1400.0)
ax.scatter(x_weak_flat, z_weak_flat,color='k', s=8, label='Weak slab top')
ax.axvline(x=trench_s_km, color='g', linestyle=':', label='Trench s-location')


ax.set_ylim(660, 0) 

ax.set_aspect('equal')
ax.set_xlabel('Distance s (km)', fontsize=fontsize)
ax.set_ylabel('Depth (km)', fontsize=fontsize)
ax.tick_params(labelsize=fontsize)

# Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
cbar.set_label("Temperature (°C)", fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize)



end = time.perf_counter()
print(f"{end - start:.2f} sec")