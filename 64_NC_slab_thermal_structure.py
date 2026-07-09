#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:45:39 2026

@author: chingchen
"""


import time
start = time.perf_counter()


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.interpolate import interp1d
model = 'NC_s050202'
model_vol = 'NC_s050201'
slab_profile='100_slab_distance'

fontsize=20

# --------------------- parameters ---------------------
R_model_km = 6371.0   
 
#### NC_060030:      
#mantle_depth_km = 660.0
#subx = 128
#subz = 32
#factor = 0.125
#ridge_s_km  = 0.0                              # s = 0 km
#trench_s_km = 1200.0                           # s = 1500 km
#cutoff_depth_km = 560   # = 440 km

#### NC_s050201:
mantle_depth_km = 800.0
subx = 160    
subz = 40   
factor = 0.159879                 
ridge_s_km  = 100.0               # was 0.0  -> ridge 內縮 100 km
trench_s_km = 1300.0              
cutoff_depth_km = mantle_depth_km      


theta_half = np.pi/8 * (subx/subz) * factor   # = 0.1963 rad = 11.25°

theta_min = -theta_half   # -11.25°
theta_max = +theta_half   # +11.25°


slab_age_myr = 65.0       # trench 處板塊年齡
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


mask_um = Depth_km <= cutoff_depth_km 

# --------------------- surface age ---------------------

# velocity (for slab age and sinking time)
v_cm_yr      = 2.0       
sec_per_myr = 1e6 * 365.25 * 24 * 3600.0
v_m_s = (v_cm_yr / 100.0) / (365.25 * 24 * 3600.0)


age_surf_myr = np.zeros_like(S_km)

# (a) 0-3000 km：150 Ma
#left_mask = S_km < ridge_s_km
#age_left_0km = 150.0
#age_at_ridge = 0.5

#age_surf_myr[left_mask] = age_left_0km + (age_at_ridge - age_left_0km) * (S_km[left_mask] / ridge_s_km)


age_at_ridge = 0.5
left_mask = S_km < ridge_s_km

spread_km_per_myr = (trench_s_km - ridge_s_km) / (slab_age_myr - age_at_ridge)
age_surf_myr[left_mask] = age_at_ridge + \
    (ridge_s_km - S_km[left_mask]) / spread_km_per_myr
    

# (b) 3000-7000 km： ridge to trench by using half-space cooling where age  = distance/velocity
target_trench_age_myr = slab_age_myr

mask_ocean = (S_km >= ridge_s_km) & (S_km < trench_s_km)
frac = (S_km[mask_ocean] - ridge_s_km) / (trench_s_km - ridge_s_km)
age_surf_myr[mask_ocean] = age_at_ridge + frac * (target_trench_age_myr - age_at_ridge)


# (c) 7000-10000 km 130 Ma 
age_surf_myr[S_km >= trench_s_km] = 130.0
AGE_s = age_surf_myr * sec_per_myr  # s


T = ts + (tm - ts) * erf(Depth_m / (2.0 * np.sqrt(kappa * AGE_s)))



coords = np.loadtxt(f'/Users/chingchen/Desktop/Rhea/rhea_process_input/{model_vol}_coordinates_vol.txt')
x, y, z = coords.T
print(f"y range from coords: {y.min():.8f} ~ {y.max():.8f}")

y_levels = np.linspace(z.min(), z.max(), 50) 

# --------------------- slab temperature  ---------------------
slab_data = np.loadtxt(f'/Users/chingchen/Desktop/Rhea/rhea_process_input/{slab_profile}.txt')


scale = R_model_km / 6371

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

print(f"slab profile: x_max = {slab_x_km.max():.1f} km, z_max = {slab_z_km.max():.1f} km")
print(f"slab tip at s = {trench_s_km + slab_x_km.max():.1f} km "
      f"(domain right = {domain_total_km:.1f} km)")

# slab arc to subducting time (s)
dx = np.diff(slab_x)
dz = np.diff(slab_z)
ds = np.sqrt(dx*dx + dz*dz)
cum_s = np.concatenate(([0.0], np.cumsum(ds)))
t_sub = cum_s / v_m_s
f_t_sub = interp1d(slab_z, t_sub, bounds_error=False, fill_value=np.nan)

# trench age 
# age_at_trench_myr = float(np.maximum((trench_s_km - ridge_s_km) * 1000.0 / v_m_s / sec_per_myr, 0.1))
age_at_trench_s = slab_age_myr * sec_per_myr 



# slab at upper mantle 
mask_um = Depth_km <= cutoff_depth_km

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

    # normal vector
    nx1, nz1 = -best_tz, best_tx
    nx2, nz2 =  best_tz, -best_tx

    rx = xp - best_xproj
    rz = zp - best_zproj

    dperp1 = rx * nx1 + rz * nz1
    dperp2 = rx * nx2 + rz * nz2

    # inner slab direction
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
    inside = active & (dperp > 0.0)
    temp_slab = ts + (tm - ts) * erf(best_d / (2.0 * np.sqrt(kappa * t_total)))

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


x_slab_3 = np.tile(x_nd_slab, len(y_levels))
z_slab_3 = np.tile(z_nd_slab, len(y_levels))
y_slab_3 = np.repeat(y_levels, len(x_nd_slab))

points_slab = np.column_stack([x_slab_3, y_slab_3, z_slab_3])
labels_slab = np.ones(points_slab.shape[0], dtype=int)  # slab = 1


print(len(points_slab))

s0_km = ridge_s_km

z_ridge_max_km = 40.0     # triangle depth
W_top_km = 5.0           # total width at surface (km)
W_bot_km = 100.0          # total width at z_max (km)

w_top = 0.5 * W_top_km    # half-width
w_bot = 0.5 * W_bot_km

nz = 180
z_ridge_km = np.linspace(0.0, z_ridge_max_km, nz)

half_w_km = w_top + (w_bot - w_top) * (z_ridge_km / z_ridge_max_km)

ns_max = 450
s_ridge_list = []
z_ridge_list = []
for zk, hw in zip(z_ridge_km, half_w_km):
    ns = max(7, int(ns_max * (hw / w_bot)))
    s_row = np.linspace(s0_km - hw, s0_km + hw, ns)
    s_ridge_list.append(s_row)
    z_ridge_list.append(np.full_like(s_row, zk))

s_ridge_wz_km = np.concatenate(s_ridge_list)
z_ridge_wz_km = np.concatenate(z_ridge_list)

# convert (s, depth) -> (theta, r_nd)
theta_ridge_wz = theta_min + (s_ridge_wz_km / R_model_km)
r_nd_ridge_wz  = 1.0 - (z_ridge_wz_km / R_model_km)

# to nondim Cartesian in X-Z plane
x_nd_ridge = r_nd_ridge_wz * np.sin(theta_ridge_wz)
z_nd_ridge = r_nd_ridge_wz * np.cos(theta_ridge_wz)

x_ridge_3 = np.tile(x_nd_ridge, len(y_levels))
z_ridge_3 = np.tile(z_nd_ridge, len(y_levels))
y_ridge_3 = np.repeat(y_levels, len(x_nd_ridge))

points_ridge = np.column_stack([x_ridge_3, y_ridge_3, z_ridge_3])
labels_ridge = np.full(points_ridge.shape[0], 2, dtype=int)  # ridge = 2

print(len(points_ridge))
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
ax.plot(theta_line, np.full_like(theta_line, 0.8964), linestyle='dashed',c='k', lw=1.5,label='660 km')

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
import numpy as np

# 1D axes
r_nd_1d = r_nd
theta_1d = theta
interp_T = RegularGridInterpolator((r_nd_1d, theta_1d), T_final, method='linear', bounds_error=False, fill_value=np.nan)

#coords = np.loadtxt(f'/Users/chingchen/Desktop/Rhea/rhea_process_input/{model_vol}_coordinates_vol.txt')
#x, y, z = coords.T
r_nd_q = np.sqrt(x*x + y*y+ z*z)
theta_q = np.arctan2(y, x)   

r_nd_q_scaled = r_nd_q * (R_model_km / 6371.0)
#mask_in = ((r_nd_q >= r_nd_1d.min()) & (r_nd_q <= r_nd_1d.max()) & (theta_q >= theta_1d.min()) & (theta_q <= theta_1d.max()))
#Tq = np.full(coords.shape[0], np.nan)
#Tq[mask_in] = interp_T(np.column_stack([r_nd_q[mask_in], theta_q[mask_in]]))

r_q  = np.clip(r_nd_q,  r_nd_1d.min(),  r_nd_1d.max())
th_q = np.clip(theta_q, theta_1d.min(), theta_1d.max())
Tq = interp_T(np.column_stack([r_q, th_q]))
final_temp_column = Tq / 1400.0
print(f"NaN after clip: {np.isnan(final_temp_column).sum()}")


points_all = np.vstack([points_slab, points_ridge])
labels_all = np.concatenate([labels_slab, labels_ridge])

print("slab weakzone points:", len(labels_slab))
print("ridge weakzone points:", len(labels_ridge))
print("total weakzone points:", len(labels_all))
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
ax.scatter(x_weak_flat, z_weak_flat,color='k', label='Weak slab top')
ax.axvline(x=trench_s_km, color='g', linestyle=':', label='Trench s-location')


ax.set_ylim(800, 0) 

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