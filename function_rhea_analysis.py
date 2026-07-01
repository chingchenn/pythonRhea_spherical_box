#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:25:29 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function_rhea_analysis.py — Rhea post-processing toolkit

Sections:
1. Geometry        - mid-plane, theta windows
2. Field sampling   - scalar/vector interpolation onto a regular grid
3. Topography       - dynamic topography from stress_norm
4. Gravity          - FFT free-air gravity from topography
5. Surface velocity - tangential/radial velocity at the surface
6. Solver diagnostics - nonlinear/linear residuals from log files
"""

import re
import glob
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter1d

# ── physical constants ────────────────────────────────────────────
R_EARTH    = 6371.0e3   # m
KAPPA      = 1.0e-6     # m^2/s, thermal diffusivity
RHO        = 3300.0     # kg/m^3
G_GRAV     = 9.81       # m/s^2
G_NEWTON   = 6.67430e-11
SEC_PER_YR = 365.25 * 24 * 3600

# ── default paths (override or pass explicitly) ──────────────────
PATH_MODEL  = '/Users/chingchen/Desktop/Rhea/rhea_model/'
PATH_OUTPUT = '/Users/chingchen/Desktop/Rhea/rhea_process_input/'
PATH_LOG    = '/Users/chingchen/pCloud Drive/Rhea_snapshot/anvil/scratch/26spring/archives/'


# 1. GEOMETRY

def get_mesh_midplane(mesh, R_earth=R_EARTH):
    """Return y_mid, th_mid, th_min, th_max, r_min, r_max, mantle_depth_km, half_width_deg."""
    _, _, ymin, ymax, _, _ = mesh.bounds
    y_mid = 0.5 * (ymin + ymax)

    pts = mesh.points
    theta = np.arctan2(pts[:, 0], pts[:, 2])
    th_min = theta.min()
    th_max = theta.max()
    th_mid = 0.5 * (th_min + th_max)

    r = np.linalg.norm(pts, axis=1)
    r_min = r.min()
    r_max = r.max()

    mantle_depth_km = (r_max - r_min) * R_earth / 1000.0
    half_width_deg = np.rad2deg(0.5 * (th_max - th_min))

    return y_mid, th_mid, th_min, th_max, r_min, r_max, mantle_depth_km, half_width_deg


def theta_window_from_deg(th_mid, th_min_mesh, th_max_mesh, target_deg, half_width_deg):
    """Theta window defined by an angular offset from th_mid (degrees), clipped to mesh range."""
    theta_min = th_mid + np.deg2rad(target_deg - half_width_deg)
    theta_max = th_mid + np.deg2rad(target_deg + half_width_deg)
    theta_min = max(theta_min, th_min_mesh)
    theta_max = min(theta_max, th_max_mesh)
    return theta_min, theta_max


def theta_window_from_km(th_mid, th_min_mesh, th_max_mesh, xmin_km, xmax_km):
    """Theta window defined by surface arc-length distance (km) from th_mid, clipped to mesh range."""
    theta_min = th_mid + (xmin_km * 1000.0) / R_EARTH
    theta_max = th_mid + (xmax_km * 1000.0) / R_EARTH
    theta_min = max(theta_min, th_min_mesh)
    theta_max = min(theta_max, th_max_mesh)
    return theta_min, theta_max


def build_sampling_cloud(th_mid, theta_min, theta_max, r_min, r_max, nx, nz, y_mid):
    """
    Build a regular (theta, r) sampling grid and convert to 3D Cartesian points.

    Returns cloud (pv.PolyData), TT, RR, X_km, Z_km (all shape (nz, nx)).
    """
    theta_lin = np.linspace(theta_min, theta_max, nx)
    r_lin = np.linspace(r_min, r_max, nz)
    TT, RR = np.meshgrid(theta_lin, r_lin)

    XX = RR * np.sin(TT)
    ZZ = RR * np.cos(TT)
    YY = np.full_like(XX, y_mid)

    cloud = pv.PolyData(np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]))
    X_km = (TT - th_mid) * R_EARTH / 1000.0
    Z_km = (1.0 - RR) * R_EARTH / 1000.0

    return cloud, TT, RR, X_km, Z_km



# 2. FIELD SAMPLING

def sample_scalar_field(mesh, cloud, field_name, nz, nx, trim=2):
    """Sample a scalar point-data field onto the cloud grid, NaN out invalid/boundary pixels."""
    samp = cloud.sample(mesh)
    F = np.asarray(samp.point_data[field_name], dtype=float)
    valid = np.asarray(samp.point_data['vtkValidPointMask']).astype(bool)
    F[~valid] = np.nan
    F = F.reshape(nz, nx)

    if trim > 0:
        F[:trim, :] = np.nan
        F[-trim:, :] = np.nan
        F[:, :trim] = np.nan
        F[:, -trim:] = np.nan

    return F


def sample_velocity_field(mesh, cloud, TT, nz, nx, trim=2):
    """Sample velocity onto the cloud grid, return tangential (Vt) and radial (Vr) components."""
    samp = cloud.sample(mesh)
    V = np.asarray(samp.point_data['velocity'])
    valid = np.asarray(samp.point_data['vtkValidPointMask']).astype(bool).reshape(nz, nx)

    vx = V[:, 0].reshape(nz, nx)
    vz = V[:, 2].reshape(nz, nx)

    sinT = np.sin(TT)
    cosT = np.cos(TT)

    Vr = vx * sinT + vz * cosT
    Vt = vx * cosT - vz * sinT

    Vr[~valid] = np.nan
    Vt[~valid] = np.nan

    if trim > 0:
        for arr in (Vt, Vr):
            arr[:trim, :] = np.nan
            arr[-trim:, :] = np.nan
            arr[:, :trim] = np.nan
            arr[:, -trim:] = np.nan

    return Vt, Vr


def get_surface_input_nodes(model_number, theta_min, theta_max, th_mid, path=PATH_MODEL):
    """Return (x_km, z_km) of surface nodes from the *_input.pvtu mesh, within a theta window."""
    mesh_input = pv.read(f"{path}{model_number}_input.pvtu")
    pts = mesh_input.points
    theta = np.arctan2(pts[:, 0], pts[:, 2])
    r = np.linalg.norm(pts, axis=1)

    win = (r > 0.9999) & (theta >= theta_min) & (theta <= theta_max)
    x_km = (theta[win] - th_mid) * R_EARTH / 1000.0
    z_km = (1.0 - r[win]) * R_EARTH / 1000.0

    idx = np.argsort(x_km)
    return x_km[idx], z_km[idx]



# 3. TOPOGRAPHY


def compute_dynamic_topography(model_number, path=PATH_MODEL, savepath=PATH_OUTPUT,
                                rho=RHO, g=G_GRAV, L=R_EARTH, save=True):
    """Compute dynamic topography from face1 stress_norm, optionally save to txt."""
    mesh_face1 = pv.read(f"{path}{model_number}_solution.face1.pvtu")

    stress = mesh_face1.point_data['stress_norm']
    sigma_dyn = stress - stress.mean()
    h_update = -sigma_dyn / (rho * g * L)
    mesh_surf = mesh_face1.points


    if save:
        np.savetxt(f"{savepath}{model_number}_mesh_xyz.txt", mesh_surf)
        np.savetxt(f"{savepath}{model_number}_mesh_xyz_topo.txt", h_update)

    return mesh_surf, h_update


def load_topography(model_number, savepath=PATH_OUTPUT,
                     theta_min=None, theta_max=None, th_mid=None):
    """Load saved topography txt, return sorted (x_km, h_km) on the mid-plane, optionally windowed."""
    x, y, z = np.loadtxt(f"{savepath}{model_number}_mesh_xyz.txt").T
    topo = np.loadtxt(f"{savepath}{model_number}_mesh_xyz_topo.txt")

    y_mid2 = 0.5 * (y.min() + y.max())
    tol = 1e-4 * (y.max() - y.min())
    sel = np.abs(y - y_mid2) < tol
    x2, z2, topo2 = x[sel], z[sel], topo[sel]

    theta = np.arctan2(x2, z2)

    if theta_min is not None and theta_max is not None:
        win = (theta >= theta_min) & (theta <= theta_max)
        theta = theta[win]
        topo2 = topo2[win]

    ref = th_mid if th_mid is not None else theta.mean()
    x_km = (theta - ref) * R_EARTH / 1000.0
    h_km = topo2 * R_EARTH / 1000.0

    idx = np.argsort(x_km)
    return x_km[idx], h_km[idx]


# 4. GRAVITY

def topo_to_gravity_fft(x_km, h_km, delta_rho=1670.0, z_obs_km=0.0, dx_km=2.0, smooth_sigma=2):
    """
    Forward-model free-air gravity from topography using the Parker (FFT) approximation.

    delta_rho: density contrast (kg/m^3), default mantle-water = 1670
    z_obs_km : observation height (km), 0 = surface
    dx_km    : output uniform grid spacing (km)
    smooth_sigma: Gaussian smoothing sigma in grid points, 0 = no smoothing

    Returns x_uniform_km, h_uniform_km, g_mgal.
    """
    x_km = np.asarray(x_km, dtype=float)
    h_km = np.asarray(h_km, dtype=float)

    mask = np.isfinite(x_km) & np.isfinite(h_km)
    x_km, h_km = x_km[mask], h_km[mask]

    idx = np.argsort(x_km)
    x_km, h_km = x_km[idx], h_km[idx]

    x_unique, uidx = np.unique(x_km, return_index=True)
    h_unique = h_km[uidx]
    if len(x_unique) < 2:
        raise ValueError("Not enough unique x points.")

    x_uniform_km = np.arange(x_unique.min(), x_unique.max() + dx_km, dx_km)
    h_uniform_km = np.interp(x_uniform_km, x_unique, h_unique)

    if smooth_sigma and smooth_sigma > 0:
        h_uniform_km = gaussian_filter1d(h_uniform_km, sigma=smooth_sigma)

    h_anom_m = (h_uniform_km - np.mean(h_uniform_km)) * 1000.0
    dx_m = dx_km * 1000.0

    k = 2.0 * np.pi * np.fft.fftfreq(len(h_anom_m), d=dx_m)
    Hk = np.fft.fft(h_anom_m)
    Gk = 2.0 * np.pi * G_NEWTON * delta_rho * np.exp(-np.abs(k) * z_obs_km * 1000.0) * Hk
    g_mgal = np.fft.ifft(Gk).real * 1e5

    return x_uniform_km, h_uniform_km, g_mgal


# 5. SURFACE VELOCITY


def get_surface_velocity(model_number, theta_min, theta_max, path=PATH_MODEL,
                           scale_mm_yr=True, pvtu_path=None, vel_field='velocity'):
    """
    Tangential (vt) and radial (vr) surface velocity from a face1 pvtu, on the mid-plane.

    pvtu_path/vel_field let you read a different file or field
    (e.g. inversion forward/observed velocity) instead of the default solution.
    """
    if pvtu_path is not None:
        mesh_face = pv.read(pvtu_path)
    else:
        mesh_face = pv.read(f"{path}{model_number}_solution.face1.pvtu")

    ymin, ymax = mesh_face.bounds[2], mesh_face.bounds[3]
    y_mid = 0.5 * (ymin + ymax)
    tol = 1e-3 * (ymax - ymin) if ymax > ymin else 1e-6

    pts = mesh_face.points
    V = np.asarray(mesh_face.point_data[vel_field])

    sel = np.abs(pts[:, 1] - y_mid) < tol
    pts2 = pts[sel]
    V2 = V[sel]

    theta = np.arctan2(pts2[:, 0], pts2[:, 2])
    win = (theta >= theta_min) & (theta <= theta_max)
    pts2, V2, theta = pts2[win], V2[win], theta[win]

    cosT = np.cos(theta)
    sinT = np.sin(theta)
    vt = V2[:, 0] * cosT - V2[:, 2] * sinT
    vr = V2[:, 0] * sinT + V2[:, 2] * cosT

    x_km = theta * R_EARTH / 1000.0
    idx = np.argsort(x_km)
    x_km, vt, vr = x_km[idx], vt[idx], vr[idx]

    if scale_mm_yr:
        scale_factor = (KAPPA / R_EARTH) * SEC_PER_YR * 1000.0
        vt = vt * scale_factor
        vr = vr * scale_factor

    return x_km, vt, vr


# 6. SOLVER DIAGNOSTICS

def nonlinear_residual(model, logpath=PATH_LOG):
    """Parse Newton nonlinear residual norm vs iteration from stdout log."""
    logfile = glob.glob(logpath + f"spb_{model}_stdout.*")[0]
    pat = re.compile(r"rhea_newton_solve_status newton_iter=(\d+), residual_norm=([0-9.eE+-]+),")

    iters, residuals = [], []
    with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pat.search(line)
            if m and 'components' not in line:
                iters.append(int(m.group(1)))
                residuals.append(float(m.group(2)))
    return iters, residuals


def linear_residual(model, logpath=PATH_LOG):
    """Parse KSP linear residual norm per Newton iteration from stdout log."""
    logfile = glob.glob(logpath + f"spb_{model}_stdout.*")[0]
    pat_ksp = re.compile(r"^\s*(\d+)\s+KSP Residual norm\s+([0-9.eE+-]+)")
    pat_done = re.compile(r"rhea_newton_solve_linear_solve_status newton_iter=(\d+),")

    linear, block = {}, []
    with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pat_ksp.search(line)
            if m:
                block.append((int(m.group(1)), float(m.group(2))))
                continue
            m = pat_done.search(line)
            if m:
                newton_iter = int(m.group(1))
                linear[newton_iter] = np.array(block)
                block.clear()
    return linear