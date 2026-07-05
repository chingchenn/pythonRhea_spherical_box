#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 19:50:13 2026

@author: chingchen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
54_plot_distribution_history_flexible.py
-----------------------------------------
Flexible corner plot for Rhea inversion.
Declare which parameters the model has via ACTIVE_PARAMS at the top;
everything else (PARAM_INFO, FILE_TO_DISPLAY, prior_mean_norm_nominal,
true_params_phys) is derived automatically.

Layout (n×n corner plot):
  diagonal    = 1D marginal distributions
  lower tri   = 2D conditional distributions (Hessian submatrix)
  upper tri   = legend / text

Space conventions (from write_initial_guess.py):
  n      : 'neg'  — phys = n_true / norm,         pm = n_true = 2.5
  yield  : 'lin'  — phys = norm * pm,              pm = 5.28e7 * YIELD_MPA
  w      : 'log'  — phys = exp(norm * |log(pm)|),  pm = 1e-6

prior_mean_norm is backed out from the itn00 gradient file (col2 =
prior gradient), so it reflects the actual perturbed prior mean when
prior-mean-perturbation-stddev is set. Falls back to nominal if the
gradient file is missing.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import glob, os, re, math

# ── Settings ────────────────────────────────────────────────────────────
model = 'NC_s040002'
path  = '/Users/chingchen/pCloud Drive/Rhea_snapshot/anvil/scratch/26summer/txt/'
fontsize = 14
MU_REF_PAS = 1.0e20
KAPPA_M2_S = 1.0e-6
DEPTH_M    = 6.371e6
YIELD_MPA  = MU_REF_PAS * KAPPA_M2_S / DEPTH_M**2 / 1.0e6

N_TRUE  = 2.5   # true stress exponent (used for 'neg' space)
R_GAS   = 8.314
DELTA_T = 1400.0
E_TRUE_KJ = 36.0 * R_GAS * DELTA_T / 1000.0   # UM activation energy (kJ/mol)
A_TRUE    = 1.0e7                              # UM scaling A

# * Only change this line: declare which parameters the model has
ACTIVE_PARAMS = ['stress_exponent', 'weakzone_factor']
# 5-par example: ACTIVE_PARAMS = ['um_scaling', 'um_activation_energy',
#                              'stress_exponent', 'yield_strength', 'weakzone_factor']
# 3-par example: ACTIVE_PARAMS = ['stress_exponent', 'yield_strength', 'weakzone_factor']
# 2-par example: ACTIVE_PARAMS = ['yield_strength', 'weakzone_factor']

# ── Full parameter table ────────────────────────────────────────────────
# name: (label, pm, space, prior_mean_norm_nominal, true_phys)
PARAM_TABLE = {
    'um_scaling':           ('UM scaling A',                 A_TRUE,             'lin',  1.0, A_TRUE),
    'um_activation_energy': ('UM Ea (kJ/mol)',               E_TRUE_KJ,          'lin',  1.0, E_TRUE_KJ),
    'stress_exponent':      ('stress exp n',                 N_TRUE,             'neg',  1.0, N_TRUE),
    'yield_strength':       ('yield stress (MPa)',           5.28e7 * YIELD_MPA, 'lin',  1.0, 5.28e7),
    'weakzone_factor':      ('weakzone w',                   1.0e-6,             'log', -1.0, 1.0e-6),
}

# Manual axis limits (looked up by parameter name; None or absent = auto range)
AXIS_LIM_MANUAL = {
    'weakzone_factor': (1e-8, 1e-2),
    'yield_strength':  (0, 280),
    'stress_exponent': (1.0, 3.5),
    # 'um_scaling':          None,   # auto
    # 'um_activation_energy': None,  # auto
}

# File order: active params taken from param_order in write_initial_guess.py:
#   5-par: [A, E, n, yield, w] (same as old 5-par PARAM_INFO order)
#   3-par: [n, yield, w]; 2-par: [yield, w]
FILE_ORDER_ALL    = ['um_scaling', 'um_activation_energy',
                     'stress_exponent', 'yield_strength', 'weakzone_factor']
# Display order extends Rudi 2022 Fig 7: active params from [w, yield, n, E, A]
DISPLAY_ORDER_ALL = ['weakzone_factor', 'yield_strength', 'stress_exponent',
                     'um_activation_energy', 'um_scaling']

file_order    = [p for p in FILE_ORDER_ALL    if p in ACTIVE_PARAMS]
display_order = [p for p in DISPLAY_ORDER_ALL if p in ACTIVE_PARAMS]

# FILE_TO_DISPLAY[k] = index in file order of the k-th display parameter
FILE_TO_DISPLAY = [file_order.index(p) for p in display_order]

# PARAM_INFO = (label, pm, space), same format as the old scripts
PARAM_INFO = [PARAM_TABLE[p][:3] for p in display_order]
n_params   = len(PARAM_INFO)

# Prior mean norm (nominal) and true physical values, display order
prior_mean_norm_nominal = np.array([PARAM_TABLE[p][3] for p in display_order])
true_params_phys        = np.array([PARAM_TABLE[p][4] for p in display_order])

def true_scale(idx):
    """Unit conversion when plotting the true model (yield -> MPa, others unchanged)"""
    return YIELD_MPA if display_order[idx] == 'yield_strength' else 1.0

print(f'[params] active (file order):    {file_order}')
print(f'[params] active (display order): {display_order}')
print(f'[params] FILE_TO_DISPLAY = {FILE_TO_DISPLAY}')

# ── norm <-> physical conversion ────────────────────────────────────────
def n2p(idx, n):
    """norm → physical"""
    _, pm, space = PARAM_INFO[idx]
    n = np.asarray(n, dtype=float)
    if space == 'log':
        return np.exp(np.clip(n * abs(math.log(pm)), -700, 700))
    elif space == 'neg':
        # phys = pm / norm  (norm must be nonzero)
        return pm / np.where(np.abs(n) < 1e-15, np.nan, n)
    else:  # 'lin'
        return n * pm

def p2n(idx, ph):
    """physical → norm"""
    _, pm, space = PARAM_INFO[idx]
    ph = np.asarray(ph, dtype=float)
    if space == 'log':
        return np.log(np.clip(ph, 1e-300, None)) / abs(math.log(pm))
    elif space == 'neg':
        return pm / np.where(np.abs(ph) < 1e-15, np.nan, ph)
    else:  # 'lin'
        return ph / pm

def jac_diag(idx, norm_val):
    """|d(phys)/d(norm)| at norm_val"""
    _, pm, space = PARAM_INFO[idx]
    if space == 'log':
        dim = abs(math.log(pm))
        return np.exp(np.clip(norm_val * dim, -700, 700)) * dim
    elif space == 'neg':
        # phys = pm/norm  →  |d(phys)/d(norm)| = pm/norm²
        return pm / (norm_val ** 2)
    else:  # 'lin'
        return pm

def phys_range(idx, center_n, sigma_n, nsig=3.5):
    """Compute physical axis range from norm-space center + sigma."""
    cp = n2p(idx, center_n)
    sp = abs(jac_diag(idx, center_n)) * sigma_n
    return max(cp - nsig * sp, 1e-10), cp + nsig * sp

# ──────────────────────────────────────────────────────────────
def permute_vec(v):
    """Permute vector from file order to display order."""
    return v[FILE_TO_DISPLAY]

def permute_mat(M):
    """Permute matrix from file order to display order (rows and cols)."""
    return M[np.ix_(FILE_TO_DISPLAY, FILE_TO_DISPLAY)]

def find_latest_hessian(path, model):
    final = os.path.join(path, f'{model}_inversion_solver_posterior_hessian.txt')
    if os.path.exists(final):
        print(f'[hessian] found final result: {os.path.basename(final)}')
        return final, None
    pattern = os.path.join(path, f'{model}_inversion_solver_itn*_hessian.txt')
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f'no hessian file found: {pattern}')
    def itn_num(f):
        m = re.search(r'itn(\d+)_hessian', f)
        return int(m.group(1)) if m else -1
    latest = max(files, key=itn_num)
    itn    = itn_num(latest)
    print(f'[hessian] not yet converged, using itn{itn:02d}: {os.path.basename(latest)}')
    return latest, itn

def find_latest_parameters(path, model):
    pattern = os.path.join(path, f'{model}_inversion_solver_itn*_parameters.txt')
    files   = glob.glob(pattern)
    def itn_num(f):
        m = re.search(r'itn(\d+)_parameters', f)
        return int(m.group(1)) if m else -1
    latest = max(files, key=itn_num)
    return latest, itn_num(latest)

def load_prior_mean_from_gradient(path, model, C_prior_file, p0_file):
    """
    Back out the prior mean (normalized space) from the itn00 gradient:
        g_prior = C_prior^{-1} (p0 - mu_prior)
        => mu_prior = p0 - C_prior @ g_prior

    The gradient file is (n_params x 4): col0=total, col1=data misfit,
    col2=prior, col3=0. Use col2 (prior gradient) for the back-out.

    Note: computation is done in file order (C_prior_file and p0_file are both
    file order); the result is then permuted to display order.
    """
    grad_file = os.path.join(path, f'{model}_inversion_solver_itn00_gradient.txt')
    if not os.path.exists(grad_file):
        print('[prior mean] itn00 gradient not found, falling back to nominal')
        return None
    g = np.loadtxt(grad_file)
    if g.ndim == 2 and g.shape[1] >= 3:
        g_prior = g[:, 2]          # <- prior gradient is in col2
    else:
        g_prior = np.ravel(g)      # fallback: single-column format
    mu = p0_file - C_prior_file @ g_prior     # file order
    mu = permute_vec(mu)                       # → display order
    print(f'[prior mean] backed out from itn00 gradient (col2): {mu}')
    return mu

def load_all_parameters(path, model):
    pattern = os.path.join(path, f'{model}_inversion_solver_itn*_parameters.txt')
    files   = glob.glob(pattern)
    def itn_num(f):
        m = re.search(r'itn(\d+)_parameters', f)
        return int(m.group(1)) if m else -1
    files = sorted(files, key=itn_num)
    itns, params = [], []
    for f in files:
        nn = itn_num(f)
        p  = permute_vec(np.loadtxt(f))   # file → display order
        itns.append(nn)
        params.append(p)
        print(f'  itn{nn:02d}: {p}')
    return np.array(itns), np.array(params)

# ── Density computation ─────────────────────────────────────────────────
def check_hessian_posdef(H_full, label='posterior'):
    """
    Check whether the Hessian is positive definite. Returns (is_posdef, eigvals).
    If indefinite (negative eigenvalues), print a warning -- this Hessian is not
    a valid precision matrix, so posterior uncertainty quantification
    (error bars / ellipses) is NOT trustworthy.
    """
    eigvals = np.linalg.eigvalsh(H_full)
    is_posdef = np.all(eigvals > 0)
    if not is_posdef:
        print(f'[warning] {label} Hessian is NOT positive definite!'
              f' eigenvalues = {eigvals}')
        print(f'          -> covariance diagonal may be negative; densities are'
              f' force-plotted with |var|, but the uncertainty is NOT trustworthy.')
    return is_posdef, eigvals


def cov_conditional_2d(idx_x, idx_y, H_full):
    """
    Conditional covariance for (idx_x, idx_y) given all other params fixed.
    Uses Schur complement of H (precision matrix).
    (Kept for reference; not called in the plotting loop.)
    """
    sel = [idx_x, idx_y]
    rest = [i for i in range(n_params) if i not in sel]

    H_AA = H_full[np.ix_(sel, sel)]
    if len(rest) == 0:
        return np.linalg.inv(H_AA)
    H_AB = H_full[np.ix_(sel, rest)]
    H_BB = H_full[np.ix_(rest, rest)]
    # Conditional precision = H_AA - H_AB @ inv(H_BB) @ H_BA
    H_cond = H_AA - H_AB @ np.linalg.inv(H_BB) @ H_AB.T
    return np.linalg.inv(H_cond)

def cov_marginal_2d(idx_x, idx_y, H_full):
    """Marginal covariance for (idx_x, idx_y), integrating out all others.
    (Kept for reference; not called in the plotting loop.)"""
    C_full = np.linalg.inv(H_full)
    sel = [idx_x, idx_y]
    return C_full[np.ix_(sel, sel)]

def conditional_density(idx_x, idx_y, H_full, mu_full_norm, xr, yr, n_grid=300):
    """
    2D conditional density on physical grid.
    Uses 2x2 submatrix of H (no Schur complement) — same as 2-par code.
    """
    H_AA = np.array([
        [H_full[idx_x, idx_x], H_full[idx_x, idx_y]],
        [H_full[idx_y, idx_x], H_full[idx_y, idx_y]],
    ])
    mu = np.array([mu_full_norm[idx_x], mu_full_norm[idx_y]])

    # 'log' -> geomspace; 'lin'/'neg' -> linspace (for 'neg', uniform in 1/n is
    # natural, but linspace in phys is fine)
    x_phys = (np.geomspace(*xr, n_grid) if PARAM_INFO[idx_x][2] == 'log'
               else np.linspace(*xr, n_grid))
    y_phys = (np.geomspace(*yr, n_grid) if PARAM_INFO[idx_y][2] == 'log'
               else np.linspace(*yr, n_grid))

    Xp, Yp = np.meshgrid(x_phys, y_phys)
    Xn = p2n(idx_x, Xp)
    Yn = p2n(idx_y, Yp)

    a, b, c = H_AA[0, 0], H_AA[0, 1], H_AA[1, 1]
    dx, dy  = Xn - mu[0], Yn - mu[1]
    quad = a*dx*dx + 2*b*dx*dy + c*dy*dy
    # If H_AA is indefinite, quad can be negative -> exp overflow. Clip the
    # exponent so the shape is still drawn without overflowing.
    expo = np.clip(-0.5 * quad, -700, 0)
    return Xp, Yp, np.exp(expo)

def marginal_1d_density(idx, H_full, mu_full_norm, xr, n_grid=400):
    """
    1D marginal density along physical axis for parameter idx.
    Marginal variance = (inv H)[idx, idx].
    """
    C_full = np.linalg.inv(H_full)
    var_norm = C_full[idx, idx]
    mu_n     = mu_full_norm[idx]

    # If H is indefinite, the diagonal of inv(H) can be negative -> dividing by a
    # negative number makes exp overflow. Use |var_norm| to force-plot the shape
    # (a warning has already been printed in check_hessian_posdef).
    if var_norm <= 0:
        var_norm = abs(var_norm)

    if PARAM_INFO[idx][2] == 'log':
        x_phys = np.geomspace(*xr, n_grid)
    else:  # 'lin' / 'neg'
        x_phys = np.linspace(*xr, n_grid)

    xn   = p2n(idx, x_phys)
    expo = np.clip(-0.5 * (xn - mu_n)**2 / var_norm, -700, 0)
    dens = np.exp(expo)
    return x_phys, dens

def safe_plot_range_1d(idx, H_prior, H_post, prior_mean_norm, m_map,
                        all_params=None, nsig=3.5):
    """
    1D axis range for parameter idx: union of prior and posterior ranges.
    """
    C_prior = np.linalg.inv(H_prior)
    C_post  = np.linalg.inv(H_post)
    sig_prior = np.sqrt(abs(C_prior[idx, idx]))
    sig_post  = np.sqrt(abs(C_post[idx, idx]))

    r_prior = phys_range(idx, prior_mean_norm[idx], sig_prior, nsig)
    r_post  = phys_range(idx, m_map[idx],           sig_post,  nsig)

    prior_w = r_prior[1] - r_prior[0]
    post_w  = r_post[1]  - r_post[0]

    if post_w > prior_w:
        print(f'  [range1d] param{idx} posterior wider than prior, clamping')
        mp = n2p(idx, m_map[idx])
        r = (min(r_prior[0], mp - 0.1*prior_w),
             max(r_prior[1], mp + 0.1*prior_w))
    else:
        r = (min(r_prior[0], r_post[0]),
             max(r_prior[1], r_post[1]))

    # include trajectory
    if all_params is not None:
        traj = np.array([n2p(idx, p[idx]) for p in all_params])
        margin = (r[1] - r[0]) * 0.1
        r = (min(r[0], traj.min() - margin),
             max(r[1], traj.max() + margin))

    r = (max(r[0], 1e-10), r[1])
    return r

def safe_plot_range_2d(idx_x, idx_y, H_prior, H_post, prior_mean_norm, m_map,
                        all_params=None, nsig=3.5):
    """
    2D axis ranges for (idx_x, idx_y): consistent with 2-par code.
    Uses submatrix inv diagonal for sigma.
    """
    H_AA_prior = np.array([[H_prior[idx_x,idx_x], H_prior[idx_x,idx_y]],
                            [H_prior[idx_y,idx_x], H_prior[idx_y,idx_y]]])
    H_AA_post  = np.array([[H_post[idx_x,idx_x],  H_post[idx_x,idx_y]],
                            [H_post[idx_y,idx_x],  H_post[idx_y,idx_y]]])
    Sigma_prior = np.linalg.inv(H_AA_prior)
    Sigma_post  = np.linalg.inv(H_AA_post)

    def axis_range(ii, sigma_val_prior, sigma_val_post):
        r_prior = phys_range(ii, prior_mean_norm[ii], np.sqrt(abs(sigma_val_prior)), nsig)
        r_post  = phys_range(ii, m_map[ii],           np.sqrt(abs(sigma_val_post)),  nsig)
        prior_w = r_prior[1] - r_prior[0]
        post_w  = r_post[1]  - r_post[0]
        if post_w > prior_w:
            mp = n2p(ii, m_map[ii])
            r = (min(r_prior[0], mp - 0.1*prior_w),
                 max(r_prior[1], mp + 0.1*prior_w))
        else:
            r = (min(r_prior[0], r_post[0]),
                 max(r_prior[1], r_post[1]))
        return (max(r[0], 1e-10), r[1])

    xr = axis_range(idx_x, Sigma_prior[0, 0], Sigma_post[0, 0])
    yr = axis_range(idx_y, Sigma_prior[1, 1], Sigma_post[1, 1])

    if all_params is not None:
        traj_x = np.array([n2p(idx_x, p[idx_x]) for p in all_params])
        traj_y = np.array([n2p(idx_y, p[idx_y]) for p in all_params])
        mx = (xr[1] - xr[0]) * 0.1
        my = (yr[1] - yr[0]) * 0.1
        xr = (min(xr[0], traj_x.min() - mx), max(xr[1], traj_x.max() + mx))
        yr = (min(yr[0], traj_y.min() - my), max(yr[1], traj_y.max() + my))
        xr = (max(xr[0], 1e-10), xr[1])
        yr = (max(yr[0], 1e-10), yr[1])

    return xr, yr

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
prior_cov_file        = os.path.join(path, f'{model}_inversion_solver_itn00_prior_covariance.txt')
hess_file, latest_itn = find_latest_hessian(path, model)
param_file, _         = find_latest_parameters(path, model)

C_prior_file = np.loadtxt(prior_cov_file)     # file order
H_post_file  = np.loadtxt(hess_file)          # file order
m_map_file   = np.loadtxt(param_file)          # file order

# Permute to display order
C_prior = permute_mat(C_prior_file)
H_prior = np.linalg.inv(C_prior)
H_post  = permute_mat(H_post_file)
m_map   = permute_vec(m_map_file)

# Check posterior Hessian positive definiteness (indefinite -> uncertainty
# not trustworthy; densities are force-plotted with |var|)
check_hessian_posdef(H_post, label=f'{model} posterior')

# ── Diagnostic prints (following chingchen's analysis code, all in file order) ──
SHORT_LABEL = {'stress_exponent': 'n', 'yield_strength': 'yield',
               'weakzone_factor': 'w', 'um_scaling': 'A',
               'um_activation_energy': 'E'}
labels_file = [SHORT_LABEL[p] for p in file_order]

H_prior_file     = np.linalg.inv(C_prior_file)
C_post_final     = np.linalg.inv(H_post_file)

print('C_prior')
print(C_prior_file)
print("\nC_post_full (= inv(posterior_hessian.txt))")
print(C_post_final)
print('\nsigma_prior')
sigma_prior = np.sqrt(np.diag(C_prior_file))
print(sigma_prior)
print('\nsigma_post_full')
sigma_post_full = np.sqrt(np.diag(C_post_final))
print(sigma_post_full)

def to_correlation(C):
    d = np.sqrt(np.abs(np.diag(C)))
    return C / np.outer(d, d)

corr_prior = to_correlation(C_prior_file)
corr_post_full = to_correlation(C_post_final)
print('\ncorr_prior')
print(corr_prior)
print("\ncorr_post_full")
print(corr_post_full)

ratio_full = np.diag(C_post_final) / np.diag(C_prior_file)
print("\nratio_full (posterior/prior variance)")
print(ratio_full)

H_data_full = H_post_file - H_prior_file
print("\nH_data_full (= H_post - H_prior)")
print(H_data_full)

cond_full = np.linalg.cond(H_post_file)
print("\ncond_full")
print(cond_full)

eig_H_data_full = np.linalg.eigvalsh(H_data_full)
print("\neig_H_data_full")
print(eig_H_data_full)

eigvals_d, eigvecs_d = np.linalg.eigh(H_data_full)
v = eigvecs_d[:, 0]
v_normalized = v / np.max(np.abs(v))
print(f'\n lamda min vector {labels_file}')
print(v_normalized)

def print_eigen_decomposition(H, labels):
    eigvals, eigvecs = np.linalg.eigh(H)
    for i in range(len(eigvals)):
        v = eigvecs[:, i]
        v_norm = v / np.max(np.abs(v))   # normalized to 1
        parts = ', '.join(f'{lab}: {val:+.2f}' for lab, val in zip(labels, v_norm))
        print(f'λ = {eigvals[i]:+.2f} → v = [{parts}]')

print_eigen_decomposition(H_data_full, labels=labels_file)
# ── End of diagnostic prints ────────────────────────────────────────────

# Back out the actual prior mean from the itn00 gradient (col2 = prior gradient).
# Computation is done in file order; result is permuted to display order.
p0_file = np.loadtxt(os.path.join(
    path, f'{model}_inversion_solver_itn00_parameters.txt'))
prior_mean_norm = load_prior_mean_from_gradient(path, model, C_prior_file, p0_file)
if prior_mean_norm is None:
    prior_mean_norm = prior_mean_norm_nominal.copy()

itn_label = 'posterior (converged)' if latest_itn is None else f'itn{latest_itn:02d}'
print(f'\n[prior cov]\n{C_prior}')
print(f'[hessian: {itn_label}]\n{H_post}')
print(f'[MAP params]\n{m_map}')
print(f'[prior mean norm]\n{prior_mean_norm}')
print(f'[prior mean norm nominal]\n{prior_mean_norm_nominal}')

print('\n[loading all iteration parameters]')
itns, all_params = load_all_parameters(path, model)

# Convert to physical
prior_mean_phys = np.array([n2p(i, prior_mean_norm[i]) for i in range(n_params)])
m_map_phys      = np.array([n2p(i, m_map[i])           for i in range(n_params)])
all_params_phys = np.array([[n2p(i, all_params[k, i]) for i in range(n_params)]
                              for k in range(len(itns))])

print(f'\n[prior mean phys] {prior_mean_phys}')
print(f'[MAP phys]        {m_map_phys}')
print(f'[true phys]       {true_params_phys}')

# ── Compute all axis ranges ─────────────────────────────────────────────
print('\n[determining plot ranges]')
ranges_1d = []
for i in range(n_params):
    r = safe_plot_range_1d(i, H_prior, H_post, prior_mean_norm, m_map,
                            all_params=all_params)
    ranges_1d.append(r)
    print(f'  param{i} ({PARAM_INFO[i][0]}): {r}')

# Axis limits: manual values looked up by parameter name; missing/None = auto range
axis_lim = [AXIS_LIM_MANUAL.get(p) or ranges_1d[i]
            for i, p in enumerate(display_order)]
for i, p in enumerate(display_order):
    print(f'  axis_lim[{i}] ({p}): {axis_lim[i]}')

# ── contour levels ──────────────────────────────────────────────────────
levels       = [np.exp(-0.5*9), np.exp(-0.5*4), np.exp(-0.5*1)]
sigma_labels = ['3σ', '2σ', '1σ']

# ── Plot: n x n corner plot ─────────────────────────────────────────────
fig = plt.figure(figsize=(4.3 * n_params + 0.1, 4 * n_params))
gs  = gridspec.GridSpec(n_params, n_params, hspace=0.08, wspace=0.08)

# colour for trajectory scatter
itn0_mask = itns > 0
sc_ref    = None   # for colorbar

for row in range(n_params):
    for col in range(n_params):
        ax = fig.add_subplot(gs[row, col])

        xr = axis_lim[col]   # x = col parameter
        yr = axis_lim[row]   # y = row parameter

        if row == col:
            # ── Diagonal: 1D marginal ───────────────────────────────────
            idx = row
            xp_prior, dp_prior = marginal_1d_density(idx, H_prior, prior_mean_norm, xr)
            xp_post,  dp_post  = marginal_1d_density(idx, H_post,  m_map,           xr)

            # Normalise to peak = 1 for display
            ax.plot(xp_prior, dp_prior / dp_prior.max(),
                    color='gray', lw=1.5, ls='--', label='prior')
            ax.plot(xp_post,  dp_post  / dp_post.max(),
                    color='red',  lw=2.0,           label='posterior')

            # Prior mean, MAP, true markers (vertical lines)
            ax.axvline(prior_mean_phys[idx], color='gray',      lw=1.2, ls=':')
            ax.axvline(m_map_phys[idx],      color='red',       lw=1.5, ls='-.')
            ax.axvline(true_params_phys[idx] * true_scale(idx),
                       color='darkgreen', lw=1.5, ls='-')

            ax.set_xlim(xr)
            ax.set_ylim(0, 1.25)
            ax.tick_params(labelsize=fontsize)

            # Only show x label on bottom row
            if row == n_params - 1:
                ax.set_xlabel(PARAM_INFO[col][0], fontsize=fontsize)
            else:
                ax.set_xticklabels([])

            ax.set_yticks([])
            if PARAM_INFO[idx][2] == 'log':
                ax.set_xscale('log')

        elif row > col:
            # ── Lower triangle: 2D conditional distribution ─────────────
            idx_x, idx_y = col, row

            Xp, Yp, D_prior = conditional_density(idx_x, idx_y, H_prior,
                                                    prior_mean_norm, xr, yr)
            Xq, Yq, D_post  = conditional_density(idx_x, idx_y, H_post,
                                                    m_map,           xr, yr)

            cs_pr = ax.contour(Xp, Yp, D_prior, levels=levels,
                                colors='gray', linewidths=1.1, linestyles='--')
            cs_po = ax.contour(Xq, Yq, D_post,  levels=levels,
                                colors='red',  linewidths=1.6)

            for cs, lbs in [(cs_pr, sigma_labels), (cs_po, sigma_labels)]:
                fmt = {lv: lb for lv, lb in zip(levels, lbs)}
                ax.clabel(cs, fmt=fmt, fontsize=fontsize)

            # Prior mean
            ax.plot(prior_mean_phys[idx_x], prior_mean_phys[idx_y],
                    'k+', ms=8, mew=1.8)
            # MAP star
            ax.plot(m_map_phys[idx_x], m_map_phys[idx_y],
                    'r*', ms=10)
            # True model
            true_x = true_params_phys[idx_x] * true_scale(idx_x)
            true_y = true_params_phys[idx_y] * true_scale(idx_y)
            ax.plot(true_x, true_y,
                    'D', color='darkgreen', ms=8,
                    markeredgecolor='darkgreen', markeredgewidth=1.2)

            # Trajectory
            ax.scatter(all_params_phys[0, idx_x], all_params_phys[0, idx_y],
                       marker='s', s=50, color='black', zorder=7)
            sc = ax.scatter(all_params_phys[itn0_mask, idx_x],
                             all_params_phys[itn0_mask, idx_y],
                             c=itns[itn0_mask], cmap='Blues', s=20, zorder=5,
                             vmin=itns[itn0_mask].min(), vmax=itns[itn0_mask].max(),
                             edgecolors='steelblue', linewidths=0.4)
            ax.plot(all_params_phys[:, idx_x], all_params_phys[:, idx_y],
                    '-', color='steelblue', lw=0.7, alpha=0.5, zorder=4)
            if sc_ref is None:
                sc_ref = sc

            if PARAM_INFO[idx_x][2] == 'log':
                ax.set_xscale('log')
            if PARAM_INFO[idx_y][2] == 'log':
                ax.set_yscale('log')

            ax.set_xlim(xr)
            ax.set_ylim(yr)
            ax.tick_params(labelsize=fontsize)

            if row == n_params - 1:
                ax.set_xlabel(PARAM_INFO[idx_x][0], fontsize=fontsize)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(PARAM_INFO[idx_y][0], fontsize=fontsize)
            else:
                ax.set_yticklabels([])

            ax.grid(True, alpha=0.25, which='both')

        else:
            # ── Upper triangle: legend / blank ──────────────────────────
            ax.axis('off')

# ── legend in top-right cell ────────────────────────────────────────────
ax_leg = fig.add_axes([0.304, 0.57, 0.38, 0.32])
ax_leg.axis('off')
legend_elements = [
    plt.Line2D([0], [0], color='gray',      lw=1.5, ls='--', label='prior'),
    plt.Line2D([0], [0], color='red',       lw=2.0,          label=f'posterior \n ({itn_label})'),
    plt.Line2D([0], [0], marker='+',  color='k',         ms=10, mew=2,  lw=0, label='prior mean'),
    plt.Line2D([0], [0], marker='*',  color='red',        ms=12,         lw=0, label='MAP'),
    plt.Line2D([0], [0], marker='D',  color='darkgreen',  ms=9,          lw=0, label='true model'),
    plt.Line2D([0], [0], marker='s',  color='black',      ms=7,          lw=0, label='itn00 (start)'),
    plt.Line2D([0], [0], marker='o',  color='steelblue',  ms=5,          lw=0, label='MAP trajectory'),
]
ax_leg.legend(handles=legend_elements, loc='center', fontsize=fontsize-2,
              frameon=True, framealpha=0.9)

# ── colorbar for iteration ───────────────────────────────────────────────
if sc_ref is not None:
    cax = fig.add_axes([0.65, 0.38, 0.02, 0.48])
    cbar = fig.colorbar(sc_ref, cax=cax, label='iteration')
    cbar.locator = ticker.MaxNLocator(integer=True)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=fontsize)

status = 'converged' if latest_itn is None else f'itn{latest_itn:02d} / ?'
plt.suptitle(f'{model}  —  {status}', fontsize=14, y=1.01)

itn_str = 'converged' if latest_itn is None else f'itn{latest_itn:02d}'
out = f'inversion_corner_{model}_{itn_str}.png'
#plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\n[saved] {out}')
plt.show()