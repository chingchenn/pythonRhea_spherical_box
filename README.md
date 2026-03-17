# pythonRhea_spherical_box
Rhea earth example
# not_active/

Exploratory scripts numbered in chronological order, documenting the research process.
For formal analysis, use the modular code in the parent directory.

## Series 08 — Initial Condition Setup

| File | Description |
|------|-------------|
| `08_slab_coordinates.py` | Read GMT slab data, convert format and output slab_top_data.txt |
| `08_slab_structure.py` | ~~Early test version, deprecated~~ |
| `08_slab_structure2.py` | Cartesian non-dimensional version, **signed distance to polyline** algorithm |
| `08_slab_structure3.py` | ~~Incomplete (30 lines), deprecated~~ |
| `08_slab_structure5.py` | ~~Cartesian dperp algorithm, superseded by box_spherical~~ |
| `08_slab_structure_3000km.py` | Cartesian version, 3000 km wide domain, dperp algorithm |
| `08_slab_structure_box_spherical.py` | **Main version**, spherical domain, dperp algorithm, outputs ridge + slab weakzone pointcloud |
| `08_temperature_coordinates.py` | Interpolate 2D temperature field onto 3D coordinates_vol.txt |

---

## Series 09 — Data Check

| File | Description |
|------|-------------|
| `09_slab2.py` | Quick read and plot of GMT slab distance data to verify format |

---

## Series 10 — Coordinate Generation

| File | Description |
|------|-------------|
| `10_make_coordinate.py` | Generate coordinates_vol.txt for Rhea input, supports arbitrary subdivision and polynomial order |

---

## Series 11 — Multi-Model Comparison

| File | Description |
|------|-------------|
| `11_topography_comparision.py` | spb_ multi-model topography profile comparison with Gaussian smoothing |

---

## Series 12 — Mesh Visualization

| File | Description |
|------|-------------|
| `12_mesh_points_line.py` | Plot mesh edges (LineCollection) to inspect AMR mesh structure |
| `12_mesh_points_scatter.py` | Plot cell centers as scatter to inspect mesh density distribution |

---

## Series 13 — Spherical Mesh

| File | Description |
|------|-------------|
| `13_plot_box_spherical_mesh.py` | Polar projection view of spb_ mesh, verify spherical domain geometry |

---

## Series 14 — Zoom-In Visualization

| File | Description |
|------|-------------|
| `14_zoom_in_mesh.py` | Read solution, AMR mesh local zoom-in, PolyCollection viscosity + velocity quiver |
| `14_zoom_in_mesh_init.py` | Read input, same as above, with option to save point data (viscosity, temperature, weakzone) |

---

## Series 15 — Mesh Resolution

| File | Description |
|------|-------------|
| `15_mesh_calculation.py` | Compute radial and tangential mesh resolution (dr_med, ds_med) as a function of depth |

---

## Series 16 — Point Data

| File | Description |
|------|-------------|
| `16_read_point_data.py` | Read saved point data txt files, multiple visualization modes: Cartesian scatter, polar, weakzone |

---

## Series 17 — Initial Field Check

| File | Description |
|------|-------------|
| `17_initial_check_temperature.py` | Read `_input.pvtu`, sample temperature onto regular grid and plot |
| `17_initial_check_temperature_or_viscosity.py` | Read `_solution_secondary.pvtu`, sample viscosity/temperature + velocity |

---

## Series 18 — Rheology

| File | Description |
|------|-------------|
| `18_viscosity_law.py` | Compute theoretical viscosity profile from Rhea parameters (non-linear + yield + final), compared against HSC temperature |

---

## Series 19 — Intermediate Iteration Check

| File | Description |
|------|-------------|
| `19_check_itn.py` | Read `_solver_itnXXX.pvtu`, inspect viscosity + velocity field at a specific Newton iteration |

---

## Series 20 — Surface Velocity

| File | Description |
|------|-------------|
| `20_vh_model.py` | spb_ multi-model surface tangential velocity (Vt) comparison, read directly from face1 |

---

## Series 21 — Weakzone

| File | Description |
|------|-------------|
| `21_output_weakzone.py` | Read `_input.pvtu`, sample weakzone field onto regular grid, log-scale visualization |
| `21_weakzone_depth.py` | Plot theoretical weakzone depth taper f(z) curves, compare different L and f_min parameters |

---

## Series 22 — Combined Visualization

| File | Description |
|------|-------------|
| `22_viscosity_and_topography.py` | spb_ model, viscosity field + topography profile + gravity anomaly + velocity quiver — the most complete single-model analysis figure |

---

## Series 23 — Solver Convergence

| File | Description |
|------|-------------|
| `23_read_residual.py` | Read archive log, plot nonlinear residual (Newton iteration) and linear residual (KSP iteration) convergence |
