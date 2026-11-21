"""
Supply code of the publication:

B. Cagri Sarar, Ivan Giorgio, Gino D’Ovidio, Francesco Fabbrocino, B. Emek Abali, and M. Erden Yildizdag (2027). Voxel-based strain-gradient modeling of TPMS metamaterials for advanced railway passenger vehicle applications
"""

__author__ = "B. Cagri Sarar"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html


import numpy as np
from scipy.optimize import brentq
from dolfin import *
import time as comp_time

start = comp_time.time()

# ---------------------------------------------------------
# Parameters and Grid Definitions
# ---------------------------------------------------------
L = 10.0
Nx = Ny = Nz = 50  
voxel_size = L / Nx 
period = 10.0


target_infill = 0.6
model_type = "FRD"
inc = 7   # fiber
mat = 8   # matrix

x_vox = np.linspace(-L/2, L/2, Nx, endpoint=False) + (voxel_size / 2.0)
y_vox = np.linspace(-L/2, L/2, Ny, endpoint=False) + (voxel_size / 2.0)
z_vox = np.linspace(-L/2, L/2, Nz, endpoint=False) + (voxel_size / 2.0)

X, Y, Z = np.meshgrid(x_vox, y_vox, z_vox, indexing='ij')
Xs = 2.0 * np.pi * X / period
Ys = 2.0 * np.pi * Y / period
Zs = 2.0 * np.pi * Z / period

tpms_models = {
    "SchwarzP": np.cos(Xs) + np.cos(Ys) + np.cos(Zs),
    "Gyroid": np.sin(Xs) * np.cos(Ys) + np.sin(Ys) * np.cos(Zs) + np.sin(Zs) * np.cos(Xs),
    "Nevious": 3 * (np.cos(Xs) + np.cos(Ys) + np.cos(Zs)) + 4 * np.cos(Xs) * np.cos(Ys) * np.cos(Zs),
    "IWP": 2 * (np.cos(Xs) * np.cos(Ys) + np.cos(Ys) * np.cos(Zs) + np.cos(Zs) * np.cos(Xs)) - (np.cos(2 * Xs) + np.cos(2 * Ys) + np.cos(2 * Zs)),
    "FischerKoch": np.cos(2 * Xs) * np.sin(Ys) * np.cos(Zs) + np.cos(2 * Ys) * np.sin(Zs) * np.cos(Xs) + np.cos(2 * Zs) * np.sin(Xs) * np.cos(Ys),
    "Lidinoid" : 0.5 * (np.sin(2 * Xs) * np.cos(Ys) * np.sin(Zs) + np.sin(2 * Ys) * np.cos(Zs) * np.sin(Xs) + np.sin(2 * Zs) * np.cos(Xs) * np.sin(Ys)) - 0.5 * (np.cos(2 * Xs) * np.cos(2 * Ys) + np.cos(2 * Ys) * np.cos(2 * Zs) + np.cos(2 * Zs) * np.cos(2 * Xs)),
    "Diamond": np.sin(Xs) * np.sin(Ys) * np.sin(Zs) + np.sin(Xs) * np.cos(Ys) * np.cos(Zs) + np.cos(Xs) * np.sin(Ys) * np.cos(Zs) + np.cos(Xs) * np.cos(Ys) * np.sin(Zs),
    "FRD" : -4 * (np.cos(Xs) * np.cos(Ys) * np.cos(Zs)) - np.cos(2 * Xs) * np.cos(2 * Ys) - np.cos(2 * Ys) * np.cos(2 * Zs) - np.cos(2 * Xs) * np.cos(2 * Zs),
}

model_matrix = tpms_models[model_type]

# ---------------------------------------------------------
# Approximation/Optimization Function
# ---------------------------------------------------------
from scipy import ndimage as ndi

def keep_main_component(voxel):
    """Keep only the periodically propagating main solid skeleton (largest component in a 3x3x3 tiling)."""
    N = voxel.shape[0]
    lab, _ = ndi.label(np.tile(voxel.astype(bool), (3, 3, 3)), structure=np.ones((3, 3, 3)))
    cnt = np.bincount(lab.ravel()); cnt[0] = 0
    return (lab[N:2*N, N:2*N, N:2*N] == cnt.argmax()).astype(np.uint8)

def connected_fraction(t):
    return keep_main_component(np.abs(model_matrix) <= t).mean()

# Smallest t for which the connected solid volume fraction reaches the target (bisection; used instead of brentq because the fraction can be discontinuous when islands are removed)
lo, hi = 0.0, np.max(np.abs(model_matrix))
while hi - lo > 1e-5:
    mid = 0.5 * (lo + hi)
    if connected_fraction(mid) < target_infill: lo = mid
    else: hi = mid
opt_thickness = hi
raw = (np.abs(model_matrix) <= opt_thickness)
voxel_matrix = keep_main_component(raw)
print("Removed floating voxels:", int(raw.sum() - voxel_matrix.sum()))

mesh = BoxMesh(Point(-L/2, -L/2, -L/2), Point(L/2, L/2, L/2), Nx, Ny, Nz)

cellsmesh = MeshFunction("size_t", mesh, mesh.topology().dim())
cellsmesh.set_all(mat)

for cell in cells(mesh):
    mp = cell.midpoint()
    
    cx = min(max(mp.x(), -L/2), L/2 - 1e-5)
    cy = min(max(mp.y(), -L/2), L/2 - 1e-5)
    cz = min(max(mp.z(), -L/2), L/2 - 1e-5)
    
    idx = int(np.floor((cx + L/2) / voxel_size))
    idy = int(np.floor((cy + L/2) / voxel_size))
    idz = int(np.floor((cz + L/2) / voxel_size))
    
    idx = min(idx, Nx - 1)
    idy = min(idy, Ny - 1)
    idz = min(idz, Nz - 1)
    
    if voxel_matrix[idx, idy, idz] == 1:
        cellsmesh[cell] = inc

# ---------------------------------------------------------
# Volume Verification (DOLFIN Integrals)
# ---------------------------------------------------------
dV = Measure("dx", domain=mesh, subdomain_data=cellsmesh,
             metadata={"quadrature_degree": 2, "quadrature_scheme": "uflacs"})

Vol = assemble(1.0 * dV)
Vol_inc = assemble(1.0 * dV(inc))
Vol_frac = Vol_inc / Vol

print(f"Final FEA Volume Fraction (Calculated Vol_frac): {Vol_frac:.4f}")

# 1. Creation of the submesh containing cells belonging to the defined 'inc' (fiber) subdomain
fiber_mesh = SubMesh(mesh, cellsmesh, inc)

# 2. Extraction of node coordinates and cell connectivity of the submesh
nodes = fiber_mesh.coordinates()
cells = fiber_mesh.cells()

# 3. Conversion and writing of the mesh data to Gmsh (.msh) format via meshio
import meshio

# The FEniCS BoxMesh structure consists of three-dimensional tetrahedral elements
cells_dict = [("tetra", cells)]

msh_mesh = meshio.Mesh(points=nodes, cells=cells_dict)

#File("output/tpms_" +str(model_type)+ "_voxel_"+ str(round(Vol_frac, 2)) +"_"+ str(Nx) +".pvd") << cellsmesh
msh_filename = f"tpms_" +str(model_type)+ "_voxel_"+ str(round(Vol_frac, 2)) +"_"+ str(Nx) +".vtu"
meshio.write(msh_filename, msh_mesh, file_format="vtu")  

end = comp_time.time()
print(f"Total Computation Time: {end - start:.2f} seconds")
