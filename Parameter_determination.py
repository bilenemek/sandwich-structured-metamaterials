"""Parameter determination with asymptotic homogenization scheme"""
__author__ = "B. Emek Abali, B. Cagri Sarar"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

"""
Supply code of the publication:

B. Cagri Sarar, Ivan Giorgio, Gino D’Ovidio, Francesco Fabbrocino, B. Emek Abali, and M. Erden Yildizdag (2027). Voxel-based strain-gradient modeling of TPMS metamaterials for advanced railway passenger vehicle applications
"""

import time as comp_time
start_time = comp_time.time()

from dolfin import *
import numpy as np
#from ufl import Identity, as_tensor
#from ufl import indices
import math
#import mshr
import matplotlib.pyplot as plt
from dolfin import XDMFFile
from scipy.optimize import brentq

import ufl_legacy as ufl
from ufl_legacy import indices, Identity, as_tensor

#Setting solver parameters
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
#parameters["allow_extrapolation"] = True
#parameters["mesh_partitioner"] = "SCOTCH"
parameters["ghost_mode"] = 'shared_vertex' #'shared_vertex'  |  'shared_facet'

krylov_params = parameters["krylov_solver"]
krylov_params["relative_tolerance"] = 1E-4
krylov_params["absolute_tolerance"] = 1E-5
#krylov_params["divergence_limit"] = 1E-8
krylov_params["maximum_iterations"] = 1000
krylov_params["nonzero_initial_guess"] = False
krylov_params["monitor_convergence"] = False
krylov_params["error_on_nonconvergence"] = True
krylov_params["report"] = True


'''
list_krylov_solver_preconditioners()

amg              |  Algebraic multigrid                       
default          |  default preconditioner                    
hypre_amg        |  Hypre algebraic multigrid (BoomerAMG)     
hypre_euclid     |  Hypre parallel incomplete LU factorization
hypre_parasails  |  Hypre parallel sparse approximate inverse 
icc              |  Incomplete Cholesky factorization         
ilu              |  Incomplete LU factorization               
jacobi           |  Jacobi iteration                          
none             |  No preconditioner                         
petsc_amg        |  PETSc algebraic multigrid                 
sor              |  Successive over-relaxation 
'''
newton_solver_parameters = {"nonlinear_solver": "newton",
	"newton_solver":{"linear_solver": "mumps",  
			"convergence_criterion": "incremental",
			"relative_tolerance": 1E-5,
			"absolute_tolerance": 1E-5, 
			"report":True,
			"krylov_solver": krylov_params, 
			"relaxation_parameter":1.0, 
			"maximum_iterations":30, 
			"error_on_nonconvergence": True
					} 
	}

solver_parameters_phi = {"linear_solver": "gmres", "preconditioner":"icc"}
solver_parameters_psi = {"linear_solver": "mumps", "preconditioner": "default"}

processID =  MPI.comm_world.Get_rank()
set_log_level(50) #50, 40, 30, 20, 16, 13, 10

# ---------------------------------------------------------
# RVE GENERATION
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

facetsmesh = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facetsmesh.set_all(0)

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


dV = Measure("dx", domain=mesh, subdomain_data=cellsmesh,
             metadata={"quadrature_degree": 2, "quadrature_scheme": "uflacs"})

Vol = assemble(1.0 * dV)
Vol_inc = assemble(1.0 * dV(inc))
Vol_frac = Vol_inc / Vol

print(f"({Nx}x{Ny}x{Nz}):")
print("(Volume Fraction):", Vol_frac)

data = str(model_type) + '_'+  str(round(Vol_frac, 2)) +'_'+ str(Nx)
##########

Dim = mesh.topology().dim()

Vol = assemble(1.*dV)
Vol_mat = assemble(1.0 * dV(8))
Vol_inc = assemble(1.0 * dV(7))
print("thickness", opt_thickness)
print(Vol_mat, Vol_inc, Vol_mat+Vol_inc, Vol, Vol_inc/Vol)

x = SpatialCoordinate(mesh)

Xc_0 = assemble( x[0] *dV ) / Vol
Xc_1 = assemble( x[1] *dV ) / Vol
Xc_2 = assemble( x[2] *dV ) / Vol

coordinates = mesh.coordinates()
xcoords, ycoords, zcoords = [], [], []
for k,coord in enumerate(coordinates):
	coordinates[k][0] = coordinates[k][0] - Xc_0
	coordinates[k][1] = coordinates[k][1] - Xc_1
	coordinates[k][2] = coordinates[k][2] - Xc_2

	xcoords.append(coordinates[k][0])
	ycoords.append(coordinates[k][1])
	zcoords.append(coordinates[k][2])

xmin, xmax = min(xcoords) , max(xcoords)
ymin, ymax = min(ycoords) , max(ycoords)
zmin, zmax = min(zcoords) , max(zcoords)
periodic_x = [xmax-xmin, 0 , 0]
periodic_y = [0, ymax-ymin , 0]
periodic_z = [0, 0 , zmax-zmin]

#Dim = mesh.topology().dim()
print(xmax, ymax, zmax,xmin, ymin, zmin)

null = 1E-10

TOL = 1E-5

class PeriodicBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return bool( ( near(x[0], xmin, TOL) or near(x[1], ymin, TOL) or near(x[2], zmin, TOL) )
			and on_boundary)

	def map(self, x, y):
		if near(x[0], xmax, TOL):
			y[0] = x[0] - periodic_x[0]
			y[1] = x[1] - periodic_x[1]
			y[2] = x[2] - periodic_x[2]
		elif near(x[1], ymax, TOL):
			y[0] = x[0] - periodic_y[0]
			y[1] = x[1] - periodic_y[1]
			y[2] = x[2] - periodic_y[2]
		elif near(x[2], zmax, TOL):
			y[0] = x[0] - periodic_z[0]
			y[1] = x[1] - periodic_z[1]
			y[2] = x[2] - periodic_z[2]
		else:
			y[0] = x[0]
			y[1] = x[1]
			y[2] = x[2]
	
	
unk_vector = VectorElement("CG", mesh.ufl_cell(), 1) 
lag_vector = VectorElement("R", mesh.ufl_cell(), 0)
Space = FunctionSpace(mesh, MixedElement([unk_vector,lag_vector]), constrained_domain=PeriodicBoundary() ) 

print('Degrees of freedom: ',Space.dim())

dunk = TrialFunction(Space)
del_unk = TestFunction(Space)
dp, dl = split(dunk)
del_p, del_l = split(del_unk)

unk00 = Function(Space)
phi00, lag00 = split(unk00)
unk01 = Function(Space)
phi01, lag01 = split(unk01)
unk02 = Function(Space)
phi02, lag02 = split(unk02)
#unk10 = unk01
unk11 = Function(Space)
phi11, lag11 = split(unk11)
unk12 = Function(Space)
phi12, lag12 = split(unk12)
#unk20 = unk02
#unk21 = unk12
unk22 = Function(Space)
phi22, lag22 = split(unk22)


unk000 = Function(Space)
psi000, lag000 = split(unk000)
unk001 = Function(Space)
psi001, lag001 = split(unk001)
unk002 = Function(Space)
psi002, lag002 = split(unk002)
unk010 = Function(Space)
psi010, lag010 = split(unk010)
unk011 = Function(Space)
psi011, lag011 = split(unk011)
unk012 = Function(Space)
psi012, lag012 = split(unk012)
unk020 = Function(Space)
psi020, lag020 = split(unk020)
unk021 = Function(Space)
psi021, lag021 = split(unk021)
unk022 = Function(Space)
psi022, lag022 = split(unk022)

#unk100 = unk010
#unk101 = unk011
#unk102 = unk012
unk110 = Function(Space)
psi110, lag110 = split(unk110)
unk111 = Function(Space)
psi111, lag111 = split(unk111)
unk112 = Function(Space)
psi112, lag112 = split(unk112)
unk120 = Function(Space)
psi120, lag120 = split(unk120)
unk121 = Function(Space)
psi121, lag121 = split(unk121)
unk122 = Function(Space)
psi122, lag122 = split(unk122)

#unk200 = unk020
#unk201 = unk021
#unk202 = unk022
#unk210 = unk120
#unk211 = unk121
#unk212 = unk122
unk220 = Function(Space)
psi220, lag220 = split(unk220)
unk221 = Function(Space)
psi221, lag221 = split(unk221)
unk222 = Function(Space)
psi222, lag222 = split(unk222)

#units: mm, Mg = 1000 kg=ton, s, MPa, mJ, K
i,j,k,l = indices(4) 
delta = Identity(Dim)

E_mat = 1E-2 #in MPa
nu_mat = 0.3
rho_mat = 1780E-12 * 1E-10  #(ton/mm^3)    #in Mg/mm3 = 1E12 kg/m3

E_inc = 3500 #in MPa 
nu_inc = 0.3
rho_inc = 1780E-12  #(ton/mm^3)

la_mat = E_mat*nu_mat/(1.+nu_mat)/(1.-2.*nu_mat)
mu_mat = E_mat/2./(1.+nu_mat)
la_inc = E_inc*nu_inc/(1.+nu_inc)/(1.-2.*nu_inc)
mu_inc = E_inc/2./(1.+nu_inc)

Cmat_m = as_tensor(la_mat*delta[i,j]*delta[k,l] + mu_mat*delta[i,k]*delta[j,l] + mu_mat*delta[i,l]*delta[j,k], (i,j,k,l))
Cinc_m = as_tensor(la_inc*delta[i,j]*delta[k,l] + mu_inc*delta[i,k]*delta[j,l] + mu_inc*delta[i,l]*delta[j,k], (i,j,k,l))

dofs = Space.dim()
Vol = assemble(1.*dV)

L00 = as_tensor( delta[i,0]*delta[j,0] + dp[i].dx(j) , (i,j)) 
L11 = as_tensor( delta[i,1]*delta[j,1] + dp[i].dx(j) , (i,j))
L22 = as_tensor( delta[i,2]*delta[j,2] + dp[i].dx(j) , (i,j))
L12 = as_tensor( delta[i,1]*delta[j,2] + dp[i].dx(j) , (i,j))
L21 = L12
L02 = as_tensor( delta[i,0]*delta[j,2] + dp[i].dx(j) , (i,j))
L20 = L02
L01 = as_tensor( delta[i,0]*delta[j,1] + dp[i].dx(j) , (i,j))
L10 = L01

#WEAK FORM 1
#Form_phiab = Cmat_m[i,j,k,l]*Lab[k,l]*del_p[i].dx(j)*dV(mat) + Cinc_m[i,j,k,l]*Lab[k,l]*del_p[i].dx(j)*dV(inc)
Form_phi00 = Cmat_m[i,j,k,l]*L00[k,l]*del_p[i].dx(j)*dV(mat) + Cinc_m[i,j,k,l]*L00[k,l]*del_p[i].dx(j)*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV
Form_phi11 = Cmat_m[i,j,k,l]*L11[k,l]*del_p[i].dx(j)*dV(mat) + Cinc_m[i,j,k,l]*L11[k,l]*del_p[i].dx(j)*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV
Form_phi22 = Cmat_m[i,j,k,l]*L22[k,l]*del_p[i].dx(j)*dV(mat) + Cinc_m[i,j,k,l]*L22[k,l]*del_p[i].dx(j)*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV
Form_phi12 = Cmat_m[i,j,k,l]*L12[k,l]*del_p[i].dx(j)*dV(mat) + Cinc_m[i,j,k,l]*L12[k,l]*del_p[i].dx(j)*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV
Form_phi02 = Cmat_m[i,j,k,l]*L02[k,l]*del_p[i].dx(j)*dV(mat) + Cinc_m[i,j,k,l]*L02[k,l]*del_p[i].dx(j)*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV
Form_phi01 = Cmat_m[i,j,k,l]*L01[k,l]*del_p[i].dx(j)*dV(mat) + Cinc_m[i,j,k,l]*L01[k,l]*del_p[i].dx(j)*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV

a_00, K_00 = lhs(Form_phi00), rhs(Form_phi00)
a_11, K_11 = lhs(Form_phi11), rhs(Form_phi11)
a_22, K_22 = lhs(Form_phi22), rhs(Form_phi22)	
a_12, K_12 = lhs(Form_phi12), rhs(Form_phi12)
a_02, K_02 = lhs(Form_phi02), rhs(Form_phi02)
a_01, K_01 = lhs(Form_phi01), rhs(Form_phi01)

print('Solving phi 00')

problem = LinearVariationalProblem(a_00, K_00, unk00, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_phi)
solver.solve()

print('Solving phi 11')

problem = LinearVariationalProblem(a_11, K_11, unk11, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_phi)
solver.solve()

print('Solving phi 22')

problem = LinearVariationalProblem(a_22, K_22, unk22, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_phi)
solver.solve()

print('Solving phi 12')

problem = LinearVariationalProblem(a_12, K_12, unk12, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_phi)
solver.solve()

print('Solving phi 02')

problem = LinearVariationalProblem(a_02, K_02, unk02, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_phi)
solver.solve()

print('Solving phi 01')

problem = LinearVariationalProblem(a_01, K_01, unk01, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_phi)
solver.solve()

if processID == 0: 
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('%.0f DOFs in %.0f h %.0f min %.0f s' % (dofs, e_h, e_m, e_s)  )

print('Determining parameters of C')
L00 = as_tensor( delta[i,0]*delta[j,0] + phi00[i].dx(j) , (i,j))
L11 = as_tensor( delta[i,1]*delta[j,1] + phi11[i].dx(j) , (i,j))
L22 = as_tensor( delta[i,2]*delta[j,2] + phi22[i].dx(j) , (i,j))
L12 = as_tensor( delta[i,1]*delta[j,2] + phi12[i].dx(j) , (i,j))
L21 = L12
L02 = as_tensor( delta[i,0]*delta[j,2] + phi02[i].dx(j) , (i,j))
L20 = L02
L01 = as_tensor( delta[i,0]*delta[j,1] + phi01[i].dx(j) , (i,j))
L10 = L01

#Cabcd = 1./Vol * assemble( Cmat_m[i,j,k,l]*Lab[i,j]*Lcd[k,l]*dV(mat) )
C0000 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*L00[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*L00[k,l]*dV(inc) )
C0011 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*L11[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*L11[k,l]*dV(inc) )
C0022 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*L22[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*L22[k,l]*dV(inc) )
C0012 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*L12[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*L12[k,l]*dV(inc) )
C0002 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*L02[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*L02[k,l]*dV(inc) )
C0001 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*L01[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*L01[k,l]*dV(inc) )

C1100 = C0011
C1111 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*L11[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*L11[k,l]*dV(inc) )
C1122 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*L22[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*L22[k,l]*dV(inc) )
C1112 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*L12[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*L12[k,l]*dV(inc) )
C1102 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*L02[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*L02[k,l]*dV(inc) )
C1101 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*L01[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*L01[k,l]*dV(inc) )

C2200 = C0022
C2211 = C1122
C2222 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*L22[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*L22[k,l]*dV(inc) )
C2212 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*L12[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*L12[k,l]*dV(inc) )
C2202 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*L02[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*L02[k,l]*dV(inc) )
C2201 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*L01[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*L01[k,l]*dV(inc) )

C1200 = C0012
C1211 = C1112
C1222 = C2212
C1212 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*L12[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*L12[k,l]*dV(inc) )
C1202 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*L02[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*L02[k,l]*dV(inc) )
C1201 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*L01[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*L01[k,l]*dV(inc) )

C0200 = C0002
C0211 = C1102
C0222 = C2202
C0212 = C1202
C0202 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*L02[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*L02[k,l]*dV(inc) )
C0201 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*L01[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*L01[k,l]*dV(inc) )

C0100 = C0001
C0111 = C1101
C0122 = C2201
C0112 = C1201
C0102 = C0201
C0101 = 1./Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*L01[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*L01[k,l]*dV(inc) )

divd = 1

if processID == 0: 
	with open(data+'.tex', 'w') as file:
		file.write(r'\documentclass[a4paper,10pt,abstracton,openbib,final]{scrartcl}'+'\n')
		file.write(r'\usepackage{amsmath, amsthm, amsxtra, amsfonts, amssymb, mathtools}'+'\n')
		file.write(r'\setcounter{MaxMatrixCols}{20}'+'\n')
		file.write(r'\newcommand{\begeq}{\begin{equation}\begin{gathered}}'+'\n')
		file.write(r'\newcommand{\eqend}{\end{gathered}\end{equation}}'+'\n')
		file.write(r'\newcommand{\mi}{^\text{m}}'+'\n')
		file.write(r'\newcommand{\ma}{^\text{M}}'+'\n')
		file.write(r'\begin{document}'+'\n')

	with open(data+'.tex', 'a') as file:
		file.write(r'\begeq')
		file.write('\n')
		file.write(r'C\ma_{AB} = \begin{pmatrix}')
		file.write('\n')
		file.write(str(round(C0000/divd))+' & '+str(round(C0011/divd))+' & '+str(round(C0022/divd))+' & '+str(round(C0012/divd))+' & '+str(round(C0002/divd))+' & '+str(round(C0001/divd))+r' \\ ')
		file.write('\n')
		file.write(str(round(C1100/divd))+' & '+str(round(C1111/divd))+' & '+str(round(C1122/divd))+' & '+str(round(C1112/divd))+' & '+str(round(C1102/divd))+' & '+str(round(C1101/divd))+r' \\ ') 
		file.write('\n')
		file.write(str(round(C2200/divd))+' & '+str(round(C2211/divd))+' & '+str(round(C2222/divd))+' & '+str(round(C2212/divd))+' & '+str(round(C2202/divd))+' & '+str(round(C2201/divd))+r' \\ ') 
		file.write('\n')
		file.write(str(round(C1200/divd))+' & '+str(round(C1211/divd))+' & '+str(round(C1222/divd))+' & '+str(round(C1212/divd))+' & '+str(round(C1202/divd))+' & '+str(round(C1201/divd))+r' \\ ')
		file.write('\n')
		file.write(str(round(C0200/divd))+' & '+str(round(C0211/divd))+' & '+str(round(C0222/divd))+' & '+str(round(C0212/divd))+' & '+str(round(C0202/divd))+' & '+str(round(C0201/divd))+r' \\ ')
		file.write('\n')
		file.write(str(round(C0100/divd))+' & '+str(round(C0111/divd))+' & '+str(round(C0122/divd))+' & '+str(round(C0112/divd))+' & '+str(round(C0102/divd))+' & '+str(round(C0101/divd))+r' \\ ')
		file.write('\n')
		file.write(r'\end{pmatrix} \text{\,MPa} \ ,')
		file.write('\n')
		file.write(r'\eqend')
		file.write('\n')

if processID == 0: 
	with open(data+'_SG.tex', 'a') as file:
		file.write(r'\begeq')
		file.write('\n')
		file.write(r'C\ma_{AB} = \begin{pmatrix}')
		file.write('\n')
		file.write(str(round(C0000/divd))+' , '+str(round(C0011/divd))+' , '+str(round(C0022/divd))+' , '+str(round(C0012/divd))+' , '+str(round(C0002/divd))+' , '+str(round(C0001/divd)))
		file.write('\n')
		file.write(str(round(C1100/divd))+' , '+str(round(C1111/divd))+' , '+str(round(C1122/divd))+' , '+str(round(C1112/divd))+' , '+str(round(C1102/divd))+' , '+str(round(C1101/divd))) 
		file.write('\n')
		file.write(str(round(C2200/divd))+' , '+str(round(C2211/divd))+' , '+str(round(C2222/divd))+' , '+str(round(C2212/divd))+' , '+str(round(C2202/divd))+' , '+str(round(C2201/divd))) 
		file.write('\n')
		file.write(str(round(C1200/divd))+' , '+str(round(C1211/divd))+' , '+str(round(C1222/divd))+' , '+str(round(C1212/divd))+' , '+str(round(C1202/divd))+' , '+str(round(C1201/divd)))
		file.write('\n')
		file.write(str(round(C0200/divd))+' , '+str(round(C0211/divd))+' , '+str(round(C0222/divd))+' , '+str(round(C0212/divd))+' , '+str(round(C0202/divd))+' , '+str(round(C0201/divd)))
		file.write('\n')
		file.write(str(round(C0100/divd))+' , '+str(round(C0111/divd))+' , '+str(round(C0122/divd))+' , '+str(round(C0112/divd))+' , '+str(round(C0102/divd))+' , '+str(round(C0101/divd)))
		file.write('\n')
		file.write(r'\end{pmatrix} \text{\,MPa} \ ,')
		file.write('\n')
		file.write(r'\eqend')
		file.write('\n')
		
def VoigtToTensor(A):
	A11, A12, A13, A14, A15, A16 = A[0,0], A[0,1], A[0,2], A[0,3], A[0,4], A[0,5]
	A22, A23, A24, A25, A26 = A[1,1], A[1,2], A[1,3], A[1,4], A[1,5]
	A33, A34, A35, A36 = A[2,2], A[2,3], A[2,4], A[2,5]
	A44, A45, A46 = A[3,3], A[3,4], A[3,5]
	A55, A56 = A[4,4], A[4,5]
	A66 = A[5,5]
	A21, A31, A41, A51, A61 = A12, A13, A14, A15, A16
	A32, A42, A52, A62 = A23, A24, A25, A26
	A43, A53, A63 = A34, A35, A36
	A54, A64 = A45, A46
	A65 = A56
	return as_tensor([ \
	[ \
	[ [A11,A16,A15], [A16,A12,A14], [A15,A14,A13]] , \
	[ [A61,A66,A65], [A66,A62,A64], [A65,A64,A63]] , \
	[ [A51,A56,A55], [A56,A52,A54], [A55,A54,A53]] \
	] , [ \
	[ [A61,A66,A65], [A66,A62,A64], [A65,A64,A63]] , \
	[ [A21,A26,A25], [A26,A22,A24], [A25,A24,A23]] , \
	[ [A41,A46,A45], [A46,A42,A44], [A45,A44,A43]] \
	] , [ \
	[ [A51,A56,A55], [A56,A52,A54], [A55,A54,A53]] , \
	[ [A41,A46,A45], [A46,A42,A44], [A45,A44,A43]] , \
	[ [A31,A36,A35], [A36,A32,A34], [A35,A34,A33]] ] \
	])

C_Voigt = np.array([ \
[C0000, C0011, C0022, C0012, C0002, C0001],\
[C1100, C1111, C1122, C1112, C1102, C1101],\
[C2200, C2211, C2222, C2212, C2202, C2201],\
[C1200, C1211, C1222, C1212, C1202, C1201],\
[C0200, C0211, C0222, C0212, C0202, C0201],\
[C0100, C0111, C0122, C0112, C0102, C0101]  ])

C_M = VoigtToTensor(C_Voigt)

N000 = as_tensor( phi00[i]*delta[j,0] + dp[i].dx(j) , (i,j))
N001 = as_tensor( phi00[i]*delta[j,1] + dp[i].dx(j) , (i,j))
N002 = as_tensor( phi00[i]*delta[j,2] + dp[i].dx(j) , (i,j))
N010 = as_tensor( phi01[i]*delta[j,0] + dp[i].dx(j) , (i,j))
N011 = as_tensor( phi01[i]*delta[j,1] + dp[i].dx(j) , (i,j))
N012 = as_tensor( phi01[i]*delta[j,2] + dp[i].dx(j) , (i,j))
N020 = as_tensor( phi02[i]*delta[j,0] + dp[i].dx(j) , (i,j))
N021 = as_tensor( phi02[i]*delta[j,1] + dp[i].dx(j) , (i,j))
N022 = as_tensor( phi02[i]*delta[j,2] + dp[i].dx(j) , (i,j))

N100 = N010 
N101 = N011 
N102 = N012 

N110 = as_tensor( phi11[i]*delta[j,0] + dp[i].dx(j) , (i,j))
N111 = as_tensor( phi11[i]*delta[j,1] + dp[i].dx(j) , (i,j))
N112 = as_tensor( phi11[i]*delta[j,2] + dp[i].dx(j) , (i,j))

N120 = as_tensor( phi12[i]*delta[j,0] + dp[i].dx(j) , (i,j))
N121 = as_tensor( phi12[i]*delta[j,1] + dp[i].dx(j) , (i,j))
N122 = as_tensor( phi12[i]*delta[j,2] + dp[i].dx(j) , (i,j))

N200 = N020 
N201 = N021 
N202 = N022 

N210 = N120 
N211 = N121 
N212 = N122 

N220 = as_tensor( phi22[i]*delta[j,0] + dp[i].dx(j) , (i,j))
N221 = as_tensor( phi22[i]*delta[j,1] + dp[i].dx(j) , (i,j))
N222 = as_tensor( phi22[i]*delta[j,2] + dp[i].dx(j) , (i,j))

rho_M = assemble(rho_mat*dV(mat) + rho_inc*dV(inc))
rho_M = rho_M/assemble(1*dV)

#Form_psiabc = ( - rho_mat/rho_M * C_M[i,c,a,b]*del_p_abc[i] + Lab[k,l]*Cmat_m[i,c,k,l]*del_p_abc[i] - Cmat_m[i,j,k,l]*Nabc[k,l]*del_p_abc[i].dx(j) )*dV(mat) + ( - rho_mat/rho_M * C_M[i,c,a,b]*del_p_abc[i] + Lab[k,l]*Cmat_m[i,c,k,l]*del_p_abc[i] - Cmat_m[i,j,k,l]*Nabc[k,l]*del_p_abc[i].dx(j) )*dV(inc) 


Form_psi000 = (- rho_mat/rho_M *C_M[i,0,0,0]*del_p[i] +  L00[k,l]*Cmat_m[i,0,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N000[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,0,0,0]*del_p[i]  + L00[k,l]*Cinc_m[i,0,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N000[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV 
Form_psi001 = (- rho_mat/rho_M *C_M[i,1,0,0]*del_p[i] +  L00[k,l]*Cmat_m[i,1,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N001[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,1,0,0]*del_p[i]  + L00[k,l]*Cinc_m[i,1,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N001[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV   
Form_psi002 = (- rho_mat/rho_M *C_M[i,2,0,0]*del_p[i] +  L00[k,l]*Cmat_m[i,2,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N002[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,2,0,0]*del_p[i]  + L00[k,l]*Cinc_m[i,2,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N002[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV 
Form_psi010 = (- rho_mat/rho_M *C_M[i,0,0,1]*del_p[i] +  L01[k,l]*Cmat_m[i,0,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N010[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,0,0,1]*del_p[i]  + L01[k,l]*Cinc_m[i,0,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N010[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV 
Form_psi011 = (- rho_mat/rho_M *C_M[i,1,0,1]*del_p[i] +  L01[k,l]*Cmat_m[i,1,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N011[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,1,0,1]*del_p[i]  + L01[k,l]*Cinc_m[i,1,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N011[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV 
Form_psi012 = (- rho_mat/rho_M *C_M[i,2,0,1]*del_p[i] +  L01[k,l]*Cmat_m[i,2,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N012[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,2,0,1]*del_p[i]  + L01[k,l]*Cinc_m[i,2,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N012[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV 
Form_psi020 = (- rho_mat/rho_M *C_M[i,0,0,2]*del_p[i] +  L02[k,l]*Cmat_m[i,0,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N020[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,0,0,2]*del_p[i]  + L02[k,l]*Cinc_m[i,0,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N020[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV 
Form_psi021 = (- rho_mat/rho_M *C_M[i,1,0,2]*del_p[i] +  L02[k,l]*Cmat_m[i,1,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N021[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,1,0,2]*del_p[i]  + L02[k,l]*Cinc_m[i,1,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N021[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV 
Form_psi022 = (- rho_mat/rho_M *C_M[i,2,0,2]*del_p[i] +  L02[k,l]*Cmat_m[i,2,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N022[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,2,0,2]*del_p[i]  + L02[k,l]*Cinc_m[i,2,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N022[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV 

a_000, L_000 = lhs(Form_psi000), rhs(Form_psi000)
a_001, L_001 = lhs(Form_psi001), rhs(Form_psi001)
a_002, L_002 = lhs(Form_psi002), rhs(Form_psi002)
a_010, L_010 = lhs(Form_psi010), rhs(Form_psi010)
a_011, L_011 = lhs(Form_psi011), rhs(Form_psi011)
a_012, L_012 = lhs(Form_psi012), rhs(Form_psi012)
a_020, L_020 = lhs(Form_psi020), rhs(Form_psi020)
a_021, L_021 = lhs(Form_psi021), rhs(Form_psi021)
a_022, L_022 = lhs(Form_psi022), rhs(Form_psi022)

#Form_psi100 = Form_psi010 
#Form_psi101 = Form_psi011 
#Form_psi102 = Form_psi012 

Form_psi110 = (- rho_mat/rho_M *C_M[i,0,1,1]*del_p[i] +  L11[k,l]*Cmat_m[i,0,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N110[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,0,1,1]*del_p[i]  + L11[k,l]*Cinc_m[i,0,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N110[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV
Form_psi111 = (- rho_mat/rho_M *C_M[i,1,1,1]*del_p[i] +  L11[k,l]*Cmat_m[i,1,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N111[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,1,1,1]*del_p[i]  + L11[k,l]*Cinc_m[i,1,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N111[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV
Form_psi112 = (- rho_mat/rho_M *C_M[i,2,1,1]*del_p[i] +  L11[k,l]*Cmat_m[i,2,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N112[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,2,1,1]*del_p[i]  + L11[k,l]*Cinc_m[i,2,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N112[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV
Form_psi120 = (- rho_mat/rho_M *C_M[i,0,1,2]*del_p[i] +  L12[k,l]*Cmat_m[i,0,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N120[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,0,1,2]*del_p[i]  + L12[k,l]*Cinc_m[i,0,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N120[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV
Form_psi121 = (- rho_mat/rho_M *C_M[i,1,1,2]*del_p[i] +  L12[k,l]*Cmat_m[i,1,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N121[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,1,1,2]*del_p[i]  + L12[k,l]*Cinc_m[i,1,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N121[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV
Form_psi122 = (- rho_mat/rho_M *C_M[i,2,1,2]*del_p[i] +  L12[k,l]*Cmat_m[i,2,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N122[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,2,1,2]*del_p[i]  + L12[k,l]*Cinc_m[i,2,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N122[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV

a_110, L_110 = lhs(Form_psi110), rhs(Form_psi110)
a_111, L_111 = lhs(Form_psi111), rhs(Form_psi111)
a_112, L_112 = lhs(Form_psi112), rhs(Form_psi112)
a_120, L_120 = lhs(Form_psi120), rhs(Form_psi120)
a_121, L_121 = lhs(Form_psi121), rhs(Form_psi121)
a_122, L_122 = lhs(Form_psi122), rhs(Form_psi122)

Form_psi200 = Form_psi020 
Form_psi201 = Form_psi021 
Form_psi202 = Form_psi022 
Form_psi210 = Form_psi120 
Form_psi211 = Form_psi121 
Form_psi212 = Form_psi122 

Form_psi220 = (- rho_mat/rho_M *C_M[i,0,2,2]*del_p[i] +  L22[k,l]*Cmat_m[i,0,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N220[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,0,2,2]*del_p[i]  + L22[k,l]*Cinc_m[i,0,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N220[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV
Form_psi221 = (- rho_mat/rho_M *C_M[i,1,2,2]*del_p[i] +  L22[k,l]*Cmat_m[i,1,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N221[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,1,2,2]*del_p[i]  + L22[k,l]*Cinc_m[i,1,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N221[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV
Form_psi222 = (- rho_mat/rho_M *C_M[i,2,2,2]*del_p[i] +  L22[k,l]*Cmat_m[i,2,k,l]*del_p[i] - Cmat_m[i,j,k,l]*N222[k,l]*del_p[i].dx(j) )*dV(mat) + ( - rho_inc/rho_M * C_M[i,2,2,2]*del_p[i]  + L22[k,l]*Cinc_m[i,2,k,l]*del_p[i] - Cinc_m[i,j,k,l]*N222[k,l]*del_p[i].dx(j) )*dV(inc) + (dp[i]*del_l[i] + del_p[i]*dl[i])*dV

a_220, L_220 = lhs(Form_psi220), rhs(Form_psi220)
a_221, L_221 = lhs(Form_psi221), rhs(Form_psi221)
a_222, L_222 = lhs(Form_psi222), rhs(Form_psi222)


print('Solving psi 00i')


problem = LinearVariationalProblem(a_000, L_000, unk000, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

problem = LinearVariationalProblem(a_001, L_001, unk001, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()


problem = LinearVariationalProblem(a_002, L_002, unk002, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

if processID == 0: 
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('%.0f DOFs in %.0f h %.0f min %.0f s' % (dofs, e_h, e_m, e_s)  )

print('Solving psi 01i')

problem = LinearVariationalProblem(a_010, L_010, unk010, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

problem = LinearVariationalProblem(a_011, L_011, unk011, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

problem = LinearVariationalProblem(a_012, L_012, unk012, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

if processID == 0: 
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('%.0f DOFs in %.0f h %.0f min %.0f s' % (dofs, e_h, e_m, e_s)  )

print('Solving psi 02i')

problem = LinearVariationalProblem(a_020, L_020, unk020, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

problem = LinearVariationalProblem(a_021, L_021, unk021, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

problem = LinearVariationalProblem(a_022, L_022, unk022, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

if processID == 0: 
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('%.0f DOFs in %.0f h %.0f min %.0f s' % (dofs, e_h, e_m, e_s)  )

print('Skipping, psi 10i = psi 01i ')

print('Solving psi 11i')

problem = LinearVariationalProblem(a_110, L_110, unk110, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

problem = LinearVariationalProblem(a_111, L_111, unk111, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

problem = LinearVariationalProblem(a_112, L_112, unk112, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

if processID == 0: 
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('%.0f DOFs in %.0f h %.0f min %.0f s' % (dofs, e_h, e_m, e_s)  )

print('Solving psi 12i')

problem = LinearVariationalProblem(a_120, L_120, unk120, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

problem = LinearVariationalProblem(a_121, L_121, unk121, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

problem = LinearVariationalProblem(a_122, L_122, unk122, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

if processID == 0: 
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('%.0f DOFs in %.0f h %.0f min %.0f s' % (dofs, e_h, e_m, e_s)  )

print('Skipping, psi 20i = psi 02i')
print('Skipping, psi 21i = psi 12i')
print('Solving psi 22i')

problem = LinearVariationalProblem(a_220, L_220, unk220, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

problem = LinearVariationalProblem(a_221, L_221, unk221, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

problem = LinearVariationalProblem(a_222, L_222, unk222, bcs=[])
solver  = LinearVariationalSolver(problem)
solver.parameters.update(solver_parameters_psi)
solver.solve()

if processID == 0: 
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('%.0f DOFs in %.0f h %.0f min %.0f s' % (dofs, e_h, e_m, e_s)  )

N000 = as_tensor( phi00[i]*delta[j,0] + psi000[i].dx(j) , (i,j))
N001 = as_tensor( phi00[i]*delta[j,1] + psi001[i].dx(j) , (i,j))
N002 = as_tensor( phi00[i]*delta[j,2] + psi002[i].dx(j) , (i,j))
N010 = as_tensor( phi01[i]*delta[j,0] + psi010[i].dx(j) , (i,j))
N011 = as_tensor( phi01[i]*delta[j,1] + psi011[i].dx(j) , (i,j))
N012 = as_tensor( phi01[i]*delta[j,2] + psi012[i].dx(j) , (i,j))
N020 = as_tensor( phi02[i]*delta[j,0] + psi020[i].dx(j) , (i,j))
N021 = as_tensor( phi02[i]*delta[j,1] + psi021[i].dx(j) , (i,j))
N022 = as_tensor( phi02[i]*delta[j,2] + psi022[i].dx(j) , (i,j))

N100 = N010 
N101 = N011 
N102 = N012 

N110 = as_tensor( phi11[i]*delta[j,0] + psi110[i].dx(j) , (i,j))
N111 = as_tensor( phi11[i]*delta[j,1] + psi111[i].dx(j) , (i,j))
N112 = as_tensor( phi11[i]*delta[j,2] + psi112[i].dx(j) , (i,j))

N120 = as_tensor( phi12[i]*delta[j,0] + psi120[i].dx(j) , (i,j))
N121 = as_tensor( phi12[i]*delta[j,1] + psi121[i].dx(j) , (i,j))
N122 = as_tensor( phi12[i]*delta[j,2] + psi122[i].dx(j) , (i,j))

N200 = N020 
N201 = N021 
N202 = N022 

N210 = N120 
N211 = N121 
N212 = N122 

N220 = as_tensor( phi22[i]*delta[j,0] + psi220[i].dx(j) , (i,j))
N221 = as_tensor( phi22[i]*delta[j,1] + psi221[i].dx(j) , (i,j))
N222 = as_tensor( phi22[i]*delta[j,2] + psi222[i].dx(j) , (i,j))

M000 = as_tensor( L00[i,j]*x[0] + N000[i,j] , (i,j))
M001 = as_tensor( L00[i,j]*x[1] + N001[i,j] , (i,j))
M002 = as_tensor( L00[i,j]*x[2] + N002[i,j] , (i,j))
M010 = as_tensor( L01[i,j]*x[0] + N010[i,j] , (i,j))
M011 = as_tensor( L01[i,j]*x[1] + N011[i,j] , (i,j))
M012 = as_tensor( L01[i,j]*x[2] + N012[i,j] , (i,j))
M020 = as_tensor( L02[i,j]*x[0] + N020[i,j] , (i,j))
M021 = as_tensor( L02[i,j]*x[1] + N021[i,j] , (i,j))
M022 = as_tensor( L02[i,j]*x[2] + N022[i,j] , (i,j))

M100 = M010
M101 = M011
M102 = M012
M110 = as_tensor( L11[i,j]*x[0] + N110[i,j] , (i,j))
M111 = as_tensor( L11[i,j]*x[1] + N111[i,j] , (i,j))
M112 = as_tensor( L11[i,j]*x[2] + N112[i,j] , (i,j))
M120 = as_tensor( L12[i,j]*x[0] + N120[i,j] , (i,j))
M121 = as_tensor( L12[i,j]*x[1] + N121[i,j] , (i,j))
M122 = as_tensor( L12[i,j]*x[2] + N122[i,j] , (i,j))

M200 = M020
M201 = M021
M202 = M022
M210 = M120
M211 = M121
M212 = M122
M220 = as_tensor( L22[i,j]*x[0] + N220[i,j] , (i,j))
M221 = as_tensor( L22[i,j]*x[1] + N221[i,j] , (i,j))
M222 = as_tensor( L22[i,j]*x[2] + N222[i,j] , (i,j))

epsilon = 1.0 #homothetic ratio, corresponding to the real size of the substructure

print('Determining parameters of G')

#Gabcde = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*Lab[i,j]*Mcde[k,l]*dV(mat) )

G00000 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M000[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M000[k,l]*dV(inc))
print(G00000)
G11000 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M000[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M000[k,l]*dV(inc))
G22000 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M000[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M000[k,l]*dV(inc))
G12000 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M000[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M000[k,l]*dV(inc))
G02000 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M000[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M000[k,l]*dV(inc))
G01000 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M000[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M000[k,l]*dV(inc))

G00110 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M110[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M110[k,l]*dV(inc))
G11110 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M110[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M110[k,l]*dV(inc))
G22110 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M110[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M110[k,l]*dV(inc))
G12110 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M110[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M110[k,l]*dV(inc))
G02110 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M110[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M110[k,l]*dV(inc))
G01110 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M110[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M110[k,l]*dV(inc))

G00220 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M220[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M220[k,l]*dV(inc))
G11220 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M220[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M220[k,l]*dV(inc))
G22220 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M220[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M220[k,l]*dV(inc))
G12220 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M220[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M220[k,l]*dV(inc))
G02220 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M220[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M220[k,l]*dV(inc))
G01220 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M220[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M220[k,l]*dV(inc))

G00120 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M120[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M120[k,l]*dV(inc))
G11120 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M120[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M120[k,l]*dV(inc))
G22120 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M120[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M120[k,l]*dV(inc))
G12120 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M120[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M120[k,l]*dV(inc))
G02120 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M120[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M120[k,l]*dV(inc))
G01120 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M120[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M120[k,l]*dV(inc))

G00020 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M020[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M020[k,l]*dV(inc))
G11020 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M020[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M020[k,l]*dV(inc))
G22020 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M020[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M020[k,l]*dV(inc))
G12020 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M020[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M020[k,l]*dV(inc))
G02020 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M020[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M020[k,l]*dV(inc))
G01020 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M020[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M020[k,l]*dV(inc))

G00010 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M010[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M010[k,l]*dV(inc))
G11010 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M010[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M010[k,l]*dV(inc))
G22010 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M010[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M010[k,l]*dV(inc))
G12010 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M010[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M010[k,l]*dV(inc))
G02010 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M010[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M010[k,l]*dV(inc))
G01010 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M010[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M010[k,l]*dV(inc))

G00001 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M001[k,l]*dV(inc))
G11001 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M001[k,l]*dV(inc))
G22001 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M001[k,l]*dV(inc))
G12001 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M001[k,l]*dV(inc))
G02001 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M001[k,l]*dV(inc))
G01001 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M001[k,l]*dV(inc))

G00111 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M111[k,l]*dV(inc))
G11111 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M111[k,l]*dV(inc))
G22111 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M111[k,l]*dV(inc))
G12111 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M111[k,l]*dV(inc))
G02111 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M111[k,l]*dV(inc))
G01111 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M111[k,l]*dV(inc))

G00221 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M221[k,l]*dV(inc))
G11221 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M221[k,l]*dV(inc))
G22221 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M221[k,l]*dV(inc))
G12221 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M221[k,l]*dV(inc))
G02221 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M221[k,l]*dV(inc))
G01221 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M221[k,l]*dV(inc))

G00121 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M121[k,l]*dV(inc))
G11121 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M121[k,l]*dV(inc))
G22121 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M121[k,l]*dV(inc))
G12121 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M121[k,l]*dV(inc))
G02121 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M121[k,l]*dV(inc))
G01121 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M121[k,l]*dV(inc))

G00021 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M021[k,l]*dV(inc))
G11021 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M021[k,l]*dV(inc))
G22021 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M021[k,l]*dV(inc))
G12021 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M021[k,l]*dV(inc))
G02021 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M021[k,l]*dV(inc))
G01021 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M021[k,l]*dV(inc))

G00011 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M011[k,l]*dV(inc))
G11011 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M011[k,l]*dV(inc))
G22011 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M011[k,l]*dV(inc))
G12011 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M011[k,l]*dV(inc))
G02011 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M011[k,l]*dV(inc))
G01011 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M011[k,l]*dV(inc))


G00002 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M002[k,l]*dV(inc))
G11002 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M002[k,l]*dV(inc))
G22002 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M002[k,l]*dV(inc))
G12002 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M002[k,l]*dV(inc))
G02002 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M002[k,l]*dV(inc))
G01002 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M002[k,l]*dV(inc))

G00112 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M112[k,l]*dV(inc))
G11112 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M112[k,l]*dV(inc))
G22112 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M112[k,l]*dV(inc))
G12112 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M112[k,l]*dV(inc))
G02112 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M112[k,l]*dV(inc))
G01112 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M112[k,l]*dV(inc))

G00222 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M222[k,l]*dV(inc))
G11222 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M222[k,l]*dV(inc))
G22222 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M222[k,l]*dV(inc))
G12222 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M222[k,l]*dV(inc))
G02222 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M222[k,l]*dV(inc))
G01222 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M222[k,l]*dV(inc))

G00122 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M122[k,l]*dV(inc))
G11122 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M122[k,l]*dV(inc))
G22122 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M122[k,l]*dV(inc))
G12122 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M122[k,l]*dV(inc))
G02122 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M122[k,l]*dV(inc))
G01122 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M122[k,l]*dV(inc))

G00022 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M022[k,l]*dV(inc))
G11022 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M022[k,l]*dV(inc))
G22022 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M022[k,l]*dV(inc))
G12022 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M022[k,l]*dV(inc))
G02022 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M022[k,l]*dV(inc))
G01022 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M022[k,l]*dV(inc))

G00012 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L00[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L00[i,j]*M012[k,l]*dV(inc))
G11012 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L11[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L11[i,j]*M012[k,l]*dV(inc))
G22012 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L22[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L22[i,j]*M012[k,l]*dV(inc))
G12012 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L12[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L12[i,j]*M012[k,l]*dV(inc))
G02012 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L02[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L02[i,j]*M012[k,l]*dV(inc))
G01012 = epsilon/Vol * assemble( Cmat_m[i,j,k,l]*L01[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*L01[i,j]*M012[k,l]*dV(inc))

if processID == 0: 
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('%.0f DOFs in %.0f h %.0f min %.0f s' % (dofs, e_h, e_m, e_s)  )

divby = 1
powerdivby = np.log10(divby)
digits = 0

# 1. G 
if processID == 0: 
	with open(data+'.tex', 'a') as file:
		file.write(r'\begeq')
		file.write('\n')
		file.write(r'G\ma_{A\alpha} =')
		file.write('\n')
		file.write(r'\resizebox{.85\textwidth}{!}{$\displaystyle')
		file.write('\n')
		file.write(r'\begin{pmatrix}')
		file.write('\n')
		file.write(str(round(G00000/divby,digits))+' & '+str(round(G00110/divby,digits))+' & '+str(round(G00220/divby,digits))+' & '+str(round(G00120/divby,digits))+' & '+str(round(G00020/divby,digits))+' & '+str(round(G00010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(G00001/divby,digits))+' & '+str(round(G00111/divby,digits))+' & '+str(round(G00221/divby,digits))+' & '+str(round(G00121/divby,digits))+' & '+str(round(G00021/divby,digits))+' & '+str(round(G00011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(G00002/divby,digits))+' & '+str(round(G00112/divby,digits))+' & '+str(round(G00222/divby,digits))+' & '+str(round(G00122/divby,digits))+' & '+str(round(G00022/divby,digits))+' & '+str(round(G00012/divby,digits))+' & ')
		file.write('\n')
		file.write(r'\\')
		file.write(str(round(G11000/divby,digits))+' & '+str(round(G11110/divby,digits))+' & '+str(round(G11220/divby,digits))+' & '+str(round(G11120/divby,digits))+' & '+str(round(G11020/divby,digits))+' & '+str(round(G11010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(G11001/divby,digits))+' & '+str(round(G11111/divby,digits))+' & '+str(round(G11221/divby,digits))+' & '+str(round(G11121/divby,digits))+' & '+str(round(G11021/divby,digits))+' & '+str(round(G11011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(G11002/divby,digits))+' & '+str(round(G11112/divby,digits))+' & '+str(round(G11222/divby,digits))+' & '+str(round(G11122/divby,digits))+' & '+str(round(G11022/divby,digits))+' & '+str(round(G11012/divby,digits))+' & ')
		file.write('\n')
		file.write(r'\\')
		file.write(str(round(G22000/divby,digits))+' & '+str(round(G22110/divby,digits))+' & '+str(round(G22220/divby,digits))+' & '+str(round(G22120/divby,digits))+' & '+str(round(G22020/divby,digits))+' & '+str(round(G22010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(G22001/divby,digits))+' & '+str(round(G22111/divby,digits))+' & '+str(round(G22221/divby,digits))+' & '+str(round(G22121/divby,digits))+' & '+str(round(G22021/divby,digits))+' & '+str(round(G22011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(G22002/divby,digits))+' & '+str(round(G22112/divby,digits))+' & '+str(round(G22222/divby,digits))+' & '+str(round(G22122/divby,digits))+' & '+str(round(G22022/divby,digits))+' & '+str(round(G22012/divby,digits))+' & ')
		file.write('\n')
		file.write(r'\\')
		file.write(str(round(G12000/divby,digits))+' & '+str(round(G12110/divby,digits))+' & '+str(round(G12220/divby,digits))+' & '+str(round(G12120/divby,digits))+' & '+str(round(G12020/divby,digits))+' & '+str(round(G12010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(G12001/divby,digits))+' & '+str(round(G12111/divby,digits))+' & '+str(round(G12221/divby,digits))+' & '+str(round(G12121/divby,digits))+' & '+str(round(G12021/divby,digits))+' & '+str(round(G12011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(G12002/divby,digits))+' & '+str(round(G12112/divby,digits))+' & '+str(round(G12222/divby,digits))+' & '+str(round(G12122/divby,digits))+' & '+str(round(G12022/divby,digits))+' & '+str(round(G12012/divby,digits))+' & ')
		file.write('\n')
		file.write(r'\\')
		file.write(str(round(G02000/divby,digits))+' & '+str(round(G02110/divby,digits))+' & '+str(round(G02220/divby,digits))+' & '+str(round(G02120/divby,digits))+' & '+str(round(G02020/divby,digits))+' & '+str(round(G02010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(G02001/divby,digits))+' & '+str(round(G02111/divby,digits))+' & '+str(round(G02221/divby,digits))+' & '+str(round(G02121/divby,digits))+' & '+str(round(G02021/divby,digits))+' & '+str(round(G02011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(G02002/divby,digits))+' & '+str(round(G02112/divby,digits))+' & '+str(round(G02222/divby,digits))+' & '+str(round(G02122/divby,digits))+' & '+str(round(G02022/divby,digits))+' & '+str(round(G02012/divby,digits))+' & ')
		file.write('\n')
		file.write(r'\\')
		file.write(str(round(G01000/divby,digits))+' & '+str(round(G01110/divby,digits))+' & '+str(round(G01220/divby,digits))+' & '+str(round(G01120/divby,digits))+' & '+str(round(G01020/divby,digits))+' & '+str(round(G01010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(G01001/divby,digits))+' & '+str(round(G01111/divby,digits))+' & '+str(round(G01221/divby,digits))+' & '+str(round(G01121/divby,digits))+' & '+str(round(G01021/divby,digits))+' & '+str(round(G01011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(G01002/divby,digits))+' & '+str(round(G01112/divby,digits))+' & '+str(round(G01222/divby,digits))+' & '+str(round(G01122/divby,digits))+' & '+str(round(G01022/divby,digits))+' & '+str(round(G01012/divby,digits))+' & ')
		file.write('\n')
		file.write(r'\\')
		file.write('\n')
		file.write(r'\end{pmatrix} $} 10^{')
		file.write(str(round(powerdivby)))
		file.write(r'} \text{\,N/mm} \ ,')
		file.write('\n')
		file.write(r'\eqend')

# 2. G
if processID == 0: 
	with open(data+'_SG.tex', 'a') as file:
		file.write(r'\begeq')
		file.write('\n')
		file.write(r'G\ma_{A\alpha} =')
		file.write('\n')
		file.write(r'\resizebox{.85\textwidth}{!}{$\displaystyle')
		file.write('\n')
		file.write(r'\begin{pmatrix}')
		file.write('\n')
		file.write(str(round(G00000/divby,digits))+' , '+str(round(G00110/divby,digits))+' , '+str(round(G00220/divby,digits))+' , '+str(round(G00120/divby,digits))+' , '+str(round(G00020/divby,digits))+' , '+str(round(G00010/divby,digits))+' , ')
		file.write(str(round(G00001/divby,digits))+' , '+str(round(G00111/divby,digits))+' , '+str(round(G00221/divby,digits))+' , '+str(round(G00121/divby,digits))+' , '+str(round(G00021/divby,digits))+' , '+str(round(G00011/divby,digits))+' , ')
		file.write(str(round(G00002/divby,digits))+' , '+str(round(G00112/divby,digits))+' , '+str(round(G00222/divby,digits))+' , '+str(round(G00122/divby,digits))+' , '+str(round(G00022/divby,digits))+' , '+str(round(G00012/divby,digits)))

		file.write('\n')		
		file.write(str(round(G11000/divby,digits))+' , '+str(round(G11110/divby,digits))+' , '+str(round(G11220/divby,digits))+' , '+str(round(G11120/divby,digits))+' , '+str(round(G11020/divby,digits))+' , '+str(round(G11010/divby,digits))+' , ')
		file.write(str(round(G11001/divby,digits))+' , '+str(round(G11111/divby,digits))+' , '+str(round(G11221/divby,digits))+' , '+str(round(G11121/divby,digits))+' , '+str(round(G11021/divby,digits))+' , '+str(round(G11011/divby,digits))+' , ')
		file.write(str(round(G11002/divby,digits))+' , '+str(round(G11112/divby,digits))+' , '+str(round(G11222/divby,digits))+' , '+str(round(G11122/divby,digits))+' , '+str(round(G11022/divby,digits))+' , '+str(round(G11012/divby,digits)))

		file.write('\n')

		file.write(str(round(G22000/divby,digits))+' , '+str(round(G22110/divby,digits))+' , '+str(round(G22220/divby,digits))+' , '+str(round(G22120/divby,digits))+' , '+str(round(G22020/divby,digits))+' , '+str(round(G22010/divby,digits))+' , ')
		file.write(str(round(G22001/divby,digits))+' , '+str(round(G22111/divby,digits))+' , '+str(round(G22221/divby,digits))+' , '+str(round(G22121/divby,digits))+' , '+str(round(G22021/divby,digits))+' , '+str(round(G22011/divby,digits))+' , ')
		file.write(str(round(G22002/divby,digits))+' , '+str(round(G22112/divby,digits))+' , '+str(round(G22222/divby,digits))+' , '+str(round(G22122/divby,digits))+' , '+str(round(G22022/divby,digits))+' , '+str(round(G22012/divby,digits)))
		file.write('\n')

		file.write(str(round(G12000/divby,digits))+' , '+str(round(G12110/divby,digits))+' , '+str(round(G12220/divby,digits))+' , '+str(round(G12120/divby,digits))+' , '+str(round(G12020/divby,digits))+' , '+str(round(G12010/divby,digits))+' , ')
		file.write(str(round(G12001/divby,digits))+' , '+str(round(G12111/divby,digits))+' , '+str(round(G12221/divby,digits))+' , '+str(round(G12121/divby,digits))+' , '+str(round(G12021/divby,digits))+' , '+str(round(G12011/divby,digits))+' , ')
		file.write(str(round(G12002/divby,digits))+' , '+str(round(G12112/divby,digits))+' , '+str(round(G12222/divby,digits))+' , '+str(round(G12122/divby,digits))+' , '+str(round(G12022/divby,digits))+' , '+str(round(G12012/divby,digits)))
		file.write('\n')

		file.write(str(round(G02000/divby,digits))+' , '+str(round(G02110/divby,digits))+' , '+str(round(G02220/divby,digits))+' , '+str(round(G02120/divby,digits))+' , '+str(round(G02020/divby,digits))+' , '+str(round(G02010/divby,digits))+' , ')
		file.write(str(round(G02001/divby,digits))+' , '+str(round(G02111/divby,digits))+' , '+str(round(G02221/divby,digits))+' , '+str(round(G02121/divby,digits))+' , '+str(round(G02021/divby,digits))+' , '+str(round(G02011/divby,digits))+' , ')
		file.write(str(round(G02002/divby,digits))+' , '+str(round(G02112/divby,digits))+' , '+str(round(G02222/divby,digits))+' , '+str(round(G02122/divby,digits))+' , '+str(round(G02022/divby,digits))+' , '+str(round(G02012/divby,digits)))
		file.write('\n')

		file.write(str(round(G01000/divby,digits))+' , '+str(round(G01110/divby,digits))+' , '+str(round(G01220/divby,digits))+' , '+str(round(G01120/divby,digits))+' , '+str(round(G01020/divby,digits))+' , '+str(round(G01010/divby,digits))+' , ')
		file.write(str(round(G01001/divby,digits))+' , '+str(round(G01111/divby,digits))+' , '+str(round(G01221/divby,digits))+' , '+str(round(G01121/divby,digits))+' , '+str(round(G01021/divby,digits))+' , '+str(round(G01011/divby,digits))+' , ')
		file.write(str(round(G01002/divby,digits))+' , '+str(round(G01112/divby,digits))+' , '+str(round(G01222/divby,digits))+' , '+str(round(G01122/divby,digits))+' , '+str(round(G01022/divby,digits))+' , '+str(round(G01012/divby,digits)))
		file.write('\n')

		file.write('\n')
		file.write(r'\end{pmatrix} $} 10^{')
		file.write(str(round(powerdivby)))
		file.write(r'} \text{\,N/mm} \ ,')
		file.write('\n')
		file.write(r'\eqend')

#Icf = assemble(x[c]*x[f]*dV)
I00 = epsilon**2 / Vol * assemble(x[0]*x[0]*dV)
I01 = epsilon**2 / Vol * assemble(x[0]*x[1]*dV)
I02 = epsilon**2 / Vol * assemble(x[0]*x[2]*dV)

I10 = I01
I11 = epsilon**2 / Vol * assemble(x[1]*x[1]*dV)
I12 = epsilon**2 / Vol * assemble(x[1]*x[2]*dV)

I20 = I02
I21 = I12
I22 = epsilon**2 / Vol * assemble(x[2]*x[2]*dV)

print('Determining parameters of D')
#c=0
D000000 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M000[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M000[k,l]*dV(inc)) - C_M[0,0,0,0]*I00
print(D000000)
D000110 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M110[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M110[k,l]*dV(inc)) - C_M[0,0,1,1]*I00
D000220 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M220[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M220[k,l]*dV(inc)) - C_M[0,0,2,2]*I00
D000120 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M120[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M120[k,l]*dV(inc)) - C_M[0,0,1,2]*I00
D000020 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M020[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M020[k,l]*dV(inc)) - C_M[0,0,0,2]*I00
D000010 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M010[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M010[k,l]*dV(inc)) - C_M[0,0,0,1]*I00
D000001 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M001[k,l]*dV(inc)) - C_M[0,0,0,0]*I01
D000111 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M111[k,l]*dV(inc)) - C_M[0,0,1,1]*I01
D000221 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M221[k,l]*dV(inc)) - C_M[0,0,2,2]*I01
D000121 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M121[k,l]*dV(inc)) - C_M[0,0,1,2]*I01
D000021 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M021[k,l]*dV(inc)) - C_M[0,0,0,2]*I01
D000011 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M011[k,l]*dV(inc)) - C_M[0,0,0,1]*I01
D000002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M002[k,l]*dV(inc)) - C_M[0,0,0,0]*I02
D000112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M112[k,l]*dV(inc)) - C_M[0,0,1,1]*I02
D000222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M222[k,l]*dV(inc)) - C_M[0,0,2,2]*I02
D000122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M122[k,l]*dV(inc)) - C_M[0,0,1,2]*I02
D000022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M022[k,l]*dV(inc)) - C_M[0,0,0,2]*I02
D000012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M000[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M000[i,j]*M012[k,l]*dV(inc)) - C_M[0,0,0,1]*I02
D110000 = D000110 #epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M000[k,l]*dV(mat) ) - C_M[1,1,0,0]*I00
D110110 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M110[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M110[k,l]*dV(inc)) - C_M[1,1,1,1]*I00
D110220 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M220[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M220[k,l]*dV(inc)) - C_M[1,1,2,2]*I00
D110120 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M120[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M120[k,l]*dV(inc)) - C_M[1,1,1,2]*I00
D110020 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M020[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M020[k,l]*dV(inc)) - C_M[1,1,0,2]*I00
D110010 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M010[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M010[k,l]*dV(inc)) - C_M[1,1,0,1]*I00
D110001 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M001[k,l]*dV(inc)) - C_M[1,1,0,0]*I01
D110111 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M111[k,l]*dV(inc)) - C_M[1,1,1,1]*I01
D110221 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M221[k,l]*dV(inc)) - C_M[1,1,2,2]*I01
D110121 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M121[k,l]*dV(inc)) - C_M[1,1,1,2]*I01
D110021 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M021[k,l]*dV(inc)) - C_M[1,1,0,2]*I01
D110011 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M011[k,l]*dV(inc)) - C_M[1,1,0,1]*I01
D110002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M002[k,l]*dV(inc)) - C_M[1,1,0,0]*I02
D110112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M112[k,l]*dV(inc)) - C_M[1,1,1,1]*I02
D110222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M222[k,l]*dV(inc)) - C_M[1,1,2,2]*I02
D110122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M122[k,l]*dV(inc)) - C_M[1,1,1,2]*I02
D110022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M022[k,l]*dV(inc)) - C_M[1,1,0,2]*I02
D110012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M110[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M110[i,j]*M012[k,l]*dV(inc)) - C_M[1,1,0,1]*I02

D220000 = D000220 
D220110 = D110220 
D220220 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M220[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M220[k,l]*dV(inc)) - C_M[2,2,2,2]*I00
D220120 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M120[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M120[k,l]*dV(inc)) - C_M[2,2,1,2]*I00
D220020 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M020[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M020[k,l]*dV(inc)) - C_M[2,2,0,2]*I00
D220010 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M010[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M010[k,l]*dV(inc)) - C_M[2,2,0,1]*I00
D220001 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M001[k,l]*dV(inc)) - C_M[2,2,0,0]*I01
D220111 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M111[k,l]*dV(inc)) - C_M[2,2,1,1]*I01
D220221 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M221[k,l]*dV(inc)) - C_M[2,2,2,2]*I01
D220121 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M121[k,l]*dV(inc)) - C_M[2,2,1,2]*I01
D220021 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M021[k,l]*dV(inc)) - C_M[2,2,0,2]*I01
D220011 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M011[k,l]*dV(inc)) - C_M[2,2,0,1]*I01
D220002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M002[k,l]*dV(inc)) - C_M[2,2,0,0]*I02
D220112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M112[k,l]*dV(inc)) - C_M[2,2,1,1]*I02
D220222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M222[k,l]*dV(inc)) - C_M[2,2,2,2]*I02
D220122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M122[k,l]*dV(inc)) - C_M[2,2,1,2]*I02
D220022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M022[k,l]*dV(inc)) - C_M[2,2,0,2]*I02
D220012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M220[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M220[i,j]*M012[k,l]*dV(inc)) - C_M[2,2,0,1]*I02

D120000 = D000120 
D120110 = D110120 
D120220 = D220120 
D120120 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M120[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M120[k,l]*dV(inc)) - C_M[1,2,1,2]*I00
D120020 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M020[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M020[k,l]*dV(inc)) - C_M[1,2,0,2]*I00
D120010 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M010[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M010[k,l]*dV(inc)) - C_M[1,2,0,1]*I00
D120001 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M001[k,l]*dV(inc)) - C_M[1,2,0,0]*I01
D120111 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M111[k,l]*dV(inc)) - C_M[1,2,1,1]*I01
D120221 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M221[k,l]*dV(inc)) - C_M[1,2,2,2]*I01
D120121 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M121[k,l]*dV(inc)) - C_M[1,2,1,2]*I01
D120021 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M021[k,l]*dV(inc)) - C_M[1,2,0,2]*I01
D120011 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M011[k,l]*dV(inc)) - C_M[1,2,0,1]*I01
D120002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M002[k,l]*dV(inc)) - C_M[1,2,0,0]*I02
D120112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M112[k,l]*dV(inc)) - C_M[1,2,1,1]*I02
D120222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M222[k,l]*dV(inc)) - C_M[1,2,2,2]*I02
D120122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M122[k,l]*dV(inc)) - C_M[1,2,1,2]*I02
D120022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M022[k,l]*dV(inc)) - C_M[1,2,0,2]*I02
D120012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M120[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M120[i,j]*M012[k,l]*dV(inc)) - C_M[1,2,0,1]*I02

D020000 = D000020 
D020110 = D110020 
D020220 = D220020 
D020120 = D120020 
D020020 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M020[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M020[k,l]*dV(inc)) - C_M[0,2,0,2]*I00
D020010 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M010[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M010[k,l]*dV(inc)) - C_M[0,2,0,1]*I00
D020001 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M001[k,l]*dV(inc)) - C_M[0,2,0,0]*I01
D020111 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M111[k,l]*dV(inc)) - C_M[0,2,1,1]*I01
D020221 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M221[k,l]*dV(inc)) - C_M[0,2,2,2]*I01
D020121 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M121[k,l]*dV(inc)) - C_M[0,2,1,2]*I01
D020021 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M021[k,l]*dV(inc)) - C_M[0,2,0,2]*I01
D020011 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M011[k,l]*dV(inc)) - C_M[0,2,0,1]*I01
D020002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M002[k,l]*dV(inc)) - C_M[0,2,0,0]*I02
D020112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M112[k,l]*dV(inc)) - C_M[0,2,1,1]*I02
D020222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M222[k,l]*dV(inc)) - C_M[0,2,2,2]*I02
D020122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M122[k,l]*dV(inc)) - C_M[0,2,1,2]*I02
D020022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M022[k,l]*dV(inc)) - C_M[0,2,0,2]*I02
D020012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M020[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M020[i,j]*M012[k,l]*dV(inc)) - C_M[0,2,0,1]*I02

D010000 = D000010 
D010110 = D110010 
D010220 = D220010 
D010120 = D120010 
D010020 = D020010 
D010010 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M010[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M010[k,l]*dV(inc)) - C_M[0,1,0,1]*I00
D010001 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M001[k,l]*dV(inc)) - C_M[0,1,0,0]*I01
D010111 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M111[k,l]*dV(inc)) - C_M[0,1,1,1]*I01
D010221 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M221[k,l]*dV(inc)) - C_M[0,1,2,2]*I01
D010121 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M121[k,l]*dV(inc)) - C_M[0,1,1,2]*I01
D010021 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M021[k,l]*dV(inc)) - C_M[0,1,0,2]*I01
D010011 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M011[k,l]*dV(inc)) - C_M[0,1,0,1]*I01
D010002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M002[k,l]*dV(inc)) - C_M[0,1,0,0]*I02
D010112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M112[k,l]*dV(inc)) - C_M[0,1,1,1]*I02
D010222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M222[k,l]*dV(inc)) - C_M[0,1,2,2]*I02
D010122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M122[k,l]*dV(inc)) - C_M[0,1,1,2]*I02
D010022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M022[k,l]*dV(inc)) - C_M[0,1,0,2]*I02
D010012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M010[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M010[i,j]*M012[k,l]*dV(inc)) - C_M[0,1,0,1]*I02

#c=1
D001000 = D000001 
D001110 = D110001 
D001220 = D220001 
D001120 = D120001 
D001020 = D020001 
D001010 = D010001 
D001001 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M001[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M001[i,j]*M001[k,l]*dV(inc)) - C_M[0,0,0,0]*I11
D001111 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M001[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M001[i,j]*M111[k,l]*dV(inc)) - C_M[0,0,1,1]*I11
D001221 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M001[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M001[i,j]*M221[k,l]*dV(inc)) - C_M[0,0,2,2]*I11
D001121 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M001[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M001[i,j]*M121[k,l]*dV(inc)) - C_M[0,0,1,2]*I11
D001021 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M001[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M001[i,j]*M021[k,l]*dV(inc)) - C_M[0,0,0,2]*I11
D001011 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M001[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M001[i,j]*M011[k,l]*dV(inc)) - C_M[0,0,0,1]*I11
D001002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M001[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M001[i,j]*M002[k,l]*dV(inc)) - C_M[0,0,0,0]*I12
D001112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M001[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M001[i,j]*M112[k,l]*dV(inc)) - C_M[0,0,1,1]*I12
D001222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M001[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M001[i,j]*M222[k,l]*dV(inc)) - C_M[0,0,2,2]*I12
D001122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M001[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M001[i,j]*M122[k,l]*dV(inc)) - C_M[0,0,1,2]*I12
D001022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M001[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M001[i,j]*M022[k,l]*dV(inc)) - C_M[0,0,0,2]*I12
D001012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M001[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M001[i,j]*M012[k,l]*dV(inc)) - C_M[0,0,0,1]*I12
D111000 = D000111
D111110 = D110111 
D111220 = D220111 
D111120 = D120111 
D111020 = D020111 
D111010 = D010111 
D111001 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M111[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M111[i,j]*M001[k,l]*dV(inc)) - C_M[1,1,0,0]*I11
D111111 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M111[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M111[i,j]*M111[k,l]*dV(inc)) - C_M[1,1,1,1]*I11
D111221 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M111[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M111[i,j]*M221[k,l]*dV(inc)) - C_M[1,1,2,2]*I11
D111121 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M111[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M111[i,j]*M121[k,l]*dV(inc)) - C_M[1,1,1,2]*I11
D111021 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M111[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M111[i,j]*M021[k,l]*dV(inc)) - C_M[1,1,0,2]*I11
D111011 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M111[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M111[i,j]*M011[k,l]*dV(inc)) - C_M[1,1,0,1]*I11
D111002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M111[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M111[i,j]*M002[k,l]*dV(inc)) - C_M[1,1,0,0]*I12
D111112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M111[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M111[i,j]*M112[k,l]*dV(inc)) - C_M[1,1,1,1]*I12
D111222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M111[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M111[i,j]*M222[k,l]*dV(inc)) - C_M[1,1,2,2]*I12
D111122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M111[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M111[i,j]*M122[k,l]*dV(inc)) - C_M[1,1,1,2]*I12
D111022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M111[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M111[i,j]*M022[k,l]*dV(inc)) - C_M[1,1,0,2]*I12
D111012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M111[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M111[i,j]*M012[k,l]*dV(inc)) - C_M[1,1,0,1]*I12
D221000 = D000221
D221110 = D110221
D221220 = D220221 
D221120 = D120221 
D221020 = D020221 
D221010 = D010221 
D221001 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M221[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M221[i,j]*M001[k,l]*dV(inc)) - C_M[2,2,0,0]*I11
D221111 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M221[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M221[i,j]*M111[k,l]*dV(inc)) - C_M[2,2,1,1]*I11
D221221 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M221[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M221[i,j]*M221[k,l]*dV(inc)) - C_M[2,2,2,2]*I11
D221121 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M221[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M221[i,j]*M121[k,l]*dV(inc)) - C_M[2,2,1,2]*I11
D221021 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M221[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M221[i,j]*M021[k,l]*dV(inc)) - C_M[2,2,0,2]*I11
D221011 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M221[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M221[i,j]*M011[k,l]*dV(inc)) - C_M[2,2,0,1]*I11
D221002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M221[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M221[i,j]*M002[k,l]*dV(inc)) - C_M[2,2,0,0]*I12
D221112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M221[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M221[i,j]*M112[k,l]*dV(inc)) - C_M[2,2,1,1]*I12
D221222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M221[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M221[i,j]*M222[k,l]*dV(inc)) - C_M[2,2,2,2]*I12
D221122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M221[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M221[i,j]*M122[k,l]*dV(inc)) - C_M[2,2,1,2]*I12
D221022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M221[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M221[i,j]*M022[k,l]*dV(inc)) - C_M[2,2,0,2]*I12
D221012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M221[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M221[i,j]*M012[k,l]*dV(inc)) - C_M[2,2,0,1]*I12
D121000 = D000121
D121110 = D110121
D121220 = D220121
D121120 = D120121 
D121020 = D020121 
D121010 = D010121 
D121001 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M121[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M121[i,j]*M001[k,l]*dV(inc)) - C_M[1,2,0,0]*I11
D121111 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M121[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M121[i,j]*M111[k,l]*dV(inc)) - C_M[1,2,1,1]*I11
D121221 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M121[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M121[i,j]*M221[k,l]*dV(inc)) - C_M[1,2,2,2]*I11
D121121 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M121[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M121[i,j]*M121[k,l]*dV(inc)) - C_M[1,2,1,2]*I11
D121021 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M121[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M121[i,j]*M021[k,l]*dV(inc)) - C_M[1,2,0,2]*I11
D121011 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M121[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M121[i,j]*M011[k,l]*dV(inc)) - C_M[1,2,0,1]*I11
D121002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M121[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M121[i,j]*M002[k,l]*dV(inc)) - C_M[1,2,0,0]*I12
D121112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M121[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M121[i,j]*M112[k,l]*dV(inc)) - C_M[1,2,1,1]*I12
D121222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M121[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M121[i,j]*M222[k,l]*dV(inc)) - C_M[1,2,2,2]*I12
D121122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M121[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M121[i,j]*M122[k,l]*dV(inc)) - C_M[1,2,1,2]*I12
D121022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M121[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M121[i,j]*M022[k,l]*dV(inc)) - C_M[1,2,0,2]*I12
D121012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M121[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M121[i,j]*M012[k,l]*dV(inc)) - C_M[1,2,0,1]*I12
D021000 = D000021
D021110 = D110021
D021220 = D220021
D021120 = D120021
D021020 = D020021 
D021010 = D010021 
D021001 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M021[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M021[i,j]*M001[k,l]*dV(inc)) - C_M[0,2,0,0]*I11
D021111 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M021[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M021[i,j]*M111[k,l]*dV(inc)) - C_M[0,2,1,1]*I11
D021221 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M021[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M021[i,j]*M221[k,l]*dV(inc)) - C_M[0,2,2,2]*I11
D021121 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M021[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M021[i,j]*M121[k,l]*dV(inc)) - C_M[0,2,1,2]*I11
D021021 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M021[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M021[i,j]*M021[k,l]*dV(inc)) - C_M[0,2,0,2]*I11
D021011 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M021[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M021[i,j]*M011[k,l]*dV(inc)) - C_M[0,2,0,1]*I11
D021002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M021[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M021[i,j]*M002[k,l]*dV(inc)) - C_M[0,2,0,0]*I12
D021112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M021[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M021[i,j]*M112[k,l]*dV(inc)) - C_M[0,2,1,1]*I12
D021222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M021[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M021[i,j]*M222[k,l]*dV(inc)) - C_M[0,2,2,2]*I12
D021122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M021[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M021[i,j]*M122[k,l]*dV(inc)) - C_M[0,2,1,2]*I12
D021022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M021[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M021[i,j]*M022[k,l]*dV(inc)) - C_M[0,2,0,2]*I12
D021012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M021[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M021[i,j]*M012[k,l]*dV(inc)) - C_M[0,2,0,1]*I12
D011000 = D000011
D011110 = D110011
D011220 = D220011
D011120 = D120011
D011020 = D020011
D011010 = D010011 
D011001 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M011[i,j]*M001[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M011[i,j]*M001[k,l]*dV(inc)) - C_M[0,1,0,0]*I11
D011111 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M011[i,j]*M111[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M011[i,j]*M111[k,l]*dV(inc)) - C_M[0,1,1,1]*I11
D011221 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M011[i,j]*M221[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M011[i,j]*M221[k,l]*dV(inc)) - C_M[0,1,2,2]*I11
D011121 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M011[i,j]*M121[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M011[i,j]*M121[k,l]*dV(inc)) - C_M[0,1,1,2]*I11
D011021 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M011[i,j]*M021[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M011[i,j]*M021[k,l]*dV(inc)) - C_M[0,1,0,2]*I11
D011011 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M011[i,j]*M011[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M011[i,j]*M011[k,l]*dV(inc)) - C_M[0,1,0,1]*I11
D011002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M011[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M011[i,j]*M002[k,l]*dV(inc)) - C_M[0,1,0,0]*I12
D011112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M011[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M011[i,j]*M112[k,l]*dV(inc)) - C_M[0,1,1,1]*I12
D011222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M011[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M011[i,j]*M222[k,l]*dV(inc)) - C_M[0,1,2,2]*I12
D011122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M011[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M011[i,j]*M122[k,l]*dV(inc)) - C_M[0,1,1,2]*I12
D011022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M011[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M011[i,j]*M022[k,l]*dV(inc)) - C_M[0,1,0,2]*I12
D011012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M011[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M011[i,j]*M012[k,l]*dV(inc)) - C_M[0,1,0,1]*I12

#c=2
D002000 = D000002 
D002110 = D110002 
D002220 = D220002 
D002120 = D120002 
D002020 = D020002 
D002010 = D010002 
D002001 = D001002 
D002111 = D111002 
D002221 = D221002 
D002121 = D121002 
D002021 = D021002 
D002011 = D011002 
D002002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M002[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M002[i,j]*M002[k,l]*dV(inc)) - C_M[0,0,0,0]*I22
D002112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M002[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M002[i,j]*M112[k,l]*dV(inc)) - C_M[0,0,1,1]*I22
D002222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M002[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M002[i,j]*M222[k,l]*dV(inc)) - C_M[0,0,2,2]*I22
D002122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M002[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M002[i,j]*M122[k,l]*dV(inc)) - C_M[0,0,1,2]*I22
D002022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M002[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M002[i,j]*M022[k,l]*dV(inc)) - C_M[0,0,0,2]*I22
D002012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M002[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M002[i,j]*M012[k,l]*dV(inc)) - C_M[0,0,0,1]*I22
D112000 = D000112
D112110 = D110112 
D112220 = D220112 
D112120 = D120112 
D112020 = D020112 
D112010 = D010112 
D112001 = D001112 
D112111 = D111112 
D112221 = D221112 
D112121 = D121112 
D112021 = D021112 
D112011 = D011112 
D112002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M112[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M112[i,j]*M002[k,l]*dV(inc)) - C_M[1,1,0,0]*I22
D112112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M112[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M112[i,j]*M112[k,l]*dV(inc)) - C_M[1,1,1,1]*I22
D112222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M112[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M112[i,j]*M222[k,l]*dV(inc)) - C_M[1,1,2,2]*I22
D112122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M112[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M112[i,j]*M122[k,l]*dV(inc)) - C_M[1,1,1,2]*I22
D112022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M112[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M112[i,j]*M022[k,l]*dV(inc)) - C_M[1,1,0,2]*I22
D112012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M112[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M112[i,j]*M012[k,l]*dV(inc)) - C_M[1,1,0,1]*I22
D222000 = D000222
D222110 = D110222
D222220 = D220222 
D222120 = D120222 
D222020 = D020222 
D222010 = D010222 
D222001 = D001222 
D222111 = D111222 
D222221 = D221222 
D222121 = D121222 
D222021 = D021222 
D222011 = D011222 
D222002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M222[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M222[i,j]*M002[k,l]*dV(inc)) - C_M[2,2,0,0]*I22
D222112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M222[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M222[i,j]*M112[k,l]*dV(inc)) - C_M[2,2,1,1]*I22
D222222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M222[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M222[i,j]*M222[k,l]*dV(inc)) - C_M[2,2,2,2]*I22
D222122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M222[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M222[i,j]*M122[k,l]*dV(inc)) - C_M[2,2,1,2]*I22
D222022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M222[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M222[i,j]*M022[k,l]*dV(inc)) - C_M[2,2,0,2]*I22
D222012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M222[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M222[i,j]*M012[k,l]*dV(inc)) - C_M[2,2,0,1]*I22
D122000 = D000122
D122110 = D110122
D122220 = D220122
D122120 = D120122
D122020 = D020122
D122010 = D010122
D122001 = D001122 
D122111 = D111122 
D122221 = D221122 
D122121 = D121122 
D122021 = D021122 
D122011 = D011122 
D122002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M122[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M122[i,j]*M002[k,l]*dV(inc)) - C_M[1,2,0,0]*I22
D122112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M122[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M122[i,j]*M112[k,l]*dV(inc)) - C_M[1,2,1,1]*I22
D122222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M122[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M122[i,j]*M222[k,l]*dV(inc)) - C_M[1,2,2,2]*I22
D122122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M122[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M122[i,j]*M122[k,l]*dV(inc)) - C_M[1,2,1,2]*I22
D122022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M122[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M122[i,j]*M022[k,l]*dV(inc)) - C_M[1,2,0,2]*I22
D122012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M122[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M122[i,j]*M012[k,l]*dV(inc)) - C_M[1,2,0,1]*I22
D022000 = D000022
D022110 = D110022
D022220 = D220022
D022120 = D120022
D022020 = D020022 
D022010 = D010022 
D022001 = D001022 
D022111 = D111022 
D022221 = D221022 
D022121 = D121022 
D022021 = D021022 
D022011 = D011022 
D022002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M022[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M022[i,j]*M002[k,l]*dV(inc)) - C_M[0,2,0,0]*I22
D022112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M022[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M022[i,j]*M112[k,l]*dV(inc)) - C_M[0,2,1,1]*I22
D022222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M022[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M022[i,j]*M222[k,l]*dV(inc)) - C_M[0,2,2,2]*I22
D022122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M022[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M022[i,j]*M122[k,l]*dV(inc)) - C_M[0,2,1,2]*I22
D022022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M022[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M022[i,j]*M022[k,l]*dV(inc)) - C_M[0,2,0,2]*I22
D022012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M022[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M022[i,j]*M012[k,l]*dV(inc)) - C_M[0,2,0,1]*I22
D012000 = D000012
D012110 = D110012
D012220 = D220012
D012120 = D120012
D012020 = D020012
D012010 = D010012 
D012001 = D001012 
D012111 = D111012 
D012221 = D221012 
D012121 = D121012 
D012021 = D021012 
D012011 = D011012 
D012002 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M012[i,j]*M002[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M012[i,j]*M002[k,l]*dV(inc)) - C_M[0,1,0,0]*I22
D012112 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M012[i,j]*M112[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M012[i,j]*M112[k,l]*dV(inc)) - C_M[0,1,1,1]*I22
D012222 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M012[i,j]*M222[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M012[i,j]*M222[k,l]*dV(inc)) - C_M[0,1,2,2]*I22
D012122 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M012[i,j]*M122[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M012[i,j]*M122[k,l]*dV(inc)) - C_M[0,1,1,2]*I22
D012022 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M012[i,j]*M022[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M012[i,j]*M022[k,l]*dV(inc)) - C_M[0,1,0,2]*I22
D012012 = epsilon**2 / Vol * assemble( Cmat_m[i,j,k,l]*M012[i,j]*M012[k,l]*dV(mat) + Cinc_m[i,j,k,l]*M012[i,j]*M012[k,l]*dV(inc)) - C_M[0,1,0,1]*I22

if processID == 0: 
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	print('%.0f DOFs in %.0f h %.0f min %.0f s' % (dofs, e_h, e_m, e_s)  )

divby = 10**0
powerdivby = np.log10(divby)
digits = 0

# 1. D
if processID == 0: 
## first version of D
	with open(data+'.tex', 'a') as file:
		file.write('\n')
		file.write(r'%first version of D')
		file.write('\n')
		file.write(r'\begeq')
		file.write('\n')
		file.write(r'D\ma_{\alpha\beta} =')
		file.write('\n')
		file.write(r'\resizebox{.85\textwidth}{!}{$\displaystyle')
		file.write('\n')
		file.write(r'\begin{pmatrix}')
		file.write('\n')
		file.write(str(round(D000000/divby,digits))+' & '+str(round(D000110/divby,digits))+' & '+str(round(D000011/divby,digits))+' & '+str(round(D000220/divby,digits))+' & '+str(round(D000022/divby,digits))+' & '+str(round(D000111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D000001/divby,digits))+' & '+str(round(D000010/divby,digits))+' & '+str(round(D000221/divby,digits))+' & '+str(round(D000122/divby,digits))+' & '+str(round(D000222/divby,digits))+' & '+str(round(D000002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D000020/divby,digits))+' & '+str(round(D000112/divby,digits))+' & '+str(round(D000121/divby,digits))+' & '+str(round(D000120/divby,digits))+' & '+str(round(D000021/divby,digits))+' & '+str(round(D000012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D110000/divby,digits))+' & '+str(round(D110110/divby,digits))+' & '+str(round(D110011/divby,digits))+' & '+str(round(D110220/divby,digits))+' & '+str(round(D110022/divby,digits))+' & '+str(round(D110111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D110001/divby,digits))+' & '+str(round(D110010/divby,digits))+' & '+str(round(D110221/divby,digits))+' & '+str(round(D110122/divby,digits))+' & '+str(round(D110222/divby,digits))+' & '+str(round(D110002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D110020/divby,digits))+' & '+str(round(D110112/divby,digits))+' & '+str(round(D110121/divby,digits))+' & '+str(round(D110120/divby,digits))+' & '+str(round(D110021/divby,digits))+' & '+str(round(D110012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D011000/divby,digits))+' & '+str(round(D011110/divby,digits))+' & '+str(round(D011011/divby,digits))+' & '+str(round(D011220/divby,digits))+' & '+str(round(D011022/divby,digits))+' & '+str(round(D011111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D011001/divby,digits))+' & '+str(round(D011010/divby,digits))+' & '+str(round(D011221/divby,digits))+' & '+str(round(D011122/divby,digits))+' & '+str(round(D011222/divby,digits))+' & '+str(round(D011002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D011020/divby,digits))+' & '+str(round(D011112/divby,digits))+' & '+str(round(D011121/divby,digits))+' & '+str(round(D011120/divby,digits))+' & '+str(round(D011021/divby,digits))+' & '+str(round(D011012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D220000/divby,digits))+' & '+str(round(D220110/divby,digits))+' & '+str(round(D220011/divby,digits))+' & '+str(round(D220220/divby,digits))+' & '+str(round(D220022/divby,digits))+' & '+str(round(D220111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D220001/divby,digits))+' & '+str(round(D220010/divby,digits))+' & '+str(round(D220221/divby,digits))+' & '+str(round(D220122/divby,digits))+' & '+str(round(D220222/divby,digits))+' & '+str(round(D220002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D220020/divby,digits))+' & '+str(round(D220112/divby,digits))+' & '+str(round(D220121/divby,digits))+' & '+str(round(D220120/divby,digits))+' & '+str(round(D220021/divby,digits))+' & '+str(round(D220012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D022000/divby,digits))+' & '+str(round(D022110/divby,digits))+' & '+str(round(D022011/divby,digits))+' & '+str(round(D022220/divby,digits))+' & '+str(round(D022022/divby,digits))+' & '+str(round(D022111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D022001/divby,digits))+' & '+str(round(D022010/divby,digits))+' & '+str(round(D022221/divby,digits))+' & '+str(round(D022122/divby,digits))+' & '+str(round(D022222/divby,digits))+' & '+str(round(D022002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D022020/divby,digits))+' & '+str(round(D022112/divby,digits))+' & '+str(round(D022121/divby,digits))+' & '+str(round(D022120/divby,digits))+' & '+str(round(D022021/divby,digits))+' & '+str(round(D022012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D111000/divby,digits))+' & '+str(round(D111110/divby,digits))+' & '+str(round(D111011/divby,digits))+' & '+str(round(D111220/divby,digits))+' & '+str(round(D111022/divby,digits))+' & '+str(round(D111111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D111001/divby,digits))+' & '+str(round(D111010/divby,digits))+' & '+str(round(D111221/divby,digits))+' & '+str(round(D111122/divby,digits))+' & '+str(round(D111222/divby,digits))+' & '+str(round(D111002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D111020/divby,digits))+' & '+str(round(D111112/divby,digits))+' & '+str(round(D111121/divby,digits))+' & '+str(round(D111120/divby,digits))+' & '+str(round(D111021/divby,digits))+' & '+str(round(D111012/divby,digits)))
		file.write('\n')
		file.write(r' \\')

		file.write('\n')
		file.write(str(round(D001000/divby,digits))+' & '+str(round(D001110/divby,digits))+' & '+str(round(D001011/divby,digits))+' & '+str(round(D001220/divby,digits))+' & '+str(round(D001022/divby,digits))+' & '+str(round(D001111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D001001/divby,digits))+' & '+str(round(D001010/divby,digits))+' & '+str(round(D001221/divby,digits))+' & '+str(round(D001122/divby,digits))+' & '+str(round(D001222/divby,digits))+' & '+str(round(D001002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D001020/divby,digits))+' & '+str(round(D001112/divby,digits))+' & '+str(round(D001121/divby,digits))+' & '+str(round(D001120/divby,digits))+' & '+str(round(D001021/divby,digits))+' & '+str(round(D001012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D010000/divby,digits))+' & '+str(round(D010110/divby,digits))+' & '+str(round(D010011/divby,digits))+' & '+str(round(D010220/divby,digits))+' & '+str(round(D010022/divby,digits))+' & '+str(round(D010111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D010001/divby,digits))+' & '+str(round(D010010/divby,digits))+' & '+str(round(D010221/divby,digits))+' & '+str(round(D010122/divby,digits))+' & '+str(round(D010222/divby,digits))+' & '+str(round(D010002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D010020/divby,digits))+' & '+str(round(D010112/divby,digits))+' & '+str(round(D010121/divby,digits))+' & '+str(round(D010120/divby,digits))+' & '+str(round(D010021/divby,digits))+' & '+str(round(D010012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D221000/divby,digits))+' & '+str(round(D221110/divby,digits))+' & '+str(round(D221011/divby,digits))+' & '+str(round(D221220/divby,digits))+' & '+str(round(D221022/divby,digits))+' & '+str(round(D221111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D221001/divby,digits))+' & '+str(round(D221010/divby,digits))+' & '+str(round(D221221/divby,digits))+' & '+str(round(D221122/divby,digits))+' & '+str(round(D221222/divby,digits))+' & '+str(round(D221002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D221020/divby,digits))+' & '+str(round(D221112/divby,digits))+' & '+str(round(D221121/divby,digits))+' & '+str(round(D221120/divby,digits))+' & '+str(round(D221021/divby,digits))+' & '+str(round(D221012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D122000/divby,digits))+' & '+str(round(D122110/divby,digits))+' & '+str(round(D122011/divby,digits))+' & '+str(round(D122220/divby,digits))+' & '+str(round(D122022/divby,digits))+' & '+str(round(D122111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D122001/divby,digits))+' & '+str(round(D122010/divby,digits))+' & '+str(round(D122221/divby,digits))+' & '+str(round(D122122/divby,digits))+' & '+str(round(D122222/divby,digits))+' & '+str(round(D122002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D122020/divby,digits))+' & '+str(round(D122112/divby,digits))+' & '+str(round(D122121/divby,digits))+' & '+str(round(D122120/divby,digits))+' & '+str(round(D122021/divby,digits))+' & '+str(round(D122012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D222000/divby,digits))+' & '+str(round(D222110/divby,digits))+' & '+str(round(D222011/divby,digits))+' & '+str(round(D222220/divby,digits))+' & '+str(round(D222022/divby,digits))+' & '+str(round(D222111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D222001/divby,digits))+' & '+str(round(D222010/divby,digits))+' & '+str(round(D222221/divby,digits))+' & '+str(round(D222122/divby,digits))+' & '+str(round(D222222/divby,digits))+' & '+str(round(D222002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D222020/divby,digits))+' & '+str(round(D222112/divby,digits))+' & '+str(round(D222121/divby,digits))+' & '+str(round(D222120/divby,digits))+' & '+str(round(D222021/divby,digits))+' & '+str(round(D222012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D002000/divby,digits))+' & '+str(round(D002110/divby,digits))+' & '+str(round(D002011/divby,digits))+' & '+str(round(D002220/divby,digits))+' & '+str(round(D002022/divby,digits))+' & '+str(round(D002111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D002001/divby,digits))+' & '+str(round(D002010/divby,digits))+' & '+str(round(D002221/divby,digits))+' & '+str(round(D002122/divby,digits))+' & '+str(round(D002222/divby,digits))+' & '+str(round(D002002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D002020/divby,digits))+' & '+str(round(D002112/divby,digits))+' & '+str(round(D002121/divby,digits))+' & '+str(round(D002120/divby,digits))+' & '+str(round(D002021/divby,digits))+' & '+str(round(D002012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D020000/divby,digits))+' & '+str(round(D020110/divby,digits))+' & '+str(round(D020011/divby,digits))+' & '+str(round(D020220/divby,digits))+' & '+str(round(D020022/divby,digits))+' & '+str(round(D020111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D020001/divby,digits))+' & '+str(round(D020010/divby,digits))+' & '+str(round(D020221/divby,digits))+' & '+str(round(D020122/divby,digits))+' & '+str(round(D020222/divby,digits))+' & '+str(round(D020002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D020020/divby,digits))+' & '+str(round(D020112/divby,digits))+' & '+str(round(D020121/divby,digits))+' & '+str(round(D020120/divby,digits))+' & '+str(round(D020021/divby,digits))+' & '+str(round(D020012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D112000/divby,digits))+' & '+str(round(D112110/divby,digits))+' & '+str(round(D112011/divby,digits))+' & '+str(round(D112220/divby,digits))+' & '+str(round(D112022/divby,digits))+' & '+str(round(D112111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D112001/divby,digits))+' & '+str(round(D112010/divby,digits))+' & '+str(round(D112221/divby,digits))+' & '+str(round(D112122/divby,digits))+' & '+str(round(D112222/divby,digits))+' & '+str(round(D112002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D112020/divby,digits))+' & '+str(round(D112112/divby,digits))+' & '+str(round(D112121/divby,digits))+' & '+str(round(D112120/divby,digits))+' & '+str(round(D112021/divby,digits))+' & '+str(round(D112012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D121000/divby,digits))+' & '+str(round(D121110/divby,digits))+' & '+str(round(D121011/divby,digits))+' & '+str(round(D121220/divby,digits))+' & '+str(round(D121022/divby,digits))+' & '+str(round(D121111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D121001/divby,digits))+' & '+str(round(D121010/divby,digits))+' & '+str(round(D121221/divby,digits))+' & '+str(round(D121122/divby,digits))+' & '+str(round(D121222/divby,digits))+' & '+str(round(D121002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D121020/divby,digits))+' & '+str(round(D121112/divby,digits))+' & '+str(round(D121121/divby,digits))+' & '+str(round(D121120/divby,digits))+' & '+str(round(D121021/divby,digits))+' & '+str(round(D121012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D120000/divby,digits))+' & '+str(round(D120110/divby,digits))+' & '+str(round(D120011/divby,digits))+' & '+str(round(D120220/divby,digits))+' & '+str(round(D120022/divby,digits))+' & '+str(round(D120111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D120001/divby,digits))+' & '+str(round(D120010/divby,digits))+' & '+str(round(D120221/divby,digits))+' & '+str(round(D120122/divby,digits))+' & '+str(round(D120222/divby,digits))+' & '+str(round(D120002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D120020/divby,digits))+' & '+str(round(D120112/divby,digits))+' & '+str(round(D120121/divby,digits))+' & '+str(round(D120120/divby,digits))+' & '+str(round(D120021/divby,digits))+' & '+str(round(D120012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D021000/divby,digits))+' & '+str(round(D021110/divby,digits))+' & '+str(round(D021011/divby,digits))+' & '+str(round(D021220/divby,digits))+' & '+str(round(D021022/divby,digits))+' & '+str(round(D021111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D021001/divby,digits))+' & '+str(round(D021010/divby,digits))+' & '+str(round(D021221/divby,digits))+' & '+str(round(D021122/divby,digits))+' & '+str(round(D021222/divby,digits))+' & '+str(round(D021002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D021020/divby,digits))+' & '+str(round(D021112/divby,digits))+' & '+str(round(D021121/divby,digits))+' & '+str(round(D021120/divby,digits))+' & '+str(round(D021021/divby,digits))+' & '+str(round(D021012/divby,digits))+r' \\')

		file.write('\n')
		file.write(str(round(D012000/divby,digits))+' & '+str(round(D012110/divby,digits))+' & '+str(round(D012011/divby,digits))+' & '+str(round(D012220/divby,digits))+' & '+str(round(D012022/divby,digits))+' & '+str(round(D012111/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D012001/divby,digits))+' & '+str(round(D012010/divby,digits))+' & '+str(round(D012221/divby,digits))+' & '+str(round(D012122/divby,digits))+' & '+str(round(D012222/divby,digits))+' & '+str(round(D012002/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D012020/divby,digits))+' & '+str(round(D012112/divby,digits))+' & '+str(round(D012121/divby,digits))+' & '+str(round(D012120/divby,digits))+' & '+str(round(D012021/divby,digits))+' & '+str(round(D012012/divby,digits)))

		file.write('\n')
		file.write('\n')
		file.write(r'\end{pmatrix} $} 10^{')
		file.write(str(round(powerdivby)))
		file.write(r'} \text{\,N} \ ,')
		file.write('\n')
		file.write(r'\eqend')

# 2. D
if processID == 0: 
## first version of D to put in SG
	with open(data+'_SG.tex', 'a') as file:
		file.write('\n')
		file.write(r'%first version of D to put in SG')
		file.write('\n')
		file.write(r'\begeq')
		file.write('\n')
		file.write(r'D\ma_{\alpha\beta} =')
		file.write('\n')
		file.write(r'\resizebox{.85\textwidth}{!}{$\displaystyle')
		file.write('\n')
		file.write(r'\begin{pmatrix}')
		file.write('\n')
		file.write(str(round(D000000/divby,digits))+' , '+str(round(D000110/divby,digits))+' , '+str(round(D000011/divby,digits))+' , '+str(round(D000220/divby,digits))+' , '+str(round(D000022/divby,digits))+' , '+str(round(D000111/divby,digits))+' , ')
		file.write(str(round(D000001/divby,digits))+' , '+str(round(D000010/divby,digits))+' , '+str(round(D000221/divby,digits))+' , '+str(round(D000122/divby,digits))+' , '+str(round(D000222/divby,digits))+' , '+str(round(D000002/divby,digits))+' , ')
		file.write(str(round(D000020/divby,digits))+' , '+str(round(D000112/divby,digits))+' , '+str(round(D000121/divby,digits))+' , '+str(round(D000120/divby,digits))+' , '+str(round(D000021/divby,digits))+' , '+str(round(D000012/divby,digits)))

		file.write('\n')
		file.write(str(round(D110000/divby,digits))+' , '+str(round(D110110/divby,digits))+' , '+str(round(D110011/divby,digits))+' , '+str(round(D110220/divby,digits))+' , '+str(round(D110022/divby,digits))+' , '+str(round(D110111/divby,digits))+' , ')
		file.write(str(round(D110001/divby,digits))+' , '+str(round(D110010/divby,digits))+' , '+str(round(D110221/divby,digits))+' , '+str(round(D110122/divby,digits))+' , '+str(round(D110222/divby,digits))+' , '+str(round(D110002/divby,digits))+' , ')
		file.write(str(round(D110020/divby,digits))+' , '+str(round(D110112/divby,digits))+' , '+str(round(D110121/divby,digits))+' , '+str(round(D110120/divby,digits))+' , '+str(round(D110021/divby,digits))+' , '+str(round(D110012/divby,digits)))

		file.write('\n')
		file.write(str(round(D011000/divby,digits))+' , '+str(round(D011110/divby,digits))+' , '+str(round(D011011/divby,digits))+' , '+str(round(D011220/divby,digits))+' , '+str(round(D011022/divby,digits))+' , '+str(round(D011111/divby,digits))+' , ')
		file.write(str(round(D011001/divby,digits))+' , '+str(round(D011010/divby,digits))+' , '+str(round(D011221/divby,digits))+' , '+str(round(D011122/divby,digits))+' , '+str(round(D011222/divby,digits))+' , '+str(round(D011002/divby,digits))+' , ')
		file.write(str(round(D011020/divby,digits))+' , '+str(round(D011112/divby,digits))+' , '+str(round(D011121/divby,digits))+' , '+str(round(D011120/divby,digits))+' , '+str(round(D011021/divby,digits))+' , '+str(round(D011012/divby,digits)))

		file.write('\n')
		file.write(str(round(D220000/divby,digits))+' , '+str(round(D220110/divby,digits))+' , '+str(round(D220011/divby,digits))+' , '+str(round(D220220/divby,digits))+' , '+str(round(D220022/divby,digits))+' , '+str(round(D220111/divby,digits))+' , ')
		file.write(str(round(D220001/divby,digits))+' , '+str(round(D220010/divby,digits))+' , '+str(round(D220221/divby,digits))+' , '+str(round(D220122/divby,digits))+' , '+str(round(D220222/divby,digits))+' , '+str(round(D220002/divby,digits))+' , ')
		file.write(str(round(D220020/divby,digits))+' , '+str(round(D220112/divby,digits))+' , '+str(round(D220121/divby,digits))+' , '+str(round(D220120/divby,digits))+' , '+str(round(D220021/divby,digits))+' , '+str(round(D220012/divby,digits)))

		file.write('\n')
		file.write(str(round(D022000/divby,digits))+' , '+str(round(D022110/divby,digits))+' , '+str(round(D022011/divby,digits))+' , '+str(round(D022220/divby,digits))+' , '+str(round(D022022/divby,digits))+' , '+str(round(D022111/divby,digits))+' , ')
		file.write(str(round(D022001/divby,digits))+' , '+str(round(D022010/divby,digits))+' , '+str(round(D022221/divby,digits))+' , '+str(round(D022122/divby,digits))+' , '+str(round(D022222/divby,digits))+' , '+str(round(D022002/divby,digits))+' , ')
		file.write(str(round(D022020/divby,digits))+' , '+str(round(D022112/divby,digits))+' , '+str(round(D022121/divby,digits))+' , '+str(round(D022120/divby,digits))+' , '+str(round(D022021/divby,digits))+' , '+str(round(D022012/divby,digits)))

		file.write('\n')
		file.write(str(round(D111000/divby,digits))+' , '+str(round(D111110/divby,digits))+' , '+str(round(D111011/divby,digits))+' , '+str(round(D111220/divby,digits))+' , '+str(round(D111022/divby,digits))+' , '+str(round(D111111/divby,digits))+' , ')
		file.write(str(round(D111001/divby,digits))+' , '+str(round(D111010/divby,digits))+' , '+str(round(D111221/divby,digits))+' , '+str(round(D111122/divby,digits))+' , '+str(round(D111222/divby,digits))+' , '+str(round(D111002/divby,digits))+' , ')
		file.write(str(round(D111020/divby,digits))+' , '+str(round(D111112/divby,digits))+' , '+str(round(D111121/divby,digits))+' , '+str(round(D111120/divby,digits))+' , '+str(round(D111021/divby,digits))+' , '+str(round(D111012/divby,digits)))
		
		file.write('\n')
		file.write(str(round(D001000/divby,digits))+' , '+str(round(D001110/divby,digits))+' , '+str(round(D001011/divby,digits))+' , '+str(round(D001220/divby,digits))+' , '+str(round(D001022/divby,digits))+' , '+str(round(D001111/divby,digits))+' , ')
		file.write(str(round(D001001/divby,digits))+' , '+str(round(D001010/divby,digits))+' , '+str(round(D001221/divby,digits))+' , '+str(round(D001122/divby,digits))+' , '+str(round(D001222/divby,digits))+' , '+str(round(D001002/divby,digits))+' , ')
		file.write(str(round(D001020/divby,digits))+' , '+str(round(D001112/divby,digits))+' , '+str(round(D001121/divby,digits))+' , '+str(round(D001120/divby,digits))+' , '+str(round(D001021/divby,digits))+' , '+str(round(D001012/divby,digits)))

		file.write('\n')
		file.write(str(round(D010000/divby,digits))+' , '+str(round(D010110/divby,digits))+' , '+str(round(D010011/divby,digits))+' , '+str(round(D010220/divby,digits))+' , '+str(round(D010022/divby,digits))+' , '+str(round(D010111/divby,digits))+' , ')
		file.write(str(round(D010001/divby,digits))+' , '+str(round(D010010/divby,digits))+' , '+str(round(D010221/divby,digits))+' , '+str(round(D010122/divby,digits))+' , '+str(round(D010222/divby,digits))+' , '+str(round(D010002/divby,digits))+' , ')
		file.write(str(round(D010020/divby,digits))+' , '+str(round(D010112/divby,digits))+' , '+str(round(D010121/divby,digits))+' , '+str(round(D010120/divby,digits))+' , '+str(round(D010021/divby,digits))+' , '+str(round(D010012/divby,digits)))

		file.write('\n')
		file.write(str(round(D221000/divby,digits))+' , '+str(round(D221110/divby,digits))+' , '+str(round(D221011/divby,digits))+' , '+str(round(D221220/divby,digits))+' , '+str(round(D221022/divby,digits))+' , '+str(round(D221111/divby,digits))+' , ')
		file.write(str(round(D221001/divby,digits))+' , '+str(round(D221010/divby,digits))+' , '+str(round(D221221/divby,digits))+' , '+str(round(D221122/divby,digits))+' , '+str(round(D221222/divby,digits))+' , '+str(round(D221002/divby,digits))+' , ')
		file.write(str(round(D221020/divby,digits))+' , '+str(round(D221112/divby,digits))+' , '+str(round(D221121/divby,digits))+' , '+str(round(D221120/divby,digits))+' , '+str(round(D221021/divby,digits))+' , '+str(round(D221012/divby,digits)))

		file.write('\n')
		file.write(str(round(D122000/divby,digits))+' , '+str(round(D122110/divby,digits))+' , '+str(round(D122011/divby,digits))+' , '+str(round(D122220/divby,digits))+' , '+str(round(D122022/divby,digits))+' , '+str(round(D122111/divby,digits))+' , ')
		file.write(str(round(D122001/divby,digits))+' , '+str(round(D122010/divby,digits))+' , '+str(round(D122221/divby,digits))+' , '+str(round(D122122/divby,digits))+' , '+str(round(D122222/divby,digits))+' , '+str(round(D122002/divby,digits))+' , ')
		file.write(str(round(D122020/divby,digits))+' , '+str(round(D122112/divby,digits))+' , '+str(round(D122121/divby,digits))+' , '+str(round(D122120/divby,digits))+' , '+str(round(D122021/divby,digits))+' , '+str(round(D122012/divby,digits)))

		file.write('\n')
		file.write(str(round(D222000/divby,digits))+' , '+str(round(D222110/divby,digits))+' , '+str(round(D222011/divby,digits))+' , '+str(round(D222220/divby,digits))+' , '+str(round(D222022/divby,digits))+' , '+str(round(D222111/divby,digits))+' , ')
		file.write(str(round(D222001/divby,digits))+' , '+str(round(D222010/divby,digits))+' , '+str(round(D222221/divby,digits))+' , '+str(round(D222122/divby,digits))+' , '+str(round(D222222/divby,digits))+' , '+str(round(D222002/divby,digits))+' , ')
		file.write(str(round(D222020/divby,digits))+' , '+str(round(D222112/divby,digits))+' , '+str(round(D222121/divby,digits))+' , '+str(round(D222120/divby,digits))+' , '+str(round(D222021/divby,digits))+' , '+str(round(D222012/divby,digits)))

		file.write('\n')
		file.write(str(round(D002000/divby,digits))+' , '+str(round(D002110/divby,digits))+' , '+str(round(D002011/divby,digits))+' , '+str(round(D002220/divby,digits))+' , '+str(round(D002022/divby,digits))+' , '+str(round(D002111/divby,digits))+' , ')
		file.write(str(round(D002001/divby,digits))+' , '+str(round(D002010/divby,digits))+' , '+str(round(D002221/divby,digits))+' , '+str(round(D002122/divby,digits))+' , '+str(round(D002222/divby,digits))+' , '+str(round(D002002/divby,digits))+' , ')
		file.write(str(round(D002020/divby,digits))+' , '+str(round(D002112/divby,digits))+' , '+str(round(D002121/divby,digits))+' , '+str(round(D002120/divby,digits))+' , '+str(round(D002021/divby,digits))+' , '+str(round(D002012/divby,digits)))

		file.write('\n')
		file.write(str(round(D020000/divby,digits))+' , '+str(round(D020110/divby,digits))+' , '+str(round(D020011/divby,digits))+' , '+str(round(D020220/divby,digits))+' , '+str(round(D020022/divby,digits))+' , '+str(round(D020111/divby,digits))+' , ')
		file.write(str(round(D020001/divby,digits))+' , '+str(round(D020010/divby,digits))+' , '+str(round(D020221/divby,digits))+' , '+str(round(D020122/divby,digits))+' , '+str(round(D020222/divby,digits))+' , '+str(round(D020002/divby,digits))+' , ')
		file.write(str(round(D020020/divby,digits))+' , '+str(round(D020112/divby,digits))+' , '+str(round(D020121/divby,digits))+' , '+str(round(D020120/divby,digits))+' , '+str(round(D020021/divby,digits))+' , '+str(round(D020012/divby,digits)))

		file.write('\n')
		file.write(str(round(D112000/divby,digits))+' , '+str(round(D112110/divby,digits))+' , '+str(round(D112011/divby,digits))+' , '+str(round(D112220/divby,digits))+' , '+str(round(D112022/divby,digits))+' , '+str(round(D112111/divby,digits))+' , ')
		file.write(str(round(D112001/divby,digits))+' , '+str(round(D112010/divby,digits))+' , '+str(round(D112221/divby,digits))+' , '+str(round(D112122/divby,digits))+' , '+str(round(D112222/divby,digits))+' , '+str(round(D112002/divby,digits))+' , ')
		file.write(str(round(D112020/divby,digits))+' , '+str(round(D112112/divby,digits))+' , '+str(round(D112121/divby,digits))+' , '+str(round(D112120/divby,digits))+' , '+str(round(D112021/divby,digits))+' , '+str(round(D112012/divby,digits)))

		file.write('\n')
		file.write(str(round(D121000/divby,digits))+' , '+str(round(D121110/divby,digits))+' , '+str(round(D121011/divby,digits))+' , '+str(round(D121220/divby,digits))+' , '+str(round(D121022/divby,digits))+' , '+str(round(D121111/divby,digits))+' , ')
		file.write(str(round(D121001/divby,digits))+' , '+str(round(D121010/divby,digits))+' , '+str(round(D121221/divby,digits))+' , '+str(round(D121122/divby,digits))+' , '+str(round(D121222/divby,digits))+' , '+str(round(D121002/divby,digits))+' , ')
		file.write(str(round(D121020/divby,digits))+' , '+str(round(D121112/divby,digits))+' , '+str(round(D121121/divby,digits))+' , '+str(round(D121120/divby,digits))+' , '+str(round(D121021/divby,digits))+' , '+str(round(D121012/divby,digits)))

		file.write('\n')
		file.write(str(round(D120000/divby,digits))+' , '+str(round(D120110/divby,digits))+' , '+str(round(D120011/divby,digits))+' , '+str(round(D120220/divby,digits))+' , '+str(round(D120022/divby,digits))+' , '+str(round(D120111/divby,digits))+' , ')
		file.write(str(round(D120001/divby,digits))+' , '+str(round(D120010/divby,digits))+' , '+str(round(D120221/divby,digits))+' , '+str(round(D120122/divby,digits))+' , '+str(round(D120222/divby,digits))+' , '+str(round(D120002/divby,digits))+' , ')
		file.write(str(round(D120020/divby,digits))+' , '+str(round(D120112/divby,digits))+' , '+str(round(D120121/divby,digits))+' , '+str(round(D120120/divby,digits))+' , '+str(round(D120021/divby,digits))+' , '+str(round(D120012/divby,digits)))

		file.write('\n')
		file.write(str(round(D021000/divby,digits))+' , '+str(round(D021110/divby,digits))+' , '+str(round(D021011/divby,digits))+' , '+str(round(D021220/divby,digits))+' , '+str(round(D021022/divby,digits))+' , '+str(round(D021111/divby,digits))+' , ')
		file.write(str(round(D021001/divby,digits))+' , '+str(round(D021010/divby,digits))+' , '+str(round(D021221/divby,digits))+' , '+str(round(D021122/divby,digits))+' , '+str(round(D021222/divby,digits))+' , '+str(round(D021002/divby,digits))+' , ')
		file.write(str(round(D021020/divby,digits))+' , '+str(round(D021112/divby,digits))+' , '+str(round(D021121/divby,digits))+' , '+str(round(D021120/divby,digits))+' , '+str(round(D021021/divby,digits))+' , '+str(round(D021012/divby,digits)))

		file.write('\n')
		file.write(str(round(D012000/divby,digits))+' , '+str(round(D012110/divby,digits))+' , '+str(round(D012011/divby,digits))+' , '+str(round(D012220/divby,digits))+' , '+str(round(D012022/divby,digits))+' , '+str(round(D012111/divby,digits))+' , ')
		file.write(str(round(D012001/divby,digits))+' , '+str(round(D012010/divby,digits))+' , '+str(round(D012221/divby,digits))+' , '+str(round(D012122/divby,digits))+' , '+str(round(D012222/divby,digits))+' , '+str(round(D012002/divby,digits))+' , ')
		file.write(str(round(D012020/divby,digits))+' , '+str(round(D012112/divby,digits))+' , '+str(round(D012121/divby,digits))+' , '+str(round(D012120/divby,digits))+' , '+str(round(D012021/divby,digits))+' , '+str(round(D012012/divby,digits)))

		file.write('\n')
		file.write('\n')
		file.write(r'\end{pmatrix} $} 10^{')
		file.write(str(round(powerdivby)))
		file.write(r'} \text{\,N} \ ,')
		file.write('\n')
		file.write(r'\eqend')


### second version of D
	with open(data+'.tex', 'a') as file:
		file.write('\n')
		file.write(r'%second version of D')
		file.write('\n')
		file.write(r'\begeq')
		file.write('\n')
		file.write(r'D\ma_{\alpha\beta} =')
		file.write('\n')
		file.write(r'\resizebox{.85\textwidth}{!}{$\displaystyle')
		file.write('\n')
		file.write(r'\begin{pmatrix}')
		file.write('\n')
		file.write(str(round(D000000/divby,digits))+' & '+str(round(D000110/divby,digits))+' & '+str(round(D000220/divby,digits))+' & '+str(round(D000120/divby,digits))+' & '+str(round(D000020/divby,digits))+' & '+str(round(D000010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D000001/divby,digits))+' & '+str(round(D000111/divby,digits))+' & '+str(round(D000221/divby,digits))+' & '+str(round(D000121/divby,digits))+' & '+str(round(D000021/divby,digits))+' & '+str(round(D000011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D000002/divby,digits))+' & '+str(round(D000112/divby,digits))+' & '+str(round(D000222/divby,digits))+' & '+str(round(D000122/divby,digits))+' & '+str(round(D000022/divby,digits))+' & '+str(round(D000012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D110000/divby,digits))+' & '+str(round(D110110/divby,digits))+' & '+str(round(D110220/divby,digits))+' & '+str(round(D110120/divby,digits))+' & '+str(round(D110020/divby,digits))+' & '+str(round(D110010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D110001/divby,digits))+' & '+str(round(D110111/divby,digits))+' & '+str(round(D110221/divby,digits))+' & '+str(round(D110121/divby,digits))+' & '+str(round(D110021/divby,digits))+' & '+str(round(D110011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D110002/divby,digits))+' & '+str(round(D110112/divby,digits))+' & '+str(round(D110222/divby,digits))+' & '+str(round(D110122/divby,digits))+' & '+str(round(D110022/divby,digits))+' & '+str(round(D110012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D220000/divby,digits))+' & '+str(round(D220110/divby,digits))+' & '+str(round(D220220/divby,digits))+' & '+str(round(D220120/divby,digits))+' & '+str(round(D220020/divby,digits))+' & '+str(round(D220010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D220001/divby,digits))+' & '+str(round(D220111/divby,digits))+' & '+str(round(D220221/divby,digits))+' & '+str(round(D220121/divby,digits))+' & '+str(round(D220021/divby,digits))+' & '+str(round(D220011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D220002/divby,digits))+' & '+str(round(D220112/divby,digits))+' & '+str(round(D220222/divby,digits))+' & '+str(round(D220122/divby,digits))+' & '+str(round(D220022/divby,digits))+' & '+str(round(D220012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D120000/divby,digits))+' & '+str(round(D120110/divby,digits))+' & '+str(round(D120220/divby,digits))+' & '+str(round(D120120/divby,digits))+' & '+str(round(D120020/divby,digits))+' & '+str(round(D120010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D120001/divby,digits))+' & '+str(round(D120111/divby,digits))+' & '+str(round(D120221/divby,digits))+' & '+str(round(D120121/divby,digits))+' & '+str(round(D120021/divby,digits))+' & '+str(round(D120011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D120002/divby,digits))+' & '+str(round(D120112/divby,digits))+' & '+str(round(D120222/divby,digits))+' & '+str(round(D120122/divby,digits))+' & '+str(round(D120022/divby,digits))+' & '+str(round(D120012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D020000/divby,digits))+' & '+str(round(D020110/divby,digits))+' & '+str(round(D020220/divby,digits))+' & '+str(round(D020120/divby,digits))+' & '+str(round(D020020/divby,digits))+' & '+str(round(D020010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D020001/divby,digits))+' & '+str(round(D020111/divby,digits))+' & '+str(round(D020221/divby,digits))+' & '+str(round(D020121/divby,digits))+' & '+str(round(D020021/divby,digits))+' & '+str(round(D020011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D020002/divby,digits))+' & '+str(round(D020112/divby,digits))+' & '+str(round(D020222/divby,digits))+' & '+str(round(D020122/divby,digits))+' & '+str(round(D020022/divby,digits))+' & '+str(round(D020012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D010000/divby,digits))+' & '+str(round(D010110/divby,digits))+' & '+str(round(D010220/divby,digits))+' & '+str(round(D010120/divby,digits))+' & '+str(round(D010020/divby,digits))+' & '+str(round(D010010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D010001/divby,digits))+' & '+str(round(D010111/divby,digits))+' & '+str(round(D010221/divby,digits))+' & '+str(round(D010121/divby,digits))+' & '+str(round(D010021/divby,digits))+' & '+str(round(D010011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D010002/divby,digits))+' & '+str(round(D010112/divby,digits))+' & '+str(round(D010222/divby,digits))+' & '+str(round(D010122/divby,digits))+' & '+str(round(D010022/divby,digits))+' & '+str(round(D010012)))
		file.write('\n')
		file.write(r' \\')
		file.write('\n')
		file.write(str(round(D001000/divby,digits))+' & '+str(round(D001110/divby,digits))+' & '+str(round(D001220/divby,digits))+' & '+str(round(D001120/divby,digits))+' & '+str(round(D001020/divby,digits))+' & '+str(round(D001010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D001001/divby,digits))+' & '+str(round(D001111/divby,digits))+' & '+str(round(D001221/divby,digits))+' & '+str(round(D001121/divby,digits))+' & '+str(round(D001021/divby,digits))+' & '+str(round(D001011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D001002/divby,digits))+' & '+str(round(D001112/divby,digits))+' & '+str(round(D001222/divby,digits))+' & '+str(round(D001122/divby,digits))+' & '+str(round(D001022/divby,digits))+' & '+str(round(D001012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D111000/divby,digits))+' & '+str(round(D111110/divby,digits))+' & '+str(round(D111220/divby,digits))+' & '+str(round(D111120/divby,digits))+' & '+str(round(D111020/divby,digits))+' & '+str(round(D111010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D111001/divby,digits))+' & '+str(round(D111111/divby,digits))+' & '+str(round(D111221/divby,digits))+' & '+str(round(D111121/divby,digits))+' & '+str(round(D111021/divby,digits))+' & '+str(round(D111011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D111002/divby,digits))+' & '+str(round(D111112/divby,digits))+' & '+str(round(D111222/divby,digits))+' & '+str(round(D111122/divby,digits))+' & '+str(round(D111022/divby,digits))+' & '+str(round(D111012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D221000/divby,digits))+' & '+str(round(D221110/divby,digits))+' & '+str(round(D221220/divby,digits))+' & '+str(round(D221120/divby,digits))+' & '+str(round(D221020/divby,digits))+' & '+str(round(D221010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D221001/divby,digits))+' & '+str(round(D221111/divby,digits))+' & '+str(round(D221221/divby,digits))+' & '+str(round(D221121/divby,digits))+' & '+str(round(D221021/divby,digits))+' & '+str(round(D221011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D221002/divby,digits))+' & '+str(round(D221112/divby,digits))+' & '+str(round(D221222/divby,digits))+' & '+str(round(D221122/divby,digits))+' & '+str(round(D221022/divby,digits))+' & '+str(round(D221012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D121000/divby,digits))+' & '+str(round(D121110/divby,digits))+' & '+str(round(D121220/divby,digits))+' & '+str(round(D121120/divby,digits))+' & '+str(round(D121020/divby,digits))+' & '+str(round(D121010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D121001/divby,digits))+' & '+str(round(D121111/divby,digits))+' & '+str(round(D121221/divby,digits))+' & '+str(round(D121121/divby,digits))+' & '+str(round(D121021/divby,digits))+' & '+str(round(D121011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D121002/divby,digits))+' & '+str(round(D121112/divby,digits))+' & '+str(round(D121222/divby,digits))+' & '+str(round(D121122/divby,digits))+' & '+str(round(D121022/divby,digits))+' & '+str(round(D121012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D021000/divby,digits))+' & '+str(round(D021110/divby,digits))+' & '+str(round(D021220/divby,digits))+' & '+str(round(D021120/divby,digits))+' & '+str(round(D021020/divby,digits))+' & '+str(round(D021010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D021001/divby,digits))+' & '+str(round(D021111/divby,digits))+' & '+str(round(D021221/divby,digits))+' & '+str(round(D021121/divby,digits))+' & '+str(round(D021021/divby,digits))+' & '+str(round(D021011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D021002/divby,digits))+' & '+str(round(D021112/divby,digits))+' & '+str(round(D021222/divby,digits))+' & '+str(round(D021122/divby,digits))+' & '+str(round(D021022/divby,digits))+' & '+str(round(D021012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D011000/divby,digits))+' & '+str(round(D011110/divby,digits))+' & '+str(round(D011220/divby,digits))+' & '+str(round(D011120/divby,digits))+' & '+str(round(D011020/divby,digits))+' & '+str(round(D011010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D011001/divby,digits))+' & '+str(round(D011111/divby,digits))+' & '+str(round(D011221/divby,digits))+' & '+str(round(D011121/divby,digits))+' & '+str(round(D011021/divby,digits))+' & '+str(round(D011011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D011002/divby,digits))+' & '+str(round(D011112/divby,digits))+' & '+str(round(D011222/divby,digits))+' & '+str(round(D011122/divby,digits))+' & '+str(round(D011022/divby,digits))+' & '+str(round(D011012)))
		file.write('\n')
		file.write(r' \\')
		file.write('\n')
		file.write(str(round(D002000/divby,digits))+' & '+str(round(D002110/divby,digits))+' & '+str(round(D002220/divby,digits))+' & '+str(round(D002120/divby,digits))+' & '+str(round(D002020/divby,digits))+' & '+str(round(D002010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D002001/divby,digits))+' & '+str(round(D002111/divby,digits))+' & '+str(round(D002221/divby,digits))+' & '+str(round(D002121/divby,digits))+' & '+str(round(D002021/divby,digits))+' & '+str(round(D002011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D002002/divby,digits))+' & '+str(round(D002112/divby,digits))+' & '+str(round(D002222/divby,digits))+' & '+str(round(D002122/divby,digits))+' & '+str(round(D002022/divby,digits))+' & '+str(round(D002012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D112000/divby,digits))+' & '+str(round(D112110/divby,digits))+' & '+str(round(D112220/divby,digits))+' & '+str(round(D112120/divby,digits))+' & '+str(round(D112020/divby,digits))+' & '+str(round(D112010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D112001/divby,digits))+' & '+str(round(D112111/divby,digits))+' & '+str(round(D112221/divby,digits))+' & '+str(round(D112121/divby,digits))+' & '+str(round(D112021/divby,digits))+' & '+str(round(D112011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D112002/divby,digits))+' & '+str(round(D112112/divby,digits))+' & '+str(round(D112222/divby,digits))+' & '+str(round(D112122/divby,digits))+' & '+str(round(D112022/divby,digits))+' & '+str(round(D112012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D222000/divby,digits))+' & '+str(round(D222110/divby,digits))+' & '+str(round(D222220/divby,digits))+' & '+str(round(D222120/divby,digits))+' & '+str(round(D222020/divby,digits))+' & '+str(round(D222010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D222001/divby,digits))+' & '+str(round(D222111/divby,digits))+' & '+str(round(D222221/divby,digits))+' & '+str(round(D222121/divby,digits))+' & '+str(round(D222021/divby,digits))+' & '+str(round(D222011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D222002/divby,digits))+' & '+str(round(D222112/divby,digits))+' & '+str(round(D222222/divby,digits))+' & '+str(round(D222122/divby,digits))+' & '+str(round(D222022/divby,digits))+' & '+str(round(D222012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D122000/divby,digits))+' & '+str(round(D122110/divby,digits))+' & '+str(round(D122220/divby,digits))+' & '+str(round(D122120/divby,digits))+' & '+str(round(D122020/divby,digits))+' & '+str(round(D122010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D122001/divby,digits))+' & '+str(round(D122111/divby,digits))+' & '+str(round(D122221/divby,digits))+' & '+str(round(D122121/divby,digits))+' & '+str(round(D122021/divby,digits))+' & '+str(round(D122011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D122002/divby,digits))+' & '+str(round(D122112/divby,digits))+' & '+str(round(D122222/divby,digits))+' & '+str(round(D122122/divby,digits))+' & '+str(round(D122022/divby,digits))+' & '+str(round(D122012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D022000/divby,digits))+' & '+str(round(D022110/divby,digits))+' & '+str(round(D022220/divby,digits))+' & '+str(round(D022120/divby,digits))+' & '+str(round(D022020/divby,digits))+' & '+str(round(D022010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D022001/divby,digits))+' & '+str(round(D022111/divby,digits))+' & '+str(round(D022221/divby,digits))+' & '+str(round(D022121/divby,digits))+' & '+str(round(D022021/divby,digits))+' & '+str(round(D022011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D022002/divby,digits))+' & '+str(round(D022112/divby,digits))+' & '+str(round(D022222/divby,digits))+' & '+str(round(D022122/divby,digits))+' & '+str(round(D022022/divby,digits))+' & '+str(round(D022012/divby,digits))+r' \\')
		file.write('\n')
		file.write(str(round(D012000/divby,digits))+' & '+str(round(D012110/divby,digits))+' & '+str(round(D012220/divby,digits))+' & '+str(round(D012120/divby,digits))+' & '+str(round(D012020/divby,digits))+' & '+str(round(D012010/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D012001/divby,digits))+' & '+str(round(D012111/divby,digits))+' & '+str(round(D012221/divby,digits))+' & '+str(round(D012121/divby,digits))+' & '+str(round(D012021/divby,digits))+' & '+str(round(D012011/divby,digits))+' & ')
		file.write('\n')
		file.write(str(round(D012002/divby,digits))+' & '+str(round(D012112/divby,digits))+' & '+str(round(D012222/divby,digits))+' & '+str(round(D012122/divby,digits))+' & '+str(round(D012022/divby,digits))+' & '+str(round(D012012/divby,digits)))
		file.write('\n')

		file.write('\n')
		file.write('\n')
		file.write(r'\end{pmatrix} $} 10^{')
		file.write(str(round(powerdivby)))
		file.write(r'} \text{\,N} \ ,')
		file.write('\n')
		file.write(r'\eqend')

## second version of D to put in SG
	with open(data+'_SG.tex', 'a') as file:
		file.write('\n')
		file.write(r'%second version of D to put in SG')
		file.write('\n')
		file.write(r'\begeq')
		file.write('\n')
		file.write(r'D\ma_{\alpha\beta} =')
		file.write('\n')
		file.write(r'\resizebox{.85\textwidth}{!}{$\displaystyle')
		file.write('\n')
		file.write(r'\begin{pmatrix}')
		file.write('\n')
		file.write(str(round(D000000/divby,digits))+' , '+str(round(D000110/divby,digits))+' , '+str(round(D000220/divby,digits))+' , '+str(round(D000120/divby,digits))+' , '+str(round(D000020/divby,digits))+' , '+str(round(D000010/divby,digits))+' , ')
		file.write(str(round(D000001/divby,digits))+' , '+str(round(D000111/divby,digits))+' , '+str(round(D000221/divby,digits))+' , '+str(round(D000121/divby,digits))+' , '+str(round(D000021/divby,digits))+' , '+str(round(D000011/divby,digits))+' , ')
		file.write(str(round(D000002/divby,digits))+' , '+str(round(D000112/divby,digits))+' , '+str(round(D000222/divby,digits))+' , '+str(round(D000122/divby,digits))+' , '+str(round(D000022/divby,digits))+' , '+str(round(D000012)))
		file.write('\n')

		file.write(str(round(D110000/divby,digits))+' , '+str(round(D110110/divby,digits))+' , '+str(round(D110220/divby,digits))+' , '+str(round(D110120/divby,digits))+' , '+str(round(D110020/divby,digits))+' , '+str(round(D110010/divby,digits))+' , ')
		file.write(str(round(D110001/divby,digits))+' , '+str(round(D110111/divby,digits))+' , '+str(round(D110221/divby,digits))+' , '+str(round(D110121/divby,digits))+' , '+str(round(D110021/divby,digits))+' , '+str(round(D110011/divby,digits))+' , ')
		file.write(str(round(D110002/divby,digits))+' , '+str(round(D110112/divby,digits))+' , '+str(round(D110222/divby,digits))+' , '+str(round(D110122/divby,digits))+' , '+str(round(D110022/divby,digits))+' , '+str(round(D110012)))
		file.write('\n')

		file.write(str(round(D220000/divby,digits))+' , '+str(round(D220110/divby,digits))+' , '+str(round(D220220/divby,digits))+' , '+str(round(D220120/divby,digits))+' , '+str(round(D220020/divby,digits))+' , '+str(round(D220010/divby,digits))+' , ')
		file.write(str(round(D220001/divby,digits))+' , '+str(round(D220111/divby,digits))+' , '+str(round(D220221/divby,digits))+' , '+str(round(D220121/divby,digits))+' , '+str(round(D220021/divby,digits))+' , '+str(round(D220011/divby,digits))+' , ')
		file.write(str(round(D220002/divby,digits))+' , '+str(round(D220112/divby,digits))+' , '+str(round(D220222/divby,digits))+' , '+str(round(D220122/divby,digits))+' , '+str(round(D220022/divby,digits))+' , '+str(round(D220012)))
		file.write('\n')

		file.write(str(round(D120000/divby,digits))+' , '+str(round(D120110/divby,digits))+' , '+str(round(D120220/divby,digits))+' , '+str(round(D120120/divby,digits))+' , '+str(round(D120020/divby,digits))+' , '+str(round(D120010/divby,digits))+' , ')
		file.write(str(round(D120001/divby,digits))+' , '+str(round(D120111/divby,digits))+' , '+str(round(D120221/divby,digits))+' , '+str(round(D120121/divby,digits))+' , '+str(round(D120021/divby,digits))+' , '+str(round(D120011/divby,digits))+' , ')
		file.write(str(round(D120002/divby,digits))+' , '+str(round(D120112/divby,digits))+' , '+str(round(D120222/divby,digits))+' , '+str(round(D120122/divby,digits))+' , '+str(round(D120022/divby,digits))+' , '+str(round(D120012)))
		file.write('\n')

		file.write(str(round(D020000/divby,digits))+' , '+str(round(D020110/divby,digits))+' , '+str(round(D020220/divby,digits))+' , '+str(round(D020120/divby,digits))+' , '+str(round(D020020/divby,digits))+' , '+str(round(D020010/divby,digits))+' , ')
		file.write(str(round(D020001/divby,digits))+' , '+str(round(D020111/divby,digits))+' , '+str(round(D020221/divby,digits))+' , '+str(round(D020121/divby,digits))+' , '+str(round(D020021/divby,digits))+' , '+str(round(D020011/divby,digits))+' , ')
		file.write(str(round(D020002/divby,digits))+' , '+str(round(D020112/divby,digits))+' , '+str(round(D020222/divby,digits))+' , '+str(round(D020122/divby,digits))+' , '+str(round(D020022/divby,digits))+' , '+str(round(D020012)))
		file.write('\n')

		file.write(str(round(D010000/divby,digits))+' , '+str(round(D010110/divby,digits))+' , '+str(round(D010220/divby,digits))+' , '+str(round(D010120/divby,digits))+' , '+str(round(D010020/divby,digits))+' , '+str(round(D010010/divby,digits))+' , ')
		file.write(str(round(D010001/divby,digits))+' , '+str(round(D010111/divby,digits))+' , '+str(round(D010221/divby,digits))+' , '+str(round(D010121/divby,digits))+' , '+str(round(D010021/divby,digits))+' , '+str(round(D010011/divby,digits))+' , ')
		file.write(str(round(D010002/divby,digits))+' , '+str(round(D010112/divby,digits))+' , '+str(round(D010222/divby,digits))+' , '+str(round(D010122/divby,digits))+' , '+str(round(D010022/divby,digits))+' , '+str(round(D010012)))
		file.write('\n')

		file.write(str(round(D001000/divby,digits))+' , '+str(round(D001110/divby,digits))+' , '+str(round(D001220/divby,digits))+' , '+str(round(D001120/divby,digits))+' , '+str(round(D001020/divby,digits))+' , '+str(round(D001010/divby,digits))+' , ')
		file.write(str(round(D001001/divby,digits))+' , '+str(round(D001111/divby,digits))+' , '+str(round(D001221/divby,digits))+' , '+str(round(D001121/divby,digits))+' , '+str(round(D001021/divby,digits))+' , '+str(round(D001011/divby,digits))+' , ')
		file.write(str(round(D001002/divby,digits))+' , '+str(round(D001112/divby,digits))+' , '+str(round(D001222/divby,digits))+' , '+str(round(D001122/divby,digits))+' , '+str(round(D001022/divby,digits))+' , '+str(round(D001012)))
		file.write('\n')

		file.write(str(round(D111000/divby,digits))+' , '+str(round(D111110/divby,digits))+' , '+str(round(D111220/divby,digits))+' , '+str(round(D111120/divby,digits))+' , '+str(round(D111020/divby,digits))+' , '+str(round(D111010/divby,digits))+' , ')
		file.write(str(round(D111001/divby,digits))+' , '+str(round(D111111/divby,digits))+' , '+str(round(D111221/divby,digits))+' , '+str(round(D111121/divby,digits))+' , '+str(round(D111021/divby,digits))+' , '+str(round(D111011/divby,digits))+' , ')
		file.write(str(round(D111002/divby,digits))+' , '+str(round(D111112/divby,digits))+' , '+str(round(D111222/divby,digits))+' , '+str(round(D111122/divby,digits))+' , '+str(round(D111022/divby,digits))+' , '+str(round(D111012)))
		file.write('\n')

		file.write(str(round(D221000/divby,digits))+' , '+str(round(D221110/divby,digits))+' , '+str(round(D221220/divby,digits))+' , '+str(round(D221120/divby,digits))+' , '+str(round(D221020/divby,digits))+' , '+str(round(D221010/divby,digits))+' , ')
		file.write(str(round(D221001/divby,digits))+' , '+str(round(D221111/divby,digits))+' , '+str(round(D221221/divby,digits))+' , '+str(round(D221121/divby,digits))+' , '+str(round(D221021/divby,digits))+' , '+str(round(D221011/divby,digits))+' , ')
		file.write(str(round(D221002/divby,digits))+' , '+str(round(D221112/divby,digits))+' , '+str(round(D221222/divby,digits))+' , '+str(round(D221122/divby,digits))+' , '+str(round(D221022/divby,digits))+' , '+str(round(D221012)))
		file.write('\n')

		file.write(str(round(D121000/divby,digits))+' , '+str(round(D121110/divby,digits))+' , '+str(round(D121220/divby,digits))+' , '+str(round(D121120/divby,digits))+' , '+str(round(D121020/divby,digits))+' , '+str(round(D121010/divby,digits))+' , ')
		file.write(str(round(D121001/divby,digits))+' , '+str(round(D121111/divby,digits))+' , '+str(round(D121221/divby,digits))+' , '+str(round(D121121/divby,digits))+' , '+str(round(D121021/divby,digits))+' , '+str(round(D121011/divby,digits))+' , ')
		file.write(str(round(D121002/divby,digits))+' , '+str(round(D121112/divby,digits))+' , '+str(round(D121222/divby,digits))+' , '+str(round(D121122/divby,digits))+' , '+str(round(D121022/divby,digits))+' , '+str(round(D121012)))
		file.write('\n')

		file.write(str(round(D021000/divby,digits))+' , '+str(round(D021110/divby,digits))+' , '+str(round(D021220/divby,digits))+' , '+str(round(D021120/divby,digits))+' , '+str(round(D021020/divby,digits))+' , '+str(round(D021010/divby,digits))+' , ')
		file.write(str(round(D021001/divby,digits))+' , '+str(round(D021111/divby,digits))+' , '+str(round(D021221/divby,digits))+' , '+str(round(D021121/divby,digits))+' , '+str(round(D021021/divby,digits))+' , '+str(round(D021011/divby,digits))+' , ')
		file.write(str(round(D021002/divby,digits))+' , '+str(round(D021112/divby,digits))+' , '+str(round(D021222/divby,digits))+' , '+str(round(D021122/divby,digits))+' , '+str(round(D021022/divby,digits))+' , '+str(round(D021012)))
		file.write('\n')

		file.write(str(round(D011000/divby,digits))+' , '+str(round(D011110/divby,digits))+' , '+str(round(D011220/divby,digits))+' , '+str(round(D011120/divby,digits))+' , '+str(round(D011020/divby,digits))+' , '+str(round(D011010/divby,digits))+' , ')
		file.write(str(round(D011001/divby,digits))+' , '+str(round(D011111/divby,digits))+' , '+str(round(D011221/divby,digits))+' , '+str(round(D011121/divby,digits))+' , '+str(round(D011021/divby,digits))+' , '+str(round(D011011/divby,digits))+' , ')
		file.write(str(round(D011002/divby,digits))+' , '+str(round(D011112/divby,digits))+' , '+str(round(D011222/divby,digits))+' , '+str(round(D011122/divby,digits))+' , '+str(round(D011022/divby,digits))+' , '+str(round(D011012)))
		file.write('\n')


		file.write(str(round(D002000/divby,digits))+' , '+str(round(D002110/divby,digits))+' , '+str(round(D002220/divby,digits))+' , '+str(round(D002120/divby,digits))+' , '+str(round(D002020/divby,digits))+' , '+str(round(D002010/divby,digits))+' , ')
		file.write(str(round(D002001/divby,digits))+' , '+str(round(D002111/divby,digits))+' , '+str(round(D002221/divby,digits))+' , '+str(round(D002121/divby,digits))+' , '+str(round(D002021/divby,digits))+' , '+str(round(D002011/divby,digits))+' , ')
		file.write(str(round(D002002/divby,digits))+' , '+str(round(D002112/divby,digits))+' , '+str(round(D002222/divby,digits))+' , '+str(round(D002122/divby,digits))+' , '+str(round(D002022/divby,digits))+' , '+str(round(D002012)))
		file.write('\n')

		file.write(str(round(D112000/divby,digits))+' , '+str(round(D112110/divby,digits))+' , '+str(round(D112220/divby,digits))+' , '+str(round(D112120/divby,digits))+' , '+str(round(D112020/divby,digits))+' , '+str(round(D112010/divby,digits))+' , ')
		file.write(str(round(D112001/divby,digits))+' , '+str(round(D112111/divby,digits))+' , '+str(round(D112221/divby,digits))+' , '+str(round(D112121/divby,digits))+' , '+str(round(D112021/divby,digits))+' , '+str(round(D112011/divby,digits))+' , ')
		file.write(str(round(D112002/divby,digits))+' , '+str(round(D112112/divby,digits))+' , '+str(round(D112222/divby,digits))+' , '+str(round(D112122/divby,digits))+' , '+str(round(D112022/divby,digits))+' , '+str(round(D112012)))
		file.write('\n')

		file.write(str(round(D222000/divby,digits))+' , '+str(round(D222110/divby,digits))+' , '+str(round(D222220/divby,digits))+' , '+str(round(D222120/divby,digits))+' , '+str(round(D222020/divby,digits))+' , '+str(round(D222010/divby,digits))+' , ')
		file.write(str(round(D222001/divby,digits))+' , '+str(round(D222111/divby,digits))+' , '+str(round(D222221/divby,digits))+' , '+str(round(D222121/divby,digits))+' , '+str(round(D222021/divby,digits))+' , '+str(round(D222011/divby,digits))+' , ')
		file.write(str(round(D222002/divby,digits))+' , '+str(round(D222112/divby,digits))+' , '+str(round(D222222/divby,digits))+' , '+str(round(D222122/divby,digits))+' , '+str(round(D222022/divby,digits))+' , '+str(round(D222012)))
		file.write('\n')

		file.write(str(round(D122000/divby,digits))+' , '+str(round(D122110/divby,digits))+' , '+str(round(D122220/divby,digits))+' , '+str(round(D122120/divby,digits))+' , '+str(round(D122020/divby,digits))+' , '+str(round(D122010/divby,digits))+' , ')
		file.write(str(round(D122001/divby,digits))+' , '+str(round(D122111/divby,digits))+' , '+str(round(D122221/divby,digits))+' , '+str(round(D122121/divby,digits))+' , '+str(round(D122021/divby,digits))+' , '+str(round(D122011/divby,digits))+' , ')
		file.write(str(round(D122002/divby,digits))+' , '+str(round(D122112/divby,digits))+' , '+str(round(D122222/divby,digits))+' , '+str(round(D122122/divby,digits))+' , '+str(round(D122022/divby,digits))+' , '+str(round(D122012)))
		file.write('\n')

		file.write(str(round(D022000/divby,digits))+' , '+str(round(D022110/divby,digits))+' , '+str(round(D022220/divby,digits))+' , '+str(round(D022120/divby,digits))+' , '+str(round(D022020/divby,digits))+' , '+str(round(D022010/divby,digits))+' , ')
		file.write(str(round(D022001/divby,digits))+' , '+str(round(D022111/divby,digits))+' , '+str(round(D022221/divby,digits))+' , '+str(round(D022121/divby,digits))+' , '+str(round(D022021/divby,digits))+' , '+str(round(D022011/divby,digits))+' , ')
		file.write(str(round(D022002/divby,digits))+' , '+str(round(D022112/divby,digits))+' , '+str(round(D022222/divby,digits))+' , '+str(round(D022122/divby,digits))+' , '+str(round(D022022/divby,digits))+' , '+str(round(D022012)))
		file.write('\n')

		file.write(str(round(D012000/divby,digits))+' , '+str(round(D012110/divby,digits))+' , '+str(round(D012220/divby,digits))+' , '+str(round(D012120/divby,digits))+' , '+str(round(D012020/divby,digits))+' , '+str(round(D012010/divby,digits))+' , ')
		file.write(str(round(D012001/divby,digits))+' , '+str(round(D012111/divby,digits))+' , '+str(round(D012221/divby,digits))+' , '+str(round(D012121/divby,digits))+' , '+str(round(D012021/divby,digits))+' , '+str(round(D012011/divby,digits))+' , ')
		file.write(str(round(D012002/divby,digits))+' , '+str(round(D012112/divby,digits))+' , '+str(round(D012222/divby,digits))+' , '+str(round(D012122/divby,digits))+' , '+str(round(D012022/divby,digits))+' , '+str(round(D012012/divby,digits)))
		file.write('\n')
		file.write('\n')
		file.write(r'\end{pmatrix} $} 10^{')
		file.write(str(round(powerdivby)))
		file.write(r'} \text{\,N} \ ,')
		file.write('\n')
		file.write(r'\eqend')
		
	with open(data+'.tex', 'a') as file:
		file.write(r'\end{document}')
		file.write('\n')
