import time as comp_time
start = comp_time.time()
#Activate firedrake
#source /home/prog/firedrake24/firedrake/bin/activate 
import os
import re
import csv

# Prepare the CSV file before starting the loop
csv_filename = "tpms_simulation_results2.csv"
file_exists = os.path.exists(csv_filename)

with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
  writer = csv.writer(f)
  # Add the header row if the file is newly created
  if not file_exists:
    writer.writerow([
        "TPMS_Type",
        "Infill",
        "Core_Thickness_mm",
        "w_min",
        "uz_top",
        "uz_bottom",
        "Calculation_Time_s",
    ])

import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYOP2_CACHE_DIR"] = "/home/cagri/.firedrakecache"
os.environ["FIREDRAKE_TSFC_KERNEL_CACHE_DIR"] = "/home/cagri/.firedrakecache"

from firedrake.petsc import PETSc
from firedrake import *
comm = COMM_WORLD
from ufl import Constant, as_tensor, derivative, ds, ds, dx, grad, indices, inner
from ufl import conditional, And, Or, gt, lt
from mpi4py import MPI

solver_parameters = {
    "snes_type": "ksponly",
    "snes_converged_reason": "",
    "ksp_type": "gmres",              # Instead of preonly: use LU as a preconditioner to enable refinement
    "ksp_rtol": 1e-10,
    "ksp_monitor_true_residual": "",
    "ksp_converged_reason": "",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_4": 0,           # MUMPS error/warning output
    "mat_mumps_icntl_14": 100,
    "mat_mumps_cntl_1": 0.1,          # Pivot threshold (default: 0.01)
    "mat_mumps_icntl_8": 77,          # Automatic scaling
    "mat_mumps_icntl_10": 5,          # Iterative refinement
}

#units: mm, Mg = 1000 kg=ton, s, MPa, mJ, K
#f = Constant((0.0, 0.0, -9810.))
import numpy as np
from voigt2tensor import *

from voigt2tensor import (
    VoigtToTensor3D_rank4,
    VoigtToTensor3D_rank6
)
def load_tensors_from_tex(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    matrices_raw = re.findall(r'\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}', content, re.DOTALL)
    
    if len(matrices_raw) < 3:
        raise ValueError(f"Error: At least 3 pmatrix blocks were not found in {file_path}.")

    target_shapes = [(6, 6), (6, 18), (18, 18)]
    parsed_matrices = []

    for idx, raw_m in enumerate(matrices_raw[:3]):
        numbers = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', raw_m)]
        
        expected_shape = target_shapes[idx]
        expected_size = expected_shape[0] * expected_shape[1]
        
        if len(numbers) != expected_size:
            raise ValueError(
                f"Expected {expected_size} elements in matrix {idx+1} of {file_path}, "
                f"but found {len(numbers)} elements."
            )
        
        mat_array = np.array(numbers).reshape(expected_shape)
        parsed_matrices.append(mat_array)

    C_Voigt = parsed_matrices[0]  # Shape: (6, 6)
    G_Voigt = parsed_matrices[1]  # Shape: (6, 18)
    D_Voigt = parsed_matrices[2]  # Shape: (18, 18)

    return C_Voigt, G_Voigt, D_Voigt

def get_strain_gradient_tensors(tex_filename):
    C_Voigt, G_Voigt, D_Voigt = load_tensors_from_tex(tex_filename)
    
    C_M = VoigtToTensor3D_rank4(C_Voigt)
    G_M = VoigtToTensor3D_rank5(G_Voigt)
    D_M = VoigtToTensor3D_rank6(D_Voigt)
    
    return C_Voigt, G_Voigt, D_Voigt, C_M, G_M, D_M

tex_files = [
    "FischerKoch_0.3_50.tex", "Lidinoid_0.3_50.tex", "IWP_0.3_50.tex", "Nevious_0.3_50.tex",
    "FRD_0.3_50.tex", "SchwarzP_0.3_50.tex", "Gyroid_0.3_50.tex", "Diamond_0.3_50.tex"]

# Core thickness values to loop over
core_thickness_values = [30.0, 40.0, 50.0]

for t_core in core_thickness_values:
    PETSc.Sys.Print("===== Core thickness t_core = {} mm =====".format(t_core))

    # Thickness parameters
    t_skin = 2.0  # skin thickness
    # t_core is now set by the outer loop
    # ---- MESH LEVEL: change only nx, ny, nz_skin, nz_core for the convergence study ----
    nz_skin = 1   # elements through each 2 mm skin
    nz_core = int(t_core/10)  # elements through the 40 mm core (core element = 40/nz_core mm)

    xmin, ymin, zmin = 0., 0., 0.
    xmax, ymax = 960.0, 580
    zmax = 2 * t_skin + t_core  # Lz total thickness

    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
    seat_lx, seat_ly = xmax/10, ymax/8     # corner patch 96 x 72.75 mm, the same for every mesh

    # ---- GRADED IN-PLANE MESH ------------------------------------------------------
    # Fine elements at the seat-patch edges (x = seat_lx, Lx-seat_lx ; y = seat_ly, Ly-seat_ly),
    # growing geometrically (ratio r) towards the panel centre and towards the outer edges.
    # Seat edges and panel centre are always nodes, so the support patch never changes with mesh level.
    # Mesh family for the convergence study: change only NA, NB (r fixed), e.g. (4,6), (6,10), (9,15)
    NA = int(os.environ.get("NA", 6))      # elements between outer edge and seat edge
    NB = int(os.environ.get("NB", 10))     # elements between seat edge and panel centre
    R  = float(os.environ.get("R", 1.3))   # growth ratio

    def geom(a, b, n, r):
        """n elements from a to b; smallest at a, each element r times larger."""
        s = (r**np.arange(n+1) - 1) / (r**n - 1)
        return a + (b - a) * s

    def levels(L, seat, na, nb, r):
        half = np.concatenate([geom(seat, 0.0, na, r)[::-1],     # [0, seat]
                               geom(seat, L/2, nb, r)[1:]])      # [seat, L/2]
        return np.concatenate([half, (L - half[::-1])[1:]])      # mirror to [L/2, L]

    x_levels = levels(Lx, seat_lx, NA, NB, R)
    y_levels = levels(Ly, seat_ly, NA, NB, R)
    nx, ny = len(x_levels) - 1, len(y_levels) - 1
    dx_min, dx_max = np.diff(x_levels).min(), np.diff(x_levels).max()
    dy_min, dy_max = np.diff(y_levels).min(), np.diff(y_levels).max()
    print("graded mesh NA={} NB={} r={}".format(NA, NB, R))
    print("dx min/max: {:.1f} / {:.1f}   dy min/max: {:.1f} / {:.1f}".format(dx_min, dx_max, dy_min, dy_max))

    # ---- Loading and supports (Sahib & Kovacs, Appl. Sci. 2023, 13, 3876) ------
    pressure = 0.004142   # 4.142 kPa on the upper face [MPa]

    # Non-uniform Z levels
    z_levels = np.concatenate([
        np.linspace(0.0, t_skin, nz_skin + 1),
        np.linspace(t_skin, t_skin + t_core, nz_core + 1)[1:],
        np.linspace(t_skin + t_core, Lz, nz_skin + 1)[1:],
    ])
    nz = len(z_levels) - 1

    PETSc.Sys.Print('setting up mesh across %d processes' % COMM_WORLD.size)

    mesh = BoxMesh(nx, ny, nz, Lx, Ly, Lz)

    # Uniformly distributed z coordinates are mapped onto z_levels
    # (element faces end up exactly on z = t_skin and z = t_skin + t_core)
    coords = mesh.coordinates.dat.data_with_halos
    z_uniform = np.linspace(0.0, Lz, nz + 1)
    coords[:, 0] = np.interp(coords[:, 0], np.linspace(0.0, Lx, nx + 1), x_levels)
    coords[:, 1] = np.interp(coords[:, 1], np.linspace(0.0, Ly, ny + 1), y_levels)
    coords[:, 2] = np.interp(coords[:, 2], z_uniform, z_levels)

    i,j,k,l,m,n,o,p = indices(8) 
    Dim = mesh.geometric_dimension()
    delta = Identity(Dim)
    print("Dimension:", Dim)
    print("nx, ny, nz:", nx, ny, nz)
    print("X,Y,Z:", xmax,ymax,zmax)
    x, y, z = SpatialCoordinate(mesh)
    bc = []

    vector_lin_element = VectorElement('CG', mesh.ufl_cell(), degree=2)
    tensor_lin_element = TensorElement('CG', mesh.ufl_cell(), degree=1)
    LM_lin_element = TensorElement('CG', mesh.ufl_cell(), degree=1)
    mixed_element = MixedElement([vector_lin_element, tensor_lin_element, LM_lin_element])
    Space = FunctionSpace(mesh, mixed_element)
    du = TrialFunction(Space)
    del_unk = TestFunction(Space)
    del_u, del_g, del_LM = split(del_unk)
    unk = Function(Space, name='unknown')
    u, g, LM = split(unk)

    #E = as_tensor(0.5 * (u[i].dx(j) + u[j].dx(i) + u[l].dx(i)*u[l].dx(j)), [i,j])
    #gE = as_tensor(0.5 * (g[i, j].dx(k) + g[j, i].dx(k) + g[l,i].dx(k)*g[l,j] + g[l,i]*g[l,j].dx(k)), [i,j,k])

    E = as_tensor(0.5 * (u[i].dx(j) + u[j].dx(i)), [i,j])
    gE = as_tensor(0.5 * (g[i, j].dx(k) + g[j, i].dx(k)), [i,j,k])


    zero_vector = Constant((0.0, 0.0, 0.0))
    zero_tensor = Constant(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))


    tol = 1e-6
    corner = conditional(And(lt(z, tol),
                             And(Or(lt(x, seat_lx + tol), gt(x, Lx - seat_lx - tol)),
                                 Or(lt(y, seat_ly + tol), gt(y, Ly - seat_ly - tol)))), 1.0, 0.0)

    print("seat footprint: {:.1f} x {:.1f} mm".format(seat_lx, seat_ly))
    def corner_nodes(degree):       # serial run assumed
        mask = Function(FunctionSpace(mesh, "CG", degree)).interpolate(corner)
        return np.where(mask.dat.data_ro > 0.5)[0].astype(PETSc.IntType)

    bc_u = DirichletBC(Space.sub(0), zero_vector, "on_boundary")
    bc_g = DirichletBC(Space.sub(1), zero_tensor, "on_boundary")
    bc_u.nodes = corner_nodes(2)
    bc_g.nodes = corner_nodes(1)
    bc = [bc_u, bc_g]

    F_constraint = (LM[i, j] * (del_g[i, j] - del_u[i].dx(j)) + del_LM[i, j] * (g[i, j] - u[i].dx(j))) * dx
    load = Constant(1.0) # initial load factor, zero in case of a load stepping 
    traction = load * pressure * del_u[2] * ds(6)

    d_scale = Constant(1.0)

    # Core (t_skin < z < t_skin + t_core): homogenized TPMS (C_M, D_M).
    # The mesh has element faces exactly at z = t_skin and z = t_skin + t_core,
    # so this indicator never cuts through an element.
    is_core = conditional(And(gt(z, t_skin), lt(z, t_skin + t_core)), 1.0, 0.0)

    tpms_types = [
        "FischerKoch", "Lidinoid", "IWP", "Nevious",
        "FRD", "Gyroid", "Diamond", "SchwarzP"
    ]

    #tpms_types = ["Lidinoid", "Diamond", "IWP", "FischerKoch"]
    #infill_values = [0.1]

    infill_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tex_dir = "/home/cagri/TPMS/voksel_SG_loop/v2_parameters"
    for tpms in tpms_types:
        for infill in infill_values:
            data = f"{tpms}_{infill}"
            print(f"Processing: {data} (t_core = {t_core} mm)")
            tex_file = os.path.join(tex_dir, f"{tpms}_{infill}_50.tex")

            if not os.path.exists(tex_file):
                PETSc.Sys.Print(f"File not found, skipping: {tex_file}")
                continue
            
            C_Voigt, G_Voigt, D_Voigt, C_M, G_M, D_M = get_strain_gradient_tensors(tex_file)

            E_mat = 3500 #in MPa
            nu_mat = 0.3
            la_mat = E_mat*nu_mat/(1.+nu_mat)/(1.-2.*nu_mat)
            mu_mat = E_mat/2./(1.+nu_mat)
            Cmat_m = as_tensor(la_mat*delta[i,j]*delta[k,l] + mu_mat*delta[i,k]*delta[j,l] + mu_mat*delta[i,l]*delta[j,k], (i,j,k,l))
            Strainterm_skin = 0.5 * E[i,j]*Cmat_m[i,j,k,l]*E[k,l]

            Strainterm_core = 0.5 * E[i,j]*C_M[i,j,k,l]*E[k,l]
            Strainterm = is_core*Strainterm_core + (1.0 - is_core)*Strainterm_skin
            StrainGradientterm = is_core*d_scale*0.5*gE[i,j,k]*D_M[i,j,k,l,m,n]*gE[l,m,n]

            Pi = (Strainterm + StrainGradientterm)*dx

            F = derivative(Pi, unk, del_unk) + F_constraint + traction
            J = derivative(F, unk, du) 
            solve(F == 0, unk, bcs=bc, J=J, solver_parameters=solver_parameters)

            u_h = unk.subfunctions[0]
            w_min = comm.allreduce(float(u_h.dat.data_ro[:, 2].min()), op=MPI.MIN)
            PETSc.Sys.Print("max downward deflection (nodal min uz) = {:.5f} mm".format(w_min))

            u, g, LM = unk.subfunctions
            pwd = "data_mixed_SG2/"
            projected = VTKFile( pwd + "u_3D.pvd", project_output=True)
            projected.write(u)

            end = comp_time.time()
            calculation_time = (end - start)

            PETSc.Sys.Print("Calculation Time: {} s\n".format(round(calculation_time)))

            uz_c = {}
            for label, z_pt in (("top", Lz), ("bottom", 0.0)):
                try:
                    uz_c[label] = float(u_h.at((xmax/2, ymax/2, z_pt))[2])
                    PETSc.Sys.Print("uz at panel centre, {} face = {:.5f} mm".format(label, uz_c[label]))
                except Exception:
                    uz_c[label] = float("nan")
                    PETSc.Sys.Print("uz at panel centre, {} face = n/a".format(label))

            # One line per run -> collect with:  grep RESULT *.log
            PETSc.Sys.Print("RESULT graded NA={} NB={} r={} nx={} ny={} nz_skin={} nz_core={} dx_min={:.2f} dy_min={:.2f} dofs={} w_min={:.5f} uz_top={:.5f} uz_bottom={:.5f} time={:.0f}s".format(
                NA, NB, R, nx, ny, nz_skin, nz_core, dx_min, dy_min, Space.dim(), w_min, uz_c["top"], uz_c["bottom"], calculation_time))


    # --- WRITING TO CSV FILE ---
        # In a parallel computing (MPI) environment, writing only from the master process (rank 0) prevents conflicts:
            if comm.rank == 0:
              with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    tpms,
                    infill,
                    t_core,
                    w_min,
                ])

            PETSc.Sys.Print(
                "RESULT graded NA={} NB={} r={} nx={} ny={} nz_skin={} nz_core={}"
                " dx_min={:.2f} dy_min={:.2f} dofs={} w_min={:.5f} uz_top={:.5f}"
                " uz_bottom={:.5f} time={:.0f}s".format(
                    NA,
                    NB,
                    R,
                    nx,
                    ny,
                    nz_skin,
                    nz_core,
                    dx_min,
                    dy_min,
                    Space.dim(),
                    w_min,
                    uz_c["top"],
                    uz_c["bottom"],
                    calculation_time,
                )
            )