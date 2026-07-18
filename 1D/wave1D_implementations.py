import numpy as np
from numba import jit

# ---------------------------------------------------------------------------
# Boundary conditions applied once per timestep (n = 1 ... Nt-1).
#
# Kept as two near-identical copies (python / numba) on purpose: sharing one
# jitted function between the two engines would make the 'python' engine
# timings include numba overhead, which would break the engine comparison.
# ---------------------------------------------------------------------------

def boundaries_python(u, u_n, u_nm1, q, C2, b, dt, dt2, n,
                       f_vals, bd0_dirichlet, bd0_vals, bdL_dirichlet, bdL_vals):
    if bd0_dirichlet:
        #Dirichlet condition
        u[0] = bd0_vals[n + 1]
    else:
        # Neumann condition
        u[0] = (1 / (1 + 0.5 * b * dt)) * ((0.5 * b * dt - 1) * u_nm1[0] + 2 * u_n[0]
                + 2 * C2 * q[0] * (u_n[1] - u_n[0]) + dt2 * f_vals[n, 0])

    if bdL_dirichlet:
        #Dirichlet condition
        u[-1] = bdL_vals[n + 1]
    else:
        # Neumann condition
        u[-1] = (1 / (1 + 0.5 * b * dt)) * ((0.5 * b * dt - 1) * u_nm1[-1] + 2 * u_n[-1]
                 + 2 * C2 * q[-1] * (u_n[-2] - u_n[-1]) + dt2 * f_vals[n, -1])
    return u


@jit(nopython=True)
def boundaries_numba(u, u_n, u_nm1, q, C2, b, dt, dt2, n,
                      f_vals, bd0_dirichlet, bd0_vals, bdL_dirichlet, bdL_vals):
    if bd0_dirichlet:
        #Dirichlet condition
        u[0] = bd0_vals[n + 1]
    else:
        # Neumann condition
        u[0] = (1 / (1 + 0.5 * b * dt)) * ((0.5 * b * dt - 1) * u_nm1[0] + 2 * u_n[0]
                + 2 * C2 * q[0] * (u_n[1] - u_n[0]) + dt2 * f_vals[n, 0])

    if bdL_dirichlet:
        #Dirichlet condition
        u[-1] = bdL_vals[n + 1]
    else:
        # Neumann condition
        u[-1] = (1 / (1 + 0.5 * b * dt)) * ((0.5 * b * dt - 1) * u_nm1[-1] + 2 * u_n[-1]
                 + 2 * C2 * q[-1] * (u_n[-2] - u_n[-1]) + dt2 * f_vals[n, -1])
    return u


# ---------------------------------------------------------------------------
# Time-marching (integration) routines: one per (engine, scheme) combination.
#
# All four share the exact same parameter list so `solver()` can call
# whichever one it picks from _RUN_FUNCTIONS with a single call site,
# instead of branching per engine/scheme.
#
# noise_vals is a pre-evaluated (Nt, Nx+1) array (zeros if noise is off) so
# neither engine ever needs to call I(x) as a closure inside the loop -
# this is what let the numba versions work at all, and it also fixes the
# NameError that random_list caused when noise=False.
#
# receiverA/receiverB are pre-allocated numpy arrays (not Python lists):
# numba's nopython mode does not reliably support reflected list.append()
# any more, and using arrays for both engines keeps behaviour identical.
# ---------------------------------------------------------------------------

def _run_python_vector(u, u_n, u_nm1, b, dt, dt2, Nt, Nx, C2, q,
                        receiverA, receiverB, f_vals, noise_vals,
                        bd0_dirichlet, bd0_vals, bdL_dirichlet, bdL_vals,
                        performance=2, callback=None, xs=None, ts=None):
    # callback/xs/ts/performance are only ever passed by the 'python' engine
    # branch in solver() - the numba versions cannot accept a Python
    # callable, so they keep the shorter signature untouched.
    for n in range(1, Nt):
        u = u + noise_vals[n]
        u_n = u_n + noise_vals[n]
        u_nm1 = u_nm1 + noise_vals[n]

        u[1:-1] = (1 / (1 + 0.5 * b * dt)) * ((0.5 * b * dt - 1) * u_nm1[1:-1] + 2 * u_n[1:-1]
                    + 0.5 * C2 * ((q[1:-1] + q[2:]) * (u_n[2:] - u_n[1:-1]) - (q[1:-1] + q[:-2]) * (u_n[1:-1] - u_n[:-2]))
                    + dt2 * f_vals[n, 1:-1])

        u = boundaries_python(u, u_n, u_nm1, q, C2, b, dt, dt2, n,
                               f_vals, bd0_dirichlet, bd0_vals, bdL_dirichlet, bdL_vals)

        receiverA[n + 1] = u[0]
        receiverB[n + 1] = u[-1]

        if performance < 1 and callback is not None:
            if callback(u, xs, ts, n + 1):
                return u, u_n, u_nm1, n + 1

        u, u_n, u_nm1 = u_nm1, u, u_n
    return u, u_n, u_nm1


def _run_python_scalar(u, u_n, u_nm1, b, dt, dt2, Nt, Nx, C2, q,
                        receiverA, receiverB, f_vals, noise_vals,
                        bd0_dirichlet, bd0_vals, bdL_dirichlet, bdL_vals,
                        performance=2, callback=None, xs=None, ts=None):
    for n in range(1, Nt):
        u = u + noise_vals[n]
        u_n = u_n + noise_vals[n]
        u_nm1 = u_nm1 + noise_vals[n]

        for i in range(1, Nx):
            u[i] = (1 / (1 + 0.5 * b * dt)) * ((0.5 * b * dt - 1) * u_nm1[i] + 2 * u_n[i]
                    + 0.5 * C2 * ((q[i] + q[i + 1]) * (u_n[i + 1] - u_n[i]) - (q[i] + q[i - 1]) * (u_n[i] - u_n[i - 1]))
                    + dt2 * f_vals[n, i])

        u = boundaries_python(u, u_n, u_nm1, q, C2, b, dt, dt2, n,
                               f_vals, bd0_dirichlet, bd0_vals, bdL_dirichlet, bdL_vals)

        receiverA[n + 1] = u[0]
        receiverB[n + 1] = u[-1]

        if performance < 1 and callback is not None:
            if callback(u, xs, ts, n + 1):
                return u, u_n, u_nm1, n + 1

        u, u_n, u_nm1 = u_nm1, u, u_n
    return u, u_n, u_nm1


@jit(nopython=True)
def _run_numba_vector(u, u_n, u_nm1, b, dt, dt2, Nt, Nx, C2, q,
                       receiverA, receiverB, f_vals, noise_vals,
                       bd0_dirichlet, bd0_vals, bdL_dirichlet, bdL_vals):
    for n in range(1, Nt):
        u = u + noise_vals[n]
        u_n = u_n + noise_vals[n]
        u_nm1 = u_nm1 + noise_vals[n]

        u[1:-1] = (1 / (1 + 0.5 * b * dt)) * ((0.5 * b * dt - 1) * u_nm1[1:-1] + 2 * u_n[1:-1]
                    + 0.5 * C2 * ((q[1:-1] + q[2:]) * (u_n[2:] - u_n[1:-1]) - (q[1:-1] + q[:-2]) * (u_n[1:-1] - u_n[:-2]))
                    + dt2 * f_vals[n, 1:-1])

        u = boundaries_numba(u, u_n, u_nm1, q, C2, b, dt, dt2, n,
                              f_vals, bd0_dirichlet, bd0_vals, bdL_dirichlet, bdL_vals)

        receiverA[n + 1] = u[0]
        receiverB[n + 1] = u[-1]

        u, u_n, u_nm1 = u_nm1, u, u_n
    return u, u_n, u_nm1


@jit(nopython=True)
def _run_numba_scalar(u, u_n, u_nm1, b, dt, dt2, Nt, Nx, C2, q,
                       receiverA, receiverB, f_vals, noise_vals,
                       bd0_dirichlet, bd0_vals, bdL_dirichlet, bdL_vals):
    for n in range(1, Nt):
        u = u + noise_vals[n]
        u_n = u_n + noise_vals[n]
        u_nm1 = u_nm1 + noise_vals[n]

        for i in range(1, Nx):
            u[i] = (1 / (1 + 0.5 * b * dt)) * ((0.5 * b * dt - 1) * u_nm1[i] + 2 * u_n[i]
                    + 0.5 * C2 * ((q[i] + q[i + 1]) * (u_n[i + 1] - u_n[i]) - (q[i] + q[i - 1]) * (u_n[i] - u_n[i - 1]))
                    + dt2 * f_vals[n, i])

        u = boundaries_numba(u, u_n, u_nm1, q, C2, b, dt, dt2, n,
                              f_vals, bd0_dirichlet, bd0_vals, bdL_dirichlet, bdL_vals)

        receiverA[n + 1] = u[0]
        receiverB[n + 1] = u[-1]

        u, u_n, u_nm1 = u_nm1, u, u_n
    return u, u_n, u_nm1


# Dispatch table: (engine, scheme) -> integration routine.
RUN_FUNCTIONS = {
    ('python', 'vector'): _run_python_vector,
    ('python', 'scalar'): _run_python_scalar,
    ('numba', 'vector'): _run_numba_vector,
    ('numba', 'scalar'): _run_numba_scalar,
}
