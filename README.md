# Underwater sound wave propagation simulation

Numerical simulation of underwater sound wave propagation using the finite difference method (central/explicit scheme), based on the 1D acoustic wave equation with depth-dependent sound speed, damping, and configurable boundary conditions. Related to my bachelor's thesis, *"Simulation of underwater sound wave propagation with central difference method."*

Includes:
- Original one-dimensional underwater sound wave propagation simulation used in bachelor's thesis `wave1D.py`
- Complete one-dimensional underwater sound wave propagation simulation with support for Dirichlet or Neumann boundary conditions, variable sound speed profiles, damping, forcing terms, and stochastic noise `wave1D_extended.py`
- Two-dimensional simulation, developed as part of follow-up research `wave2D.py`

## Features

- **Two integration engines**: `python` (plain NumPy) and `numba` (JIT-compiled via `@jit(nopython=True)`) for performance comparison
- **Two update schemes**: `vector` (fully vectorized NumPy slicing) and `scalar` (explicit index loop) — useful for comparing vectorization overhead against numba's JIT
- **Optional parallel execution** (`parallel=True`, `numba` engine only): parallelizes the spatial update within each timestep using `prange`. The time-marching loop itself stays sequential, since each timestep depends on the previous one — only the independent per-point updates within a single step are parallelized. Available for both `scheme` values.
- **CFL stability check** printed at runtime, based on the Courant number
- **Configurable boundary conditions**: Dirichlet (fixed value) or a damped/absorbing condition at each end independently
- **Depth-dependent sound speed** via the empirical Del Grosso/Mackenzie-style formula (temperature- and salinity-dependent, adjustable with depth)
- **Pulse shapes**: Gaussian, cosine-hat, and half-cosine-hat initial conditions
- **Optional stochastic forcing** (noise), with true random numbers sourced from random.org (falls back to NumPy's PRNG if the request fails or the step count is too high)
- Runtime performance metrics (CPU time, peak memory via `tracemalloc`)


## Installation

```bash
git clone https://github.com/thienantrieu/Sound-wave-simulation.git
cd Sound-wave-simulation
pip install -r requirements.txt
```

## Usage

Run the 1D simulation directly:

```bash
cd 1D
python wave1D_extended.py
```

Or call `solver()` / `main()` from your own script to customize parameters (depth, CFL number, damping, mesh resolution, pulse type, engine, scheme, etc.) — see the `main()` function in `wave1D_extended.py` for the full parameter list and defaults.

## Method

The wave equation is discretized with a standard second-order central difference scheme in both space and time (explicit, conditionally stable — see the CFL check at runtime). The core update kernels are implemented separately for each (engine, scheme) combination in `wave1D_implementations.py` and dispatched via a lookup table, so the four variants can be benchmarked against each other without any engine seeing overhead from the others (see `RUN_FUNCTIONS`).

A Cython implementation was evaluated as an alternative/addition to the numba backend (see [issue #1](https://github.com/thienantrieu/Sound-wave-simulation/issues/1)) — not pursued, since numba's JIT already reaches near-C performance for this kind of stencil loop.

## Performance notes

### Scalar vs. vector under parallel execution

With `parallel=True`, the `scalar` scheme (explicit index loop, parallelized with
`prange`) consistently outperforms the `vector` scheme (NumPy slice-based update,
parallelized via numba's automatic array fusion) — benchmarked at `Nx=5000` over
repeated runs.

The likely explanation is how each scheme handles intermediate values within the
stencil update. The vectorized update

```python
u[1:-1] = (1 / (1 + 0.5*b*dt)) * ((0.5*b*dt - 1)*u_nm1[1:-1] + 2*u_n[1:-1]
            + 0.5*C2*((q[1:-1]+q[2:])*(u_n[2:]-u_n[1:-1]) - (q[1:-1]+q[:-2])*(u_n[1:-1]-u_n[:-2]))
            + dt2*f_vals[n, 1:-1])
```

chains several NumPy slice operations (`q[1:-1]+q[2:]`, `u_n[2:]-u_n[1:-1]`, etc.).
Even though numba fuses this into a single parallel region (confirmed with
`parallel_diagnostics(level=4)`), each sub-expression may still be materialized as
an intermediate array before being combined. The scalar version

```python
u[i] = (1 / (1 + 0.5*b*dt)) * ((0.5*b*dt - 1)*u_nm1[i] + 2*u_n[i]
        + 0.5*C2*((q[i]+q[i+1])*(u_n[i+1]-u_n[i]) - (q[i]+q[i-1])*(u_n[i]-u_n[i-1]))
        + dt2*f_vals[n, i])
```

computes the same quantity per grid point using only scalar values, which stay in
registers rather than being written to and read back from memory.

Since this stencil update is memory-bandwidth-bound rather than compute-bound
(few floating-point operations per byte read/written), the extra memory traffic
from intermediate arrays in the vectorized version outweighs whatever benefit
NumPy's slice syntax would normally offer. This is consistent with the `scalar`
scheme's noise-application and boundary-update code also being written as
explicit per-index loops rather than chained array operations, for the same reason.

This result was measured at `Nx=5000`; it has not been verified across a wider
range of grid sizes or hardware, and the relative overhead of thread
dispatch/synchronization under `parallel=True` may behave differently at much
smaller `Nx`.

## Acknowledgments

Code adapted in part from Langtangen and Linge, *Finite Difference Methods for Wave Equations* ([fdm-book](https://github.com/hplgit/fdm-book)), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Development was assisted by **Claude Sonnet 5** (Anthropic), used for:
- Code review and debugging (e.g. `wave2D.py` vectorization and code rework for `numba` compatibility)
- Design/architecture discussion (e.g. numba vs. Cython evaluation)
- Git/GitHub workflow guidance


All code decisions, implementation, and testing were carried out by the repository owner. AI tools were used as a development aid, not as an autonomous contributor.
