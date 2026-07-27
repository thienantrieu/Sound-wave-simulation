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
`prange`) outperformed the `vector` scheme (NumPy slice-based update, parallelized
via numba's automatic array fusion) on a multi-core machine, benchmarked at
`Nx=5000` over repeated runs.

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

**This result is core-count dependent.** On a single-core machine, `prange`
cannot parallelize at all — `parallel=True` then only adds thread-management
overhead with no benefit, and in that setting `vector_parallel` was observed
to be faster than `scalar_parallel` instead (fewer, more fused parallel regions
per timestep). The `scalar`-faster result should therefore only be assumed to
hold on hardware with multiple available cores.

This result was measured at `Nx=5000`; it has not been verified across a wider
range of grid sizes, core counts, or hardware. Both schemes were confirmed to
produce bit-identical output (`receiverA`, `receiverB`) under `parallel=True`,
so the timing difference is not caused by a correctness discrepancy between
the two implementations.

## Testing

`1D/wave1D_tests.py` is a `pytest` sanity-test suite covering `wave1D.py` and
`wave1D_extended.py`, in the exact-reproduction/verification style used in
fdm-book's `wave1D_dn.py`: initial conditions and forcing terms are chosen so
that the *exact* solution is known in closed form (a constant field, a
manufactured quadratic solution, a plug pulse under reflecting boundaries),
and the scheme's output is checked against it to machine precision, rather
than just checked for "reasonable-looking" behaviour.

Covers:
- **Exact-reproduction tests**: constant solution (any damping), manufactured
  quadratic solution with matching forcing term (Dirichlet boundaries),
  plug-pulse round trip under reflecting (Neumann) boundaries
- **Empirical convergence rate**: a standing-wave eigenfunction of the
  Neumann-Neumann Laplacian, confirming ~2nd-order accuracy as the mesh is
  refined at fixed Courant number
- **Symmetry and damping sanity checks**: symmetric input stays symmetric;
  added damping strictly reduces peak amplitude
- **Cross-consistency**: `scalar` vs. `vector`, `python` vs. `numba`, and
  `parallel=True` vs. `parallel=False` all reproduce identical output for
  the same input; `wave1D.py` and `wave1D_extended.py` (with Neumann
  boundaries) agree with each other
- **Error handling**: unknown `scheme`/`engine` names, and
  `engine='python'` combined with `parallel=True`, raise `ValueError`

Run from the `1D` directory:

```bash
pip install pytest
pytest wave1D_tests.py -v
```

## Acknowledgments
Code adapted in part from Langtangen and Linge, *Finite Difference Methods for
Wave Equations* ([fdm-book](https://github.com/hplgit/fdm-book)), licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Development was assisted by **Claude** (Anthropic) for code review, numba
refactoring, test suite design, and documentation drafting. All design
decisions, implementation, and testing were carried out by the repository
owner.
