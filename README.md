# Underwater sound wave propagation simulation

Numerical simulation of underwater sound wave propagation using the finite difference method (central/explicit scheme), based on the 1D acoustic wave equation with depth-dependent sound speed, damping, and configurable boundary conditions. Related to my bachelor's thesis, *"Simulation of underwater sound wave propagation with central difference method."*

Includes:
- Original one-dimensional underwater sound wave propagation simulation used in bachelor's thesis `wave1D.py`
- Complete one-dimensional underwater sound wave propagation simulation with support for Dirichlet or Neumann boundary conditions, variable sound speed profiles, damping, forcing terms, and stochastic noise `wave1D_extended.py`
- Two-dimensional simulation, developed as part of follow-up research `wave2D.py`

## Features

- **Two integration engines**: `python` (plain NumPy) and `numba` (JIT-compiled via `@jit(nopython=True)`) for performance comparison
- **Two update schemes**: `vector` (fully vectorized NumPy slicing) and `scalar` (explicit index loop) — useful for comparing vectorization overhead against numba's JIT
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

## Acknowledgments

Code adapted in part from Langtangen and Linge, *Finite Difference Methods for Wave Equations* ([fdm-book](https://github.com/hplgit/fdm-book)), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Development was assisted by **Claude Sonnet 5** (Anthropic), used for:
- Code review and debugging (e.g. vectorization, CFL condition handling, boundary condition fixes)
- Design/architecture discussion (e.g. numba vs. Cython evaluation)
- Git/GitHub workflow guidance


All code decisions, implementation, and testing were carried out by the repository owner. AI tools were used as a development aid, not as an autonomous contributor.
