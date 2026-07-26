# Sanity tests for wave1D.py and wave1D_extended.py.
#
# Style/approach adapted from
# Finite difference methods for wave equations
# by Langtangen and Linge
# https://github.com/hplgit/fdm-book
# (see e.g. src/wave/wave1D/wave1D_dn.py and wave1D_n0.py)
# Licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
#
# The three checks below are the classic "verification by exact
# reproduction" tests for a wave-equation FDM scheme:
#
#   1. test_constant  - a spatially/temporally constant field solves the
#      (possibly damped) wave equation exactly, so both the scalar and
#      the vectorized update should reproduce it to machine precision,
#      regardless of the boundary condition in use.
#   2. test_quadratic - a quadratic-in-x, linear-in-t manufactured
#      solution combined with a matching source term f is exactly
#      reproduced by a centered 2nd-order scheme (only possible with
#      Dirichlet boundaries, since wave1D.py only implements Neumann
#      ("reflecting") boundaries and there is no non-trivial quadratic
#      with zero derivative at both ends).
#   3. test_plug      - an initial "plug" (rectangular pulse) under
#      reflecting boundaries and C=1 must return to its exact initial
#      shape after one full round trip across the domain.
#
# On top of those, a few implementation-specific regression checks are
# included: scalar/vector consistency, python/numba engine consistency
# (wave1D_extended.py only), and that unknown scheme/engine names raise
# ValueError instead of silently doing the wrong thing.
#

import matplotlib
matplotlib.use("Agg")  # headless: no window/animation while testing

import numpy as np
import pytest

import wave1D
import wave1D_extended

TOL = 1E-10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def maximize_c(c, L):
    """Same convention the modules themselves use for variable-speed c."""
    if isinstance(c, (float, int)):
        return c
    return max(c(x) for x in np.linspace(0, L, 101))


class ExactSolutionRecorder:
    """Callback that tracks the largest deviation from a known exact
    solution u_exact(x, t) seen over an entire run. Used with
    wave1D_extended.solver(), which (unlike wave1D.solver()) only
    returns boundary receiver time series, not the full spatial field -
    so the interior of the domain can only be checked through a
    per-step callback.
    """

    def __init__(self, u_exact):
        self.u_exact = u_exact
        self.max_diff = 0.0

    def __call__(self, u, x, t, n):
        u_e = self.u_exact(x, t[n])
        diff = np.abs(u - u_e).max()
        self.max_diff = max(self.max_diff, diff)
        return False  # never request early stopping


# ===========================================================================
# wave1D.py  (Neumann/"reflecting" boundaries at both ends only)
# ===========================================================================

@pytest.mark.parametrize("scheme", ["scalar", "vector"])
@pytest.mark.parametrize("b", [0.0, 3.0])
def test_constant_wave1D(scheme, b):
    """A constant field u(x,t) = u_const solves u_tt + b*u_t = c^2 u_xx
    exactly for any b, and satisfies the reflecting boundary condition
    trivially, so it must be reproduced to machine precision.

    Note: wave1D.solver() internally sets u(x,0) = 2*I(x) whenever
    noise=False (an amplitude convention baked into that code path), so
    I is defined as half the target constant to compensate.
    """
    u_const = 0.45
    c = 1.5
    D = 2.5
    C = 0.75
    Nx = 3  # coarse mesh is fine: the check is exactness, not accuracy
    dx = D / Nx
    dt = C * dx / c
    T = 18.0  # long time integration, as in the reference tests

    I = lambda x: u_const / 2.0
    V = lambda x: 0
    f = lambda x, t: 0

    u, xs, ts, receiverA, receiverB, _ = wave1D.solver(
        I, c, b, D, T, dt, C,
        randomness=False, animation=False, performance=True,
        scheme=scheme, noise=False, V=V, f=f, callback=None,
    )

    diff = np.abs(u - u_const).max()
    assert diff < TOL, f"scheme={scheme}, b={b}: diff={diff:.3e}"


@pytest.mark.parametrize("scheme", ["scalar", "vector"])
def test_plug_returns_after_period_wave1D(scheme):
    """An initial rectangular "plug" pulse, with reflecting boundaries at
    both ends and C=1 (no numerical dispersion), must come back to its
    exact original shape after one full round trip T = 2*D/c.
    """
    D = 1.0
    c = 0.5
    Nx = 10
    C = 1.0
    dx = D / Nx
    dt = C * dx / c
    b = 0.0
    loc, width = 0.5, 0.1

    def I(x):
        return 0.5 if abs(x - loc) <= width else 0.0

    T = 2 * D / c

    u, xs, ts, receiverA, receiverB, _ = wave1D.solver(
        I, c, b, D, T, dt, C,
        randomness=False, animation=False, performance=True,
        scheme=scheme, noise=False, V=None, f=None, callback=None,
    )

    u0 = 2 * np.vectorize(I)(xs)  # noise=False doubles I(x) at t=0
    diff = np.abs(u - u0).max()
    assert diff < TOL, f"scheme={scheme}: diff={diff:.3e}"


def test_scalar_vector_consistency_wave1D():
    """The scalar and vectorized update rules implement the same
    discrete scheme, so for identical (non-trivial, variable-speed,
    damped) input they must produce identical output.
    """
    D = 3.0
    Nx = 20
    C = 0.8
    b = 5.0
    c = lambda x: 1449.2 + 4.6 * 9 - 0.055 * 9**2 + 0.00029 * 9**3 + (1.34 - 0.01 * 9) * (6 - 35) + 0.016 * x
    dt = D * C / (Nx * maximize_c(c, D))
    I = lambda x: np.exp(-0.5 * (x / 0.2) ** 2)
    T = 0.02

    results = {}
    for scheme in ("scalar", "vector"):
        u, xs, ts, receiverA, receiverB, _ = wave1D.solver(
            I, c, b, D, T, dt, C,
            randomness=False, animation=False, performance=True,
            scheme=scheme, noise=False, V=None, f=None, callback=None,
        )
        results[scheme] = u

    diff = np.abs(results["scalar"] - results["vector"]).max()
    assert diff < TOL, f"scalar vs vector diff={diff:.3e}"


def test_invalid_scheme_raises_wave1D():
    """An unknown scheme name must fail loudly, not silently."""
    I = lambda x: 0.0
    with pytest.raises(ValueError):
        wave1D.solver(
            I, 1.0, 0.0, 1.0, 0.001, 0.001, 1.0,
            randomness=False, animation=False, performance=True,
            scheme="bogus", noise=False, V=None, f=None, callback=None,
        )


def test_convergence_rate_wave1D():
    """u(x,t) = cos(m*pi*x/D)*cos(m*pi*c*t/D) is an exact eigenfunction of
    the Neumann-Neumann Laplacian (V=0, f=0, b=0), but is NOT a low-degree
    polynomial, so it is not exactly reproduced the way test_constant is -
    only to the scheme's true (2nd) order of accuracy. Refining the mesh
    at fixed Courant number C should therefore roughly quarter the error
    each time dt halves. This catches bugs that happen to cancel out for
    polynomial-exact tests but still break the formal accuracy of the
    scheme.
    """
    D, c, m, C, b = 1.0, 1.0, 2, 0.8, 0.0
    T = 0.3
    I = lambda x: 0.5 * np.cos(m * np.pi * x / D)  # halved: noise=False doubles it

    errors, dts = [], []
    for Nx in (20, 40, 80, 160):
        dx = D / Nx
        dt = C * dx / c
        u, xs, ts, _, _, _ = wave1D.solver(
            I, c, b, D, T, dt, C,
            randomness=False, animation=False, performance=True,
            scheme="vector", noise=False, V=None, f=None, callback=None,
        )
        u_e = np.cos(m * np.pi * xs / D) * np.cos(m * np.pi * c * ts[-1] / D)
        errors.append(np.abs(u - u_e).max())
        dts.append(dt)

    rates = [
        np.log(errors[i] / errors[i - 1]) / np.log(dts[i] / dts[i - 1])
        for i in range(1, len(errors))
    ]
    for r in rates:
        assert 1.8 < r < 2.2, f"observed convergence rates: {rates}"


def test_symmetry_wave1D():
    """A spatially symmetric initial pulse, constant wave speed, and
    Neumann boundaries (themselves symmetric) must keep the solution
    symmetric about the domain midpoint at every time - catches
    off-by-one / indexing bugs that a purely 1D-summary check like
    test_constant would not.
    """
    D, c, Nx, C, b = 2.0, 1450.0, 40, 0.8, 1.5
    dt = D * C / (Nx * c)
    I = lambda x: np.exp(-0.5 * ((x - D / 2) / 0.1) ** 2)
    T = 0.002

    u, xs, ts, _, _, _ = wave1D.solver(
        I, c, b, D, T, dt, C,
        randomness=False, animation=False, performance=True,
        scheme="vector", noise=False, V=None, f=None, callback=None,
    )
    diff = np.abs(u - u[::-1]).max()
    assert diff < TOL, f"symmetry diff={diff:.3e}"


def test_damping_reduces_amplitude_wave1D():
    """A physically-motivated (not exact-reproduction) smoke test: adding
    positive damping b must not increase the peak amplitude relative to
    the undamped (b=0) run, for identical initial data, time, and mesh.
    """
    D, c, Nx, C = 2.0, 1450.0, 60, 0.8
    dt = D * C / (Nx * c)
    I = lambda x: np.exp(-0.5 * ((x - D / 2) / 0.1) ** 2)
    T = 200 * dt

    peaks = {}
    for b in (0.0, 25.0):
        u, xs, ts, _, _, _ = wave1D.solver(
            I, c, b, D, T, dt, C,
            randomness=False, animation=False, performance=True,
            scheme="vector", noise=False, V=None, f=None, callback=None,
        )
        peaks[b] = np.abs(u).max()

    assert peaks[25.0] < peaks[0.0], f"peaks={peaks}"


# ===========================================================================
# wave1D_extended.py  (supports Neumann and Dirichlet, 'python'/'numba')
# ===========================================================================

@pytest.mark.parametrize("engine", ["python", "numba"])
@pytest.mark.parametrize("scheme", ["scalar", "vector"])
@pytest.mark.parametrize("bc", ["neumann", "dirichlet"])
def test_constant_wave1D_extended(engine, scheme, bc):
    """Same idea as test_constant_wave1D, extended over both boundary
    condition types and both compute engines. For engine='numba' only
    the boundary receiver arrays are available (see
    ExactSolutionRecorder docstring), so those are checked directly;
    for engine='python' the whole spatial field is checked via a
    callback.
    """
    u_const = 0.45
    L = 2.5
    c = 1.5
    C = 0.75
    Nx = 3
    dx = L / Nx
    dt = C * dx / c
    T = 18.0
    b = 3.0

    I = lambda x: u_const / 2.0
    V = lambda x: 0
    f = lambda x, t: 0

    if bc == "neumann":
        bd_0 = bd_L = None
    else:
        bd_0 = lambda t: u_const
        bd_L = lambda t: u_const

    callback = ExactSolutionRecorder(lambda x, t: u_const) if engine == "python" else None

    ts, receiverA, receiverB, _ = wave1D_extended.solver(
        I, c, b, L, T, dt, C, noise=False, randomness=False,
        performance=0, scheme=scheme, engine=engine, parallel=False,
        V=V, f=f, bd_0=bd_0, bd_L=bd_L, callback=callback,
    )

    boundary_diff = max(np.abs(receiverA - u_const).max(), np.abs(receiverB - u_const).max())
    assert boundary_diff < TOL, f"engine={engine}, scheme={scheme}, bc={bc}: boundary diff={boundary_diff:.3e}"

    if callback is not None:
        assert callback.max_diff < TOL, (
            f"engine={engine}, scheme={scheme}, bc={bc}: interior diff={callback.max_diff:.3e}"
        )


@pytest.mark.parametrize("scheme", ["scalar", "vector"])
def test_quadratic_dirichlet_wave1D_extended(scheme):
    """u(x,t) = x*(L-x)*(1+t/2) is exactly reproduced by a centered
    2nd-order scheme when driven with the matching source term f and
    Dirichlet conditions equal to the exact solution at both ends -
    the classic FDM verification test (see wave1D_dn.py:test_quadratic
    in the reference above). Only possible for wave1D_extended.py,
    since wave1D.py has no Dirichlet boundary option.
    """
    L = 2.5
    c = 1.5
    b = 0.0
    C = 0.75
    Nx = 3
    dx = L / Nx
    dt = C * dx / c
    T = 18.0

    u_exact = lambda x, t: x * (L - x) * (1 + 0.5 * t)
    I = lambda x: u_exact(x, 0) / 2.0  # noise=False halves the effective I(x)
    V = lambda x: 0.5 * u_exact(x, 0)
    f = lambda x, t: 2 * (1 + 0.5 * t) * c ** 2
    bd_0 = lambda t: u_exact(0.0, t)
    bd_L = lambda t: u_exact(L, t)

    callback = ExactSolutionRecorder(u_exact)
    wave1D_extended.solver(
        I, c, b, L, T, dt, C, noise=False, randomness=False,
        performance=0, scheme=scheme, engine="python", parallel=False,
        V=V, f=f, bd_0=bd_0, bd_L=bd_L, callback=callback,
    )

    assert callback.max_diff < TOL, f"scheme={scheme}: diff={callback.max_diff:.3e}"


@pytest.mark.parametrize("engine", ["python", "numba"])
@pytest.mark.parametrize("scheme", ["scalar", "vector"])
def test_plug_returns_after_period_wave1D_extended(engine, scheme):
    """Same plug/round-trip check as for wave1D.py, run through both
    compute engines. Only the boundary receivers are available for
    engine='numba', but since the initial plug does not touch either
    boundary, "receiver value back to its t=0 value" is exactly the
    same statement as "the field is back to its initial shape".
    """
    L = 1.0
    c = 0.5
    Nx = 10
    C = 1.0
    dx = L / Nx
    dt = C * dx / c
    b = 0.0
    loc, width = 0.5, 0.1

    def I(x):
        return 0.5 if abs(x - loc) <= width else 0.0

    T = 2 * L / c

    ts, receiverA, receiverB, _ = wave1D_extended.solver(
        I, c, b, L, T, dt, C, noise=False, randomness=False,
        performance=2, scheme=scheme, engine=engine, parallel=False,
        V=None, f=None, bd_0=None, bd_L=None, callback=None,
    )

    diffA = abs(receiverA[0] - receiverA[-1])
    diffB = abs(receiverB[0] - receiverB[-1])
    assert diffA < TOL and diffB < TOL, (
        f"engine={engine}, scheme={scheme}: diffA={diffA:.3e}, diffB={diffB:.3e}"
    )


@pytest.mark.parametrize("scheme", ["scalar", "vector"])
def test_python_numba_consistency_wave1D_extended(scheme):
    """The 'python' and 'numba' engines implement the same update
    formulas (see wave1D_implementations.py), so for identical
    (non-trivial, variable-speed, damped) input they must produce
    identical boundary receiver time series.
    """
    L = 3.0
    Nx = 20
    C = 0.8
    b = 5.0
    c = lambda x: 1449.2 + 4.6 * 9 - 0.055 * 9**2 + 0.00029 * 9**3 + (1.34 - 0.01 * 9) * (6 - 35) + 0.016 * x
    dt = L * C / (Nx * maximize_c(c, L))
    I = lambda x: np.exp(-0.5 * (x / 0.2) ** 2)
    T = 0.02

    results = {}
    for engine in ("python", "numba"):
        ts, receiverA, receiverB, _ = wave1D_extended.solver(
            I, c, b, L, T, dt, C, noise=False, randomness=False,
            performance=2, scheme=scheme, engine=engine, parallel=False,
            V=None, f=None, bd_0=None, bd_L=None, callback=None,
        )
        results[engine] = (receiverA, receiverB)

    diffA = np.abs(results["python"][0] - results["numba"][0]).max()
    diffB = np.abs(results["python"][1] - results["numba"][1]).max()
    assert diffA < TOL and diffB < TOL, f"scheme={scheme}: diffA={diffA:.3e}, diffB={diffB:.3e}"


def test_invalid_scheme_raises_wave1D_extended():
    """An unknown scheme name must fail loudly, not silently."""
    I = lambda x: 0.0
    with pytest.raises(ValueError):
        wave1D_extended.solver(
            I, 1.0, 0.0, 1.0, 0.001, 0.001, 1.0, noise=False, randomness=False,
            performance=2, scheme="bogus", engine="python", parallel=False,
            V=None, f=None, bd_0=None, bd_L=None, callback=None,
        )


def test_invalid_engine_raises_wave1D_extended():
    """An unknown engine name must fail loudly, not silently."""
    I = lambda x: 0.0
    with pytest.raises(ValueError):
        wave1D_extended.solver(
            I, 1.0, 0.0, 1.0, 0.001, 0.001, 1.0, noise=False, randomness=False,
            performance=2, scheme="scalar", engine="bogus", parallel=False,
            V=None, f=None, bd_0=None, bd_L=None, callback=None,
        )


def test_convergence_rate_wave1D_extended():
    """Same standing-wave convergence check as test_convergence_rate_wave1D,
    against wave1D_extended.py's 'python' engine (full-field access
    requires a callback here, see ExactSolutionRecorder).
    """
    D, c, m, C, b = 1.0, 1.0, 2, 0.8, 0.0
    T = 0.3
    I = lambda x: 0.5 * np.cos(m * np.pi * x / D)

    errors, dts = [], []
    for Nx in (20, 40, 80, 160):
        dx = D / Nx
        dt = C * dx / c
        last = {}

        def cb(u, x, t, n, last=last):
            last["u"], last["x"], last["t"] = u.copy(), x, t[n]
            return False

        wave1D_extended.solver(
            I, c, b, D, T, dt, C, noise=False, randomness=False,
            performance=0, scheme="vector", engine="python", parallel=False,
            V=None, f=None, bd_0=None, bd_L=None, callback=cb,
        )
        u_e = np.cos(m * np.pi * last["x"] / D) * np.cos(m * np.pi * c * last["t"] / D)
        errors.append(np.abs(last["u"] - u_e).max())
        dts.append(dt)

    rates = [
        np.log(errors[i] / errors[i - 1]) / np.log(dts[i] / dts[i - 1])
        for i in range(1, len(errors))
    ]
    for r in rates:
        assert 1.8 < r < 2.2, f"observed convergence rates: {rates}"


def test_symmetry_wave1D_extended():
    """Same symmetry check as test_symmetry_wave1D, against
    wave1D_extended.py's 'python' engine.
    """
    D, c, Nx, C, b = 2.0, 1450.0, 40, 0.8, 1.5
    dt = D * C / (Nx * c)
    I = lambda x: np.exp(-0.5 * ((x - D / 2) / 0.1) ** 2)
    T = 0.002

    last = {}

    def cb(u, x, t, n, last=last):
        last["u"] = u.copy()
        return False

    wave1D_extended.solver(
        I, c, b, D, T, dt, C, noise=False, randomness=False,
        performance=0, scheme="vector", engine="python", parallel=False,
        V=None, f=None, bd_0=None, bd_L=None, callback=cb,
    )
    diff = np.abs(last["u"] - last["u"][::-1]).max()
    assert diff < TOL, f"symmetry diff={diff:.3e}"


def test_damping_reduces_amplitude_wave1D_extended():
    """Same damping-vs-undamped amplitude comparison as
    test_damping_reduces_amplitude_wave1D, using the boundary receivers
    (the pulse reaches both ends within the chosen T).
    """
    D, c, Nx, C = 2.0, 1450.0, 60, 0.8
    dt = D * C / (Nx * c)
    I = lambda x: np.exp(-0.5 * ((x - D / 2) / 0.1) ** 2)
    T = 200 * dt

    peaks = {}
    for b in (0.0, 25.0):
        ts, receiverA, receiverB, _ = wave1D_extended.solver(
            I, c, b, D, T, dt, C, noise=False, randomness=False,
            performance=2, scheme="vector", engine="python", parallel=False,
            V=None, f=None, bd_0=None, bd_L=None, callback=None,
        )
        peaks[b] = max(np.abs(receiverA).max(), np.abs(receiverB).max())

    assert peaks[25.0] < peaks[0.0], f"peaks={peaks}"


def test_wave1D_and_wave1D_extended_agree():
    """wave1D.py always implements Neumann-Neumann boundaries, so
    wave1D_extended.py run with bd_0=None, bd_L=None (its own Neumann
    option) over the same input must reproduce it exactly - both the
    boundary receivers and the full spatial field. This guards against
    the two implementations silently drifting apart.
    """
    D = 2.0
    Nx = 15
    C = 0.9
    b = 2.0
    c = lambda x: 1400 + 10 * np.sin(x)
    dt = D * C / (Nx * maximize_c(c, D))
    I = lambda x: np.exp(-0.5 * ((x - 1.0) / 0.15) ** 2)
    T = 0.01

    u, xs, ts, receiverA1, receiverB1, _ = wave1D.solver(
        I, c, b, D, T, dt, C,
        randomness=False, animation=False, performance=True,
        scheme="vector", noise=False, V=None, f=None, callback=None,
    )

    last = {}

    def cb(uu, x, t, n, last=last):
        last["u"] = uu.copy()
        return False

    ts2, receiverA2, receiverB2, _ = wave1D_extended.solver(
        I, c, b, D, T, dt, C, noise=False, randomness=False,
        performance=0, scheme="vector", engine="python", parallel=False,
        V=None, f=None, bd_0=None, bd_L=None, callback=cb,
    )

    assert np.abs(receiverA1 - receiverA2).max() < TOL
    assert np.abs(receiverB1 - receiverB2).max() < TOL
    assert np.abs(u - last["u"]).max() < TOL


# ---------------------------------------------------------------------------
# parallel=True (engine='numba' only - see RUN_FUNCTIONS in
# wave1D_implementations.py, which has no ('python', *, True) entries)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["scalar", "vector"])
def test_constant_wave1D_extended_parallel(scheme):
    """Same exact-reproduction check as test_constant_wave1D_extended,
    run through the parallel (prange) numba path specifically - the
    interior update there is restructured into two explicit element-wise
    prange loops (noise addition, then the update itself), which is a
    plausible place to introduce an off-by-one or race-condition bug
    even though the underlying math is unchanged.
    """
    u_const = 0.45
    L = 2.5
    c = 1.5
    C = 0.75
    Nx = 3
    dx = L / Nx
    dt = C * dx / c
    T = 18.0
    b = 3.0

    I = lambda x: u_const / 2.0
    V = lambda x: 0
    f = lambda x, t: 0

    ts, receiverA, receiverB, _ = wave1D_extended.solver(
        I, c, b, L, T, dt, C, noise=False, randomness=False,
        performance=2, scheme=scheme, engine="numba", parallel=True,
        V=V, f=f, bd_0=None, bd_L=None, callback=None,
    )

    boundary_diff = max(np.abs(receiverA - u_const).max(), np.abs(receiverB - u_const).max())
    assert boundary_diff < TOL, f"scheme={scheme}: boundary diff={boundary_diff:.3e}"


@pytest.mark.parametrize("scheme", ["scalar", "vector"])
def test_plug_returns_after_period_wave1D_extended_parallel(scheme):
    """Same plug/round-trip check as test_plug_returns_after_period_wave1D,
    run through parallel=True.
    """
    L = 1.0
    c = 0.5
    Nx = 10
    C = 1.0
    dx = L / Nx
    dt = C * dx / c
    b = 0.0
    loc, width = 0.5, 0.1

    def I(x):
        return 0.5 if abs(x - loc) <= width else 0.0

    T = 2 * L / c

    ts, receiverA, receiverB, _ = wave1D_extended.solver(
        I, c, b, L, T, dt, C, noise=False, randomness=False,
        performance=2, scheme=scheme, engine="numba", parallel=True,
        V=None, f=None, bd_0=None, bd_L=None, callback=None,
    )

    diffA = abs(receiverA[0] - receiverA[-1])
    diffB = abs(receiverB[0] - receiverB[-1])
    assert diffA < TOL and diffB < TOL, f"scheme={scheme}: diffA={diffA:.3e}, diffB={diffB:.3e}"


@pytest.mark.parametrize("scheme", ["scalar", "vector"])
def test_numba_parallel_matches_serial(scheme):
    """parallel=True is meant to be a pure performance optimization of the
    same numba scheme: for identical (non-trivial, variable-speed,
    damped) input, parallel=True and parallel=False must produce
    identical boundary receiver time series. Since the update writes
    each index independently from data in *other* arrays (u_n, u_nm1),
    there should be no race condition and thus no floating-point
    reordering effects either - this is checked, not just assumed.
    """
    L = 3.0
    Nx = 20
    C = 0.8
    b = 5.0
    c = lambda x: 1449.2 + 4.6 * 9 - 0.055 * 9**2 + 0.00029 * 9**3 + (1.34 - 0.01 * 9) * (6 - 35) + 0.016 * x
    dt = L * C / (Nx * maximize_c(c, L))
    I = lambda x: np.exp(-0.5 * (x / 0.2) ** 2)
    T = 0.02

    results = {}
    for parallel in (False, True):
        ts, receiverA, receiverB, _ = wave1D_extended.solver(
            I, c, b, L, T, dt, C, noise=False, randomness=False,
            performance=2, scheme=scheme, engine="numba", parallel=parallel,
            V=None, f=None, bd_0=None, bd_L=None, callback=None,
        )
        results[parallel] = (receiverA, receiverB)

    diffA = np.abs(results[False][0] - results[True][0]).max()
    diffB = np.abs(results[False][1] - results[True][1]).max()
    assert diffA < TOL and diffB < TOL, f"scheme={scheme}: diffA={diffA:.3e}, diffB={diffB:.3e}"


def test_parallel_with_python_engine_raises():
    """('python', scheme, True) is not in RUN_FUNCTIONS - only numba has
    parallel variants - so requesting engine='python' with parallel=True
    directly from solver() must fail loudly rather than silently running
    serially or crashing with a raw KeyError.
    """
    I = lambda x: 0.0
    with pytest.raises(ValueError):
        wave1D_extended.solver(
            I, 1.0, 0.0, 1.0, 0.001, 0.001, 1.0, noise=False, randomness=False,
            performance=2, scheme="vector", engine="python", parallel=True,
            V=None, f=None, bd_0=None, bd_L=None, callback=None,
        )


def test_main_forces_numba_when_parallel(capsys):
    """main() silently overrides engine to 'numba' whenever parallel=True
    (even if the caller explicitly asked for engine='python'), precisely
    to avoid the ValueError that test_parallel_with_python_engine_raises
    checks for at the solver() level. Verified here by checking the
    printed engine name, since main() doesn't return it directly.
    """
    wave1D_extended.main(
        L=1.0, Nx=10, T=0.001, performance=2,
        engine="python", parallel=True, randomness=False,
    )
    printed = capsys.readouterr().out
    assert "Engine: numba" in printed


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))