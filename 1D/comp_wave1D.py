# Code adapted from
# Finite difference methods for wave equations
# by Langtangen and Linge
# https://github.com/hplgit/fdm-book
# Licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import tracemalloc

from integrators_1D import RUN_FUNCTIONS

def gen_noise(Nt, I, xs, randomness):
    # define list of random numbers to define when to generate new drop
    if randomness == True and Nt <= 5000:
        try:
            url = "https://www.random.org/decimal-fractions/?num=" + str(2*Nt)+ "&dec=2&col=1&format=plain&rnd=new"
            randomrequest = requests.get(url)
            str_list = list(randomrequest.text)
            temp_list = []
            helper = ''
            for i in str_list:
                if i == '\n':
                    temp_list.append(float(helper))
                    helper = ''
                else: 
                    helper = helper + str(i)
            random_list = (np.array(temp_list[:int(Nt)]) - np.array(temp_list[int(Nt):]))/10
        except:
            # Failsafe
            print('Random.org allowance is negative')
            print('Using Numpy pseudo-random number generation')
            random_list = (np.random.rand(Nt)-np.random.rand(Nt))/10
    else:
        # Uniform distribution from [-0.1,0.1]
        random_list = (np.random.rand(Nt)-np.random.rand(Nt))/10

    # initial condition
    u_n = random_list[0]*np.vectorize(I)(xs).astype('float64')

    return u_n, random_list


def maximize_c(c,L):
    if isinstance(c, (float, int)):
        return c
    elif callable(c):
        return max([c(x) for x in np.linspace(0, L, 101)])

def solver(I, c,b, L, T, dt, C, noise, randomness, performance, scheme, engine, V, f, bd_0, bd_L, callback):

    s = time.process_time()

    Nt = int(round(T / dt))
    ts = np.linspace(0, Nt * dt, Nt + 1)
    receiverA = np.zeros(Nt + 1)
    receiverB = np.zeros(Nt + 1)

    c_max = maximize_c(c,L)

    dx = dt * c_max / C
    Nx = int(round(L / dx))
    xs = np.linspace(0, L, Nx + 1)

    dt = ts[1] - ts[0]
    dx = xs[1] - xs[0]

    courant = c_max*dt/dx

    if courant <= 1:
        print('CLF criterion met')
        print('Solution stable')
    else:
        print('Solution unstable')

    print(f'Courant number: {courant}')

    if isinstance(c, (float, int)):
        q = np.full(Nx + 1, c * c, dtype='float64')
    elif callable(c):
        cs = np.vectorize(c)(xs).astype('float64')
        q = cs * cs

    dt2 = dt * dt
    C2 = dt2 / (dx * dx)

    # allow ease of setting initial conditions or forcing term
    if f is None or f == 0:
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0

    # bd0_dirichlet/bdL_dirichlet must be recorded BEFORE bd_0==0 / bd_L==0
    # get turned into a zero-lambda below, otherwise a "Dirichlet, always 0"
    # boundary would be misclassified as Neumann.
    bd0_dirichlet = bd_0 is not None
    bdL_dirichlet = bd_L is not None
    if bd_0 == 0:
        bd_0 = lambda t: 0
    if bd_L == 0:
        bd_L = lambda t: 0
 
    # Pre-evaluate everything that depends on Python callables (lambdas /
    # closures), since numba cannot call them from inside a jitted function.
    f_vals = np.zeros((Nt, Nx + 1))
    for n in range(Nt):
        f_vals[n, :] = f(xs, ts[n])
 
    bd0_vals = np.array([bd_0(t) for t in ts]) if bd0_dirichlet else np.zeros(Nt + 1)
    bdL_vals = np.array([bd_L(t) for t in ts]) if bdL_dirichlet else np.zeros(Nt + 1)

    # initialize u arrays
    u = np.zeros(Nx + 1) # u array at new timestep
    u_n = np.zeros(Nx + 1) # u array at current timestep
    u_nm1 = np.zeros(Nx + 1) # u array at previous timestep

    # Pre-evaluate the noise perturbation for every timestep too, for the
    # same reason as f_vals: this way both engines just add a plain array at
    # each step instead of calling I(x) as a closure. 
    noise_vals = np.zeros((Nt, Nx + 1))
    if noise == True:
        u_n, random_list = gen_noise(Nt, I, xs, randomness)
        I_vals = np.vectorize(I)(xs).astype('float64')
        for n in range(1, Nt):
            if n == round(Nt / 10):
                noise_vals[n] = 2 * I_vals
            else:
                noise_vals[n] = random_list[n] * I_vals
    else:
        u_n = 2 * np.vectorize(I)(xs).astype('float64')

    receiverA[0] = u_n[0]
    receiverB[0] = u_n[-1]

    if performance < 1:
        if callback is not None:
            callback(u_n, xs, ts, 0)

    if scheme == 'vector':
        u[1:-1] = u_n[1:-1] - (0.5 * b * dt - 1) * V(xs[1:-1]) * dt + 0.25 * C2 * (
            (q[1:-1] + q[2:]) * (u_n[2:] - u_n[1:-1]) - (q[1:-1] + q[:-2]) * (u_n[1:-1] - u_n[:-2])
        ) + 0.5 * dt2 * f(xs[1:-1], 0)
    elif scheme == 'scalar':
        for i in range(1, Nx):
            u[i] = u_n[i] - (0.5 * b * dt - 1) * V(xs[i]) * dt + 0.25 * C2 * (
                (q[i] + q[i + 1]) * (u_n[i + 1] - u_n[i]) - (q[i] + q[i - 1]) * (u_n[i] - u_n[i - 1])
            ) + 0.5 * dt2 * f(xs[i], 0)
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}. Allowed values: 'vector', 'scalar'.")
    
    if bd0_dirichlet:
        u[0] = bd_0(ts[1])
    else:
        u[0] = u_n[0] - (0.5 * b * dt - 1) * V(xs[0]) * dt + C2 * q[0] * (u_n[1] - u_n[0]) + 0.5 * dt2 * f(xs[0], 0)
    
    if bdL_dirichlet:
        u[-1] = bd_L(ts[1])
    else:
        u[-1] = u_n[-1] - (0.5 * b * dt - 1) * V(xs[-1]) * dt + C2 * q[-1] * (u_n[-2] - u_n[-1]) + dt2 * f(xs[-1], 0)

    if performance < 1:
        if callback is not None:
            callback(u, xs, ts, 1)
 
    u, u_n, u_nm1 = u_nm1, u, u_n
    receiverA[1] = u_n[0]
    receiverB[1] = u_n[-1]   

    try:
        run_fn = RUN_FUNCTIONS[(engine, scheme)]
    except KeyError:
        raise ValueError(
            f"Unknown combination engine={engine!r}, scheme={scheme!r}. "
            f"Allowed engines: 'python', 'numba'. Allowed schemes: 'vector', 'scalar'."
        )

    # Per-step callback/live-plotting is only possible with engine='python':
    # a jitted numba function can't call an arbitrary Python callable, so
    # the numba run functions never receive performance/callback/xs/ts.
    if engine == 'python':
        u, u_n, u_nm1 = run_fn(u, u_n, u_nm1, b, dt, dt2, Nt, Nx, C2, q,
                                        receiverA, receiverB, f_vals, noise_vals,
                                        bd0_dirichlet, bd0_vals, bdL_dirichlet, bdL_vals,
                                        performance=performance, callback=callback, xs=xs, ts=ts)
    else:
        u, u_n, u_nm1 = run_fn(u, u_n, u_nm1, b, dt, dt2, Nt, Nx, C2, q,
                                receiverA, receiverB, f_vals, noise_vals,
                                bd0_dirichlet, bd0_vals, bdL_dirichlet, bdL_vals)
 
    u = u_n
    e = time.process_time()
    return ts, receiverA, receiverB, e-s


class PlotVariableSpeed():
    def __init__(
            self,
            medium,
            umin = -1, umax = 1,
            title='',
            ):
        self.medium = medium
        self.yaxis = [umin, umax]
        self.title = title
        self.plt = plt

    def __call__(self, u, x, t, n):
        umin, umax = self.yaxis

        title = f"Nx={x.size - 1}"
        if self.title:
            title = self.title + ' ' + title

        self.plt.clf()
        self.plt.plot(x, u, 'r-')
        self.plt.axis([x[0], x[-1], umin, umax])
        self.plt.xlabel('x')
        self.plt.ylabel('Paine-ero')
        self.plt.title(title)
        self.plt.legend([f"y={t[n]:.3f}"], loc='lower left')
        self.plt.grid()
        self.plt.pause(0.001)

def main(
        L = 37,                     # Depth (metres)
        C = 1,                      # CFL condition
        b = 10,                     # Dampening factor
        Nx = 500,                   # Spatial mesh
        T = 0.1,                    # Animation time
        pulse_type = 'gaussian',
        medium = [0, 0],
        sigma = 0.05,
        temp1 = 9,                  # Temperature (Celsius)
        sal1 = 6,                   # Salinity (ppt)
        noise = True,               # Noise generation
        randomness = False,         # Pseudo-random on default
        performance = 0,            # >= 0 shows everything
                                    # 1 shows performance metrics and pressure difference at sea bottom and sea surface
                                    # <= 2 shows only performance metrics 
        scheme = 'vector',          # 'vector' uses vector calculation
                                    # 'scalar' uses scalar calculation
        engine = 'python'           # 'python' runs through python
                                    # 'numba'  runs through numba
        ):   

    tracemalloc.start()

    if pulse_type == 'gaussian':
        I = lambda x: np.exp(-0.5*(x / sigma)**2)
    elif pulse_type == 'cosinehat':
        I = lambda x: 0.5 * (1 + np.cos(np.pi * x / (2 * sigma))) if -2 * sigma <= x <= 2 * sigma else 0
    elif pulse_type == 'half-cosinehat':
        I = lambda x: np.cos(np.pi * x / (4 * sigma)) if -2 * sigma <= x <= 2 * sigma else 0
    else:
        raise ValueError(f"Unknown pulse type {pulse_type!r}. " 
                         f"Allowed values: 'gaussian', 'cosinehat', 'half-cosinehat'.")
    
    c = lambda x: 1449.2 + 4.6*temp1 - 0.055*temp1**2 + 0.00029*temp1**3 + (1.34 - 0.01*temp1)*(sal1-35) + 0.016*x

    umin = -1.5
    umax = 1.5 

    plotter = PlotVariableSpeed(medium, umin=umin, umax=umax)

    c_max = maximize_c(c,L)

    dt = L / (Nx * c_max)

    ts, receiverA, receiverB, timer = solver(I, c,b, L, T, dt, C, noise, randomness, performance, scheme, engine, V=None, f = None, bd_0=None, bd_L=None, callback=plotter)

    _, peak = tracemalloc.get_traced_memory()
    print(f'Nx: {Nx}')
    print(f'Engine: {engine}')
    print(f'Scheme: {scheme}')
    print(f'CPU time: {timer} seconds')
    print(f'Peak memory usage: {peak} bytes')
    tracemalloc.stop()

    if performance < 2:
        _, axs = plt.subplots(2,1,layout='constrained')
        axs[0].plot(ts,receiverA)
        axs[0].set_xlabel('y')
        axs[0].set_ylabel('Pressure difference at surface')
        axs[0].grid()

        axs[1].plot(ts,receiverB)
        axs[1].set_xlabel('y')
        axs[1].set_ylabel('Pressure difference at bottom')
        axs[1].grid()
        plt.show()
    return Nx, timer, peak


if __name__ == '__main__':
    main()
    