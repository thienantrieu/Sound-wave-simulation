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

def solver(I, c, b, D, T, dt, C, randomness, animation, performance, scheme, noise, V, f, callback):
    s = time.process_time()

    Nt = int(round(T / dt))
    ts = np.linspace(0, Nt * dt, Nt + 1)
    receiverA = []
    receiverB = []

    c_max = maximize_c(c,D)

    dx = dt * c_max/C
    Nx = int(round(D / dx))
    xs = np.linspace(0, D, Nx + 1)

    dt = ts[1] - ts[0]
    dx = xs[1] - xs[0]

    courant = (c_max*dt)/dx

    if courant <= 1:
        print('CFL criterion met')
        print('Solution stable')
    else:
        print('Solution unstable')

    print(f'Courant-number: {courant}')

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

    # initialize u arrays
    u = np.zeros(Nx + 1) # u array at new timestep
    u_n = np.zeros(Nx + 1) # u array at current timestep
    u_nm1 = np.zeros(Nx + 1) # u array at previous timestep

    # define list of random numbers to define when to generate new drop
    if noise == True:
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
    if noise == True:
        u_n = random_list[0]*np.vectorize(I)(xs).astype('float64')
    else:
        u_n = 2*np.vectorize(I)(xs).astype('float64')

    receiverA.append(u_n[0])
    receiverB.append(u_n[-1])

    if animation == True and performance == False:
        if callback is not None:
            callback(u_n, xs, ts, 0)

    # special first timestep formula
    if scheme == 'vector':
        u[1:-1] = u_n[1:-1] - (0.5*b*dt-1)*V(xs[1:-1])*dt + 0.25*C2*((q[1:-1] + q[2:]) * (u_n[2:] - u_n[1:-1]) - (q[1:-1] + q[:-2]) * (u_n[1:-1] - u_n[:-2])) + 0.5*dt2 * f(xs[1:-1], 0)
    elif scheme == 'scalar':
        for i in range(1, Nx):
            u[i] = u_n[i] - (0.5*b*dt-1)*V(xs[i])*dt + 0.25*C2*((q[i] + q[i+1]) * (u_n[i+1] - u_n[i]) - (q[i] + q[i-1]) * (u_n[i] - u_n[i-1])) + 0.5*dt2 * f(xs[i], 0)
    else: 
        raise ValueError(
        f"Unknown scheme={scheme!r}. "
        f"Allowed schemes: 'vector', 'scalar'."
    )
    
    u[0] = u_n[0] - (0.5*b*dt-1)*V(xs[0])*dt + C2 * q[0] * (u_n[1] - u_n[0]) + 0.5*dt2*f(xs[0], 0)
    u[-1] = u_n[-1] - (0.5*b*dt-1)*V(xs[-1])*dt + C2 * q[-1] * (u_n[-2] - u_n[-1]) + 0.5*dt2*f(xs[-1], 0)

    if animation == True and performance == False:
        if callback is not None:
            callback(u, xs, ts, 1)

    # reference swap
    u, u_n, u_nm1 = u_nm1, u, u_n
    receiverA.append(u[0])
    receiverB.append(u[-1])

    # compute each timestep until the end of the simulation
    for n in range(1, Nt):
        if noise == True:
            if n == round(Nt/10):
                u2 = 2*np.vectorize(I)(xs).astype('float64')
            else:
                u2 = random_list[n]*np.vectorize(I)(xs).astype('float64')
        else:
            u2 = 0
        u = u + u2
        u_n = u_n + u2
        u_nm1 = u_nm1 + u2

        # Dampening u[1:-1]
        if scheme == 'vector':
            u[1:-1] = (1/(1+0.5*b*dt))*((0.5*b*dt-1)*u_nm1[1:-1] + 2 * u_n[1:-1] + 0.5*C2 * ((q[1:-1] + q[2:]) * (u_n[2:] - u_n[1:-1]) - (q[1:-1] + q[:-2]) * (u_n[1:-1] - u_n[:-2])) + dt2 * f(xs[1:-1], ts[n]))
        elif scheme == 'scalar':
            for i in range(1, Nx):
                u[i] = (1/(1+0.5*b*dt))*((0.5*b*dt-1)*u_nm1[i] + 2 * u_n[i] + 0.5*C2 * ((q[i] + q[i+1]) * (u_n[i+1] - u_n[i]) - (q[i] + q[i-1]) * (u_n[i] - u_n[i-1])) + dt2 * f(xs[i], ts[n]))
        else: 
            raise ValueError(
                f"Unknown scheme={scheme!r}. "
                f"Allowed schemes: 'vector', 'scalar'."
            )

        u[0] = (1/(1+0.5*b*dt))*((0.5*b*dt-1)*u_nm1[0] + 2 * u_n[0] + 2*C2 * q[0] * (u_n[1] - u_n[0]) + dt2 * f(xs[0], ts[n]))
        u[-1] = (1/(1+0.5*b*dt))*((0.5*b*dt-1)*u_nm1[-1] + 2 * u_n[-1] + 2*C2 * q[-1] * (u_n[-2] - u_n[-1]) + dt2 * f(xs[-1], ts[n]))

        receiverA.append(u[0])
        receiverB.append(u[-1])

        if animation == True and performance == False:        
            if callback is not None:
                if callback(u, xs, ts, n+1):
                    break

        u, u_n, u_nm1 = u_nm1, u, u_n

    u = u_n
    e = time.process_time()
    return u, xs, ts, receiverA, receiverB, e-s


class PlotVariableSpeed():
    def __init__(
            self,
            umin = -1, 
            umax = 1,
            ):
        self.yaxis = [umin, umax]
        self.plt = plt

    def __call__(self, u, x, t, n):
        umin, umax = self.yaxis

        self.plt.clf()
        self.plt.plot(x, u, 'r-')
        self.plt.axis([x[0], x[-1], umin, umax])
        self.plt.xlabel('$x$')
        self.plt.ylabel('Paine-ero')
        self.plt.legend([f"y={t[n]:.3f}"], loc='lower left')
        self.plt.grid()
        self.plt.pause(0.001)

def maximize_c(c,D):
    if isinstance(c, (float, int)):
        return c
    elif callable(c):
        return max([c(x) for x in np.linspace(0, D, 101)])

def main(
        D = 37,                 # Depth (metres)
        C = 1,                  # CFL condition
        b = 10,                 # dampening factor
        Nx = 500,               # Spatial mesh
        T = 0.1,                # Animation time
        sigma = 0.05,
        temp1 = 9,              # Temperature (Celsius)
        sal1 = 6,               # Salinity (ppt)
        randomness = False,     # Pseudo-random on default
        animation = True,
        performance = False,
        scheme = 'vector',
        noise = True
        ):   

    tracemalloc.start()

    I = lambda x: np.exp(-0.5*(x / sigma)**2)
    
    c = lambda x: 1449.2 + 4.6*temp1 - 0.055*temp1**2 + 0.00029*temp1**3 + (1.34 - 0.01*temp1)*(sal1-35) + 0.016*x

    umin = -1.5
    umax = 1.5 

    plotter = PlotVariableSpeed(umin=umin, umax=umax)

    c_max= maximize_c(c,D)
    dt = D*C/ (Nx * c_max)

    _, _, ts, receiverA, receiverB, timer = solver(I, c,b, D, T, dt, C, randomness, animation, performance, scheme,noise, V=None, f=None, callback=plotter)

    _, peak = tracemalloc.get_traced_memory()
    print(f'CPU time: {timer} seconds')
    print(f'Peak memory usage: {peak} bytes')
    tracemalloc.stop()

    if performance == False:
        _, axs = plt.subplots(2,1,layout='constrained')
        axs[0].plot(ts,receiverA)
        axs[0].set_xlabel('y')
        axs[0].set_ylabel('Paine-ero')
        axs[0].grid()

        axs[1].plot(ts,receiverB)
        axs[1].set_xlabel('y')
        axs[1].set_ylabel('Paine-ero')
        axs[1].grid()
        plt.show()

    return Nx, timer, peak

if __name__ == '__main__':
    main()