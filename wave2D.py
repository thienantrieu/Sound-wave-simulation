import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import requests
import time
import tracemalloc

def scheme_ij(u, u_1, u_2, k_1, k_2, k_3, k_4,
              f, dt2, Cx2, Cy2, x, y, t_1,
              i, j, im1, ip1, jm1, jp1):
    
    u_ij = - k_2*u_2[i,j] + k_1*2*u_1[i,j]
    u_xx = k_3*Cx2*(u_1[im1,j] - 2*u_1[i,j] + u_1[ip1,j])
    u_yy = k_3*Cy2*(u_1[i,jm1] - 2*u_1[i,j] + u_1[i,jp1])
    f_term = k_4*dt2*f(x, y, t_1)
    return u_ij + u_xx + u_yy + f_term

def scheme_scalar_mesh(u, u_1, u_2, k_1, k_2, k_3, k_4,
                       f, dt2, Cx2, Cy2, x, y, t_1, Nx, Ny,
                       bc):
    Ix = range(0, u.shape[0])
    Iy = range(0, u.shape[1])

    # Interior points
    for i in Ix[1:-1]:
        for j in Iy[1:-1]:
            im1 = i-1; ip1 = i+1; jm1 = j-1; jp1 = j+1
            u[i,j] = scheme_ij(
                u, u_1, u_2, k_1, k_2, k_3, k_4,
                f, dt2, Cx2, Cy2, x[i], y[j], t_1,
                i, j, im1, ip1, jm1, jp1)
    # Boundary points
    i = Ix[0]
    ip1 = i+1
    im1 = ip1
    if bc['W'] is None:
        for j in Iy[1:-1]:
            jm1 = j-1; jp1 = j+1
            u[i,j] = scheme_ij(
                u, u_1, u_2, k_1, k_2, k_3, k_4,
                f, dt2, Cx2, Cy2, x[i], y[j], t_1,
                i, j, im1, ip1, jm1, jp1)
    else:
        for j in Iy[1:-1]:
            u[i,j] = bc['W'](x[i], y[j])
    i = Ix[-1]
    im1 = i-1
    ip1 = im1
    if bc['E'] is None:
        for j in Iy[1:-1]:
            jm1 = j-1; jp1 = j+1
            u[i,j] = scheme_ij(
                u, u_1, u_2, k_1, k_2, k_3, k_4,
                f, dt2, Cx2, Cy2, x[i], y[j], t_1,
                i, j, im1, ip1, jm1, jp1)
    else:
        for j in Iy[1:-1]:
            u[i,j] = bc['E'](x[i], y[j])
    j = Iy[0]
    jp1 = j+1
    jm1 = jp1
    if bc['S'] is None:
        for i in Ix[1:-1]:
            im1 = i-1; ip1 = i+1
            u[i,j] = scheme_ij(
                u, u_1, u_2, k_1, k_2, k_3, k_4,
                f, dt2, Cx2, Cy2, x[i], y[j], t_1,
                i, j, im1, ip1, jm1, jp1)
    else:
        for i in Ix[1:-1]:
            u[i,j] = bc['S'](x[i], y[j])
    j = Iy[-1]
    jm1 = j-1
    jp1 = jm1
    if bc['N'] is None:
        for i in Ix[1:-1]:
            im1 = i-1; ip1 = i+1
            u[i,j] = scheme_ij(
                u, u_1, u_2, k_1, k_2, k_3, k_4,
                f, dt2, Cx2, Cy2, x[i], y[j], t_1,
                i, j, im1, ip1, jm1, jp1)
    else:
        for i in Ix[1:-1]:
            u[i,j] = bc['N'](x[i], y[j])

    # Corner points
    i = j = Iy[0]
    ip1 = i+1; jp1 = j+1
    im1 = ip1; jm1 = jp1
    if bc['S'] is None:
        u[i,j] = scheme_ij(
            u, u_1, u_2, k_1, k_2, k_3, k_4,
            f, dt2, Cx2, Cy2, x[i], y[j], t_1,
            i, j, im1, ip1, jm1, jp1)
    else:
        u[i,j] = bc['S'](x[i], y[j])

    i = Ix[-1]; j = Iy[0]
    im1 = i-1; jp1 = j+1
    ip1 = im1; jm1 = jp1
    if bc['S'] is None:
        u[i,j] = scheme_ij(
            u, u_1, u_2, k_1, k_2, k_3, k_4,
            f, dt2, Cx2, Cy2, x[i], y[j], t_1,
            i, j, im1, ip1, jm1, jp1)
    else:
        u[i,j] = bc['S'](x[i], y[j])

    i = Ix[-1]; j = Iy[-1]
    im1 = i-1; jm1 = j-1
    ip1 = im1; jp1 = jm1
    if bc['N'] is None:
        u[i,j] = scheme_ij(
            u, u_1, u_2, k_1, k_2, k_3, k_4,
            f, dt2, Cx2, Cy2, x[i], y[j], t_1,
            i, j, im1, ip1, jm1, jp1)
    else:
        u[i,j] = bc['N'](x[i], y[j])

    i = Ix[0]; j = Iy[-1]
    ip1 = i+1; jm1 = j-1
    im1 = ip1; jp1 = jm1
    if bc['N'] is None:
        u[i,j] = scheme_ij(
            u, u_1, u_2, k_1, k_2, k_3, k_4,
            f, dt2, Cx2, Cy2, x[i], y[j], t_1,
            i, j, im1, ip1, jm1, jp1)
    else:
        u[i,j] = bc['N'](x[i], y[j])

    return u

def init_wave(u, u_1, u_2, Ix, Iy, Nx,Ny, Cx2, Cy2, x, y, f, t, dt2, I, bc, pos_x, pos_y):
# Load initial condition into u_1
    for i in Ix:
        for j in Iy:
            u_1[i,j] = I(x[i], y[j],pos_x,pos_y)

    u = scheme_scalar_mesh(u, u_1, u_2, 0.5, 0, 0.5, 0.5,
                               f, dt2, Cx2, Cy2, x, y, t[0],
                               Nx, Ny, bc)

    u_2[:,:] = u_1
    u_1[:,:] = u
    return u, u_1, u_2 

def gen_random(It, x):
    min = -x
    max = x
    url = "https://www.random.org/integers/?num=" + str(len(It)) + "&min=" + str(min) + "&max="+ str(max) +"&col=1&base=10&format=plain&rnd=new"
    randomrequest = requests.get(url)
    str_list = list(randomrequest.text)
    random_list = []
    helper = ''
    for i in str_list:
        if i == '\n':
            random_list.append(int(helper))
            helper = ''
        else: 
            helper = helper + str(i)
    return random_list

def solver(I, f, bc, Lx, Ly, Nx, Ny, dt, T ,c ,turbulence , randomness, noise,
           user_action=None,
           verbose=True):
    
    s = time.process_time()

    x = np.linspace(0, Lx, Nx+1)  
    y = np.linspace(0, Ly, Ny+1)  
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xv = x[:,np.newaxis]          # for vectorized function evaluations
    yv = y[np.newaxis,:]

    if isinstance(c, (float, int)):
        c_max = c
    elif callable(c):
        c_max = max([c(x) for x in np.linspace(0, Lx, 101)])

    if dt <= 0:                # max time step ?
        dt = (1/float(c_max))*(1/np.sqrt(1/dx**2 + 1/dy**2))

    Nt = int(round(T/float(dt)))
    t = np.linspace(0, T, Nt+1)  

    Cx2 = (c_max*dt/dx)**2
    Cy2 = (c_max*dt/dy)**2 
    dt2 = dt**2

    u   = np.zeros((Nx+1,Ny+1))   
    u_1 = np.zeros((Nx+1,Ny+1))  
    u_2 = np.zeros((Nx+1,Ny+1))  

    Ix = range(0, Nx+1)
    Iy = range(0, Ny+1)
    It = range(0, Nt+1)

    u, u_1, u_2 = init_wave(u, u_1, u_2, Ix, Iy, Nx,Ny, Cx2, Cy2, x, y, f, t, dt2, I, bc, 0, 0)

    wave_length = 4

    if noise == True:
        if randomness == True and len(It) <= 1000:
            try:
                random_list_x = gen_random(It,Lx*2)
                random_list_y = gen_random(It,Ly*2)
                random_list = np.zeros([2,max(len(random_list_x),len(random_list_y))])
                random_list[0,:len(random_list_x)] = random_list_x
                random_list[1,:len(random_list_y)] = random_list_y
            except: 
                # Failsafe
                print('Random.org allowance is negative')
                print('Using Numpy pseudo-random number generation')
                random_list = np.random.rand(2,len(It))*4*Lx-2*Lx
        else:
            # Random values from uniform distribtion [-2*Lx, 2*Lx]
            random_list = np.random.rand(2,len(It))*4*Lx-2*Lx

    for n in It[1:-1]:
        u = scheme_scalar_mesh(u, u_1, u_2, 1, 1, 1, 1,
                                   f, dt2, Cx2, Cy2, x, y, t[n],
                                   Nx, Ny, bc)
        
        if noise == True:
            if turbulence == True:
                if n % 10 == 0:
                    for i in range(0,round(wave_length/2)+1):
                        u2, u_12, u_22 = init_wave(np.zeros((Nx+1,Ny+1)), np.zeros((Nx+1,Ny+1)), np.zeros((Nx+1,Ny+1)), range(0, Nx+1), range(0, Ny+1), Nx,Ny, Cx2, Cy2, x, y, f, t, dt2, I, bc, -Lx/2+t[n],i)
                        u = u + u2
                        u_1 = u_1 + u_12
                        u_2 = u_2 + u_22
            else:
                u2, u_12, u_22 = init_wave(np.zeros((Nx+1,Ny+1)), np.zeros((Nx+1,Ny+1)), np.zeros((Nx+1,Ny+1)), range(0, Nx+1), range(0, Ny+1), Nx,Ny, Cx2, Cy2, x, y, f, t, dt2, I, bc, random_list[0,n],random_list[1,n])
                u = u + u2
                u_1 = u_1 + u_12
                u_2 = u_2 + u_22

        if user_action is not None:
            if user_action(u, x, xv, y, yv, t, n+1):
                break

        u_2, u_1, u = u_1, u, u_2

    e = time.process_time()
    return e-s


def main(Lx = 200,            # Spatial length in x-axis
         Ly = 200,            # Spatial length in y-axis
         Nx = 50,             # Spatial mesh in x-axis
         Ny = 50,             # Spatial mesh in y-axis
         T = 0.5,             # Animation time
         pulse_type = 'gaussian',
         temp = 9,            # Temperature (Celsius)
         sal = 6,             # Salinity (ppt)
         z = 0,               # Depth (metres)
         turbulence = False,  # Simulates turbulent rain
         randomness = False,  # Pseudo-random on default
         noise = True         # Simulates random drops
         ):
    
    tracemalloc.start()

    L = np.sqrt(Lx^2+Ly^2)

    c =  1449.2 + 4.6*temp - 0.055*temp**2 + 0.00029*temp**3 + (1.34 - 0.01*temp)*(sal-35) + 0.016*z

    if pulse_type == 'gaussian':
        I = lambda x, y, pos_x, pos_y: np.exp(-(x-Lx/2.0+pos_x)**2/2.0 -(y-Ly/2.0+pos_y)**2/2.0)
    elif pulse_type == 'drop':
        I = lambda x, y, pos_x, pos_y: -np.exp(-(x-Lx/2.0+pos_x)**2/2.0 -(y-Ly/2.0+pos_y)**2/2.0)
    elif pulse_type == '1D':
        I = lambda x, y, pos_x, pos_y: 0 if abs(x-L/2.0) > 0.1 else 1
            
    f = lambda x, y, t: 0 if isinstance(x, (float,int)) else np.zeros(x.size)

    bc = dict(N=None, W=None, E=None, S=None)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def action(u, x, xv, y, yv, t, n):
        ax.clear()
        ax.plot_surface(xv, yv, u,cmap = cm.Blues)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([min(x),max(x)])
        ax.set_ylim([min(y),max(y)])
        ax.set_zlim([-1,1])
        ax.set_title('t=%g' % t[n])
        plt.pause(0.001) 

    timer = solver(I, f, bc, Lx, Ly, Nx, Ny, 0, T, c, turbulence, randomness, noise, user_action=action)
    _, peak = tracemalloc.get_traced_memory()
    print(f'Nx: {Nx}')
    print(f'Ny: {Ny}')
    print(f'CPU time: {timer} seconds')
    print(f'Peak memory usage: {peak} bytes')
    tracemalloc.stop()



if __name__ == '__main__':
    main()