import numpy as np
from numba import jit

#-------------------------------------------------------------------------------------------------------#
#----------------------------------------| Initial Parameters |-----------------------------------------#
#-------------------------------------------------------------------------------------------------------#

Niter = 5000
ncx = 21
ncy = 21
ncz = 21
nc_total = (ncx)*(ncy)*(ncz)

h = 1
h2 = h/2

Lx = (ncx-1)*h                  #Lx & Ly are length of simulation box NOT grid
Ly = (ncy-1)*h
Lz = (ncz-1)*h

center_b_x = 0
center_b_y = 0
center_b_z = 0

x_min = center_b_x - Lx/2
y_min = center_b_y - Ly/2
z_min = center_b_z - Lz/2

x_max = x_min + Lx
y_max = y_min + Ly
z_max = z_min + Lz

np_avg = 10
N = (ncx-1)*(ncy-1)*(ncz-1)*np_avg
N_v = 333
R1 = 3*h
R2 = R1 - 1.73*h
mass = 1

res = 5
hcf = h/res
ncx_fp = res*(ncx-1)
ncy_fp = res*(ncy-1)
ncz_fp = res*(ncz-1)
nct_profile = ncx_fp*ncy_fp*ncz_fp

mu = 0
sigma = 1
lamda = 0.1
kBT = 0.1
beta = 3
B1 = 0.1*np.sqrt(kBT/mass)
dt = kBT/sigma
dt2 = dt/2


#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------#
#-----------------------------------------|  Initialisation  |------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
x = np.random.uniform(x_min,x_max,(N,1))
y = np.random.uniform(y_min,y_max,(N,1))
z = np.random.uniform(y_min,y_max,(N,1))

a = []
for i in range(N):
    if (x[i]**2 + y[i]**2 + z[i]**2) < (R1**2):
        a.append(i)
N = N-len(a)

x = np.delete(x, a, axis=0)
y = np.delete(y, a, axis=0)
z = np.delete(z, a, axis=0)
u = np.random.normal(mu,sigma,(N,1))
v = np.random.normal(mu,sigma,(N,1))
w = np.random.normal(mu,sigma,(N,1))

@jit
def gen_vir_part_sq():
    u = np.random.uniform(0,1,(N_v,1))
    R = (R1**3 + (R2**3 - R1**3)*u)**(1/3)
    eta = np.random.uniform(-1, 1, (N_v,1))
    phi = np.random.uniform(0, 2*np.pi, (N_v,1))

    x = R * (1-eta**2)**(1/2) * np.cos(phi)
    y = R * (1-eta**2)**(1/2) * np.sin(phi)
    z = R * eta
    return x, y, z
    
x_v, y_v, z_v = gen_vir_part_sq()
u_v = np.random.normal(mu,sigma,(N_v,1))
v_v = np.random.normal(mu,sigma,(N_v,1))
w_v = np.random.normal(mu,sigma,(N_v,1))

for j in range(N_v):
    mag_position_v = np.sqrt(x_v[j]**2 + y_v[j]**2 + z_v[j]**2)
    x_v_rad = R1*x_v/mag_position_v
    y_v_rad = R1*y_v/mag_position_v
    z_v_rad = R1*z_v/mag_position_v
    u_v[j,0] = -u_v[j,0] + 2*(B1*(1+beta*x_v_rad[j,0])*(x_v_rad[j,0]**2 - 1) + 0 + 0)
    v_v[j,0] = -v_v[j,0] + 2*(B1*(1+beta*x_v_rad[j,0])*(z_v_rad[j,0]*x_v_rad[j,0]) + 0 + 0)
    w_v[j,0] = -w_v[j,0] + 2*(B1*(1+beta*x_v_rad[j,0])*(x_v_rad[j,0]*z_v_rad[j,0]) + 0 + 0)

grid_shift_x = h*np.random.rand(Niter,1)-h2
grid_shift_y = h*np.random.rand(Niter,1)-h2
grid_shift_z = h*np.random.rand(Niter,1)-h2

n_avg_fp = 300000
u_cm_fp = np.zeros((nct_profile,1))
v_cm_fp = np.zeros((nct_profile,1))
w_cm_fp = np.zeros((nct_profile,1))

#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#