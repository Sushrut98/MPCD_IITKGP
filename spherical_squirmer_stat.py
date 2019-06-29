import numpy as np
import time
from numba import jit,vectorize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

#-------------------------------------------------------------------------------------------------------#
#----------------------------------------| Initial Parameters |-----------------------------------------#
#-------------------------------------------------------------------------------------------------------#

Niter = 50000
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

                         #------------| Flow Stabilization Run  |---------------#
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


             #------------| For Velocity Profiles  |---------------#
# x=np.load('x_poiseullie.npy')
# y=np.load('y_poiseullie.npy')
# z=np.load('z_poiseullie.npy')
# u=np.load('u_poiseullie.npy')
# v=np.load('v_poiseullie.npy')
# w=np.load('w_poiseullie.npy')

grid_shift_x = h*np.random.rand(Niter,1)-h2
grid_shift_y = h*np.random.rand(Niter,1)-h2
grid_shift_z = h*np.random.rand(Niter,1)-h2

#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------#
#--------------------------------------------| Functions |----------------------------------------------#
#-------------------------------------------------------------------------------------------------------#

# Streaming
@jit
def pos_stream(pos, vel, acc, dt):
    pos_new = np.zeros((N,1))			
    for i in range(N):
        pos_new[i,0] = pos[i,0] + vel[i,0]*dt + 0.5*acc*(dt**2)
    return pos_new

@jit
def vel_stream(vel, acc, dt):
    new_vel = np.zeros((N,1))
    for i in range(N):
        new_vel[i,0] = vel[i,0] + acc * dt
    return new_vel

@vectorize
def periodic(pos, LowerBound, L):
    pos  = LowerBound + ((pos-LowerBound)%L)
    return pos

@vectorize
def CellIndex(pos, LowerBound, grid_shift):
    in_a = np.floor((pos-LowerBound+h2-grid_shift)/h)
    return in_a

@jit
def head_list(locations, Num):
    #head, list array must be initialized for every iteration

    head = -1*(np.ones((nc_total,1)))
    list_particle = -2*np.ones((Num,1))
    count = np.zeros((nc_total,1))
    
    for j in range(0,Num):
        list_particle[j,0] = head[locations[j,0],0]
        head[locations[j,0],0] = j
        count[locations[j,0],0] += 1

    return head, list_particle, count

@jit
def cm_quantity(head, list_particle, count, quant):
    quant_cm = np.zeros((nc_total, 1))

    for j in range(0,nc_total):
        k = head[j,0]

        while(k != -1):
            quant_cm[j,0] = quant_cm[j,0] + quant[k,0]
            k = list_particle[k,0]    
    return quant_cm

@jit
def cm_vel(veloc, count):
    veloc_cm = np.zeros((nc_total,1))
    for j in range(nc_total):
        if count[j,0] == 0:
            continue
        else:
            veloc_cm[j,0] = veloc[j,0]/count[j,0]
    
    return veloc_cm

@jit
def anderson_thermostat(vel_cm, vel_rand, vel_si, locations, Num):    
    vel = np.zeros((Num,1))
    
    for i in range(Num):
        vel[i,0] = vel_cm[locations[i,0],0] + vel_rand[i,0] - vel_si[locations[i,0],0]

    return vel

#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
plt.show()
s_sim = time.time()
for i in range(0,Niter):
    x = pos_stream(x,u,0,dt)
    y = pos_stream(y,v,0,dt)	
    z = pos_stream(z,w,0,dt)

    x = periodic(x,x_min, Lx)
    #print('aa')
    y = periodic(y,y_min, Ly)
    #print('ab')
    z = periodic(z,z_min, Lz)
    #print('ac')
    
    #print('a')
    #--------------------------------| Squirmer BC |------------------------------------#
    sq_in = np.nonzero((x**2+y**2+z**2)<(R1**2))
    sq_in = np.asarray(sq_in)
    sq_in = np.delete(sq_in,1,axis=0).ravel()
    x[sq_in,0] = x[sq_in,0] - u[sq_in,0]*dt2
    y[sq_in,0] = y[sq_in,0] - v[sq_in,0]*dt2
    z[sq_in,0] = z[sq_in,0] - w[sq_in,0]*dt2

    mag_position = np.sqrt(x[sq_in,0]**2 + y[sq_in,0]**2 + z[sq_in,0]**2)
    
    x[sq_in,0] = R1*x[sq_in,0]/mag_position
    y[sq_in,0] = R1*y[sq_in,0]/mag_position
    z[sq_in,0] = R1*z[sq_in,0]/mag_position

    u[sq_in,0] = -u[sq_in,0] + 2*(B1*(1+beta*x[sq_in,0])*(x[sq_in,0]**2 - 1) + 0 + 0)
    v[sq_in,0] = -v[sq_in,0] + 2*(B1*(1+beta*x[sq_in,0])*(y[sq_in,0]*x[sq_in,0]) + 0 + 0)
    w[sq_in,0] = -w[sq_in,0] + 2*(B1*(1+beta*x[sq_in,0])*(x[sq_in,0]*z[sq_in,0]) + 0 + 0)

    x[sq_in,0] = x[sq_in,0] + u[sq_in,0]*dt2
    y[sq_in,0] = y[sq_in,0] + v[sq_in,0]*dt2
    z[sq_in,0] = z[sq_in,0] + w[sq_in,0]*dt2
    
    #print('b')

    #--------------------------------| Periodic BC |------------------------------------#
    x = periodic(x,x_min, Lx)
    #print('ca')
    y = periodic(y,y_min, Ly)
    #print('cb')
    z = periodic(z,z_min, Lz)
    #print('cc')
    
    #print('c')
    locations = np.zeros((N,1))
    in_x = CellIndex(x, x_min, grid_shift_x[i])
    in_y = CellIndex(y, y_min, grid_shift_y[i])
    in_z = CellIndex(z, z_min, grid_shift_z[i])
    locations = ncx*ncy*in_z + in_y*ncx + in_x
    locations = locations.astype(int)
    
    #print('d')
    locations_v = np.zeros((N_v,1))
    in_x_v = CellIndex(x_v, x_min, grid_shift_x[i])
    in_y_v = CellIndex(y_v, y_min, grid_shift_y[i])
    in_z_v = CellIndex(z_v, z_min, grid_shift_z[i])
    locations_v = ncx*ncy*in_z_v + in_y_v*ncx + in_x_v
    locations_v = locations_v.astype(int)
    #print('d')

    head, list_particle, count_f = head_list(locations,N) 
    head = head.astype(int)
    list_particle = list_particle.astype(int)
    
    x_cm_f = cm_quantity(head, list_particle, count_f, x)
    y_cm_f = cm_quantity(head, list_particle, count_f, y)
    z_cm_f = cm_quantity(head, list_particle, count_f, z)
    u_cm_f = cm_quantity(head, list_particle, count_f, u)
    v_cm_f = cm_quantity(head, list_particle, count_f, v)
    w_cm_f = cm_quantity(head, list_particle, count_f, w)
    
    #print('e')
    head_v, list_particle_v, count_v = head_list(locations_v,N_v)
    head_v = head_v.astype(int)
    list_particle_v = list_particle_v.astype(int)
    
    x_cm_v = cm_quantity(head_v, list_particle_v, count_v, x_v)
    y_cm_v = cm_quantity(head_v, list_particle_v, count_v, y_v)
    z_cm_v = cm_quantity(head_v, list_particle_v, count_v, z_v)
    u_cm_v = cm_quantity(head_v, list_particle_v, count_v, u_v)
    v_cm_v = cm_quantity(head_v, list_particle_v, count_v, v_v)
    w_cm_v = cm_quantity(head_v, list_particle_v, count_v, w_v)
    
    #print('f')

    count = count_v + count_f

    u_cm = cm_vel((u_cm_f + u_cm_v),count)
    v_cm = cm_vel((v_cm_f + v_cm_v),count)
    w_cm = cm_vel((w_cm_f + w_cm_v),count)
    
    #print('g')
    
    u_r_at = np.random.normal(mu,sigma,(N,1))
    v_r_at = np.random.normal(mu,sigma,(N,1))
    w_r_at = np.random.normal(mu,sigma,(N,1))

    u_cm_r = cm_quantity(head, list_particle, count_f, u_r_at)
    v_cm_r = cm_quantity(head, list_particle, count_f, v_r_at)
    w_cm_r = cm_quantity(head, list_particle, count_f, w_r_at)

    u_si =cm_vel(u_cm_r, count)
    v_si =cm_vel(v_cm_r, count)
    w_si =cm_vel(w_cm_r, count)

    u = anderson_thermostat(u_cm, u_r_at, u_si, locations, N)
    v = anderson_thermostat(v_cm, v_r_at, v_si, locations, N)
    w = anderson_thermostat(w_cm, w_r_at, w_si, locations, N)

    #if i%1000==1:
    print("ITERATION - " + 	str(i))
    

e_sim = time.time()
print(str(e_sim-s_sim))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
plt.show()
