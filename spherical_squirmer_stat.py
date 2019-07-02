import numpy as np
import time
from numba import jit,vectorize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

#-------------------------------------------------------------------------------------------------------#
#----------------------------------------| Initial Parameters |-----------------------------------------#
#-------------------------------------------------------------------------------------------------------#

Niter = 50000
ncx = 49
ncy = 49
ncz = 49
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
R1 = 4*h
R2 = R1 - 1.73*h
N_v = (4/3)*np.pi*(R1**3-R2**3)*np_avg
N_v=int(N_v)
mass = 1

res = 2
hcf = h/res
ncx_fp = res*(ncx-1)
ncy_fp = res*(ncy-1)
ncz_fp = res*(ncz-1)
nct_profile = ncx_fp*ncy_fp*ncz_fp

mu = 0
kBT = 36
sigma = np.sqrt(kBT/mass)
lamda = 0.1

beta = 0
B1 = 0.01*np.sqrt(kBT/mass)
dt = lamda/sigma
dt2 = dt/2

#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------#
#-----------------------------------------|  Initialisation  |------------------------------------------#
#-------------------------------------------------------------------------------------------------------#

x = np.random.uniform(x_min,x_max,(N,1))
y = np.random.uniform(y_min,y_max,(N,1))
z = np.random.uniform(z_min,z_max,(N,1))

a = []
for i in range(N):
    if (x[i,0]**2 + y[i,0]**2 + z[i,0]**2) < (R1**2):
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
    au = np.random.uniform(0,1,(N_v,1))
    R = (R1**3 + (R2**3 - R1**3)*au)**(1/3)
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

x_v_rad = np.zeros((N_v,1))
y_v_rad = np.zeros((N_v,1))
z_v_rad = np.zeros((N_v,1))

for j in range(N_v):
    mag_position_v = np.sqrt(x_v[j]**2 + y_v[j]**2 + z_v[j]**2)
    x_v_rad[j,0] = x_v[j,0]/mag_position_v
    y_v_rad[j,0] = y_v[j,0]/mag_position_v
    z_v_rad[j,0] = z_v[j,0]/mag_position_v
    u_v[j,0] = u_v[j,0] + (B1*(1+beta*x_v_rad[j,0])*(x_v_rad[j,0]**2 - 1) + 0 + 0)
    v_v[j,0] = v_v[j,0] + (B1*(1+beta*x_v_rad[j,0])*(x_v_rad[j,0]*y_v_rad[j,0]) + 0 + 0)
    w_v[j,0] = w_v[j,0] + (B1*(1+beta*x_v_rad[j,0])*(x_v_rad[j,0]*z_v_rad[j,0]) + 0 + 0)

grid_shift_x = h*np.random.rand(Niter,1)-h2
grid_shift_y = h*np.random.rand(Niter,1)-h2
grid_shift_z = h*np.random.rand(Niter,1)-h2

n_avg_fp = 30000
u_cm_fp = np.zeros((nct_profile,1))
v_cm_fp = np.zeros((nct_profile,1))
w_cm_fp = np.zeros((nct_profile,1))

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
def head_list(locations, Num, Num_cell):
    #head, list array must be initialized for every iteration

    head = -1*(np.ones((Num_cell,1)))
    list_particle = -2*np.ones((Num,1))
    count = np.zeros((Num_cell,1))
    
    for j in range(0,Num):
        list_particle[j,0] = head[locations[j,0],0]
        head[locations[j,0],0] = j
        count[locations[j,0],0] += 1

    return head, list_particle, count

@jit
def cm_quantity(head, list_particle, count, quant, Num_cell):
    quant_cm = np.zeros((Num_cell, 1))

    for j in range(0,Num_cell):
        k = head[j,0]

        while(k != -1):
            quant_cm[j,0] = quant_cm[j,0] + quant[k,0]
            k = list_particle[k,0]    
    return quant_cm

@jit
def cm_vel(veloc, count, Num_cell):
    veloc_cm = np.zeros((Num_cell,1))
    for j in range(Num_cell):
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
    sq_in = np.nonzero((x**2+y**2+z**2)<=(R1**2))
    sq_in = np.asarray(sq_in)
    sq_in = np.delete(sq_in,1,axis=0).ravel()
    x[sq_in,0] = x[sq_in,0] - u[sq_in,0]*dt2
    y[sq_in,0] = y[sq_in,0] - v[sq_in,0]*dt2
    z[sq_in,0] = z[sq_in,0] - w[sq_in,0]*dt2

    mag_position = np.sqrt(x[sq_in,0]**2 + y[sq_in,0]**2 + z[sq_in,0]**2)
    
    x[sq_in,0] = x[sq_in,0]/mag_position
    y[sq_in,0] = y[sq_in,0]/mag_position
    z[sq_in,0] = z[sq_in,0]/mag_position

    u[sq_in,0] = -u[sq_in,0] + 2*(B1*(1+beta*x[sq_in,0])*(x[sq_in,0]**2 - 1) + 0 + 0)
    v[sq_in,0] = -v[sq_in,0] + 2*(B1*(1+beta*x[sq_in,0])*(x[sq_in,0]*y[sq_in,0]) + 0 + 0)
    w[sq_in,0] = -w[sq_in,0] + 2*(B1*(1+beta*x[sq_in,0])*(x[sq_in,0]*z[sq_in,0]) + 0 + 0)

    x[sq_in,0] = R1*x[sq_in,0]
    y[sq_in,0] = R1*y[sq_in,0]
    z[sq_in,0] = R1*z[sq_in,0]

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

    head, list_particle, count_f = head_list(locations,N,nc_total) 
    head = head.astype(int)
    list_particle = list_particle.astype(int)
    
    u_cm_f = cm_quantity(head, list_particle, count_f, u, nc_total)
    v_cm_f = cm_quantity(head, list_particle, count_f, v, nc_total)
    w_cm_f = cm_quantity(head, list_particle, count_f, w, nc_total)
    
    #print('e')
    head_v, list_particle_v, count_v = head_list(locations_v,N_v,nc_total)
    head_v = head_v.astype(int)
    list_particle_v = list_particle_v.astype(int)
    
    u_cm_v = cm_quantity(head_v, list_particle_v, count_v, u_v, nc_total)
    v_cm_v = cm_quantity(head_v, list_particle_v, count_v, v_v, nc_total)
    w_cm_v = cm_quantity(head_v, list_particle_v, count_v, w_v, nc_total)
    
    #print('f')

    count = count_v + count_f

    u_cm = cm_vel((u_cm_f + u_cm_v),count, nc_total)
    v_cm = cm_vel((v_cm_f + v_cm_v),count, nc_total)
    w_cm = cm_vel((w_cm_f + w_cm_v),count, nc_total)
    
    #print('g')
    
    u_r_at = np.random.normal(mu,sigma,(N,1))
    v_r_at = np.random.normal(mu,sigma,(N,1))
    w_r_at = np.random.normal(mu,sigma,(N,1))

    u_cm_r = cm_quantity(head, list_particle, count_f, u_r_at, nc_total)
    v_cm_r = cm_quantity(head, list_particle, count_f, v_r_at, nc_total)
    w_cm_r = cm_quantity(head, list_particle, count_f, w_r_at, nc_total)

    u_si = cm_vel(u_cm_r, count, nc_total)
    v_si = cm_vel(v_cm_r, count, nc_total)
    w_si = cm_vel(w_cm_r, count, nc_total)

    u = anderson_thermostat(u_cm, u_r_at, u_si, locations, N)
    v = anderson_thermostat(v_cm, v_r_at, v_si, locations, N)
    w = anderson_thermostat(w_cm, w_r_at, w_si, locations, N)

    #if i>=n_avg_fp:
    in_x_fp = np.floor((x-x_min)/hcf)
    in_y_fp = np.floor((y-y_min)/hcf)
    in_z_fp = np.floor((z-z_min)/hcf)
    locations_fp = ncx_fp*ncy_fp*in_z_fp + ncx_fp*in_y_fp + in_x_fp
    locations_fp = locations_fp.astype(int)
    
    head_fp, list_particle_fp, count_fp = head_list(locations_fp, N, nct_profile)
    head_fp = head_fp.astype(int)
    list_particle_fp = list_particle_fp.astype(int)

    u_cm_fp1 = cm_quantity(head_fp, list_particle_fp, count_fp, u, nct_profile)
    v_cm_fp1 = cm_quantity(head_fp, list_particle_fp, count_fp, v, nct_profile)
    w_cm_fp1 = cm_quantity(head_fp, list_particle_fp, count_fp, w, nct_profile)

    u_cm_fp2 = cm_vel(u_cm_fp1, count_fp, nct_profile)
    v_cm_fp2 = cm_vel(v_cm_fp1, count_fp, nct_profile)
    w_cm_fp2 = cm_vel(w_cm_fp1, count_fp, nct_profile)

    u_cm_fp = u_cm_fp + u_cm_fp2
    v_cm_fp = v_cm_fp + v_cm_fp2                        
    w_cm_fp = w_cm_fp + w_cm_fp2

    if i%100==1:
        print("ITERATION - " + 	str(i))

u_cm_fp = u_cm_fp/Niter #(Niter-n_avg_fp)
v_cm_fp = v_cm_fp/Niter #(Niter-n_avg_fp)                       
w_cm_fp = w_cm_fp/Niter #(Niter-n_avg_fp)

u_new = u_cm_fp.reshape(ncx_fp,ncy_fp,ncz_fp)
v_new = v_cm_fp.reshape(ncx_fp,ncy_fp,ncz_fp)
w_new = w_cm_fp.reshape(ncx_fp,ncy_fp,ncz_fp)

u_new = u_new.transpose()
v_new = v_new.transpose()
w_new = w_new.transpose()

e_sim = time.time()
print(str(e_sim-s_sim))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)
plt.show()
