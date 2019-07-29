import numpy as np
import time
from numba import jit,vectorize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

s_sim = time.time()
Niter = 100000
n_avg_fp = 50000
NRuns = 10

ncx = 33
ncy = 33
ncz = 33
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

beta = -5
B1 = 0.01*np.sqrt(kBT/mass)
dt = lamda/sigma
dt2 = dt/2

u_f_run = np.zeros((ncx_fp, ncy_fp, ncz_fp))
v_f_run = np.zeros((ncx_fp, ncy_fp, ncz_fp))
w_f_run = np.zeros((ncx_fp, ncy_fp, ncz_fp))

#-------------------------------------------------------------------------------------------------------#
#--------------------------------------------| Functions |----------------------------------------------#
#-------------------------------------------------------------------------------------------------------#

@jit
def gen_vir_part_sq(N_v):
    au = np.random.uniform(0,1,(N_v,1))
    R = (R1**3 + (R2**3 - R1**3)*au)**(1/3)
    eta = np.random.uniform(-1, 1, (N_v,1))
    phi = np.random.uniform(0, 2*np.pi, (N_v,1))

    x = R * (1-eta**2)**(1/2) * np.cos(phi)
    y = R * (1-eta**2)**(1/2) * np.sin(phi)
    z = R * eta
    return x, y, z

@jit
def virtual_vel(x_v, y_v, z_v):
    N_v = len(x_v)
    u_v = np.random.normal(mu,sigma,(N_v,1))
    v_v = np.random.normal(mu,sigma,(N_v,1))
    w_v = np.random.normal(mu,sigma,(N_v,1))
    
    x_v_rad = np.zeros((N_v,1))
    y_v_rad = np.zeros((N_v,1))
    z_v_rad = np.zeros((N_v,1))

    for j in range(0,N_v):
        mag_position_v = np.sqrt(x_v[j,0]**2 + y_v[j,0]**2 + z_v[j,0]**2)
        x_v_rad[j,0] = x_v[j,0]/mag_position_v
        y_v_rad[j,0] = y_v[j,0]/mag_position_v
        z_v_rad[j,0] = z_v[j,0]/mag_position_v
        u_v[j,0] = u_v[j,0] + (B1*(1+beta*x_v_rad[j,0])*(x_v_rad[j,0]**2 - 1) + 0 + 0)
        v_v[j,0] = v_v[j,0] + (B1*(1+beta*x_v_rad[j,0])*(x_v_rad[j,0]*y_v_rad[j,0]) + 0 + 0)
        w_v[j,0] = w_v[j,0] + (B1*(1+beta*x_v_rad[j,0])*(x_v_rad[j,0]*z_v_rad[j,0]) + 0 + 0)

    u_v_diff = np.sum(u_v,axis=0)/len(u_v)
    v_v_diff = np.sum(v_v,axis=0)/len(v_v)
    w_v_diff = np.sum(w_v,axis=0)/len(w_v)
    
    u_v = u_v - u_v_diff
    v_v = v_v - v_v_diff
    w_v = w_v - w_v_diff
    return u_v, v_v, w_v

@jit
def pos_stream(pos, vel, acc, dt):
    pos_new = np.zeros((len(pos),1))			
    for i in range(len(pos)):
        pos_new[i,0] = pos[i,0] + vel[i,0]*dt + 0.5*acc*(dt**2)
    return pos_new

@jit
def vel_stream(vel, acc, dt):
    N = len(vel)
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

#-----------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#

for k in range(NRuns):
    #-------------------------------------------------------------------------------------------------------#
    #----------------------------------------| Initial Parameters |-----------------------------------------#
    #-------------------------------------------------------------------------------------------------------#

    N = (ncx-1)*(ncy-1)*(ncz-1)*np_avg
    R1 = 4*h
    R2 = R1 - 1.73*h
    N_v = (4/3)*np.pi*(R1**3-R2**3)*np_avg
    N_v = int(N_v)

    #-------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------#


    #-------------------------------------------------------------------------------------------------------#
    #-----------------------------------------|  Initialisation  |------------------------------------------#
    #-------------------------------------------------------------------------------------------------------#

    x_f = np.random.uniform(x_min,x_max,(N,1))
    y_f = np.random.uniform(y_min,y_max,(N,1))
    z_f = np.random.uniform(z_min,z_max,(N,1))

    a = []
    for i in range(N):
        if (x_f[i,0]**2 + y_f[i,0]**2 + z_f[i,0]**2) < (R1**2):
            a.append(i)
    N = N-len(a)

    x_f = np.delete(x_f, a, axis=0)
    y_f = np.delete(y_f, a, axis=0)
    z_f = np.delete(z_f, a, axis=0)

    u_f = np.random.normal(mu,sigma,(N,1))
    v_f = np.random.normal(mu,sigma,(N,1))
    w_f = np.random.normal(mu,sigma,(N,1))

    grid_shift_x = h*np.random.rand(Niter,1)-h2
    grid_shift_y = h*np.random.rand(Niter,1)-h2
    grid_shift_z = h*np.random.rand(Niter,1)-h2

    u_cm_fp = np.zeros((nct_profile,1))
    v_cm_fp = np.zeros((nct_profile,1))
    w_cm_fp = np.zeros((nct_profile,1))

    #-------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------#

    for i in range(0,Niter):

        #--------------------------------| Streaming Step |------------------------------------#
        x_f = pos_stream(x_f,u_f,0,dt)
        y_f = pos_stream(y_f,v_f,0,dt)
        z_f = pos_stream(z_f,w_f,0,dt)

        #--------------------------------| Periodic BC |------------------------------------#
        x_f = periodic(x_f,x_min, Lx)
        y_f = periodic(y_f,y_min, Ly)
        z_f = periodic(z_f,z_min, Lz)    

        #--------------------------------| Squirmer BC |------------------------------------#
        sq_in = np.nonzero((x_f**2+y_f**2+z_f**2)<=(R1**2))
        sq_in = np.asarray(sq_in)
        sq_in = np.delete(sq_in,1,axis=0).ravel()
        x_f[sq_in,0] = x_f[sq_in,0] - u_f[sq_in,0]*dt2
        y_f[sq_in,0] = y_f[sq_in,0] - v_f[sq_in,0]*dt2
        z_f[sq_in,0] = z_f[sq_in,0] - w_f[sq_in,0]*dt2

        mag_position = np.sqrt(x_f[sq_in,0]**2 + y_f[sq_in,0]**2 + z_f[sq_in,0]**2)
        
        x_f[sq_in,0] = x_f[sq_in,0]/mag_position
        y_f[sq_in,0] = y_f[sq_in,0]/mag_position
        z_f[sq_in,0] = z_f[sq_in,0]/mag_position

        u_f[sq_in,0] = -u_f[sq_in,0] + 2*(B1*(1+beta*x_f[sq_in,0])*(x_f[sq_in,0]**2 - 1) + 0 + 0)
        v_f[sq_in,0] = -v_f[sq_in,0] + 2*(B1*(1+beta*x_f[sq_in,0])*(x_f[sq_in,0]*y_f[sq_in,0]) + 0 + 0)
        w_f[sq_in,0] = -w_f[sq_in,0] + 2*(B1*(1+beta*x_f[sq_in,0])*(x_f[sq_in,0]*z_f[sq_in,0]) + 0 + 0)

        x_f[sq_in,0] = R1*x_f[sq_in,0]
        y_f[sq_in,0] = R1*y_f[sq_in,0]
        z_f[sq_in,0] = R1*z_f[sq_in,0]

        x_f[sq_in,0] = x_f[sq_in,0] + u_f[sq_in,0]*dt2
        y_f[sq_in,0] = y_f[sq_in,0] + v_f[sq_in,0]*dt2
        z_f[sq_in,0] = z_f[sq_in,0] + w_f[sq_in,0]*dt2

        #--------------------------------| Periodic BC |------------------------------------#
        x_f = periodic(x_f,x_min, Lx)
        y_f = periodic(y_f,y_min, Ly)
        z_f = periodic(z_f,z_min, Lz)

        #--------------------------------| Collision Step |------------------------------------#
        x_v, y_v, z_v = gen_vir_part_sq(N_v)
        u_v, v_v, w_v = virtual_vel(x_v, y_v, z_v)

        x = np.append(x_f, x_v, axis=0)
        y = np.append(y_f, y_v, axis=0)
        z = np.append(z_f, z_v, axis=0)
        u = np.append(u_f, u_v, axis=0)
        v = np.append(v_f, v_v, axis=0)
        w = np.append(w_f, w_v, axis=0)
        
        locations = np.zeros((N,1))
        in_x = CellIndex(x, x_min, grid_shift_x[i])
        in_y = CellIndex(y, y_min, grid_shift_y[i])
        in_z = CellIndex(z, z_min, grid_shift_z[i])
        locations = ncx*ncy*in_z + in_y*ncx + in_x
        locations = locations.astype(int)

        head, list_particle, count = head_list(locations,N+N_v,nc_total) 
        head = head.astype(int)
        list_particle = list_particle.astype(int)
        
        u_sum = cm_quantity(head, list_particle, count, u, nc_total)
        v_sum = cm_quantity(head, list_particle, count, v, nc_total)
        w_sum = cm_quantity(head, list_particle, count, w, nc_total)

        u_cm = cm_vel(u_sum, count, nc_total)
        v_cm = cm_vel(v_sum, count, nc_total)
        w_cm = cm_vel(w_sum, count, nc_total)
        
        # Anderson Thermostat
        u_r_at = np.random.normal(mu,sigma,(N+N_v,1))
        v_r_at = np.random.normal(mu,sigma,(N+N_v,1))
        w_r_at = np.random.normal(mu,sigma,(N+N_v,1))

        u_cm_r_sum = cm_quantity(head, list_particle, count, u_r_at, nc_total)
        v_cm_r_sum = cm_quantity(head, list_particle, count, v_r_at, nc_total)
        w_cm_r_sum = cm_quantity(head, list_particle, count, w_r_at, nc_total)

        u_si = cm_vel(u_cm_r_sum, count, nc_total)
        v_si = cm_vel(v_cm_r_sum, count, nc_total)
        w_si = cm_vel(w_cm_r_sum, count, nc_total)

        u_at = anderson_thermostat(u_cm, u_r_at, u_si, locations, N+N_v)
        v_at = anderson_thermostat(v_cm, v_r_at, v_si, locations, N+N_v)
        w_at = anderson_thermostat(w_cm, w_r_at, w_si, locations, N+N_v)
        
        u_diff = u_at[N:N+N_v,0].reshape(N_v,1) - u_v
        v_diff = v_at[N:N+N_v,0].reshape(N_v,1) - v_v
        w_diff = w_at[N:N+N_v,0].reshape(N_v,1) - w_v

        deltap_x = mass*np.sum(u_diff,axis=0)/N
        deltap_y = mass*np.sum(v_diff,axis=0)/N
        deltap_z = mass*np.sum(w_diff,axis=0)/N

        u_f = u_at[0:N,0].reshape(N,1) + deltap_x
        v_f = v_at[0:N,0].reshape(N,1) + deltap_y
        w_f = w_at[0:N,0].reshape(N,1) + deltap_z

        if i>=n_avg_fp:
            in_x_fp = np.floor((x_f-x_min)/hcf)
            in_y_fp = np.floor((y_f-y_min)/hcf)
            in_z_fp = np.floor((z_f-z_min)/hcf)
            locations_fp = ncx_fp*ncy_fp*in_z_fp + ncx_fp*in_y_fp + in_x_fp
            locations_fp = locations_fp.astype(int)
            
            head_fp, list_particle_fp, count_fp = head_list(locations_fp, N, nct_profile)
            head_fp = head_fp.astype(int)
            list_particle_fp = list_particle_fp.astype(int)

            u_cm_fp1 = cm_quantity(head_fp, list_particle_fp, count_fp, u_f, nct_profile)
            v_cm_fp1 = cm_quantity(head_fp, list_particle_fp, count_fp, v_f, nct_profile)
            w_cm_fp1 = cm_quantity(head_fp, list_particle_fp, count_fp, w_f, nct_profile)

            u_cm_fp2 = cm_vel(u_cm_fp1, count_fp, nct_profile)
            v_cm_fp2 = cm_vel(v_cm_fp1, count_fp, nct_profile)
            w_cm_fp2 = cm_vel(w_cm_fp1, count_fp, nct_profile)

            u_cm_fp = u_cm_fp + u_cm_fp2
            v_cm_fp = v_cm_fp + v_cm_fp2                        
            w_cm_fp = w_cm_fp + w_cm_fp2

        if i%100==1:
            print("ITERATION - " + 	str(i))

    u_cm_fp = u_cm_fp/(Niter-n_avg_fp)
    v_cm_fp = v_cm_fp/(Niter-n_avg_fp)                       
    w_cm_fp = w_cm_fp/(Niter-n_avg_fp)

    u_new = u_cm_fp.reshape(ncx_fp,ncy_fp,ncz_fp)
    v_new = v_cm_fp.reshape(ncx_fp,ncy_fp,ncz_fp)
    w_new = w_cm_fp.reshape(ncx_fp,ncy_fp,ncz_fp)

    u_new = u_new.transpose()
    v_new = v_new.transpose()
    w_new = w_new.transpose()

    u_f_run = u_f_run + u_new
    v_f_run = v_f_run + v_new
    w_f_run = w_f_run + w_new

    print("Run Number : " + str(k))

u_f_run = u_f_run/NRuns
v_f_run = v_f_run/NRuns
w_f_run = w_f_run/NRuns

e_sim = time.time()
print(str(e_sim-s_sim))