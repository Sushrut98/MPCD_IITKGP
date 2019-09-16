import numpy as np 
import functions as func 
import initialize as init
import time

x = init.x
y = init.y
z = init.z
u = init.u
v = init.v
w = init.w

x_v = init.x_v
y_v = init.y_v
z_v = init.z_v
u_v = init.u_v
v_v = init.v_v
w_v = init.w_v

u_cm_fp = init.u_cm_fp
v_cm_fp = init.v_cm_fp
w_cm_fp = init.w_cm_fp

grid_shift_x = init.grid_shift_x
grid_shift_y = init.grid_shift_y
grid_shift_z = init.grid_shift_z

Niter = init.Niter
s_sim = time.time()
for i in range(0,Niter):
    x = func.pos_stream(x,u,0,init.dt,init.N)
    y = func.pos_stream(y,v,0,init.dt,init.N)	
    z = func.pos_stream(z,w,0,init.dt,init.N)

    x = func.periodic(x,init.x_min, init.Lx)
    #print('aa')
    y = func.periodic(y,init.y_min, init.Ly)
    #print('ab')
    z = func.periodic(z,init.z_min, init.Lz)
    #print('ac')
    
    #print('a')
    #--------------------------------| Squirmer BC |------------------------------------#
    sq_in = np.nonzero((x**2+y**2+z**2)<(init.R1**2))
    sq_in = np.asarray(sq_in)
    sq_in = np.delete(sq_in,1,axis=0).ravel()
    x[sq_in,0] = x[sq_in,0] - u[sq_in,0]*init.dt2
    y[sq_in,0] = y[sq_in,0] - v[sq_in,0]*init.dt2
    z[sq_in,0] = z[sq_in,0] - w[sq_in,0]*init.dt2

    mag_position = np.sqrt(x[sq_in,0]**2 + y[sq_in,0]**2 + z[sq_in,0]**2)
    
    x[sq_in,0] = init.R1*x[sq_in,0]/mag_position
    y[sq_in,0] = init.R1*y[sq_in,0]/mag_position
    z[sq_in,0] = init.R1*z[sq_in,0]/mag_position

    u[sq_in,0] = -u[sq_in,0] + 2*(init.B1*(1+init.beta*x[sq_in,0])*(x[sq_in,0]**2 - 1) + 0 + 0)
    v[sq_in,0] = -v[sq_in,0] + 2*(init.B1*(1+init.beta*x[sq_in,0])*(y[sq_in,0]*x[sq_in,0]) + 0 + 0)
    w[sq_in,0] = -w[sq_in,0] + 2*(init.B1*(1+init.beta*x[sq_in,0])*(x[sq_in,0]*z[sq_in,0]) + 0 + 0)

    x[sq_in,0] = x[sq_in,0] + u[sq_in,0]*init.dt2
    y[sq_in,0] = y[sq_in,0] + v[sq_in,0]*init.dt2
    z[sq_in,0] = z[sq_in,0] + w[sq_in,0]*init.dt2
    
    #print('b')

    #--------------------------------| func.periodic BC |------------------------------------#
    x = func.periodic(x,init.x_min, init.Lx)
    #print('ca')
    y = func.periodic(y,init.y_min, init.Ly)
    #print('cb')
    z = func.periodic(z,init.z_min, init.Lz)
    #print('cc')
    
    #print('c')
    locations = np.zeros((init.N,1))
    in_x = func.CellIndex(x, init.x_min, grid_shift_x[i])
    in_y = func.CellIndex(y, init.y_min, grid_shift_y[i])
    in_z = func.CellIndex(z, init.z_min, grid_shift_z[i])
    locations = init.ncx*init.ncy*in_z + in_y*init.ncx + in_x
    locations = locations.astype(int)
    
    #print('d')
    locations_v = np.zeros((init.N_v,1))
    in_x_v = func.CellIndex(x_v, init.x_min, init.grid_shift_x[i])
    in_y_v = func.CellIndex(y_v, init.y_min, init.grid_shift_y[i])
    in_z_v = func.CellIndex(z_v, init.z_min, init.grid_shift_z[i])
    locations_v = init.ncx*init.ncy*in_z_v + in_y_v*init.ncx + in_x_v
    locations_v = locations_v.astype(int)
    #print('d')

    head, list_particle, count_f = func.head_list(locations,init.N,init.nc_total) 
    head = head.astype(int)
    list_particle = list_particle.astype(int)
    
    u_cm_f = func.cm_quantity(head, list_particle, count_f, u, init.nc_total)
    v_cm_f = func.cm_quantity(head, list_particle, count_f, v, init.nc_total)
    w_cm_f = func.cm_quantity(head, list_particle, count_f, w, init.nc_total)
    
    #print('e')
    head_v, list_particle_v, count_v = func.head_list(locations_v,init.N_v,init.nc_total)
    head_v = head_v.astype(int)
    list_particle_v = list_particle_v.astype(int)
    
    u_cm_v = func.cm_quantity(head_v, list_particle_v, count_v, u_v, init.nc_total)
    v_cm_v = func.cm_quantity(head_v, list_particle_v, count_v, v_v, init.nc_total)
    w_cm_v = func.cm_quantity(head_v, list_particle_v, count_v, w_v, init.nc_total)
    
    #print('f')

    count = count_v + count_f

    u_cm = func.cm_vel((u_cm_f + u_cm_v),count, init.nc_total)
    v_cm = func.cm_vel((v_cm_f + v_cm_v),count, init.nc_total)
    w_cm = func.cm_vel((w_cm_f + w_cm_v),count, init.nc_total)
    
    #print('g')
    
    u_r_at = np.random.normal(init.mu,init.sigma,(init.N,1))
    v_r_at = np.random.normal(init.mu,init.sigma,(init.N,1))
    w_r_at = np.random.normal(init.mu,init.sigma,(init.N,1))

    u_cm_r = func.cm_quantity(head, list_particle, count_f, u_r_at, init.nc_total)
    v_cm_r = func.cm_quantity(head, list_particle, count_f, v_r_at, init.nc_total)
    w_cm_r = func.cm_quantity(head, list_particle, count_f, w_r_at, init.nc_total)

    u_si =func.cm_vel(u_cm_r, count, init.nc_total)
    v_si =func.cm_vel(v_cm_r, count, init.nc_total)
    w_si =func.cm_vel(w_cm_r, count, init.nc_total)

    u = func.anderson_thermostat(u_cm, u_r_at, u_si, locations, init.N)
    v = func.anderson_thermostat(v_cm, v_r_at, v_si, locations, init.N)
    w = func.anderson_thermostat(w_cm, w_r_at, w_si, locations, init.N)

    # if i>=init.n_avg_fp:
    in_x_fp = np.floor((x-init.x_min)/init.hcf)
    in_y_fp = np.floor((y-init.y_min)/init.hcf)
    in_z_fp = np.floor((z-init.z_min)/init.hcf)
    locations_fp = init.ncx_fp*init.ncy_fp*in_z_fp + init.ncx_fp*in_y_fp + in_x_fp
    locations_fp = locations_fp.astype(int)
    
    head_fp, list_particle_fp, count_fp = func.head_list(locations_fp, init.N, init.nct_profile)
    head_fp = head_fp.astype(int)
    list_particle_fp = list_particle_fp.astype(int)

    u_cm_fp1 = func.cm_quantity(head_fp, list_particle_fp, count_fp, u, init.nct_profile)
    v_cm_fp1 = func.cm_quantity(head_fp, list_particle_fp, count_fp, v, init.nct_profile)
    w_cm_fp1 = func.cm_quantity(head_fp, list_particle_fp, count_fp, w, init.nct_profile)

    u_cm_fp1 = func.cm_vel(u_cm_fp1, count_fp, init.nct_profile)
    v_cm_fp1 = func.cm_vel(v_cm_fp1, count_fp, init.nct_profile)
    w_cm_fp1 = func.cm_vel(w_cm_fp1, count_fp, init.nct_profile)

    u_cm_fp = u_cm_fp + u_cm_fp1
    v_cm_fp = v_cm_fp + v_cm_fp1                        
    w_cm_fp = w_cm_fp + w_cm_fp1

    if i%1000==1:
        print("ITERATION - " + 	str(i))

u_cm_fp = u_cm_fp/(init.Niter-init.n_avg_fp)
v_cm_fp = v_cm_fp/(init.Niter-init.n_avg_fp)                       
w_cm_fp = w_cm_fp/(init.Niter-init.n_avg_fp)

e_sim = time.time()
print(str(e_sim-s_sim))