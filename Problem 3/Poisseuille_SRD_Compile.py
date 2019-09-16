import numpy as np
import time
from numba import jit,vectorize,prange,cuda
import matplotlib.pyplot as plt

# Initial Parameters
Niter = 50000
ncx = 201
ncy = 51
nc_total = (ncx)*(ncy)
h = 1
h2 = h/2
Lx = (ncx-1)*h                  #Lx & Ly are length of simulation box NOT grid
Ly = (ncy-1)*h
x_min = 0
y_min = 0
x_max = x_min+Lx
y_max = y_min+Ly
np_avg = 10
N = (ncx-1)*(ncy-1)*np_avg
mass = 1
force_x = 0.0002
acc_x = force_x/mass
mu = 0
sigma = 1
lamda = 0.1
sin_alpha = 1
cos_alpha = 0
dt = lamda/sigma


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
def vel_cm(locations, u, v):
	#head, list array must be initialized for every iteration
	head = -1*(np.ones((nc_total,1)))
	list_particle = -2*np.ones((N,1))
	count = np.zeros((nc_total,1))
	u_c = np.zeros((nc_total, 1))
	v_c = np.zeros((nc_total, 1))
	
	for j in range(0,N):
		list_particle[j,0] = head[locations[j,0],0]
		head[locations[j,0],0] = j
		count[locations[j,0],0] += 1

	head = head.astype(int)
	list_particle = list_particle.astype(int)

	for j in range(0,nc_total):
		k = head[j,0]

		while(k != -1):
			u_c[j,0] = u_c[j,0] + u[k,0]
			v_c[j,0] = v_c[j,0] + v[k,0]
			k = list_particle[k,0]    

		if count[j,0] == 0:
			continue
		else: 
			u_c[j,0] = u_c[j,0]/count[j,0]
			v_c[j,0] = v_c[j,0]/count[j,0]

	return u_c, v_c

@jit
def collide(u, v, u_cm, v_cm, locations):
	u_rel = np.zeros((N,1))
	v_rel = np.zeros((N,1))
	u_rot = np.zeros((N,1))
	v_rot = np.zeros((N,1))
	RandNumRot = 2*np.random.random_integers(0, 1, size=(nc_total,1))-1

	for i in range(N):
		u_rel[i,0] = u[i,0] - u_cm[locations[i,0],0]
		v_rel[i,0] = v[i,0] - v_cm[locations[i,0],0]
		u_rot[i,0] = cos_alpha*u_rel[i,0] - np.multiply((sin_alpha*RandNumRot[locations[i,0],0]),v_rel[i,0])
		v_rot[i,0] = np.multiply((sin_alpha*RandNumRot[locations[i,0],0]),u_rel[i,0]) + cos_alpha*v_rel[i,0]
		u[i,0] = u_rot[i,0] + u_cm[locations[i,0],0]
		v[i,0] = v_rot[i,0] + v_cm[locations[i,0],0]
	return u,v

#----------------------------|  Initialisation  |----------------------------------#

			 #------------| Flow Stabilization Run  |---------------#
#x = np.random.uniform(x_min,x_max, (N,1))
#y = np.random.uniform(y_min,y_max, (N,1))
#u = np.random.normal(mu,sigma,(N,1))
#v = np.random.normal(mu,sigma,(N,1))

			 #------------| For Velocity Profiles  |---------------#
x=np.load('x_poiseullie.npy')
y=np.load('y_poiseullie.npy')
u=np.load('u_poiseullie.npy')
v=np.load('v_poiseullie.npy')

grid_shift_x = h*np.random.rand(Niter,1)-h2
grid_shift_y = h*np.random.rand(Niter,1)-h2
vel_data = np.zeros((ncy,Niter))

s_sim = time.time()

for i in range(0,Niter):
	x = pos_stream(x,u,acc_x,dt)
	y = pos_stream(y,v,0,dt)	
	u = vel_stream(u,acc_x,dt)

	top_escape = np.nonzero(y>y_max)
	top_escape = np.asarray(top_escape)
	top_escape = np.delete(top_escape,1, axis=0).ravel()
	time_above_top = np.divide((y[top_escape,0]-y_max),v[top_escape,0])
	y[top_escape,0] = 2*y_max - y[top_escape,0]
	x[top_escape,0] = x[top_escape,0] - 2*np.multiply((u[top_escape,0]), time_above_top) + 0.5*acc_x*np.square(time_above_top)
	u[top_escape,0] = -u[top_escape, 0] + 2*acc_x*time_above_top
	v[top_escape,0] = -v[top_escape,0]

	bottom_escape = np.nonzero(y<y_min)
	bottom_escape = np.asarray(bottom_escape)
	bottom_escape = np.delete(bottom_escape,1, axis=0).ravel()
	time_below_bottom = np.divide((y[bottom_escape,0] - y_min),v[bottom_escape,0])
	y[bottom_escape,0] = -y[bottom_escape,0]
	x[bottom_escape,0] = x[bottom_escape,0] - np.multiply((2*u[bottom_escape,0]), time_below_bottom) + 0.5*acc_x*np.square(time_below_bottom)
	u[bottom_escape,0] = -u[bottom_escape,0] + 2*acc_x*time_below_bottom
	v[bottom_escape,0] = -v[bottom_escape,0]
	
	# Periodic BC in x
	x = periodic(x,x_min, Lx)

	in_x = CellIndex(x, x_min, grid_shift_x[i])
	in_y = CellIndex(y, y_min, grid_shift_y[i])
	locations = np.zeros((N,1))
	locations = (in_y*ncx+in_x)
	locations = locations.astype(int)

	u_cm, v_cm = vel_cm(locations, u, v)

	u,v = collide(u,v,u_cm,v_cm,locations)

	if i%1000==1:
		print("ITERATION - " + 	str(i))
		
	eta = np.floor((y-y_min)/h)
	for j in range(0,ncy):
		same_eta = np.nonzero(eta == j)
		same_eta = np.asarray(same_eta)
		same_eta = np.delete(same_eta,1, axis=0).ravel()
		if len(eta) == 0:
			continue
		else:
			vel_data[j,i] = np.sum(u[same_eta,0])/len(same_eta)

vel_data = np.sum(vel_data,axis=1)/Niter
dis_y = np.arange(0,ncy)
plt.figure()
plt.plot(vel_data, dis_y)
plt.savefig('vel_profile.png')
e_sim = time.time()
print(str(e_sim-s_sim))