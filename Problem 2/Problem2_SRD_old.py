"""
Code to simulate Flow using Stochastic Rotation Dynamics

Collaborator    :   Sushrut
Place           :   Indian Institute of Technology, Kharagpur
Date            :   25/5/2019

Problem 2: Couette Flow
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Initial Parameters
Niter = 50000
ncx = 201						#Number of cells in GRID Box
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
force_x = 0
acc_x = force_x/mass
mu = 0
sigma = 1
lamda = 0.1
sin_alpha = 1
cos_alpha = 0
dt = lamda/sigma
U = 1	
#----------------------------|  Functions  |----------------------------------#

#Velocity Loading for generating profiles
# def velload(file,column):
# 	x = np.loadtxt('file', delimiter='', usecols=(0,column))
# 	x = np.delete(x, 1, axis=1)
# 	return x
#----------------------------|  Initialisation  |----------------------------------#

			 #------------| Flow Stabilization Run  |---------------#

# x = np.random.uniform(x_min,x_max,(N,1))
# y = np.random.uniform(y_min,y_max,(N,1))
# u = np.random.normal(mu,sigma,(N,1))
# v = np.random.normal(mu,sigma,(N,1))

			 #------------| For Velocity Profiles  |---------------#
x = np.genfromtxt('pos.txt', delimiter=' ', usecols = (0)).reshape(N,1)
y = np.genfromtxt('pos.txt', delimiter=' ', usecols = (1)).reshape(N,1)
u = np.genfromtxt('vel.txt', delimiter=' ', usecols = (0)).reshape(N,1)
v = np.genfromtxt('vel.txt', delimiter=' ', usecols = (1)).reshape(N,1)

grid_shift_x = h*np.random.rand(Niter,1)-h2
grid_shift_y = h*np.random.rand(Niter,1)-h2
vel_data = np.zeros((ncy,Niter))

s_sim = time.time()

# plt.figure(1)
# plt.scatter(x,y,color='teal',s = 10, alpha = 0.8)
# plt.xlim(0,ncx)
# plt.ylim(0,ncy)
# plt.grid(linestyle='dotted')

for i in range(0,Niter):

	u_rel = np.zeros((N,1))
	v_rel = np.zeros((N,1))
	u_rot = np.zeros((N,1))
	v_rot = np.zeros((N,1))
	u_cm = np.zeros((N,1))
	v_cm = np.zeros((N,1))
	u_c = np.zeros((nc_total, 1))
	v_c = np.zeros((nc_total, 1))

	# Streaming Step
	x = x + u*dt + 0.5*acc_x*(dt**2)
	y = y + v*dt
	u = u+acc_x*dt

	#No Slip/Bounceback BC
	#Top wall is moving with velocity U = 1
	top_escape = np.nonzero(y>y_max)
	top_escape = np.asarray(top_escape)
	top_escape = np.delete(top_escape,1, axis=0).ravel()
	time_above_top = np.divide((y[top_escape,0]-y_max),v[top_escape,0])
	y[top_escape,0] = 2*y_max - y[top_escape,0]
	x[top_escape,0] = x[top_escape,0] - np.multiply((2*(u[top_escape,0] - U)), time_above_top)
	u[top_escape,0] = -u[top_escape, 0] + 2*U
	v[top_escape,0] = -v[top_escape,0]

	bottom_escape = np.nonzero(y<y_min)
	bottom_escape = np.asarray(bottom_escape)
	bottom_escape = np.delete(bottom_escape,1, axis=0).ravel()
	time_below_bottom = np.divide((y[bottom_escape,0] - y_min),v[bottom_escape,0])
	y[bottom_escape,0] = -y[bottom_escape,0]
	x[bottom_escape,0] = x[bottom_escape,0] - np.multiply((2*u[bottom_escape,0]), time_below_bottom)
	u[bottom_escape,0] = -u[bottom_escape,0]
	v[bottom_escape,0] = -v[bottom_escape,0]
	
	# Periodic BC in x
	x = x_min+(x-x_min)%Lx
	y = y_min+(y-y_min)%Ly
		
	locations = np.zeros((N,1))                   #locations, head, list array must be initialized for every iteration
	head = -1*(np.ones((nc_total,1)))
	list_particle = -2*np.ones((N,1))
	count = np.zeros((nc_total,1))

	# Assigning particles to grid cells
	# in_x = CellIndex(x, x_min, grid_shift_x[i])
	# in_y = CellIndex(y, y_min, grid_shift_y[i])
	in_x = np.floor((x-x_min+h2-grid_shift_x[i])/h)
	in_y = np.floor((y-y_min+h2-grid_shift_y[i])/h)

	locations = (in_y*ncx+in_x)
	locations = locations.astype(int)

	#---- Usual Method  ----#
	# for j in range(0,nc_total):
	#     neighbour = np.nonzero(locations==j)
	#     neighbour = np.asarray(neighbour).transpose()
	#     u_c[j,0] = np.sum(u[neighbour,0])/len(neighbour) #gives runtime warning because of NaN
	#     u_c = np.nan_to_num(u_cm)
	#     v_c[j,0] = np.sum(v[neighbour,0])/len(neighbour)
	#     v_c = np.nan_to_num(v_cm)

	#---- Linked List Method  ----#
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

	#Collision Step
	Rot = 2*np.random.random_integers(0, 1, size=(nc_total,1))-1
	RandNumRot = np.zeros((N,1))
	N_l = np.arange(0,N,dtype = int).reshape(N,1)

	for m,n in zip(N_l,locations):
			u_cm[m] = u_c[n]           # shape of u_cm = (N,1) & shape of u_c = (nc_total,1)
			v_cm[m] = v_c[n]
			RandNumRot[m] = Rot[n]     # shape of RandNumRot = (N,1) & shape of Rot = (nc_total,1)

	# Collision Step
	u_rel = u - u_cm
	v_rel = v - v_cm

	u_rot = cos_alpha*u_rel - np.multiply((sin_alpha*RandNumRot),v_rel)
	v_rot = np.multiply((sin_alpha*RandNumRot),u_rel) + cos_alpha*v_rel

	u = u_rot + u_cm
	v = v_rot + v_cm

	if i%1000==1:
		print("ITERATION - " + 	str(i))

	# if i%10000==1:
	# 	plt.figure()
	# 	plt.scatter(x,y,color='blue',s = 10, alpha = 0.8)+
	# 	plt.grid(linestyle='dotted')
	# 	plt.savefig('%i.png'%i)
	# 	#np.savetxt('x_%i'%i, x, delimiter=' ')

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
np.savetxt('vel_plot.txt', vel_data)
dis_y = np.arange(0,ncy)
plt.figure()
plt.plot(vel_data, dis_y)
plt.savefig('Vel_profile.png')

e_sim = time.time()
print('Time taken in running simulation = ' + str(e_sim-s_sim) + ' seconds')
s_save = time.time()
a = np.concatenate((x,y,locations), axis=1)
np.savetxt('pos.txt', a, delimiter=' ')
a = np.concatenate((u, v, u_rel, v_rel), axis=1)
np.savetxt('vel.txt', a, delimiter=' ')
a = np.concatenate((u_c, v_c, count), axis=1)
np.savetxt('cell_cm.txt', a, delimiter=' ')
e_save = time.time()
print('Time taken in saving values = ' + str(e_save-s_save) + ' seconds')

# #Plotting
# plt.figure()
# plt.scatter(x,y,color='red',s = 10, alpha = 0.8)
# plt.grid(linestyle='dotted')
# plt.savefig('img.png')
