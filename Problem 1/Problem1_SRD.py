"""
Code to simulate Flow using Stochastic Rotation Dynamics

Collaborator    :   Sushrut
Place           :   Indian Institute of Technology, Kharagpur
Date            :   25/5/2019

Problem 1: Two dimensional square domain with Periodic Boundary condition on all sides
"""
import numpy as np 
import matplotlib.pyplot as plt 
import time

#----------------------------|  Initial Parameters  |----------------------------------#
Niter = 50000
ncx = 11
ncy = 11
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
U = 10

#----------------------------|  Functions  |----------------------------------#
# Streaming
def stream(pos, vel, acc, dt):
    pos = pos + vel*dt + 0.5*acc*(dt**2)
    return pos

#Periodic BC
def periodic(pos, LowerBound, L):
        pos  = LowerBound + ((pos-LowerBound)%L)
        return pos

def CellIndex(pos, LowerBound, grid_shift):
       in_a = np.floor((pos-LowerBound+h2-grid_shift)/h) 
       return in_a

#----------------------------|  Initialisation  |----------------------------------#

             #------------| Flow Stabilization run  |---------------#

x = np.random.uniform(x_min,x_max,(N,1))
y = np.random.uniform(y_min,y_max,(N,1))
u = np.random.normal(mu,sigma,(N,1))
v = np.random.normal(mu,sigma,(N,1))

             #------------| For Velocity Profiles  |---------------#
#x = np.genfromtxt('pos.txt', )

grid_shift_x = h*np.random.rand(Niter,1)-h2
grid_shift_y = h*np.random.rand(Niter,1)-h2
#Add velocity storing array

#Plotting 
# plt.scatter(x,y, s = 10, alpha = 0.8)
# plt.grid(linestyle='dotted')
# plt.show()

e_old = 0
mom_x_old = 0
mom_y_old = 0

s_sim = time.time()
for i in range(0,Niter):
        #streaming Step
        x = stream(x, u, 0, dt)
        y = stream(y, v, 0, dt)

        #Periodic BC
        x = periodic(x,x_min, Lx)
        y = periodic(y,y_min, Ly)

        locations = -2*np.ones((N,1))                   #locations, head, list array must be initialized for every iteration
        head = -1*(np.ones((nc_total,1)))
        list_particle = -2*np.ones((N,1))
        count = np.zeros((nc_total,1))

        #Assigning particles to grid cells
        in_x = CellIndex(x, x_min, grid_shift_x[i])
        in_y = CellIndex(y, y_min, grid_shift_y[i])
        locations = in_y*ncx+in_x
        locations = locations.astype(int)

        u_rel = np.zeros((N,1))
        v_rel = np.zeros((N,1))
        u_rot = np.zeros((N,1))
        v_rot = np.zeros((N,1))
        u_cm = np.zeros((N,1))
        v_cm = np.zeros((N,1))
        u_c = np.zeros((nc_total, 1))
        v_c = np.zeros((nc_total, 1))

        
        #Calculation of Center of Mass Velocity of each cell in a grid
                # Usual Method
        # for j in range(0,nc_total):
        #         neighbour = np.nonzero(locations==j)
        #         neighbour = np.asarray(neighbour).transpose()
        #         u_cm[j] = np.sum(u[neighbour,0])/len(neighbour) #gives runtime warning because of NaN
        #         u_cm = np.nan_to_num(u_cm)
        #         v_cm[j] = np.sum(v[neighbour,0])/len(neighbour)
        #         v_cm = np.nan_to_num(v_cm)
        
                # Linked List Method

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
                u_c[j,0] = u_c[j,0]/count[j,0]
                u_c = np.nan_to_num(u_cm)
                v_c[j,0] = v_c[j,0]/count[j,0]
                v_c = np.nan_to_num(v_cm)        

        #Collision Step
        Rot = 2*np.random.random_integers(0, 1, size=(nc_total,1))-1
        RandNumRot = np.zeros((N,1))
        N_l = np.arange(0,N,dtype = int).reshape(N,1)
        
        for m,n in zip(N_l,locations):
                u_cm[m] = u_c[n]           # shape of u_cm = (N,1) & shape of u_c = (nc_total,1)
                RandNumRot[m] = Rot[n]     # shape of RandNumRot = (N,1) & shape of Rot = (nc_total,1)

        # Collision Step
        u_rel = u - u_cm
        v_rel = v - v_cm
        u_rot = cos_alpha*u_rel - np.multiply((sin_alpha*RandNumRot),v_rel)
        v_rot = np.multiply((sin_alpha*RandNumRot),u_rel) + cos_alpha*v_rel 
        u = u_rot + u_cm
        v = v_rot + v_cm

        e = 0.5*m*np.sum(np.square(u)+np.square(v)) 
        mom_x = m*np.sum(u)
        mom_y = m*np.sum(u)     

        if i%1000==1:
                print(i)
                print(" Energy difference" + str((e-e_old)/e_old))
                print(" Energy difference" + str((mom_x-mom_x_old)/mom_x_old))
                print(" Energy difference" + str((mom_y-mom_y_old)/mom_y_old))
                # plt.figure()
                # plt.scatter(x,y, s = 10, alpha = 0.8)
                # plt.grid(linestyle='dotted')
                # plt.savefig('%i.png'%i)
                # np.savetxt('x_%i'%i, x, delimiter=' ')
        e_old = e
        mom_x_old = mom_x
        mom_y_old = mom_y

e_sim = time.time()
print('Time taken in running simulation = ' + str(e_sim-s_sim) + ' seconds')
s_save = time.time()
a = np.concatenate((x,y,locations), axis=1)
np.savetxt('pos.txt', a, delimiter='  ')
a = np.concatenate((u, v, u_rel, v_rel), axis=1)
np.savetxt('vel.txt', a, delimiter='  ')
a = np.concatenate((u_cm, v_cm), axis=1)
np.savetxt('cell_cm.txt', a, delimiter='  ')
e_save = time.time()
print('Time taken in saving values = ' + str(e_save-s_save) + ' seconds')

#Plotting 
plt.scatter(x,y, s = 10, alpha = 0.8)
plt.grid(linestyle='dotted')
plt.savefig('img.png')