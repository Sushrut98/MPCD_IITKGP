"""
Code to simulate Flow using Stochastic Rotation Dynamics

Collaborator    :   Sushrut
Place           :   Indian Institute of Technology, Kharagpur
Date            :   25/15/2019

Problem 1: Periodic Boundary on all sides
"""

import numpy as np 
import matplotlib.pyplot as plt 
import time

# Function 

# Streaming
def stream(x, y, u, v, dt):
    x = x + u*dt
    y = y + v*dt
    return x,y

#Periodic BC
def periodic(a, a_min, La):
        a  = a_min + (a%10)
        # for j in range(0,N):

        #         if a[j,0] >= a_max:
        #                 a[j,0] -= La 

        #         elif a[j,0] <= a_min:
        #                 a[j,0] += La    

        #         else:
        #                 a[j,0] = a[j,0]
        # print(a)

        # np.place(a, a < a_min, x_min + a%La)
        # np.place(a, a > a_max, x_min + a%La)
        return a 

# Checking momentum and energy conservation

# Rotation

s_init = time.time()
# Initial Parameters
Niter = 50000
ncx = 11
ncy = 11
nc_total = (ncx)*(ncy)
h = 1
h2 = h/2
Lx = (ncx-1)*h
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

# Initialisation
x = np.random.uniform(x_min,Lx, (N,1))
y = np.random.uniform(y_min,Lx, (N,1))
u = np.random.normal(mu,sigma,(N,1))
v = np.random.normal(mu,sigma,(N,1))

u_rel = np.zeros((N,1))
v_rel = np.zeros((N,1))
u_rel = np.zeros((N,1))
v_rel = np.zeros((N,1))
u_cm = np.zeros((nc_total, 1))
v_cm = np.zeros((nc_total, 1))
locations = np.zeros((N,1))

grid_shift_x = h*np.random.rand(Niter,1)-h2
grid_shift_y = h*np.random.rand(Niter,1)-h2

e_init = time.time()

print('Time taken in initialisation = ' + str(e_init-s_init) + ' seconds')

#Add velocity storing array

#Plotting 
# plt.scatter(x,y, s = 10, alpha = 0.8)
# plt.grid(linestyle='dotted')
# plt.show()

s_sim = time.time()
for i in range(0,Niter):
        #streaming Step
        x = x + u*dt
        y = y + v*dt

        #Periodic BC for x
        x = periodic(x,x_min, Lx)

        #Periodic BC for y
        y = periodic(y,y_min, Ly)

        #Assigning particles to grid cells
        in_x = np.floor((x-x_min+h2-grid_shift_x[i])/h)
        in_y = np.floor((y-y_min+h2-grid_shift_y[i])/h)
        locations = in_y*ncx+in_x+1

        
        #Calculation of Center of Mass Velocity of each cell in a grid
        for j in range(nc_total):
                neighbour = np.nonzero(locations==j)
                neighbour = np.asarray(neighbour).transpose()
                u_cm[j] = np.sum(u[neighbour])/len(neighbour) #gives runtime warning because of NaN
                u = np.nan_to_num(u)
                v_cm[j] = np.sum(v[neighbour])/len(neighbour)
                v = np.nan_to_num(v)

        u_rel = u - u_cm[locations]
        v_rel = v - v_cm[locations]

        #Collision Step


        if i%1000==1:
                print(i)
                # plt.figure()
                # plt.scatter(x,y, s = 10, alpha = 0.8)
                # plt.grid(linestyle='dotted')
                # plt.savefig('%i.png'%i)
                #np.savetxt('x_%i'%i, x, delimiter=' ')

e_sim = time.time()
print('Time taken in running simulation = ' + str(e_sim-s_sim) + ' seconds')

s_save = time.time()
a = np.concatenate((x,y,locations), axis=1)
np.savetxt('a.txt', a, delimiter='  ')
e_save = time.time()
print('Time taken in saving values = ' + str(e_save-s_save) + ' seconds')

#Plotting 
# plt.scatter(x,y, s = 10, alpha = 0.8)
# plt.grid(linestyle='dotted')
# plt.show()
