"""
Code to simulate Flow using Stochastic Rotation Dynamics

Collaborator    :   Sushrut
Place           :   Indian Institute of Technology, Kharagpur
Date            :   25/15/2019

Problem 1: Periodic Boundary on all sides
"""

import numpy as np 
import matplotlib.pyplot as plt 

# Function 

# Streaming
def stream(x, y, u, v, dt):
    x = x + u*dt
    y = y + v*dt
    return x,y

#Periodic BC
def periodic(a, a_min, a_max, La, N):
        
        for j in range(0,N):

                if a[j,0] >= a_max:
                        a[j,0] -= La 

                elif a[j,0] <= a_min:
                        a[j,0] += La

                else:
                        a[j,0] = a[j,0]
        
        return a 


# Checking momentum and energy conservation

# Rotation

# Initial Parameters
Niter = 500
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
x = np.random.uniform(x_min,Lx, N)
y = np.random.uniform(y_min,Lx, N)
u = np.random.normal(mu,sigma,N)
v = np.random.normal(mu,sigma,N)

u_rel = np.zeros((1,N))
v_rel = np.zeros((1,N))
u_rel = np.zeros((1,N))
v_rel = np.zeros((1,N))
u_cm = np.zeros((1,nc_total))
v_cm = np.zeros((1,nc_total))
locations = np.zeros((1,N))

grid_shift_x = h*np.random.rand(1,Niter)-h2
grid_shift_y = h*np.random.rand(1,Niter)-h2

#Add velocity storing array

#Plotting 
plt.scatter(x,y, s = 10, alpha = 0.8)
plt.grid(linestyle='dotted')
plt.show()

for i in range(0,Niter):
        if i%50==1:
                print(i)

        #streaming Step
        x = x + u*dt
        y = y + v*dt

        #Assigning particles to grid cells
        in_x = np.floor((x-x_min+h2-grid_shift_x[0,i])/h)
        in_y = np.floor((y-y_min+h2-grid_shift_y[0,i])/h)

        #Calculation of Center of Mass Velocity of each cell in a grid
  
#Plotting 
plt.scatter(x,y, s = 10, alpha = 0.8)
plt.grid(linestyle='dotted')
plt.show()
