"""
Code to simulate Flow using Stochastic Rotation Dynamics

Collaborator    :   Sushrut
Place           :   Indian Institute of Technology, Kharagpur
Date            :   25/15/2019

Problem 3: Poiseuille Flow
"""

import numpy as np 
import matplotlib.pyplot as plt 
import time

# Function 

# Streaming
def stream(a, a_u, dt):
    a = a + a_u*dt
    return a

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

def index(a, a_min, h, grid_shift):
       in_a = np.floor((x-x_min+(h/2)-grid_shift)/h) 
       return in_a

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
U = 1

# Initialisation
x = np.random.uniform(x_min,Lx, (N,1))
y = np.random.uniform(y_min,Lx, (N,1))
u = np.random.normal(mu,sigma,(N,1))
v = np.random.normal(mu,sigma,(N,1))

u_rel = np.zeros((N,1))
v_rel = np.zeros((N,1))
u_rot = np.zeros((N,1))
v_rot = np.zeros((N,1))
u_cm = np.zeros((nc_total, 1))
v_cm = np.zeros((nc_total, 1))
locations = np.zeros((N,1))

grid_shift_x = h*np.random.rand(Niter,1)-h2
grid_shift_y = h*np.random.rand(Niter,1)-h2

e_init = time.time()
#print(v)
print('Time taken in initialisation = ' + str(e_init-s_init) + ' seconds')

#Add velocity storing array

#Plotting 
# plt.scatter(x,y, s = 10, alpha = 0.8)
# plt.grid(linestyle='dotted')
# plt.show()

s_sim = time.time()
for i in range(0,Niter):
    #streaming Step
    x = stream(x, u, dt)
    y = stream(y, v, dt)

    #Periodic BC for x & y
    y = periodic(y,y_min, Ly)

    #print(v.shape)

    #No Slip/Bounceback BC
    #Top wall is moving with velocity U = 1
    top_escape = np.nonzero(y>y_max)
    top_escape = np.asarray(top_escape)
    top_escape = np.delete(top_escape,1, axis=0).ravel()
    #print(top_escape.shape, v[top_escape,0].shape) #top_escape + y[top_escape,0], 
    time_above_top = np.divide((y[top_escape,0]-y_max),v[top_escape,0])
    y[top_escape,0] = 2*y_max - y[top_escape,0]
    x[top_escape,0] = x[top_escape,0] - np.multiply((2*(u[top_escape,0] - U)), time_above_top)
    u[top_escape,0] = -u[top_escape, 0] + 2*U
    v[top_escape,0] = -v[top_escape,0]

    bottom_escape = np.nonzero(y<y_min)
    bottom_escape = np.asarray(bottom_escape)
    bottom_escape = np.delete(bottom_escape,1, axis=0).ravel()
    time_below_bottom = np.divide((y[bottom_escape,0] - y_min),v[bottom_escape])
    y[bottom_escape,0] = -y[bottom_escape,0]
    x[bottom_escape,0] = x[bottom_escape,0] - np.multiply((2*u[bottom_escape,0]), time_below_bottom)
    u[bottom_escape,0] = -u[bottom_escape,0]
    v[bottom_escape,0] = -v[bottom_escape,0]

    x = periodic(x,x_min, Lx)

    #Assigning particles to grid cells
    in_x = index(x, x_min, h, grid_shift_x[i])
    in_y = index(y, y_min, h, grid_shift_y[i])
    locations = in_y*ncx+in_x+1
    locations = locations.astype(int)

    
    #Calculation of Center of Mass Velocity of each cell in a grid
    for j in range(0,nc_total):
        neighbour = np.nonzero(locations==j)
        neighbour = np.asarray(neighbour).transpose()
        u_cm[j,0] = np.sum(u[neighbour,0])/len(neighbour) #gives runtime warning because of NaN
        u_cm = np.nan_to_num(u_cm)
        v_cm[j,0] = np.sum(v[neighbour,0])/len(neighbour)
        v_cm = np.nan_to_num(v_cm)

    n = 0
    RandNumRot = 2*np.random.random_integers(0, 1, size=(nc_total,1))-1

    #Collision Step
    for m in range(N), n in locations:
        u_rel[m] = u[m] - u_cm[n]
        v_rel[m] = v[m] - v_cm[n]

    for m in range(N), n in locations:
        u_rot[m] = cos_alpha*u_rel[m] - (sin_alpha*RandNumRot[n,0])*v_rel[m]
        v_rot[m] = (sin_alpha*RandNumRot[n,0])*u_rel[m] + cos_alpha*v_rel[m] 

    for m in range(N), n in locations:
        u[m] = u_rot[m] + u_cm[n]
        v[m] = v_rot[m] + v_cm[n]

    #Position change in collision       

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
a = np.concatenate((u, v, u_rel, v_rel), axis=1)
np.savetxt('v.txt', a, delimiter='  ')
a = np.concatenate((u_cm, v_cm), axis=1)
np.savetxt('s.txt', a, delimiter='  ')
e_save = time.time()
print('Time taken in saving values = ' + str(e_save-s_save) + ' seconds')

#Plotting 
plt.scatter(x,y, s = 10, alpha = 0.8)
plt.grid(linestyle='dotted')
plt.savefig('img.png')
