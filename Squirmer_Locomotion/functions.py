import numpy as np 
from numba import jit, vectorize
import initialize as init

#-------------------------------------------------------------------------------------------------------#
#--------------------------------------------| Functions |----------------------------------------------#
#-------------------------------------------------------------------------------------------------------#

h = init.h
h2 = init.h2

@jit
def pos_stream(pos, vel, acc, dt, Num):
    pos_new = np.zeros((Num,1))			
    for i in range(Num):
        pos_new[i,0] = pos[i,0] + vel[i,0]*dt + 0.5*acc*(dt**2)
    return pos_new

@jit
def vel_stream(vel, acc, dt, Num):
    new_vel = np.zeros((Num,1))
    for i in range(Num):
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