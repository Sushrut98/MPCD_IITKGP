#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:09:47 2019

@author: sushrut98
"""

import numpy as np
from matplotlib import pyplot as plt

x = np.genfromtxt('vel_data_correct.txt')
y = np. arange(0,50)

plt.figure()
plt.plot(x,y,color='teal')
plt.xlabel('Velocity')
plt.ylabel('Distance')
plt.grid(linestyle='dotted')
plt.savefig('plot.png')