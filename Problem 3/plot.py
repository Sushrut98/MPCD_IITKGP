#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:36:26 2019

@author: sushrut98
"""

from matplotlib import pyplot as plt
import numpy as np

vel_data = np.load('vel_data.npy')
vel_data = np.sum(vel_data,axis=1)/50000
y = np.arange(0,51)

plt.figure()
plt.plot(vel_data, y)
plt.show()