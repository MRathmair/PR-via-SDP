#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:50:12 2024

@author: martin

Test run of reconstruction algorithm; 
Step 1: random signal f is generated
Step 2: compute noisy spectrogram samples of f
Step 3: feed spectrogram samples into algorithm to estimate f(with phase!)
Step 4: compare reconstruction with f
"""

import os
os.chdir("/home/martin/verbund/code")

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from classes import *
from auxiliary import *
from reconstruction import phaserec
a0=1/np.sqrt(2)


#--parameter configuration-----------------------------
T,S=2.5,2.2
R=1.5
r=np.sqrt(2)
nu=0.1 #noise level
epsilon=nu+0.05 #tolerance
h=0.5


#--Step 1: generate random f------------------------------
T_Beta, S_Beta = T+2, S-0.7
Beta = Lattice(T_Beta,S_Beta,a0)

print('#Beta= ', Beta.nop)

m,n=Beta.X.shape


Coeffs = np.sqrt(np.random.rand(m,n))*np.exp(2*pi*1j*np.random.rand(m,n))

    

f=Signal(Beta,Coeffs)

f.plotspectrogram(max(T_Beta, T)+1, max(S_Beta, S)+1, Beta, titletext='spectrogram + generating lattice')



#--Step 2: evaluate spectrogram and add random noise-----------------------------

Omega = Lattice(T+R/2,S+R/2, h)

print('#Omega = ', Omega.nop)
p,q=Omega.X.shape
Vals = f.evalSpectrogramonLattice(Omega)+nu*(np.random.rand(p,q)-0.5)
Sf = Samples(Omega.X.flatten(), Omega.Y.flatten(), Vals.flatten())



#-- Step 3: run reconstruction scheme 

Lambda,VfLambda_estd=phaserec(Sf,T,S,R,epsilon,r)

f.plotspectrogram(max(T_Beta, T)+1, max(S_Beta, S)+1, Lambda, titletext='spectrogram + Lambda')

VfLambda_true = f.evalSTFTonLattice(Lambda)



#--Step 4: compare reconstruction with original f

#determine optimal relative phase factor
c_true, c_estd=VfLambda_true.flatten(), VfLambda_estd.flatten()
u = np.vdot(c_true,c_estd)
rpf_opt = u/np.abs(u)



#compute and print mean squared error on Lambda
error_meansq = np.linalg.norm(c_estd-c_true*rpf_opt)**2/len(c_true)
print('mean squared error = ', error_meansq)



#comparison on function level 
pts = np.linspace(-T-2,T+2,int(T*100))
f_true = tfsum(Beta.X.flatten(), Beta.Y.flatten(), Coeffs.flatten(), pts, dual=False)

f_estd = tfsum(Lambda.X.flatten(), Lambda.Y.flatten(), c_estd, pts, dual=True)

f_true_rotated = f_true*rpf_opt

plt.plot(pts, np.real(f_true_rotated), pts, np.real(f_estd))
plt.title('real part of function and reconstruction')
plt.legend(['f', 'f_{estd}'])
plt.show()

plt.plot(pts, np.imag(f_true_rotated), pts, np.imag(f_estd))
plt.title('imaginary part of function and reconstruction')
plt.legend(['f', 'f_{estd}'])
plt.show()

