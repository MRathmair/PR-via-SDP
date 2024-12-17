#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:54:45 2024

@author: martin

contains the code of the reconstruction scheme
"""
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from classes import *
from auxiliary import *

a0=1/np.sqrt(2)

def phaserec(Sf:Samples, T,S,R, epsilon=0.1, r=2*a0):
    """
    

    Parameters
    ----------
    Sf : Samples
        given spectrogram samples.
    T : >0
        where samples are to be reconstructed(horizontally).
    S : >0
        where samples are to be reconstructed(vertically).
    R : >0
        buffer parameter.
    epsilon : >0, optional
        tolerance parameter. The default is 0.1.
    r : >0, optional
        parameter determining the connectedness. The default is 2*a0.

    Returns
    -------
    Lambda : Lattice
        lattice where coefficients are reconstructed.
    V : list
        the estimated values attatched to Lambda.

    """

    X_Samples,Y_Samples, Vals_samples = Sf.X, Sf.Y, Sf.Values
    Gamma = Lattice(T+R,S+R,a0)
    print('#Gamma = ', Gamma.nop)
    
    Lambda = Lattice(T,S,a0)
    print('#Lambda = ', Lambda.nop)
    
    #Step one: Lifting & Extension----------------------------------------------
    #find C s.t. Psi_C(omega,omega)~Sf(omega), omega \in Omega
    
    d=Gamma.nop
    As,bs=[],[]
    
    for i in range(len(X_Samples)):
        z=[X_Samples[i], Y_Samples[i]]
        val=Vals_samples[i]
        
        As.append(Gamma.evalPsi(z,z))
        bs.append(val)
    
    
    C=cp.Variable((d,d), hermitian=True)
    constraints = [C>>0]
    constraints += [cp.trace(C @ np.conjugate(As[i])) == bs[i] for i in range(len(bs))]
    
    prob = cp.Problem(cp.Minimize(cp.max(cp.abs(cp.diag(C)))), constraints)
    prob.solve(verbose=True, eps = epsilon, max_iters=8000)
    
    C_opt=C.value
    
    ploteigs(C_opt, 'C_opt')
    
    PsiC=Ansatzfunction(Gamma,C_opt)
    
    #return PsiC

    
    #Step two: Laplacian null space
            #1) assemble Laplacian L
    nop=Lambda.nop
    L=np.zeros((nop,nop), dtype=complex)
    
    Xflat,Yflat=Lambda.X.flatten(), Lambda.Y.flatten()
    
    temp = 0
    for i in range(nop):
        zi=[Xflat[i],Yflat[i]]
        
        temp += PsiC.evaluate(zi,zi)
        for j in range(nop):
            zj=[Xflat[j],Yflat[j]]
            
            dst= dist(zi,zj)
            
            if (dst>10**(-10) and dst< r+10**(-10)):
                L[i,i] +=  PsiC.evaluate(zj,zj)
                L[i,j]  = -PsiC.evaluate(zi,zj)
                
    estdnorm=np.sqrt(temp)
    #print('norm Vf on Lambda estimated =', estdnorm)
    
    ploteigs(L,'Laplacian L')
    
            #2) extract minimal eigenvector
    
    v= extractmineigenvector(L)*estdnorm
    V = np.reshape(v, Lambda.X.shape)
    
    return Lambda, V

