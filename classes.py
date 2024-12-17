#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:55:40 2024

@author: martin

contains important classes
"""


import numpy as np
import matplotlib.pyplot as plt

pi = np.pi


#------------------------------------------------------------------------------

class Lattice:
    """
    Lattice class
    
    Attributes: 
        stepsize: h
        x: x coordinates
        y: y coordinates
        X: x coordinates of the grid
        Y: y coordinates of the grid
        nop: number of lattice points
        
    Methods:
        evalSigma(z): returns a matrix w s.t. (matrix) inner product <c,w> = Sigma_c(z) (i.e. evaluation of STFT of Sum(c_i*pi(i)phi))
        evalPsi(z1,z2): returns a matrix W s.t. (matrix) inner product <C,W> = Psi_C(z1,z2) (i.e. the lifted version of Sigma_c)
        
    """
    def __init__(self, T,S,h):
        
        self.stepsize = h
        
        M=np.floor(T/h).astype(int)
        N=np.floor(S/h).astype(int)
        x=np.arange(-M,M+1)*h
        y=np.arange(-N,N+1)*h
        
        self.x, self.y = x,y
        
        X,Y=np.meshgrid(x,y)
        
        self.X, self.Y = X,Y
        
        self.nop = X.size
        
    def evalSigma(self, z):
        #return matrix w s.t. <c,w> = Sigma_c(z)
        
        p,q=z
        X,Y = self.X, self.Y
        return np.exp(-pi/2*(p-X)**2-pi/2*(q-Y)**2-pi*1j*(p+X)*(q-Y))
        
        
    def evalPsi(self, z1,z2):
        #return matrix W s.t. <C,W> = Psi_C(z1,z2)
        
        w1 = self.evalSigma(z1).flatten()
        w2 = self.evalSigma(z2).flatten()
        return np.outer(np.conjugate(w2),w1)

#------------------------------------------------------------------------------


class Signal:
    """
    class Signal: represents a L²-function f on the real line
    
    Attributes:
        Lattice: Lambda (the underlying Lattice)
        Coeeffs: c (coefficients attatched to each of the lattice points)
        
    Methods:
        evalSTFT(z): evaluation of the STFT of f at z
        evalSTFTonLattice(Beta): evaluation of the STFT of f on a Lattice Beta
        evalSpectrogramonLattice(Beta): evaluation of the spectrogram of f on Beta
        plotspectrogram(xmax,ymax,Lambda): plot spectrogram in region [-xmax,xmax]x[-ymax,ymax] and Lambda on top of it
    
    """
    def __init__(self, Lambda: Lattice, c):
        self.Lattice = Lambda
        self.Coeffs = c
        
    def evalSTFT(self,z):
        c=self.Coeffs
        w=self.Lattice.evalSigma(z)
        return (c.flatten() @ w.flatten())
        
    
    def evalSTFTonLattice(self, Beta: Lattice):
        X,Y = Beta.X, Beta.Y
        m,n = np.shape(X)
        G = np.zeros((m,n), dtype = complex)
        for k in range(m):
            for l in range(n):
                G[k,l] = self.evalSTFT([X[k,l],Y[k,l]])
        return G
            
    def evalSpectrogramonLattice(self, Omega: Lattice):
        G = self.evalSTFTonLattice(Omega)
        return np.abs(G)**2
    
    def plotspectrogram(self, xmax, ymax, Lambda: Lattice, titletext=''):
        Omega = Lattice(xmax,ymax,0.05)
        Sf = self.evalSpectrogramonLattice(Omega)
        
        fig, ax = plt.subplots()

        ax.imshow(Sf, interpolation='none', extent=[-xmax, xmax, -ymax, ymax])
        ax.set_aspect(1) 
        
        X=Lambda.X.flatten()
        Y=Lambda.Y.flatten()
        
        plt.scatter(X,Y, facecolors='none', edgecolors='w',s=7, marker ='P')
        plt.title(titletext)
        plt.show()
#-----------------------------------------------------------------------------

class Ansatzfunction:
    """
    class Ansatzfunction: represents an Ansatzfunction (lifting!)
    
    Attributes:
        Lattice: Lambda (the underlying lattice)
        Coeffs: C (the corresponding coefficients)
        
    Methods:
        evaluate(z1,z2): evaluation of the Ansatzfunction at (z1,z2)
    """
    def __init__(self, Lambda: Lattice, C):
        self.Lattice = Lambda
        self.Coeffs = C
    
    def evaluate(self,z1,z2):
        Lambda=self.Lattice
        W=Lambda.evalPsi(z1,z2)
        return (self.Coeffs.flatten() @ W.flatten())
    
#-----------------------------------------------------------------------------

class Samples:
    """
    class Samples: contains information of positions and values of a sampled function
        
    Attributes:
        X: X (x coordinates of the grid)
        Y: Y (y coordinates of the grid)
        Values: Vals (values at the corresponding points)
    """
    def __init__(self, X, Y, Vals):
        self.X = X
        self.Y = Y
        self.Values = Vals
        