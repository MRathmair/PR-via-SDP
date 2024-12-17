#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:43:01 2024

@author: martin

contains a few auxiliary functions
"""

import numpy as np
import matplotlib.pyplot as plt

from windows import *

def ploteigs(A, name_A:str):
    """
    Parameters
    ----------
    A : square matrix
        DESCRIPTION.
    name_A : str
        name of A.

    Returns
    -------
    None.
    
    plots the eigenvalues of A

    """
    Ev=np.linalg.eigvals(A)
    x, y = np.real(Ev), np.imag(Ev)
    
    plt.scatter(x,y, marker='o')
    plt.title('eigvals of ' + name_A)

    plt.show()
    
def extractmineigenvector(L):
    """
    

    Parameters
    ----------
    L : square matrix

    Returns
    -------
    vector
        eigenvector corresponding to the minimal eigenvalue of L.

    """
    vals,Vecs = np.linalg.eig(L)
    
    minpos=np.argmin(np.abs(vals))
    
    return Vecs[:,minpos]

def dist(z1,z2):
    """
    

    Parameters
    ----------
    z1 : list
        list of coordinates (typically of lenght 2).
    z2 : list
        list of coordinates (typically of length 2).

    Returns
    -------
    float
        euclidean distance between the two vectors z1,z2

    """
    return np.linalg.norm(np.array(z1)-np.array(z2))


def tfsum(X,Y,C,T,dual=False):
    """
    

    Parameters
    ----------
    X : list
        list of x coordinates.
    Y : list
        list of y coordinates.
    C : list
        list of coefficients.
    T : list
        array of positions where the resulting function is to be evaluated.
    dual : boolean, optional
        whether the dual window or the Gaussian is used in the expansion. The default is False.

    Returns
    -------
    V : list
        list of evaluations.

    """
    
    V=np.zeros(len(T), dtype=complex)
    
    if dual==True:
        generator=phidual
    else:
        generator=phi
    
    
    for k in range(len(X)):
        a,b=X[k],Y[k]
        V += C[k]*np.exp(2*1j*np.pi*b*T)*generator(T-a)

    return V

#-------