#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:04:32 2024

@author: martin

contains the definition of the gaussian window phi and its canonical dual phidual
"""

import numpy as np
import mpmath


pi=np.pi

phi = lambda T: 2**(1/4)*np.exp(-pi*T**2)


#--------------------dual window phi0---------------------------------
temp=0
for n in range(-10,11):
    temp += (-1)**n*(n+0.5)*np.exp(-pi*(n+0.5)**2)

nmz = temp

ck=[]
for k in range(-10,11):
    sm=0
    for m in range(11):
        sm += (-1)**(k+m)*np.exp(-pi*(m+0.5)*(2*abs(k)+m+0.5))
    
    ck.append(sm/nmz)


def phi0(t):
    k=-10
    temp = 0
    for c in ck:
        temp += c*phi(t-np.sqrt(2)*k)
        k+=1
    
    return temp/(2*float(mpmath.jtheta(3,np.sqrt(2)*pi*t, np.exp(-pi))))
    
    
phidual = np.vectorize(phi0)
#-------------------------------------------------------------------




    