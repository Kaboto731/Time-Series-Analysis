#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:58:39 2019

@author: manuel TSA Final
"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import copy
#Compute ACF Here
def ACF(x):
    gam= []
    #the number of values that we have
    vals = len(x)
    #the variance of X
    
    #the value from the mean

    #Since this is based off of data we cannot assume a weakly stationary process: 
    for j in range(vals):
        xa = (x[0:vals-j]-np.mean(x))
        xb = (x[j:vals]-np.mean(x))
        xvar= np.var(x)
        gam.append(np.mean(xa*xb)/(xvar))
    return gam

        
x = np.zeros(1000)
t = np.arange(0.,1000.,1)
t2 = np.arange(0,1000,4)
theory = []
for i in range(0,20):
    theory.append(0.1**i) 
#Graph's the first functions theory ACF
plt.plot(t[0:10],theory[0:10])
plt.title('Theoretical Autocorrelation')
plt.xlabel('j')
plt.ylabel('rho j')
plt.show()
#Graphs the second functions theory ACF
plt.plot(t2[0:10],theory[0:10])
plt.title('Theoretical Autocorrelation')
plt.xlabel('rho j')
plt.xlabel('j')
plt.show()
#Now for the third function theory ACF
rho0 = 1
rho2 = 0.1/(9.0-0.1)
rho1 = 9*rho2
rho3 = rho2
rho4 = 0.1*rho2+0.1
x2 = []
x2.append(rho0)
x2.append(rho1)
x2.append(rho2)
x2.append(rho3)
x2.append(rho4)
for i in range(5,30):
    x2.append(0.1*x2[i-1]+0.1*x2[i-4])
plt.plot(x2)
plt.title('Theoretical Autocorrelation')
plt.xlabel('j')
plt.ylabel('rho j')
plt.show()
#White noise with a mean of 0 and a standard deviation of 1
epsilon = np.random.normal(0,1,len(x))
x[0] = 0.5+epsilon[0]
for i in range (len(x)-1):
    x[i+1] = 0.5+0.1*x[i]+epsilon[i]
#plot the first ACF
acf1 = ACF(x)
plt.plot(t[0:10],acf1[0:10])
plt.ylabel('Autocorrelation')
plt.xlabel('x(t)')
plt.show()
#plot the second ACF, spacing has changed
plt.plot(t2[0:10],acf1[0:10])
plt.ylabel('Autocorrelation')
plt.xlabel('x(t)')
plt.show()
x3 = np.zeros(1000)
#building the last function
for i in range(len(x)-5):
    if i<5:
        x3[i] = 0.5+epsilon[i]
    else:
        x3[i+4] = 0.5+0.1*x3[i+3]+0.1*x3[i]+epsilon[i]
#plot of all the times
plt.plot(t,x)
plt.title('Plot of x(t) = 0.5+0.1x(t-1)+epsilon')
plt.xlabel('time')
plt.ylabel('x(t)')
plt.show()
plt.plot(t2,x[0:250])
plt.title('Plot of x(t) = 0.5+0.1x(t-4)+epsilon')
plt.xlabel('time')
plt.ylabel('x(t)')
plt.show()
plt.plot(t,x3)
plt.title('Plot of x(t) = 0.5+0.1x(t-1)+0.1x(t-4)+epsilon')
plt.xlabel('time')
plt.ylabel('x(t)')
#plot of ACF
acf3=ACF(x3)
plt.plot(acf3)
plt.title('ACF of x(t) = 0.5+0.1x(t-1)+0.1x(t-4)+epsilon')
plt.xlabel('j')
plt.ylabel('rho j')
plt.show()
plt.plot(acf3[0:30])
plt.title('ACF of x(t) = 0.5+0.1x(t-1)+0.1x(t-4)+epsilon')
plt.xlabel('j')
plt.ylabel('rho j')
plt.show()