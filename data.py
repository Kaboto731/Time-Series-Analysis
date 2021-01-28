#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:12:28 2019

@author: manuel
"""

import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
file = 'Ocean Salinity.csv'
file2 = 'OceanTemp.csv'
file3 = 'Oscillator1993expo.csv'
data = pd.read_csv(file)
plt.show()
data = data.values
Year= []
Month = []
Time = []
Sal = []
data2 = pd.read_csv(file2)
data2 = data2.values
Year2= []
Month2 = []
Time2 = []
Temp = []
data3 = pd.read_csv(file3)
data3= data3.values
t=np.arange(1,len(data3)+1,1)
plt.plot(t,data3)
plot_acf(data3)
plot_pacf(data3)
plt.title('Oscillator 1993 Challenge')
plt.show()
#breaking up data in various ways for plotting      
for i in range(len(data)):
    Year.append(data[i][0])
    Month.append(data[i][1])
    Sal.append(data[i][2])
    Time.append((Year[i]-70)*12+(Month[i]-1))
for i in range(len(data2)):
    Year2.append(data2[i][0])
    Month2.append(data2[i][1])
    Temp.append(data2[i][2])
    #months since the 1970's
    Time2.append((Year2[i]-70)*12+(Month2[i]-1))
plt.plot(Time,Sal)
plt.title(' Ocean Salinity in months since 1970')
plt.xlabel('Months')
plt.ylabel('Ocean Salinity')
plt.show()
plt.plot(Time2,Temp)
plt.title(' Ocean Temp in months since 1970')
plt.xlabel('Months')
plt.ylabel('Ocean Temp')
plt.show()
#Detrending and deseasoning Oscillator
#we see 8 seasons in the data
xmat3 = np.zeros((130,8))
mu1xmat3 = np.zeros(130)
mu2xmat3 = np.zeros(8)
j=0
k=0
#reformatting into matrix
for i in range(len(data3)):
    xmat3[j][k] = data3[i]
    j = j+1
    if j==130:
        j=j-130
        k=k+1
#taking means
k=0
j=0
for i in range(len(mu1xmat3)):
    mu1xmat3[i]=np.mean(xmat3[i])
    j = j+1
    if j==130:
        j=j-130
        k=k+1
    if xmat3[j][k]==0:
        continue
    else:
        xmat3[i] = xmat3[i]-mu1xmat3[i]
k=0
j=0
transposexmat3 = xmat3.T
for i in range(len(mu2xmat3)):
    mu2xmat3[i]=np.mean(xmat3[i])
    j = j+1
    if j==130:
        j=j-130
        k=k+1
    if transposexmat3[k][j]==0:
        continue
    else:
        transposexmat3[i] = transposexmat3[i]-mu2xmat3[i]
dedata3=np.zeros(len(data3))
for i in range(len(data3)):
    dedata3[i] = xmat3[j][k]
    j = j+1
    if j==130:
        j=j-130
        k=k+1
#detrending and de-seasoning Sal about 4 samples each year
xmat = np.zeros((15,4))
mu1xmat = np.zeros(15)
mu2xmat = np.zeros(4)
j=0
k=0
#reformatting into matrix
for i in range(len(Sal)):
    xmat[j][k] = Sal[i]
    j = j+1
    if j==15:
        j=j-15
        k=k+1
#taking row means
k=0
j=0
for i in range(len(mu1xmat)):
    mu1xmat[i]=np.mean(xmat[i])
    j = j+1
    if j==15:
        j=j-15
        k=k+1
    if xmat[j][k]==0:
        continue
    else:
        xmat[i] = xmat[i]-mu1xmat[i]
#taking column means
k=0
j=0
transposexmat = xmat.T
for i in range(len(mu2xmat)):
    mu2xmat[i]=np.mean(xmat[i])
    j = j+1
    if j==15:
        j=j-15
        k=k+1
    if transposexmat[k][j]==0:
        continue
    else:
        transposexmat[i] = transposexmat[i]-mu2xmat[i]
dedata=np.zeros(len(Sal))
k=0
j=0
for i in range(len(Sal)):
    dedata[i] = xmat[j][k]
    j = j+1
    if j==15:
        j=j-15
        k=k+1
#detrending and de-seasoning Temp
xmat2 = np.zeros((15,5))
mu1xmat2 = np.zeros(15)
mu2xmat2 = np.zeros(5)
j=0
k=0
#reformatting into matrix
for i in range(len(Temp)):
    xmat2[j][k] = Temp[i]
    j = j+1
    if j==15:
        j=j-15
        k=k+1
#taking means
k=0
j=0
for i in range(len(mu1xmat)):
    mu1xmat2[i]=np.mean(xmat2[i])
    j = j+1
    if j==15:
        j=j-15
        k=k+1
    if xmat2[j][k]==0:
        continue
    else:
        xmat2[i] = xmat2[i]-mu1xmat2[i]
k=0
j=0
transposexmat2 = xmat2.T
for i in range(len(mu2xmat2)):
    mu2xmat2[i]=np.mean(xmat2[i])
    j = j+1
    if j==15:
        j=j-15
        k=k+1
    if transposexmat2[k][j]==0:
        continue
    else:
        transposexmat2[i] = transposexmat2[i]-mu2xmat2[i]
k=0
j=0
dedata2=np.zeros(len(Temp))
for i in range(len(Temp)):
    dedata2[i] = xmat2[j][k]
    j = j+1
    if j==15:
        j=j-15
        k=k+1
        #plotting the ACF and PACF of the detrended and deseasoned data
plot_acf(dedata)
plt.title('ACF of Ocean Salinity ')
plot_pacf(dedata)
plt.title('PACF of Ocean Salinity')
plot_acf(dedata2)
plt.title('ACF of Ocean Temp')
plot_pacf(dedata2)
plt.title('PACF of Ocean Temp')
plot_acf(dedata3)
plt.title('ACF of Oscillator')
plot_pacf(dedata3)
plt.title('PACF of Oscillator')
#First model for Oscillating data
model = ARIMA(dedata3, order =(7,0,8))
modelfit = model.fit(disp=0)
print(modelfit.summary())
#plot residual errors of first model
residual = DataFrame(modelfit.resid)
residual.plot()
plt.title('Oscillator Residual for ARMA(7,8)')
plt.show()
residual.plot(kind='kde')
plt.title('Oscillator Residual Density for ARMA(7,8)')
plt.show()
print(residual.describe())
#Seconds model for Oscillating data
model2 = ARIMA(dedata3, order =(3,0,1))
model2fit = model2.fit(disp=0)
print(model2fit.summary())
#plot residual errors for second model
residual2 = DataFrame(model2fit.resid)
residual2.plot()
plt.show()
residual2.plot(kind='kde')
plt.show()
print(residual2.describe())

#First model for Temp 
model3 = ARIMA(dedata2, order =(4,0,4))
model3fit = model3.fit(disp=0)
print(model3fit.summary())
#plot residual errors for first model
residual3 = DataFrame(model3fit.resid)
residual3.plot()
plt.title('ARMA(4,4) Residual Plot for Ocean Temp')
plt.show()
residual3.plot(kind='kde')
plt.title('ARMA(4,4) Residual Density for Ocean Temp')
plt.show()
#Second model for Temp
model4 = ARIMA(dedata2,order = (2,0,0))
model4fit = model4.fit(disp=0)
print(model4fit.summary())
#plot residual errors for the second model
residual4 = DataFrame(model4fit.resid)
residual4.plot()
plt.title('ARMA(2,0) Residual Plot for Ocean Temp')
plt.show()
residual4.plot(kind='kde')
plt.title('ARMA(2,0) Residual Density for Ocean Temp')
plt.show()
print(residual4.describe())

#First model for Salinity 
model5 = ARIMA(dedata, order =(2,0,2))
model5fit = model5.fit(disp=0)
print(model5fit.summary())
#plot residual errors for first model
residual5 = DataFrame(model5fit.resid)
residual5.plot()
plt.title('ARMA(2,2) Residual Plot for Ocean Sal')
plt.show()
residual5.plot(kind='kde')
plt.title('ARMA(2,2) Residual Density for Ocean Sal')
plt.show()
print(residual5.describe())
#Second model for Salinity
model6 = ARIMA(dedata, order =(4,0,3))
model6fit = model6.fit(disp=0)
print(model6fit.summary())
#plot residual errors for first model
residual6 = DataFrame(model6fit.resid)
residual6.plot()
plt.title('ARMA(4,3) Residual Plot for Ocean Sal')
plt.show()
residual6.plot(kind='kde')
plt.title('ARMA(4,3) Residual Density for Ocean Sal')
plt.show()
 

#SCRAPPED CODE HERE: SAVED FOR SOME TIME LATER
#plt.scatter(Year3,Eggs)
#def ACF(x):
#    gam= [] 
    #the number of values that we have
#    vals = len(x)
    #the variance of X
    #the value from the mean
    #Since this is based off of data we cannot assume a weakly stationary process: 
#    for j in range(vals):
#        xa = (x[0:vals-j]-np.mean(x))
#        xb = (x[j:vals]-np.mean(x))
#        xvar= np.var(x)
#        gam.append(np.mean(xa*xb)/(xvar))
#    return gam
#def PACF(x,k):
#    acfs = ACF(x)
#    y = np.zeros((k,k))
#    for i in range(k):
#        for j in range(k):
#            if i==j:
#                y[i][j]=1
#            else:
#                y[i][j] = x[abs(i-j)]
 #   y2 = []
 #   y2= copy.deepcopy(y)
 #   dety = np.linalg.det(y)
 #   for i in range(k):
#        y2[i][k-1] =acfs[i]
#    dety2= np.linalg.det(y2)
#    return dety2/dety
#acf1 = ACF(Sal)
#acf2 = ACF(Temp)
#acf3 = ACF(dedata3)
#pacf1 = []
#pacf2 = []
#pacf3 = []
#for i in range(1,len(Sal)):
#   pacf1.append(PACF(Sal,i))
#for i in range(1,len(Temp)):
#   pacf2.append(PACF(Temp,i))
#for i in range(1,100):
#   pacf3.append(PACF(dedata3,i))
#acf3 = ACF(Eggs)
#plt.plot(acf1)
#plt.title('ACF of Ocean Salinity in months since 1970')
#plt.xlabel('Months')
#plt.ylabel('Ocean Salinity')
#plt.show()
#plt.scatter(t[0:len(pacf1)],pacf1)
#plt.title('PACF of Ocean Salinity in months since 1970')
#plt.xlabel('Months')
#plt.ylabel('Ocean Salinity')
#plt.show()
#plt.plot(acf2)
#plt.title('ACF of Ocean Temp in months since 1970')
#plt.xlabel('Months')
#plt.ylabel('Ocean Temp')
#plt.show()
#need de-trend and de-seasonlize
#plt.plot(t[0:len(pacf2)],pacf2)
#plt.title('PACF of Ocean Temp in months since 1970')
#plt.xlabel('Months')
#plt.ylabel('Ocean Temp')
#plt.show()
#plt.plot(acf3)
#plt.title('ACF of Oscillator Challenge')
#plt.xlabel('Months')
#plt.ylabel('Ocean Temp')
#plt.show()
#plt.scatter(t[0:len(pacf3)],pacf3)
#plt.title('PACF of Oscillator Challenge')
#plt.xlabel('Months')
#plt.ylabel('Ocean Temp')
#plt.show()
#plt.plot(acf3)
#plt.title('ACF of Crab Eggs  per year ')
#plt.xlabel('Years')5
#plt.ylabel('Eggs')
#plt.show()
#def PACF()
#def PACF(x):
#    gam= []
#    phigam = []
    #the number of values that we have
#    vals = len(x)   
#    for j in range(vals):
#        xa = (x[0:vals-j]-np.mean(x[0:vals-j]))
#        xb = (x[j:vals]-np.mean(x[j:vals]))
#        xvar= np.var(x[0:vals-j])
#        xvar2 = np.var(x[j:vals])
#        gam.append(np.mean(xa*xb)/(xvar**0.5*xvar2**0.5))
#    for j in range(vals):
        #insert sigma here
#        rgam = gam[::-1]
#        matrgam = np.matrix(rgam)
#        matgam = np.matrix(gam)
#        Tmatgam = matgam.transpose()
#        Tmatrgam = matrgam.transpose()
#        Cov = gam[j]-rgam[0:j] gam[0:j]
#        pvar1 = np.var(x[j:vals])
#        pvar2 = np.var(x[0:vals-j])
        #insert sigma here
#        varet= pvar2 - gam[0:j] gam[0:j]
        #insert sigma here
#        lagvaret = pvar1-rgam[len(gam)-j:len(gam)] rgam[len(gam)-j:len(gam)]
#        phigam.append(Cov/(varet*lagvaret)**0.5)
#    return phigam
#def Sigmatrix(X):
#    sigma = np.zeros(len(X),len(X))
#    for i in range(len(X)):
#        for j in range(len(X)):
#            index = abs(i-j)
#            sigma[i][j] = X[index]
#    return sigma
