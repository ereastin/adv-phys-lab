#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:22:58 2020

@author: evaneastin
"""

# PHYS 128AL Lab 3: Microwave Optics
# need to accurately confirm the wavelength of the transmitted microwave - DONE
# all experiments checked for accuracy except 7, getting good results and uncertainties
# can you combine these average measurements to get another average??? -- not necessary, either weight them or don't average at all, can have a table of all measurementws at the end


import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

Nerr = 1/np.sqrt(12)
derr = .1/np.sqrt(12)
thetaerr = 1/np.sqrt(12)

def wavelength(d, n):
    return 2 * abs(d) / n

def lloyd(h, x):
    return 2 * (np.sqrt(h**2 + x**2) - x)

def relderr(m):
    return np.sqrt(2 * m**2)

def totaluncert(value, a, da, b, db):
    return abs(value) * np.sqrt((da/a)**2 + (db/b)**2)

def exp6uncert(d, n, angle, err):
    return  d * np.cos(np.radians(angle)) * np.radians(err)

def exp6lamb(d, n, angle):
    return (d/n) * np.sin(np.radians(angle))

def gaussian(x, mu, sig, amp):
    return amp * np.exp(-(x - mu)**2/(2 * sig**2))/np.sqrt(2*np.pi*sig**2)

def mean(x, N):
    return sum(x)/N

def stddev(x, mean, N):
    return np.sqrt(sum((x - mean)**2)/N)

def residuals(func, x, y):
    return sum((abs(func(x, popt[0], popt[1], popt[2]) - y))**2)

def residplot(func, x, y):
    return np.array([func(x, popt[0], popt[1], popt[2]) - y])

# Experiment 3: Measuring wavelength by creating standing waves

r_t = np.array([75.0, 75.0]) # initial position of transitter in cm
d_i = np.array([41.80, 28.15]) # initial position of receiver in cm
d_f = np.array([27.40, 42.50]) # final position of receiver in cm
d1 = r_t - d_i
d2 = r_t - d_f
D = abs(d1 - d2) # relative change in transmitter/receiver separation
wvlngth = wavelength(D, 10) # wavelength
wvlerr = relderr(derr) # error in wavelength
trueval = mean(wvlngth, 2) # average wavelength value
truevalerr = np.sqrt(2 * (wvlerr**2))/2 # average error in wavelength
print(trueval, '\pm', truevalerr)
print(wvlngth, wvlerr)

# Experiment 6: Double Slit Interference
# Part I: Wide Slit Spacer - 90 mm (labman), 1.5 cm slit width
# only looking around the first maxima (away from center) 
# expect max around 15.8 deg
# calculate wavelength by finding mean and standard deviation of local data

thetaw = np.array([10, 15, 20, 25, 11, 13, 14, 15, 16, 17, 19, 21, 23]) # angle in degrees
currentw = np.array([0.8, 5.5, 4.7, 0.7, 0.8, 1.3, 2.9, 5.4, 7.5, 8.6, 7.4, 3.5, 1.3]) # current reading in mA
muw = mean(thetaw, len(thetaw)) # mean of angle data
sigmaw = stddev(thetaw, muw, len(thetaw)) # standard deviation of angle data 
currentwerr = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])
thetawerr = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
plt.errorbar(thetaw, currentw, yerr=currentwerr, xerr=thetawerr, fmt='.')
dw = 9.0 + 1.5 # slit spacing in cm
nw = 1 # nth maxima away from center
beat = exp6lamb(dw, nw, muw) # calculated wavelength
beaten = exp6uncert(dw, nw, muw, sigmaw) # calculated error in wavelength
print(beat, '\pm', beaten, muw, sigmaw)
xw = np.linspace(9.0, 26.0, 1000)
popt, _ = optimize.curve_fit(gaussian, thetaw, currentw, p0=[muw, sigmaw, 8.9])
plt.plot(xw, gaussian(xw, popt[0], popt[1], popt[2]), label='Gaussian Fit')
werk = residplot(gaussian, thetaw, currentw)
plt.plot(thetaw, werk.flatten(), '.', label='residuals')
plt.hlines(0, min(xw), max(xw))
plt.xlabel(r'Angle of Incident Radiation [$^{\circ}$]')
plt.ylabel('Current Reading [mA]')
plt.title('Current reading as a function of incident angle (wide spacer)')
plt.legend()
plt.grid()

plt.show()

lmno = residuals(gaussian, thetaw, currentw)
print(lmno)
# Part II: Narrow Slit Spacer - 60 mm (labman), 1.5 cm slit width
# only looking around the first maxima (away from center)
# expect max around 22.3 deg
# calculate wavelength by finding mean and standard deviation of local data

thetan = np.array([15, 20, 25, 30, 35, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29]) # angle in degrees
currentn = np.array([2.1, 8.9, 8.9, 4.0, 0.8, 2.1, 5.1, 8.5, 8.7, 8.8, 8.8, 8.9, 8.9, 8.9, 8.9, 8.8, 7.1]) # current reading in mA
mun = mean(thetan, len(thetan)) # mean of angle data
sigman = stddev(thetan, mun, len(thetan)) # standard deviation of angle data
currentnerr = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])
thetanerr = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
plt.errorbar(thetan, currentn, yerr=currentnerr, xerr=thetanerr, fmt='.')
dn = 6.0 + 1.5 # slit spacing in cm
nn = 1 # nth maxima away from center
lit = exp6lamb(dn, nn, mun) # calculated wavelength
litted = exp6uncert(dn, nn, mun, sigman) # calculated error in wavelength
print(lit, '\pm', litted, mun, sigman)
xn = np.linspace(10.0, 40.0, 1000)
popt, _ = optimize.curve_fit(gaussian, thetan, currentn, p0=[mun, sigman, 8.9])
plt.plot(xn, gaussian(xn, popt[0], popt[1], popt[2]), label='Gaussian Fit')
merp = residplot(gaussian, thetan, currentn)
plt.plot(thetan, merp.flatten(), '.', label='residuals')
plt.hlines(0, min(xn), max(xn))
plt.xlabel(r'Angle of Incident Radiation [$^{\circ}$]')
plt.ylabel('Current Reading [mA]')
plt.title('Current reading as a function of incident angle (narrow spacer)')
plt.legend()
plt.grid()

plt.show()

# residuals
pqrs = residuals(gaussian, thetan, currentn)
print(pqrs)


# Experiment 8: Fabry-Perot Interferometer - Cross 10 minima to return to new maxima
# Did not return a great result, but there is a significant outlier result of 2.1 cm used in average

ntotal = 10 # total number of minima crossed

d_0 = np.array([48.5, 35.1, 40, 30, 45]) # initial position of partial reflector closest to transmitter in cm
d_i8 = np.array([73.5, 69.4, 68.5, 68.5, 68.8])# initial position of partial reflector closest to receiver in cm
d_f8 = np.array([84, 83.5, 81.6, 83.1, 82.1]) # final position of partial reflector closest to receiver in cm
d_init = abs(d_0 - d_i8)
d_fin = abs(d_0 - d_f8)
deld = abs(d_fin - d_init) # relative change in separation of partial reflectors

lamb = wavelength(deld, ntotal) # calculated wavelength
dellamb = totaluncert(lamb, deld, relderr(derr), ntotal, Nerr) # calculated error in wavelength
newerr = np.sqrt(sum(dellamb**2))/5 # average error in calculated wavelength 
lambavg = mean(lamb, 5) # average calculated wavelength
lambx = mean(lamb[1:], 4)
lambxerr = np.sqrt(sum(dellamb[1:]**2))/5
print(lambavg, '\pm', newerr)
print(lambx, lambxerr)
print(lamb)

# Experiment 9: Michelson Iterferometer - Cross 10 minima to return to new maxima

d_i9 = np.array([76.5, 83.9, 78.0, 92.3, 88.5]) # initial position of reflector A
d_f9 = np.array([90.9, 98.1, 92.3, 106.7, 102.7]) # final position of reflector A
d_rel = abs(d_f9 - d_i9)# relative change in position of reflector A

wv = wavelength(d_rel, ntotal) # calculated wavelength
uncertd9 = relderr(derr) # error in relative measured position
delwv = totaluncert(wv, d_rel, uncertd9, ntotal, Nerr) # error in calculated wavelength
wverr = np.sqrt(sum(delwv**2))/5 # average error in calculated wavelength
wvavg = mean(wv, 5) # average calculated wavelength
print(wvavg, '\pm', wverr)
print(wv)

