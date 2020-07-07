#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:25:05 2020

@author: evaneastin
"""
# plotting for lab2: interferometry

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# Experiment 1: calc wavelength 
n = np.array([ 14, 27, 39, 53, 68, 82, 95, 109, 123, 137])
d = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]) # steps of 5 micrometers, find uncert for di - df
# expect distances to be \pm 1% accuracy ref lab manual, also remember to talk about backlash and whatnot
derr = 1/np.sqrt(12)
Nerr = 1/np.sqrt(12)
x = np.linspace(0, 55, 100)
coeff1, covar1 = np.polyfit(d, n, 1, cov=True)
dm1 = np.sqrt(covar1[0][0])

print(coeff1, dm1)
plt.errorbar(d, n, xerr=derr, yerr=Nerr, fmt='.')
y1 = coeff1[1] + coeff1[0] * x
plt.plot(x, y1, '-')
plt.xlabel(r'Relative change in position of movable mirror [$\mu$m]')
plt.ylabel(r'Number of observed fringe transitions')
plt.xlim(0, 55)
plt.ylim(0)
plt.title(r'Fringe transitions N per change in position ($\Delta$d)')
plt.grid()

plt.show()

# Experiment 2: calc index of refraction of air
kpa = -1 * np.array([20, 40, 60, 74]) # CHECK THESE VALUES NOT TOO SURE HOW MAN POINTS TO USE
N = np.array([4, 8, 12, 15])
coeff2, covar2 = np.polyfit(kpa, N, 1, cov=True)
dm2 = np.sqrt(covar2[0][0])
kpaerr = 2/np.sqrt(12)
print(coeff2, dm2)
u = np.linspace(-800, 0, 100)
plt.errorbar(kpa, N, xerr=kpaerr, yerr=Nerr, fmt='.')
plt.plot(u, coeff2[1] + coeff2[0] * u, '-')
plt.xlabel('Change in pressure of the vacuum cell below local pressure [kPa]')
plt.ylabel('Number of observed fringe transitions')
plt.title('Fringe transitions N per change in cell pressure')
plt.xlim(-85, 0)
plt.ylim(0, 16)
plt.grid()
plt.show()


# Experiment 3: calc index of refraction of glass using custom fit equation
theta = np.radians(np.array([2.4, 3.4, 4.2, 4.7, 5.7, 6.7, 7.4, 8.2, 8.7, 9.4, 10.2, 10.8, 11.5, 12.2, 12.7]))
dN = np.array([5, 10, 15, 20, 30, 40, 50, 60, 70, 85, 100, 115, 130, 145, 160])
a = np.radians(0.1/np.sqrt(12))
thetaerr = np.array([a, a, a, a, a, a, a, a, a, a, a, a, a, a, a])

def Ntheta(deg, n_g):
    return (0.0118/(633*10**(-9)))*(n_g*(1-np.cos(deg))+np.cos(deg)-1)/(n_g+np.cos(deg)-1)


popt, pcov = optimize.curve_fit(Ntheta, theta, dN, p0=1.0, bounds=(-5, 5))
n_g = popt
print(n_g)
xplot = np.linspace(0, np.pi/12, 100)
y_plot = Ntheta(xplot, n_g)
perr = np.sqrt(np.diag(pcov))
print(perr) # random error, systematic uncert calc from uncert of measuring devices
plt.errorbar(theta, dN, xerr=thetaerr, yerr=Nerr, fmt='.')
plt.plot(xplot, y_plot, '-', label='Custom fit curve')
plt.xlabel('Change in angle of incidence from normal [radians]')
plt.ylabel('Number of observed fringe transitions')
plt.title('Fringe transitions as a function of incidence angle on glass')
plt.xlim(0, 0.25)
plt.ylim(0)
plt.legend()
plt.grid()

plt.show()

