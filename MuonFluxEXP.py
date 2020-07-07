#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:49:25 2020

@author: evaneastin
"""

# PHYS 128AL FINAL LAB: Muon Flux Experiment
# aiming to quantify flux of cosmic ray particles, angled N-S, form zenith angles 0-90
# want to show discrepancy between flat and round earth distributions

import numpy as np
# approximate rectangular detector dimensions, use for flux, found using vernier caliper
w = 10.2 * 10**(-2) # m
l = 10.7 * 10**(-2) # m
A =  l * w # area estimate of rect. detector in m^2
derr = .001/np.sqrt(12) # uncertianty in vernier caliper
Aerr = A * np.sqrt((derr/w)**2 + (derr/l)**2) # uncertainty in detector area
alpha = 9.08 # uncertainty in zenith angle **important one**
beta = 10.8 # uncertainty in azimuthal angle, half of total sep angle

# defined constants for the fitting curve
R = 6371.371 * 10**3 # Earth's radius in m
# calculated using latitude (34.41406135) and elevation (~26 m) from https://www.maps.ie/coordinates.html
# https://rechneronline.de/earth-radius/ combined above values to get radius
d_low = 10 * 10**3 # lowest expected height above ground of muon production in m
d_high = 15 * 10**3 # greatest expected height above ground of muon production in m

def D_round(R, d, x): # closed column density expression for curved atmosphere
    return np.sqrt((R ** 2 / d ** 2) * np.cos(x) **2 + (2 * R / d) + 1) - (R * np.cos(x) / d) 

def MuonEFlux(Dfunc, I_0, n): # muon energy integrated flux
    return I_0 * (Dfunc) ** (-(n - 1))

def cosfit(x, I_0, n): # fit function from zenith angle dependence of cosmic muons
    return I_0 * np.cos(x)**(n - 1)

def MuonFlux(counts, area, time): # observed muon flux from experiment
    return counts / (area * time)

def FluxErr(count, counterr, area, aerr): # equal to the flux uncertainty divided by calculated flux
    return np.sqrt((counterr/count)**2 + (aerr/area)**2)

def regression(fitfunc, y): # returns residual sum of squares
    return sum((fitfunc - y)**2)

def residuals(fitfunc, y): # returns residuals
    return np.array(fitfunc - y)

import matplotlib.pyplot as plt
from scipy import optimize, stats

theta = np.radians(90 - np.array([10, 25, 40, 55, 70, 85])) # zenith angle
N = np.array([482, 652, 61, 85, 107, 172]) # observed counts
Nerr = np.array([np.sqrt(N)]) # count uncertainty
observ_time = np.array([70000, 70000, 4000, 4000, 4000, 4000]) # observation time in seconds
flux= MuonFlux(N, A, observ_time) # calculated flux without subtracting accidental rate
newflux = ((N / observ_time) - 0.00001536765222) / A # final calculated flux removing rate of accidentals
observerr = FluxErr(N, Nerr, A, Aerr) 
newerr = newflux * observerr # total flux uncertainty 

x = np.linspace(0, np.pi / 2, 2000)

xnew = D_round(R, d_low, theta)

a = np.radians(alpha)/2
thetaerr = np.array([a, a, a, a, a, a]) # angle uncertianty calculated from alpha

# plotting of full results

# plot 1: overall distributions with log10 y scale
plt.errorbar(theta, newflux, yerr=newerr.flatten(), xerr=thetaerr, fmt='.', label='Observed Data')

popt, pcov = optimize.curve_fit(MuonEFlux, xnew, newflux, p0=[4, 3], bounds=(0, 5))
n_fit = popt
fiterr = np.sqrt(np.diag(pcov))
xplot = np.linspace(0, np.pi / 2, 2000)
yplot1 = MuonEFlux(D_round(R, d_high, xplot), n_fit[0], n_fit[1])
plt.plot(xplot, yplot1, '-', color='green', label='Fitted Round Earth Distribution')

popt2, pcov2 = optimize.curve_fit(cosfit, theta, newflux, p0=[4, 3], bounds=(0, 5))
fiterr2 = np.sqrt(np.diag(pcov2))
print(popt2, fiterr2)

plt.plot(xplot, cosfit(xplot, *popt2), color='black', label='Fitted Flat Earth Distribution')
plt.plot(xplot, 4 * np.cos(xplot)**2, color='red', label=r'$\cos^{2}(\theta_{z})$')
plt.ylabel(r'Muon Flux [counts $m^{-2} s^{-1}$]')
plt.xlabel(r'Angle $\theta_{z}$ from zenith [radians]')
plt.yscale('log')
plt.ylim(10**(-2), 5)
plt.xlim(0, np.pi / 2)
plt.xticks(np.arange(0, np.pi / 2 + np.pi / 10, np.pi / 10), labels=[0, r'$\frac{\pi}{10}$', r'$\frac{\pi}{5}$', r'$\frac{3\pi}{10}$', r'$\frac{2\pi}{5}$', r'$\frac{\pi}{2}$'])
plt.title('Muon Flux Distribution as a Function of Zenith Angle')
plt.grid()
plt.legend()
plt.show()

# plot 2: detailed plot around 90 deg, should show discrepancy between flat and round earth distributions
plt.plot(xplot, yplot1, '-', color='green', label='Fitted Round Earth Distribution')
plt.plot(xplot, cosfit(xplot, *popt2), color='black', label='Fitted Flat Earth Distribution')
plt.plot(xplot, 4 * np.cos(xplot)**2, color='r', label=r'$\cos^{2}(\theta_{z})$')
plt.ylabel(r'Muon Flux [counts $m^{-2} s^{-1}$]')
plt.xlabel(r'Angle $\theta_{z}$ from zenith [radians]')
plt.xlim(11 * np.pi / 24, np.pi / 2)
plt.xticks(np.arange(11 * np.pi / 24, np.pi / 2, np.pi / 24), labels=[r'$\frac{11\pi}{24}$', r'$\frac{\pi}{2}$'])
plt.ylim(0, 0.1)
plt.grid()
plt.legend()
plt.title(r'Muon Flux Distribution around $\theta_{z} = \frac{\pi}{2}$')
plt.show()

rss = regression(MuonEFlux(D_round(R, d_high, theta), n_fit[0], n_fit[1]), flux) # residual sum of squares
resid = residuals(MuonEFlux(D_round(R, d_high, theta), n_fit[0], n_fit[1]), flux) # residuals


chisquare1 = stats.chisquare(flux, MuonEFlux(D_round(R, d_high, theta), n_fit[0], n_fit[1]), 2) # chi-squared test for round earth distribution
chisquare2 = stats.chisquare(flux, cosfit(theta, *popt2), 2) # chi-squared test for flat earth distribution

print(MuonEFlux(D_round(R, d_high, np.pi/2), n_fit[0], n_fit[1]))

# making some new changes in order to see if push/pull requests work??
