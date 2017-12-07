#!/usr/bin/env python

"""
Finite difference time domain model
1-dimensional

For solving the Maxwell Equations

Based on lectures from CEM
https://www.youtube.com/watch?v=y7hJAhKp2d8

Author: Ben Hills
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import math

############################################################
### Setup ###

# Constants
c0 = 3e8        # Speed of Light m/s
e0 = 8.85e-12   # free space permittivity 1/m
u0 = 1.26e-6    # free space permeability 1/m
fmax   = 5e9    # Source Frequency 1/s

# slab properties
dslab = 0.25/3.1    # Slab Thickness
erair = 1.          # relative permittivity of air
erslab = 12.        # relative permittivity of slab

### Grid Parameters ###
ermax = max([erair, erslab])        # relative permittivity
nmax  = 1#np.sqrt(ermax)              # maximum refractive index
NLAM  = 10                          # grid resolution, resolve nmax with 10pts
NDIM  = 1                           # number of dimensions
NBUFZ = [100, 100]                  # space before and after device
lam0  = c0/fmax                     # min wavelength in simulation
# grid resolution, min between first two
dz1   = lam0/nmax/NLAM
dz2   = dslab/NDIM
dz    = dz1#min([dz1,dz2])
# number of points in grid
nz    = math.ceil(dslab/dz)
Nz    = sum(NBUFZ)+3#int(nz) + sum(NBUFZ) + 3    # number of points in grid
Z     = dz*np.arange(0,Nz)          # distance array for domain

# Initialize material constants
Er = erair*np.ones(Nz)              # relative permittivity
Ur = np.ones(Nz)                    # relative permeability

nz1 = 2 + NBUFZ[0] +1               # slab start location
nz2 = nz1 + math.ceil(dslab/dz) -1  # slab end location
#Er[nz1:nz2] = erslab                # relative permittivity in the slab

# Time Domain
nbc   = np.sqrt(Ur[0]*Er[0])        # refractive index at boundary
dt    = nbc*dz/(2*c0)               # time step
tau   = 0.5/fmax                    # duration of Gaussian source
t0    = 5.*tau                      # initial time, offset of Gaussian source
tprop = nmax*Nz*dz/c0               # time for wave accross grid
t     = 2.*t0 + 3.*tprop            # total simulation time
steps = math.ceil(t/dt)             # number of time steps
t     = np.arange(0,steps)*dt       # update simulation time

# Source
nz_src = math.ceil(Nz/2.)                   # Index of Source Location
Esrc   = np.exp(-((t-t0)/tau)**2.)          # Electricity source, Gaussian

# Initialize FDTD parametrs
mEy = (c0*dt)/Er    # Electricity multiplication parameter
mHx = (c0*dt)/Ur    # Magnetism multiplication parameter
# Initialize fields to zero
Ey = np.zeros(Nz)   # Electric Field
Hx = np.zeros(Nz)   # Magnetic Field

############################################################
### Matrices ###

# Define transformation matrices
A = sp.lil_matrix((Nz,Nz))
A.setdiag(-1.*np.ones(Nz),k=0)
A.setdiag(np.ones(Nz-1),k=1)
A = A.tocsr()

B = sp.lil_matrix((Nz,Nz))
B.setdiag(np.ones(Nz),k=0)
B.setdiag(-1.*np.ones(Nz-1),k=-1)
B = B.tocsr()

# Dirichlet BCs
A[-1,:] = 0
B[0,:] = 0

# Perfectly absorbing BC
H1,H2,H3 = 0,0,0
E1,E2,E3 = 0,0,0

############################################################
### Figure ###

fig = plt.figure()
ax = plt.subplot()
ax.set_ylim(-1.5,1.5)
ax.set_xlim(min(Z),max(Z))

#plt.fill_betweenx(np.linspace(-5,5,10),Z[nz1],Z[nz2])
plt.ion()
H_line, = plt.plot([],[],'r',zorder=1)
E_line, = plt.plot([],[],'b',zorder=0)

############################################################
### Algorithm ###

for t_i in np.arange(steps):

    # Update Magnetic Field
    Hx += (mHx/dz)*(A*Ey)
    Hx[-1] = Hx[-1] + mHx[-1]*(E3 - Ey[-1])/dz
    # Record H-field at Boundary
    H3 = H2
    H2 = H1
    H1 = Hx[0]
    # Update Electric Field
    Ey += (mEy/dz)*(B*Hx)
    Ey[0] = Ey[0] + mEy[0]*(Hx[0] - H3)/dz
    # Record E-field at Boundary
    E3 = E2
    E2 = E1
    E1 = Ey[-1]
    # Apply the source
    Ey[nz_src] = Ey[nz_src] + Esrc[t_i]

    # Plot
    E_line.set_xdata(Z)
    E_line.set_ydata(Ey)
    H_line.set_xdata(Z)
    H_line.set_ydata(Hx)
    plt.pause(0.000001)


