# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Finite difference time domain model
2-dimensional

For solving the Maxwell Equations
See 1d_model.py for the 1-d analogous model

Author: Ben Hills
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import math

############################################################
### Setup ###

# Constants
c0   = 3e8                                      # Speed of Light m/s
e0   = 8.85e-12                                 # free space permittivity 1/m
u0   = 1.26e-6                                  # free space permeability 1/m
fmax = 30e6                                     # Source Frequency 1/s
erice = 3.2                                     # relative permittivity of ice
erdebris = 2.                                   # relative permittivity of debris laden ice
erbed = 12.                                     # relative permittivity of bed

### Grid Parameters ###
ermax = max([erice, erbed])                     # maximum relative permittivity
nmax  = np.sqrt(ermax)                          # maximum refractive index
NLAM  = 10                                      # grid resolution, resolve nmax with 10pts
lam0  = c0/fmax                                 # min wavelength in simulation
dz = lam0/nmax/NLAM                             # step size in x/z-direction
zmax = 100.
Z   = np.arange(0.,zmax,dz)                     # X-distance and Z-depth arrays for domain
Nz = len(Z)                                     # number of x/z points in grid

# Initialize material constants
Er = erice*np.ones(Nz)                          # relative permittivity
Ur = (1/erice)*np.ones(Nz)                      # relative permeability

# change the permittivity for some random layers that have debris and for the bed
nbed_1 = np.argmin(abs(Z-0.8*zmax))             # bed start location
nbed_2 = -1                                     # bed start location
rand_ind  = np.where(np.random.rand(Nz)>0.8)
Er[rand_ind] = erdebris                         # relative permittivity in the slab
Er[nbed_1:nbed_2] = erbed                       # relative permittivity in the bed

# Time Domain
nbc   = np.sqrt(Ur[0]*Er[0])                    # refractive index at boundary
dt    = nbc*dz/(2*c0)                           # time step
tau   = 0.5/fmax                                # duration of Gaussian source
t0    = 5.*tau                                  # initial time, offset of Gaussian source
tprop = nmax*Nz*dz/c0                           # time for wave accross grid
t_f   = 2.*t0 + 0.3*tprop                       # total simulation time
steps = math.floor((t_f/dt))                    # number of time steps
t     = np.arange(0,steps)*dt                   # update simulation time

# Source
nz_src = np.argmin(abs(Z-zmax/2.))              # Index of Source Location (centered)
Esrc   = np.exp(-((t-t0)/tau)**2.)              # Electricity source, Gaussian

# Initialize FDTD parametrs
mEy = (c0*dt)/Er                                # Electricity multiplication parameter
mHx = (c0*dt)/Ur                                # Magnetism multiplication parameter
# Initialize fields to zero
Ey = np.zeros(Nz)                               # Electric Field
Hx = np.zeros(Nz)                               # Magnetic Field

############################################################
### Matrices ###

# Define transformation matrices for forward difference
A = sp.lil_matrix((Nz,Nz))                      # Sparse Matrix for Hx update
A.setdiag(-1.*np.ones(Nz),k=0)                  # Matrix diagonal to -1
A.setdiag(np.ones(Nz-1),k=1)                    # Matrix off-diagonal to 1

B = sp.lil_matrix((Nz,Nz))                      # Sparse Matrix for Ey update
B.setdiag(np.ones(Nz),k=0)
B.setdiag(-1.*np.ones(Nz-1),k=-1)

# Dirichlet BCs
A[-1,:] = 0
B[0,:] = 0

# Perfectly absorbing BC
PABC = True
H1,H2,H3 = 0,0,0
E1,E2,E3 = 0,0,0

############################################################
### Figure ###

fig = plt.figure(figsize=(6,6))
ax1 = plt.subplot(111)
ax1.set_ylim(-1.5,1.5)
ax1.set_xlim(0,zmax)
plt.xlabel('Distance')
plt.ylabel('Normalized EM Field')
time_text = ax1.text(0.5,1.1,'',ha='center',transform=ax1.transAxes)

# plot Electric and Magnetic field
H_line, = ax1.plot([],[],'b',zorder=1)
E_line, = ax1.plot([],[],'r',zorder=2)

plt.ion()

############################################################

### Algorithm ###
for t_i in np.arange(steps):
    # Update Magnetic Field
    Hx += (mHx/dz)*(A*Ey)
    if PABC == True:
        Hx[-1] = Hx[-1] + mHx[-1]*(E3 - Ey[-1])/dz
        # Record H-field at Boundary
        H3 = H2
        H2 = H1
        H1 = Hx[0]
    # Update Electric Field
    Ey += (mEy/dz)*(B*Hx)
    if PABC == True:
        Ey[0] = Ey[0] + mEy[0]*(Hx[0] - H3)/dz
        # Record E-field at Boundary
        E3 = E2
        E2 = E1
        E1 = Ey[-1]
    # Apply the source
    Ey[nz_src] = Ey[nz_src] + Esrc[t_i]

    # Update the Plot
    E_line.set_ydata(Z)
    E_line.set_xdata(Ey)
    H_line.set_ydata(Z+0.5*dz)
    H_line.set_xdata(Hx/erice)
    time_text.set_text('Time Step = %0.0f of %0.0f' % (t_i,steps))
    plt.pause(0.00001)

