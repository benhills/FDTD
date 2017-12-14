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
c0    = 3e8                                     # Speed of Light m/s
e0    = 8.85e-12                                # free space permittivity 1/m
u0    = 1.26e-6                                 # free space permeability 1/m
fmax  = 30e6                                    # Source Frequency 1/s
erice = 3.2                                     # relative permittivity of ice
erbed = 12                                      # relative permittivity of bed

### Grid Parameters ###
ermax = max([erice, erbed])                     # maximum relative permittivity
nmax  = np.sqrt(ermax)                          # maximum refractive index
NLAM  = 10                                      # grid resolution, resolve nmax with 10pts
lam0  = c0/fmax                                 # min wavelength in simulation
dx,dy = 2*lam0/nmax/NLAM, 2*lam0/nmax/NLAM      # step size in x/z-direction
X,Y   = np.arange(0,1.,dx), np.arange(0,1.,dy)  # X-distance and Z-depth arrays for domain
Nx,Ny = len(X),len(Y)                           # number of x/z points in grid

# Initialize material constants
N    = Nx*Ny
epsz = erice*np.ones(N)                         # relative permittivity in the z direction
mux  = np.ones(N)                               # relative permeability in the x direction
muy  = np.ones(N)                               # relative permeability in the y direction

# Time Domain
nbc   = np.sqrt(mux[0]*epsz[0])                 # refractive index at boundary
dt    = nbc*dy/(2*c0)                           # time step
tau   = 0.2/fmax                                # duration of Gaussian source
t0    = 2.*tau                                  # initial time, offset of Gaussian source
tprop = nmax*Ny*dy/c0                           # time for wave accross grid
t_f   = 2.*t0 + 3.*tprop                        # total simulation time
steps = math.ceil(t_f/dt)                       # number of time steps
t     = np.arange(0,steps)*dt                   # update simulation time

# Source
nx_src = math.ceil(Nx/4.)                       # x Index of Source Location
ny_src = math.ceil(Ny/4.)                       # y Index of Source Location
n_src  = int(ny_src*Nx+nx_src)                  # Source location in vector
Esrc   = 40*np.exp(-((t-t0)/tau)**2.)           # Electricity source, Gaussian

# Initialize FDTD parametrs
mEz = (c0*dt)/epsz                              # Electricity multiplication parameter
mHx = (c0*dt)/mux                               # Magnetism multiplication parameter
mHy = (c0*dt)/muy                               # Magnetism multiplication parameter
# Initialize fields to zero
Ez = np.zeros((N))                              # Electric Field in z direction
Hx = np.zeros((N))                              # Magnetic Field in x direction
Hy = np.zeros((N))                              # Magnetic Field in y direction

############################################################
### Matrices ###

# Define transformation matrices for forward difference
Ea = sp.lil_matrix((N,N))                       # Sparse Matrix for Hx update
Ea.setdiag(-1.*np.ones(N),k=0)                  # Matrix diagonal to -1 for the node itself
Ea.setdiag(np.ones(N-Nx),k=Nx)                  # Matrix off-diagonal to 1 for the node in the y-direction
Ea/=dy

Eb = sp.lil_matrix((N,N))                       # Sparse Matrix for Hy update
Eb.setdiag(-1.*np.ones(N),k=0)                  # Matrix diagonal to -1 for the node itself
Eb.setdiag(np.ones(N-1),k=1)                    # Matrix off-diagonal to 1 for the node in the x-direction
Eb/=dx

Ha = sp.lil_matrix((N,N))                       # Sparse Matrix to be multiplied by Hx
Ha.setdiag(np.ones(N),k=0)
Ha.setdiag(-1.*np.ones(N-Nx),k=-Nx)             # Matrix off-diagonal to -1 for the node in the y-direction
Ha/=dy

Hb = sp.lil_matrix((N,N))                       # Sparse Matrix to be multiplied by Hy
Hb.setdiag(np.ones(N),k=0)
Hb.setdiag(-1.*np.ones(N-1),k=-1)               # Matrix off-diagonal to -1 for the node in the x-direction
Hb/=dx

# Dirichlet BCs
Ea[N-Nx:,:] = 0
Eb[np.arange(Ny)*Nx-1,:] = 0
Ha[:Nx+1,:] = 0
Hb[np.arange(Ny)*Nx,:] = 0

############################################################
### Figure ###

fig = plt.figure(figsize=(12,9))
ax = plt.subplot()

plt.ion()
im = plt.imshow(Ez.reshape(Nx,Ny),vmin=-1.,vmax=1.,cmap='RdYlBu')
plt.colorbar()
time_text = ax.text(0.5,1.05,'',ha='center',transform=ax.transAxes)

############################################################

### Algorithm ###
for t_i in np.arange(steps):

    # Update Magnetic Field
    Hx += -mHx*(Ea*Ez)
    Hy += mHy*(Eb*Ez)
    # Update Electric Field
    Ez += mEz*(Hb*Hy-Ha*Hx)
    # Apply the source
    Ez[n_src] += Esrc[t_i]

    # Update the Plot
    im.set_data(Ez.reshape(Nx,Ny))
    time_text.set_text('Time Step = %0.0f of %0.0f' % (t_i,steps))
    plt.pause(0.000001)
