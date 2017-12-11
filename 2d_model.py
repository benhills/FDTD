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
import matplotlib.animation as animation

############################################################
### Setup ###

# Constants
c0 = 3e8        # Speed of Light m/s
e0 = 8.85e-12   # free space permittivity 1/m
u0 = 1.26e-6    # free space permeability 1/m
fmax   = 5e9    # Source Frequency 1/s

# bed properties
erice = 1.          # relative permittivity of ice
erbed = 12.         # relative permittivity of bed
bed_thick = 0.1

### Grid Parameters ###
ermax = max([erice, erbed])                     # maximum relative permittivity
nmax  = np.sqrt(ermax)                          # maximum refractive index
NLAM  = 10                                      # grid resolution, resolve nmax with 10pts
lam0  = c0/fmax                                 # min wavelength in simulation
dx,dy = lam0/nmax/NLAM, lam0/nmax/NLAM          # step size in x/z-direction
Nx,Ny = 200,200                                 # number of x/z points in grid
X,Y   = dx*np.arange(0,Nx), dy*np.arange(0,Ny)  # X-distance and Z-depth arrays for domain
nslab_1 = int(Ny/2)                             # bed start location
nslab_2 = nslab_1 + math.ceil(bed_thick/dy)  -1      # slab end location

# Initialize material constants
N = Nx*Ny
epsz = erice*np.ones(N)              # relative permittivity
mux = np.ones(N)                    # relative permeability
muy = np.ones(N)                    # relative permeability

mux[Nx*120:Nx*140] = 10.        # relative permeability in the anisotropic zone

# Time Domain
nbc   = np.sqrt(mux[0]*epsz[0])        # refractive index at boundary
dt    = nbc*dy/(2*c0)               # time step
tau   = 0.2/fmax                    # duration of Gaussian source
t0    = 2.*tau                      # initial time, offset of Gaussian source
tprop = nmax*Ny*dy/c0               # time for wave accross grid
t_f     = 2.*t0 + 3.*tprop          # total simulation time
steps = 1000#math.ceil(t_f/dt)           # number of time steps
t     = np.arange(0,steps)*dt       # update simulation time

# Source
nx_src = math.ceil(Nx/4.)                   # x Index of Source Location
ny_src = math.ceil(Ny/4.)                   # y Index of Source Location
n_src = int(ny_src*Nx+nx_src)
Esrc   = 40*np.exp(-((t-t0)/tau)**2.)          # Electricity source, Gaussian

# Initialize FDTD parametrs
mEz = (c0*dt)/epsz    # Electricity multiplication parameter
mHx = (c0*dt)/mux    # Magnetism multiplication parameter
mHy = (c0*dt)/muy    # Magnetism multiplication parameter
# Initialize fields to zero
Ez = np.zeros((N))   # Electric Field
Hx = np.zeros((N))  # Magnetic Field in x direction
Hy = np.zeros((N))  # Magnetic Field in y direction

############################################################
### Matrices ###

# Define transformation matrices for forward difference
Ea = sp.lil_matrix((N,N))               # Sparse Matrix for Hx update
Ea.setdiag(-1.*np.ones(N),k=0)          # Matrix diagonal to -1 for the node itself
Ea.setdiag(np.ones(N-Nx),k=Nx)          # Matrix off-diagonal to 1 for the node in the y-direction
Ea/=dy

Eb = sp.lil_matrix((N,N))               # Sparse Matrix for Hx update
Eb.setdiag(-1.*np.ones(N),k=0)          # Matrix diagonal to -1 for the node itself
Eb.setdiag(np.ones(N-1),k=1)          # Matrix off-diagonal to 1 for the node in the y-direction
Eb/=dx

Ha = sp.lil_matrix((N,N))               # Sparse Matrix for Ey update
Ha.setdiag(np.ones(N),k=0)
Ha.setdiag(-1.*np.ones(N-Nx),k=-Nx)       # Matrix off-diagonal to 1 for the node in the x-direction
Ha/=dy

Hb = sp.lil_matrix((N,N))               # Sparse Matrix for Ey update
Hb.setdiag(np.ones(N),k=0)
Hb.setdiag(-1.*np.ones(N-1),k=-1)       # Matrix off-diagonal to 1 for the node in the x-direction
Hb/=dx

# Dirichlet BCs
Ea[N-Nx:,:] = 0
Eb[np.arange(Ny)*Nx-1,:] = 0
Ha[:Nx+1,:] = 0
Hb[np.arange(Ny)*Nx,:] = 0

# Perfectly absorbing BC
PABC = False
H1,H2,H3 = 0,0,0
E1,E2,E3 = 0,0,0

############################################################
### Figure ###

fig = plt.figure()
ax = plt.subplot()

plt.ion()
im = plt.imshow(Ez.reshape(Nx,Ny),vmin=-1.,vmax=1.,cmap='RdYlBu')
plt.colorbar()
time_text = ax.text(0.5,1.05,'',ha='center',transform=ax.transAxes)

############################################################

### Algorithm ###

E_out = np.empty((steps,len(Ez)))

for t_i in np.arange(steps):

    # Update Magnetic Field
    Hx += -mHx*(Ea*Ez)
    Hy += mHy*(Eb*Ez)
    if PABC == True:
        # Record H-field at Boundary
        H3 = H2
        H2 = H1
        H1 = Hx[0]

    # Update Electric Field
    Ez += mEz*(Hb*Hy-Ha*Hx)
    if PABC == True:
        # Record E-field at Boundary
        E3 = E2
        E2 = E1
        E1 = Ez[-1]
    # Apply the source
    Ez[n_src] += Esrc[t_i]

    # Plot
    #im.set_data(Ez.reshape(Nx,Ny))
    #time_text.set_text('Time Step = %0.0f of %0.0f' % (t_i,steps))
    #plt.pause(0.000001)

    E_out[t_i] = Ez

############################################################

def init():
    im.set_data([[],[]])
    return im,

def animate(i):
    im.set_data(E_out[i].reshape(Nx,Ny))
    time_text.set_text('Time Step = %0.0f of %0.0f' % (i,steps))
    return im, time_text,

ani = animation.FuncAnimation(fig,animate,init_func=init,frames=np.arange(0,steps,2),interval=20,blit=True)

# Save the animation
ani.save('Anisotropic.mp4',writer="ffmpeg")

