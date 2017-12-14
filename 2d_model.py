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

### Grid Parameters ###
ermax = max([erice, erbed])                     # maximum relative permittivity
nmax  = np.sqrt(ermax)                          # maximum refractive index
NLAM  = 10                                      # grid resolution, resolve nmax with 10pts
lam0  = c0/fmax                                 # min wavelength in simulation
dx,dy = 2*lam0/nmax/NLAM, 2*lam0/nmax/NLAM          # step size in x/z-direction
X,Y   = np.arange(0,1.,dx), np.arange(0,1.,dy)  # X-distance and Z-depth arrays for domain
Nx,Ny = len(X),len(Y)                                 # number of x/z points in grid

# Initialize material constants
N = Nx*Ny
epsz = erice*np.ones(N)              # relative permittivity
mux = np.ones(N)                    # relative permeability
muy = np.ones(N)                    # relative permeability

#muy[Nx*120:Nx*140] = 10.        # relative permeability in the anisotropic zone

# Time Domain
nbc   = np.sqrt(mux[0]*epsz[0])        # refractive index at boundary
dt    = nbc*dy/(2*c0)               # time step
tau   = 0.2/fmax                    # duration of Gaussian source
t0    = 2.*tau                      # initial time, offset of Gaussian source
tprop = nmax*Ny*dy/c0               # time for wave accross grid
t_f     = 2.*t0 + 3.*tprop          # total simulation time
steps = 500#math.ceil(t_f/dt)           # number of time steps
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

Eb = sp.lil_matrix((N,N))               # Sparse Matrix for Hy update
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

fig = plt.figure(figsize=(12,9))
"""
ax = plt.subplot()

plt.ion()
im = plt.imshow(Ez.reshape(Nx,Ny),vmin=-1.,vmax=1.,cmap='RdYlBu')
plt.colorbar()
time_text = ax.text(0.5,1.05,'',ha='center',transform=ax.transAxes)
"""

Xs,Ys = np.meshgrid(X,Y)

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

    #E_out[t_i] = Ez

    print t_i,steps

    if t_i == int(steps/10):
        ax1 = plt.subplot(221)
        im = plt.pcolor(Xs,Ys,Ez.reshape(Nx,Ny),vmin=-1.,vmax=1.,cmap='RdYlBu')
        plt.title('%0.2E sec'%t[t_i])
        plt.xlabel('m')
        plt.ylabel('m')

    if t_i == int(3*steps/10):
        ax2 = plt.subplot(222)
        plt.pcolor(Xs,Ys,Ez.reshape(Nx,Ny),vmin=-1.,vmax=1.,cmap='RdYlBu')
        plt.title('%0.2E sec'%t[t_i])
        plt.xlabel('m')
        plt.ylabel('m')

    if t_i == int(6*steps/10):
        ax3 = plt.subplot(223)
        plt.pcolor(Xs,Ys,Ez.reshape(Nx,Ny),vmin=-1.,vmax=1.,cmap='RdYlBu')
        plt.title('%0.2E sec'%t[t_i])
        plt.xlabel('m')
        plt.ylabel('m')

    if t_i == int(10*steps/10)-1:
        ax4 = plt.subplot(224)
        plt.pcolor(Xs,Ys,Ez.reshape(Nx,Ny),vmin=-1.,vmax=1.,cmap='RdYlBu')
        plt.title('%0.2E sec'%t[t_i])
        plt.xlabel('m')
        plt.ylabel('m')

plt.tight_layout()

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85 0.1, 0.05, 0.8])
cbar = plt.colorbar(cax=cax)
cbar.set_label('Normalized EM Field')

plt.savefig('2D_figure.png',dpi=300)

############################################################
"""
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
"""
