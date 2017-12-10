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
import matplotlib.animation as animation
#plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

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
NLAM  = 5                                      # grid resolution, resolve nmax with 10pts
lam0  = c0/fmax                                 # min wavelength in simulation
dx,dz = lam0/nmax/NLAM, lam0/nmax/NLAM          # step size in x/z-direction
Z   = np.arange(0,1.,dz)  # X-distance and Z-depth arrays for domain
Nz = len(Z)                                 # number of x/z points in grid
nslab_1 = int(Nz/2)                             # bed start location
nslab_2 = nslab_1 + math.ceil(bed_thick/dz)  -1      # slab end location

# Initialize material constants
Er = erice*np.ones(Nz)              # relative permittivity
Ur = np.ones(Nz)                    # relative permeability
#Er[nslab_1:nslab_2] = erbed        # relative permittivity in the slab

# Time Domain
nbc   = np.sqrt(Ur[0]*Er[0])        # refractive index at boundary
dt    = nbc*dz/(2*c0)               # time step
tau   = 0.5/fmax                    # duration of Gaussian source
t0    = 5.*tau                      # initial time, offset of Gaussian source
tprop = nmax*Nz*dz/c0               # time for wave accross grid
t_f     = 2.*t0 + 3.*tprop          # total simulation time
steps = math.ceil(t_f/dt)           # number of time steps
t     = np.arange(0,steps)*dt       # update simulation time

# Source
nx_src = math.ceil(Nz/2.)                   # Index of Source Location
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

# Define transformation matrices for forward difference
A = sp.lil_matrix((Nz,Nz))          # Sparse Matrix for Hx update
A.setdiag(-1.*np.ones(Nz),k=0)      # Matrix diagonal to -1
A.setdiag(np.ones(Nz-1),k=1)        # Matrix off-diagonal to 1

B = sp.lil_matrix((Nz,Nz))          # Sparse Matrix for Ey update
B.setdiag(np.ones(Nz),k=0)
B.setdiag(-1.*np.ones(Nz-1),k=-1)

# Dirichlet BCs
A[-1,:] = 0
B[0,:] = 0

# Perfectly absorbing BC
PABC = False
H1,H2,H3 = 0,0,0
E1,E2,E3 = 0,0,0

############################################################
### Figure ###

fig = plt.figure()
ax = plt.subplot()
ax.set_ylim(-1.5,1.5)
ax.set_xlim(min(Z),max(Z))

#plt.fill_betweenx(np.linspace(-5,5,10),Z[nz1],Z[nz2])
H_line, = ax.plot([],[],'r',zorder=1)
E_line, = ax.plot([],[],'b',zorder=0)
time_text = ax.text(0.02,0.9,'',transform=ax.transAxes)

############################################################
### Algorithm ###

def init():
    E_line.set_data([],[])
    H_line.set_data([],[])
    return E_line, H_line,

#for t_i in np.arange(steps):
def animate(t_i,Hx=Hx,Ey=Ey,H3=H3,H2=H2,H1=H1,E3=E3,E2=E2,E1=E1):

    # Update Magnetic Field
    Hx += (mHx/dz)*(A*Ey)
    if PABC == True:
        # Record H-field at Boundary
        H3 = H2
        H2 = H1
        H1 = Hx[0]
    # Update Electric Field
    Ey += (mEy/dz)*(B*Hx)
    if PABC == True:
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
    time_text.set_text('Time Step = %0.0f of %0.0f' % (t_i,t_f))

    return E_line, H_line, time_text

ani = animation.FuncAnimation(fig,animate,init_func=init,frames=np.arange(steps),interval=0.001,blit=True)

# Save the animation
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15,metadata=dict(artist='Me'),bitrate=1800)
ani.save('DirichletBC.mp4',writer=writer)
