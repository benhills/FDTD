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

############################################################
### Setup ###

# Constants
c0 = 3e8        # Speed of Light m/s
e0 = 8.85e-12   # free space permittivity 1/m
u0 = 1.26e-6    # free space permeability 1/m
fmax   = 30e6    # Source Frequency 1/s

# bed properties
erice = 3.2          # relative permittivity of ice
erdebris = 2.         # relative permittivity of debris laden ice
erbed = 12.         # relative permittivity of bed

### Grid Parameters ###
ermax = max([erice, erbed])                     # maximum relative permittivity
nmax  = np.sqrt(ermax)                          # maximum refractive index
NLAM  = 10                                      # grid resolution, resolve nmax with 10pts
lam0  = c0/fmax                                 # min wavelength in simulation
dz = lam0/nmax/NLAM          # step size in x/z-direction
zmax = 100.
Z   = np.arange(-zmax/2.,1.5*zmax,dz)  # X-distance and Z-depth arrays for domain
Nz = len(Z)                                 # number of x/z points in grid
nbed_1 = np.argmin(abs(Z-0.8*zmax))                              # bed start location
nbed_2 = -1                              # bed start location

# Initialize material constants
Er = erice*np.ones(Nz)              # relative permittivity
Ur = (1/erice)*np.ones(Nz)                    # relative permeability
# change the permittivity for some random layers that have debris and for the bed
rand_ind  = np.where(np.random.rand(Nz)>0.8)
Er[rand_ind] = erdebris       # relative permittivity in the slab
Er[nbed_1:nbed_2] = erbed                   # relative permittivity in the bed

# Time Domain
nbc   = np.sqrt(Ur[0]*Er[0])        # refractive index at boundary
dt    = nbc*dz/(2*c0)               # time step
tau   = 0.5/fmax                    # duration of Gaussian source
t0    = 5.*tau                      # initial time, offset of Gaussian source
tprop = nmax*Nz*dz/c0               # time for wave accross grid
t_f     = 2.*t0 + 0.3*tprop          # total simulation time
steps = math.floor((t_f/dt))           # number of time steps
t     = np.arange(0,steps)*dt       # update simulation time

# Source
#nz_src = math.ceil(Nz/2.)                   # Index of Source Location (centered)
nz_src = np.argmin(abs(Z-0.))               # Source at the start
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
PABC = True
H1,H2,H3 = 0,0,0
E1,E2,E3 = 0,0,0

############################################################
### Figure ###

"""
fig = plt.figure(figsize=(6,6))
ax1 = plt.subplot(111)
ax1.set_xlim(-1.5,1.5)
ax1.set_ylim(1.,0)
plt.ylabel('Distance')
plt.xlabel('Normalized EM Field')
plt.ion()
time_text = ax1.text(0.5,1.1,'',ha='center',transform=ax1.transAxes)

ax2 = plt.subplot(122)
plt.xlim(0,dt*steps)
plt.ylim(-1,1)
plt.xlabel('seconds')
plt.ylabel('E-return')

# fill the bed and debris locations
for i in rand_ind[0]:
    ax1.axhline(Z[i],c='k',alpha=0.2)
ax1.fill_between(np.linspace(-5,5,10),Z[nbed_1],Z[nbed_2],color='k',zorder=0)
# plot Electric and Magnetic field
H_line, = ax1.plot([],[],'b',zorder=1)
E_line, = ax1.plot([],[],'r',zorder=2)

# plot the power output through time
#P_line, = ax2.plot([],[],'k')

#plt.tight_layout()
"""

############################################################
### Algorithm ###

E_out = np.empty((steps,len(Ey)))
H_out = np.empty((steps,len(Hx)))
P_out = [[],[]]

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

    # save the fields to an array
    #E_out[t_i] = Ey
    #H_out[t_i] = Hx/erice

    # Save the E-field at the top to an array
    P_out[0].append(dt*t_i)
    P_out[1].append(Ey[nz_src])
    """
    E_line.set_ydata(Z)
    E_line.set_xdata(Ey)
    H_line.set_ydata(Z+0.5*dz)
    H_line.set_xdata(Hx/erice)
    #P_line.set_xdata(P_out[0])
    #P_line.set_ydata(P_out[1])
    #time_text.set_text('Time Step = %0.0f of %0.0f' % (t_i,steps))
    plt.pause(0.00001)
    """
    print t_i, steps

    if t_i == int(steps/10):
        fig = plt.figure(figsize=(12,9))

        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(4,4)

        ax1 = plt.subplot(gs[:3,0])
        ax1.set_xlim(-1.5,1.5)
        ax1.set_ylim(zmax,0.)
        plt.ylabel('Depth')
        plt.xlabel('Normalized EM Field')
        plt.plot(Hx/erice,Z+0.5*dz,'b')
        plt.plot(Ey,Z,'r')
        for zi in rand_ind[0]:
            ax1.axhline(Z[zi],c='k',alpha=0.2)
        ax1.fill_between(np.linspace(-5,5,10),Z[nbed_1],Z[nbed_2],color='k',zorder=0)
        ax1.text(0.1,0.95,'(a)',size='large',weight='bold',ha='center',transform=ax1.transAxes)
        plt.title('%0.2E sec'%t[t_i])

    elif t_i == int(3*steps/10):
        ax2 = plt.subplot(gs[:3,1])
        ax2.set_xlim(-1.5,1.5)
        ax2.set_ylim(zmax,0.)
        ax2.tick_params(which='both',labelleft='off')
        plt.plot(Hx/erice,Z+0.5*dz,'b')
        plt.plot(Ey,Z,'r')
        for zi in rand_ind[0]:
            ax2.axhline(Z[zi],c='k',alpha=0.2)
        ax2.fill_between(np.linspace(-5,5,10),Z[nbed_1],Z[nbed_2],color='k',zorder=0)
        ax2.text(0.1,0.95,'(b)',size='large',weight='bold',ha='center',transform=ax2.transAxes)
        plt.title('%0.2E sec'%t[t_i])

    elif t_i == int(5*steps/10):
        ax3 = plt.subplot(gs[:3,2])
        ax3.set_xlim(-1.5,1.5)
        ax3.set_ylim(zmax,0.)
        ax3.tick_params(which='both',labelleft='off')
        plt.plot(Hx/erice,Z+0.5*dz,'b')
        plt.plot(Ey,Z,'r')
        for zi in rand_ind[0]:
            ax3.axhline(Z[zi],c='k',alpha=0.2)
        ax3.fill_between(np.linspace(-5,5,10),Z[nbed_1],Z[nbed_2],color='k',zorder=0)
        ax3.text(0.1,0.95,'(c)',size='large',weight='bold',ha='center',transform=ax3.transAxes)
        plt.title('%0.2E sec'%t[t_i])

    elif t_i == int(7*steps/10):
        ax4 = plt.subplot(gs[:3,3])
        ax4.set_xlim(-1.5,1.5)
        ax4.set_ylim(zmax,0.)
        ax4.tick_params(which='both',labelleft='off')
        plt.plot(Hx/erice,Z+0.5*dz,'b')
        plt.plot(Ey,Z,'r')
        for zi in rand_ind[0]:
            ax4.axhline(Z[zi],c='k',alpha=0.2)
        ax4.fill_between(np.linspace(-5,5,10),Z[nbed_1],Z[nbed_2],color='k',zorder=0)
        ax4.text(0.1,0.95,'(d)',size='large',weight='bold',ha='center',transform=ax4.transAxes)
        plt.title('%0.2E sec'%t[t_i])

ax5 = plt.subplot(gs[3,:])
ax5.set_ylim(-1.,1.)
plt.plot(P_out[0],P_out[1])
ax5.text(0.04,0.85,'(e)',size='large',weight='bold',ha='center',transform=ax5.transAxes)
plt.ylabel('E-return')
plt.xlabel('seconds')

plt.tight_layout()

plt.savefig('IceSimulation.png',dpi=300)

############################################################
"""
def init():
    E_line.set_data([],[])
    H_line.set_data([],[])
    P_line.set_data([],[])
    return E_line, H_line, P_line,

def animate(i):
    E_line.set_ydata(Z)
    E_line.set_xdata(E_out[i])
    H_line.set_ydata(Z+0.5*dz)
    H_line.set_xdata(H_out[i])
    P_line.set_xdata(P_out[0][:i])
    P_line.set_ydata(P_out[1][:i])
    time_text.set_text('Time Step = %0.0f of %0.0f' % (i,steps))
    return E_line, H_line, P_line, time_text,

ani = animation.FuncAnimation(fig,animate,init_func=init,frames=np.arange(0,steps,2),interval=20,blit=True)

# Save the animation
ani.save('WithLayers.mp4',writer="ffmpeg")
"""
