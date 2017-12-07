#!/usr/bin/env python

"""
Finite difference time domain model
1-dimensional

For solving the Maxwell Equations

Author: Ben Hills
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import math

c0 = 3e8        #m/s
e0 = 8.85e-12   # 1/m
u0 = 1.26e-6    # 1/m

# slab properties
dslab = 0.25/3.1
erair = 1.
erslab = 12.

# Source Parameters
fmax  = 5e9  # 1/s

# Grid Parameters
ermax = max([erair, erslab])
nmax  = np.sqrt(ermax)
NLAM  = 10
NDIM  = 1
NBUFZ = [100, 100]
lam0  = c0/fmax
dz1   = lam0/nmax/NLAM
dz2   = dslab/NDIM
dz    = min([dz1,dz2])
nz    = math.ceil(dslab/dz)
dz    = dslab/nz
Nz    = int(nz) + sum(NBUFZ) + 3
Z     = dz*np.arange(0,Nz)

# Initialize material constants
Er = erair*np.ones(Nz)
Ur = np.ones(Nz)

nz1 = 2 + NBUFZ[0] +1
nz2 = nz1 + math.ceil(dslab/dz) -1

Er[nz1:nz2] = erslab

# Time Domain
nbc   = np.sqrt(Ur[0]*Er[0])
dt    = nbc*dz/(2*c0)
tau   = 0.5/fmax
t0    = 5.*tau
tprop = nmax*Nz*dz/c0
t     = 2.*t0 + 3.*tprop
steps = math.ceil(t/dt)
# Source
t      = np.arange(0,steps)*dt
s      = dz/(2.*c0)+dt/2.
nz_src = math.ceil(Nz/6.)
Esrc   = np.exp(-((t-t0)/tau)**2.)
A      = -np.sqrt(Er[nz_src]/Ur[nz_src])
Hsrc   = A*np.exp(-((t-t0 +s)/tau)**2.)

# Initialize FDTD parametrs
mEy = (c0*dt)/Er
mHx = (c0*dt)/Ur

Ey = np.zeros(Nz)
Hx = np.zeros(Nz)

# Define transformation matrices
A = sp.lil_matrix((Nz,Nz))
A.setdiag(-1.*np.ones(Nz),k=0)
A.setdiag(np.ones(Nz-1),k=1)

B = sp.lil_matrix((Nz,Nz))
B.setdiag(np.ones(Nz),k=0)
B.setdiag(-1.*np.ones(Nz-1),k=-1)

A = A.tocsr()
B = B.tocsr()

# Dirichlet BCs
A[-1,:] = 0
B[0,:] = 0

# Perfectly absorbing BC
H1,H2,H3 = 0,0,0
E1,E2,E3 = 0,0,0

# Figure Setup
fig = plt.figure()
ax = plt.subplot()
ax.set_ylim(-1,1)
ax.set_xlim(min(Z),max(Z))

plt.fill_betweenx(np.linspace(-5,5,10),Z[nz1],Z[nz2])

plt.ion()

H_line, = plt.plot([],[],'b')
E_line, = plt.plot([],[],'r')

for t_i in np.arange(steps):
    Hx += (mHx/dz)*(A*Ey)
    Hx[-1] = Hx[-1] + mHx[-1]*(E3 - Ey[-1])/dz
    # Source
    Hx[nz_src-1] -= mHx[nz_src-1]*Esrc[t_i]/dz
    # Record H-field at Boundary
    H3 = H2
    H2 = H1
    H1 = Hx[0]

    Ey += (mEy/dz)*(B*Hx)
    Ey[0] = Ey[0] + mEy[0]*(Hx[0] - H3)/dz
    # Source
    Ey[nz_src] -= mEy[nz_src]*Hsrc[t_i]/dz
    # Record E-field at Boundary
    E3 = E2
    E2 = E1
    E1 = Ey[-1]

    # Plot
    H_line.set_xdata(Z)
    H_line.set_ydata(Hx)
    E_line.set_xdata(Z)
    E_line.set_ydata(Ey)
    plt.pause(0.000001)


"""
c0 = 3e8    # m/s
eps = 3.2#*np.ones_like(Z)
mu = 3.2#*np.ones_like(Z)
sigma = 7e-5#*np.ones_like(Z)

zi = 0
N = 100
dz = 1000.
Z = np.linspace(zi,N*dz+zi,N)

dt = 6/7.*np.sqrt(mu*eps/(1/dz**2.))   # s
tf = 1e-4   # s
ts = np.arange(0,tf,dt)

mE = c0*dt/eps
mH = c0*dt/mu

Ey = np.zeros_like(Z)
Hx = np.zeros_like(Z)

A = scipy.sparse.lil_matrix((N,N))
A.setdiag(-1.*np.ones(N),k=0)
A.setdiag(np.ones(N-1),k=1)

B = scipy.sparse.lil_matrix((N,N))
B.setdiag(np.ones(N),k=0)
B.setdiag(-1.*np.ones(N-1),k=-1)

A = A.tocsr()
B = B.tocsr()

# Dirichlet BCs
A[0,:],A[-1,:] = 0,0
B[0,:],B[-1,:] = 0,0

Source = np.exp(-((ts-tf/8.)/(dt*2.))**2.)

fig = plt.figure()
ax = plt.subplot()
ax.set_ylim(-1,1)
ax.set_xlim(min(Z),max(Z))

plt.ion()

H_line, = plt.plot([],[],'b')
E_line, = plt.plot([],[],'r')

t = 0.
count = 0
while t < tf:
    Ey[50] += Source[int(t/dt)]

    Hx += (mE/dz)*A*Ey
    t += (1/2.)*(dt)

    Ey += (mH/dz)*B*Hx
    t += (1/2.)*(dt)

    H_line.set_xdata(Z)
    H_line.set_ydata(Hx)
    E_line.set_xdata(Z)
    E_line.set_ydata(Ey)
    plt.pause(0.01)
    plt.draw()
"""
