import os
import sys
import math
import numpy as np
import torch
from torch import nn
from common.derivatives import WENO, FDM
from temporal.solvers import *
from numpy import pi
from scipy.fftpack import fft, ifft
from tqdm import tqdm
import matplotlib.pyplot as plt

class PDE(nn.Module):
    """Generic PDE template"""
    def __init__(self):
        # Data params for grid and initial conditions
        super().__init__()
        pass

    def __repr__(self):
        return "PDE"

    def FDM_reconstruction(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """A finite differences method template"""
        pass

    def FVM_reconstruction(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """A finite volumes method template"""
        pass

    def WENO_reconstruction(self, t: float, u: torch.Tensor) -> torch.Tensor:
        """A WENO reconstruction template"""
        pass


class CE(PDE):
    """
    Combined equation with Burgers and KdV as edge cases
    ut = -alpha*uux + beta*uxx + -gamma*uxxx = 0
    alpha = 6 for KdV and alpha = 1. for Burgers
    beta = nu for Burgers
    gamma = 1 for KdV
    alpha = 0, beta = nu, gamma = 0 for heat equation
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 L: float=None,
                 flux_splitting: str=None,
                 alpha: float=3.,
                 beta: float=0.,
                 gamma: float=1.,
                 device: torch.cuda.device = "cpu") -> None:
        """
        Args:
            tmin (float): starting time
            tmax (float): end time
            grid_size (list): grid points [nt, nx]
            L (float): periodicity
            flux_splitting (str): flux splitting used for WENO reconstruction (Godunov, Lax-Friedrichs)
            alpha (float): shock term
            beta (float): viscosity/diffusion parameter
            gamma (float): dispersive parameter
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        # Data params for grid and initial conditions
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 0.5 if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1
        self.lmax = 3
        # Number of different waves
        self.N = 5
        # Length of the spatial domain / periodicity
        self.L = 16 if L is None else L
        self.grid_size = (2 ** 4, 2 ** 6) if grid_size is None else grid_size
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device

        # Initialize WENO reconstrution object
        self.weno = WENO(self, order=3, device=self.device)
        self.fdm = FDM(self, device=self.device)
        self.force = None
        self.flux_splitting = f'godunov' if flux_splitting is None else flux_splitting

        assert (self.flux_splitting == f'godunov') or (self.flux_splitting == f'laxfriedrichs')


    def __repr__(self):
        return f'CE'

    def flux(self, input: torch.Tensor) -> torch.Tensor:
        """
        Flux as used in weno scheme for CE equations
        """
        return 0.5 * input ** 2

    def FDM_reconstruction(self, t: float, u):
        raise


    def WENO_reconstruction(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute derivatives using WENO scheme
        update = -alpha*uux + beta*uxx - gamma*uxxx
        weno reconstruction for uux
        FDM reconstruction gives uxx, uxxx
        Args:
            t (torch.Tensor): timepoint at which spatial terms are reconstructed, only important for time-dependent forcing term
            u (torch.Tensor): input fields at given timepoint
        Returns:
            torch.Tensor: reconstructed spatial derivatives
        """
        dudt = torch.zeros_like(u)

        # WENO reconstruction of advection term
        u_padded_weno = self.weno.pad(u)
        if self.flux_splitting == f'godunov':
            dudt = - self.alpha * self.weno.reconstruct_godunov(u_padded_weno, self.dx)

        if self.flux_splitting == f'laxfriedrichs':
            dudt = - self.alpha * self.weno.reconstruct_laxfriedrichs(u_padded_weno, self.dx)

        # reconstruction of diffusion term
        u_padded_fdm = self.fdm.pad(u)
        uxx = self.fdm.second_derivative(u_padded_fdm)
        uxxx = self.fdm.third_derivative(u_padded_fdm)

        dudt += self.beta*uxx
        dudt -= self.gamma*uxxx

        # Forcing term
        if self.force:
            dudt += self.force(t)

        return dudt


class WE(PDE):
    """
    utt = c2uxx
    Dirichlet BCs:  u(−1,t)=0  and  u(+1,t)=0  for all  t>0
    Neumann BCs:  ux(−1,t)=0 and  ux(+1,t)=0  for all  t>0
    We implement the 2nd-order in time PDE as a 1st-order augmented state-space equation.
    We introduce a new variable  v, such that  ut=vut=v , so  ut=v, so utt=vt.
    For discretization, it is better to just use v as a storage variable for ut and compute utt directly from u.
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 xmin: float=None,
                 xmax: float=None,
                 grid_size: list=None,
                 bc_left: str=None,
                 bc_right: str=None,
                 device: torch.cuda.device = "cpu") -> None:
        """
        Args:
            tmin (float): starting time
            tmax (float): end time
            xmin (float): left spatial boundary
            xmax (float): right spatial boundary
            grid_size (list): grid points [nt, nx]
            bc_left (str): left boundary condition [Dirichlet, Neumann]
            bc_right (str): right boundary condition [Dirichlet, Neumann]
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        # Data params for grid and initial conditions
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 20 if tmax is None else tmax
        # Left and right spatial boundaries
        self.xmin = -8 if xmin is None else xmin
        self.xmax = 8 if xmax is None else xmax
        # Length of the spatial domain
        self.L = abs(self.xmax - self.xmin)
        self.grid_size = (2 ** 4, 2 ** 6) if grid_size is None else grid_size
        self.dt = self.tmax / (self.grid_size[0] - 1)
        self.dx = self.L / (self.grid_size[1] - 1)
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        self.bc_left = "dirichlet" if bc_left is None else bc_left
        self.bc_right = "dirichlet" if bc_right is None else bc_right
        # Initialize Chebyshev pseudospectral solvers
        self.cheb = Cheb()

    def __repr__(self):
        return f'WE'

    def chebdx(self, t: np.array, u: np.array, x: np.array, c: float = 1.) -> np.array:
        """
        Compute the spatial derivatives using the pseudo-spectral method.
        Args:
            t (np.array): timepoint at which spatial terms are reconstructed, only important for time-dependent forcing term
            x (np.array): non-regular spatial grid
            u (np.array): input fields at given timepoint
            c (float): wave speed
        Returns:
            np.array: reconstructed spatial derivatives
        """
        N = len(u)
        # We store the first derivative in v and 2nd derivative in w. We do not use an extra
        # array dimension, because solve_ivp only works with 1D arrays.
        v, w = u[:N // 2], u[N // 2:]

        # Compute
        vt = w

        # The BCs are implemented with the dictionary. The key is the spatial derivative order,
        # the left and right values are the boundary condition values
        boundary_condition_left = 0
        boundary_condition_right = 0
        if self.bc_left == "dirichlet":
            boundary_condition_left = 0
        elif self.bc_left == "neumann":
            boundary_condition_left = 1
        if self.bc_right == "dirichlet":
            boundary_condition_right = 0
        elif self.bc_right == "neumann":
            boundary_condition_right = 1

        if(boundary_condition_left is not boundary_condition_right):
            wt = c ** 2 * self.cheb.solve(v, x, {boundary_condition_left: (0, None), boundary_condition_right: (None, 0)}, m=2)
        else:
            wt = c ** 2 * self.cheb.solve(v, x, {boundary_condition_left: (0, 0)}, m=2)

        # Compute du/dt.
        dudt = np.concatenate([vt, wt])

        return dudt

class AD(PDE):
    """
    Linear advection system in one space dimension with speeds a and b
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 L: float=None,
                 a: float = 1.,
                 b: float = 1.,
                 device: torch.cuda.device = "cpu") -> None:
        """
        Args:
            tmin (float): starting time
            tmax (float): end time
            grid_size (list): grid points [nt, nx]
            L (float): periodicity
            a (float): first dim speed
            b (float): second dim speed
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        # Data params for grid and initial conditions
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 0.5 if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1
        self.lmax = 3
        # Number of different waves
        self.N = 5
        # Length of the spatial domain / periodicity
        self.L = 16 if L is None else L
        self.grid_size = (2 ** 4, 2 ** 6) if grid_size is None else grid_size
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.a = a
        self.b = b

        self.device = device

        #transition matrix
        self.R = torch.tensor([[-1.,1.],[1.,1.]]).to(device)
        self.Rinv = torch.tensor([[-1/2,1/2],[1/2,1/2]]).to(device)

        #unstructured grid boolean
        self.untructured_grid = False

    def __repr__(self):
        return f'AD'

    def get_sol(self,u0_f,x,t):
        self.lam1 = 2*self.a
        self.lam2 = 2*self.b
        solved = torch.empty(2,1,len(t),len(x))
        w0_f = lambda x,u0_f=u0_f,Rinv = self.Rinv : Rinv@u0_f(x)[:,0,:]
        for i,ti in enumerate(t):
            w1 = w0_f(x-self.lam1*ti)[0][None,]
            w2 = w0_f(x-self.lam2*ti)[1][None,]
            w = torch.cat((w1,w2),dim=0)
            u = self.R@w
            solved[:,0,i,:] = u
        return solved

def cheb_points(N: int) -> np.array:
    """
    Get point grid array of Chebyshev extremal points
    """
    return np.cos(np.arange(0, N) * np.pi / (N-1))

class KF(PDE):
    """
    1D reaction diffusion equation (we use the kolmogorov fisher case where f(u) = u(1-u))
    ut = D*u_xx + r*f(u)
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 L: float=None,
                 r: float=1.,
                 D : float = 0.1,
                 f = None,
                 device: torch.cuda.device = "cpu",
                 bc : str = "dirichlet") -> None:
        """
        Args:
            tmin (float): starting time
            tmax (float): end time
            grid_size (list): grid points [nt, nx]
            L (float): periodicity
            r (float): reaction parameter
            D (float): viscosity/diffusion parameter
            f (float): reaction function
            device (torch.cuda.device): device (cpu/gpu)
            bc : boundary conditions type, atm support only dirichlet (in utils.py downsampling)
        Returns:
            None
        """
        if f is None :
            f = lambda u: -u*(u-1)
        self.f = f 

        # Data params for grid and initial conditions
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 0.5 if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1
        self.lmax = 8
        # Number of different waves
        self.N = 5
        # Length of the spatial domain / periodicity
        self.L = 16 if L is None else L
        self.grid_size = (2 ** 4, 2 ** 6) if grid_size is None else grid_size
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.r = r
        self.D = D
        self.device = device

        # Initialize FDM object
        self.fdm = FDM(self, device=self.device)

        self.bc = bc

        #compute dirichlet finite difference matrix
        a = torch.ones((1, self.grid_size[1]))[0]*(-49/18)
        b = torch.ones((1, self.grid_size[1]-1))[0]*(3/2)
        c = torch.ones((1, self.grid_size[1]-2))[0]*(-3/20)
        d = torch.ones((1, self.grid_size[1]-3))[0]*(1/90)

        m = torch.diag(a, 0) + np.diag(b, -1) + np.diag(b, 1) + np.diag(c, -2) + np.diag(c, 2) + np.diag(d,-3) + np.diag(d, 3)
        self.m = ((1 / self.dx)**2 * m).to(device)



    def __repr__(self):
        return f'KF'

    def RHS(self, t: np.array, u: np.array, a: float = 1.0):
        #diffusion term
        if self.bc == "periodic":
            u_padded_fdm = self.fdm.pad(u)
            uxx = self.fdm.second_derivative(u_padded_fdm[None,].permute(1,0,2)).squeeze(1)
            out = self.D*uxx + self.r*self.f(u)
        elif self.bc == "dirichlet":
            out = self.D*torch.einsum('ii,ki->ki',self.m,u) - self.r*u*(u-1)
        return out


class Cheb:
    """
    Class for pseudospectral reconstruction using Chebyshev basis
    """
    # TODO: implement Cheb class in PyTorch
    def __init__(self):
        self.diffmat = {}
        self.bc_mat = {}
        self.basis = {}
        self.D = {}

    def poly(self, n: np.array, x: np.array) -> np.array:
        """
        Chebyshev polynomials for x \in [-1,1],
        see https://en.wikipedia.org/wiki/Chebyshev_polynomials#Trigonometric_definition
        """
        return np.cos(n * np.arccos(x))

    def chebder(self, N: int, m: int) -> np.array:
        """
        Get Chebyshev derivatives of order m
        Args:
            N (int): number of grid points
            m (int): order of derivative
        Returns:
            np.array: derivatives of m-th order
        """
        diffmat = np.zeros((N - m, N))
        for i in range(N):
            c = np.zeros((N,))
            c[i] = 1
            diffmat[:, i] = np.polynomial.chebyshev.chebder(c, m=m)
        return diffmat

    def get_basis(self, x: np.array) -> np.array:
        """Get polynomial basis matrix
        Args:
            x: num points
        Returns:
            np.array: [N, N] basis matrix where N = len(x)
        """
        N = len(x)
        # Memoization
        if N in self.basis:
            return self.basis[N]

        # Get domain
        L = np.abs(x[0] - x[-1])
        x = cheb_points(N)[:, None]
        n = np.arange(N)[None, :]

        # Compute basis
        self.basis[N] = self.poly(n, x)
        return self.basis[N]

    def solve(self, u: np.array, x: np.array, bcs: list, m: int=1) -> np.array:
        """Differentiation matrix with boundary conditions on [-1, 1]
        f = T @ f_hat
        Args:
            u (np.array): field values
            x (np.array): spatial points
            bcs (list): boundary conditions [left boundary, right boundary]
            m (int): order of derivative
        Returns:
            np.array: [N, N] differentation matrix where N = len(x)
        """
        N = len(x)
        key = hash((N, m) + tuple(bcs.items()))

        # Memoization
        if key not in self.diffmat:
            T = self.get_basis(x)
            L = np.abs(x[0] - x[-1])

            # Cut off boundaries
            t0 = T[:1, :]
            t1 = T[-1:, :]
            T_int = T[1:-1, :]

            # Boundary bordering LHS
            bc_mat = []
            for order, bc in bcs.items():
                if order > 0:
                    # Basis derivative matrix
                    D = self.chebder(N, m=order)
                    D *= (-2 / L) ** order

                    # Differentiate the basis on the boundary
                    t0m = t0[:, :-order] @ D
                    t1m = t1[:, :-order] @ D
                else:
                    t0m = t0
                    t1m = t1

                # Add BCs
                if bc[0] is not None and bc[1] is not None:
                    T_int = np.concatenate([t0m, t1m, T_int], 0)
                    bc_mat = np.concatenate([[bc[0]], [bc[1]], bc_mat], 0)

                else:
                    if bc[0] is not None:
                        T_int = np.concatenate([t0m, T_int], 0)
                        bc_mat = np.concatenate([[bc[0]], bc_mat], 0)
                    if bc[1] is not None:
                        T_int = np.concatenate([t1m, T_int], 0)
                        bc_mat = np.concatenate([[bc[1]], bc_mat], 0)


            # Compute inverse
            Tinv = np.linalg.pinv(T_int)
            diffmat = self.chebder(N, m=m)
            diffmat *= (-2 / L) ** m

            self.diffmat[key] = T[:, :-m] @ diffmat @ Tinv
            self.bc_mat[key] = bc_mat

        # Boundary bordering RHS
        # Cut off boundaries
        u = u[1:-1]
        # Add BCs
        u = np.concatenate([self.bc_mat[key], u], 0)

        return self.diffmat[key] @ u

class KS:
    # Code from CSE Lab ETH/Harvard : https://github.com/cselab/LED/blob/main/Code/Data/KSGP64L22/Utils/KS.py
    # Solution of the 1D Kuramoto-Sivashinsky equation
    #
    # u_t + u*u_x + u_xx + u_xxxx = 0,
    # with periodic BCs on x \in [0, 2*pi*L]: u(x+2*pi*L,t) = u(x,t).
    #
    # The nature of the solution depends on the system size L and on the initial
    # condition u(x,0).  Energy enters the system at long wavelengths via u_xx
    # (an unstable diffusion term), cascades to short wavelengths due to the
    # nonlinearity u*u_x, and dissipates via diffusion with u_xxxx.
    #
    # Spatial  discretization: spectral (Fourier)
    # Temporal discretization: exponential time differencing fourth-order Runge-Kutta
    # see AK Kassam and LN Trefethen, SISC 2005

    def __init__(self, L=16, nx=128, dt=0.25, nsteps=None, tend=150, iout=1, u0=None, tstart=None, dt_downsampled=250):
        #
        # Initialize
        
        if tstart is not None:
            self.tstart = tstart
            tend = tend - tstart
        else:
            self.tstart = 0.0

        L  = float(L); dt = float(dt); tend = float(tend)
        if (nsteps is None):
            nsteps = int(tend/dt)
            nsteps_downsampled = int(tend/dt_downsampled)
        else:
            nsteps = int(nsteps)

            # override tend
            tend = dt*nsteps
        self.tend = tend
        #
        # save to self
        self.L      = L
        self.nx     = nx
        self.dx     = 2*pi*L/nx
        self.dt     = dt
        self.dt_downsampled = dt_downsampled
        self.nsteps = nsteps
        self.nsteps_downsampled = nsteps_downsampled
        self.iout   = iout
        self.nout   = int(nsteps/iout)

        # Sine frequencies for initial conditions
        self.lmin = 1
        self.lmax = 3
        # Number of different waves
        self.N = 5
        
        #
        # set initial condition
        if u0 is None:
            #print("NO INITIAL CONDITION.")
            self.IC()
        else:
            #print("STARTING FROM GIVEN INITIAL CONDITION.")
            self.IC(u0)
        #
        # initialize simulation arrays

        self.setup_timeseries()
        #
        # precompute Fourier-related quantities
        self.setup_fourier()
        #
        # precompute ETDRK4 scalar quantities:
        self.setup_etdrk4()

    def __repr__(self):
        return f'KS'

    def setup_timeseries(self, nout=None):
        if (nout != None):
            self.nout = int(nout)
        # nout+1 so we store the IC as well
        self.vv = np.zeros([self.nout+1, self.nx], dtype=np.complex64)
        self.tt = np.zeros(self.nout+1)
        #
        # store the IC in [0]
        self.vv[0,:] = self.v0
        self.tt[0]   = 0.


    def setup_fourier(self, coeffs=None):
        self.x  = 2*pi*self.L*np.r_[0:self.nx]/self.nx
        self.k  = np.r_[0:self.nx/2, 0, -self.nx/2+1:0]/self.L # Wave numbers
        # Fourier multipliers for the linear term Lu
        if (coeffs is None):
            # normal-form equation
            self.l = self.k**2 - self.k**4
        else:
            # altered-coefficients 
            self.l = -      coeffs[0]*np.ones(self.k.shape) \
                     -      coeffs[1]*1j*self.k             \
                     + (1 + coeffs[2])  *self.k**2          \
                     +      coeffs[3]*1j*self.k**3          \
                     - (1 + coeffs[4])  *self.k**4


    def setup_etdrk4(self):
        self.E  = np.exp(self.dt*self.l)
        self.E2 = np.exp(self.dt*self.l/2.)
        self.M  = 62                                           # no. of points for complex means
        self.r  = np.exp(1j*pi*(np.r_[1:self.M+1]-0.5)/self.M) # roots of unity
        self.LR = self.dt*np.repeat(self.l[:,np.newaxis], self.M, axis=1) + np.repeat(self.r[np.newaxis,:], self.nx, axis=0)
        self.Q  = self.dt*np.real(np.mean((np.exp(self.LR/2.) - 1.)/self.LR, 1))
        self.f1 = self.dt*np.real( np.mean( (-4. -    self.LR              + np.exp(self.LR)*( 4. - 3.*self.LR + self.LR**2) )/(self.LR**3) , 1) )
        self.f2 = self.dt*np.real( np.mean( ( 2. +    self.LR              + np.exp(self.LR)*(-2. +    self.LR             ) )/(self.LR**3) , 1) )
        self.f3 = self.dt*np.real( np.mean( (-4. - 3.*self.LR - self.LR**2 + np.exp(self.LR)*( 4. -    self.LR             ) )/(self.LR**3) , 1) )
        self.g  = -0.5j*self.k


    def IC(self, u0=None, v0=None, testing=False):
        #
        # Set initial condition, either provided by user or by "template"
        if (v0 is None):
            # IC provided in u0 or use template
            if (u0 is None):
                # set u0
                if testing:
                    # template from AK Kassam and LN Trefethen, SISC 2005
                    u0 = np.cos(self.x/self.L)*(1. + np.sin(self.x/self.L))
                else:
                    # random noise
                    u0 = (np.random.rand(self.nx) -0.5)*0.01
            else:
                # check the input size
                if (np.size(u0,0) != self.nx):
                    print('Error: wrong IC array size')
                    return -1
                else:
                    # if ok cast to np.array
                    u0 = np.array(u0)
            # in any case, set v0:
            v0 = fft(u0)
        else:
            # the initial condition is provided in v0
            # check the input size
            if (np.size(v0,0) != self.nx):
                print('Error: wrong IC array size')
                return -1
            else:
                # if ok cast to np.array
                v0 = np.array(v0)
                # and transform to physical space
                u0 = ifft(v0)
        #
        # and save to self
        self.u0  = u0
        self.v0  = v0
        self.v   = v0
        self.t   = 0.
        self.stepnum = 0
        self.ioutnum = 0 # [0] is the initial condition
        

    def step(self):
        #
        # Computation is based on v = fft(u), so linear term is diagonal.
        # The time-discretization is done via ETDRK4
        # (exponential time differencing - 4th order Runge Kutta)
        #
        v = self.v;                           Nv = self.g*fft(np.real(ifft(v))**2)
        a = self.E2*v + self.Q*Nv;            Na = self.g*fft(np.real(ifft(a))**2)
        b = self.E2*v + self.Q*Na;            Nb = self.g*fft(np.real(ifft(b))**2)
        c = self.E2*a + self.Q*(2.*Nb - Nv);  Nc = self.g*fft(np.real(ifft(c))**2)
        #
        self.v = self.E*v + Nv*self.f1 + 2.*(Na + Nb)*self.f2 + Nc*self.f3
        self.stepnum += 1
        self.t       += self.dt


    def simulate(self, nsteps=None, iout=None, restart=False, correction=[]):
        #
        # If not provided explicitly, get internal values
        if (nsteps is None):
            nsteps = self.nsteps
        else:
            nsteps = int(nsteps)
            self.nsteps = nsteps
        if (iout is None):
            iout = self.iout
            nout = self.nout
        else:
            self.iout = iout
        if restart:
            # update nout in case nsteps or iout were changed
            nout      = int(nsteps/iout)
            self.nout = nout
            # reset simulation arrays with possibly updated size
            self.setup_timeseries(nout=self.nout)

        #self.tqdm = tqdm(total=self.nsteps)

        # advance in time for nsteps steps
        if (correction==[]):
            for n in range(1,self.nsteps+1):
                try:
                    self.step()
                except FloatingPointError:
                    #
                    # something exploded
                    # cut time series to last saved solution and return
                    self.nout = self.ioutnum
                    self.vv.resize((self.nout+1,self.nx)) # nout+1 because the IC is in [0]
                    self.tt.resize(self.nout+1)          # nout+1 because the IC is in [0]
                    return -1
                if ( (self.iout>0) and (n%self.iout==0) ):
                    self.ioutnum += 1
                    self.vv[self.ioutnum,:] = self.v
                    self.tt[self.ioutnum]   = self.t

                #self.tqdm.update(1)
        else:
            # lots of code duplication here, but should improve speed instead of having the 'if correction' at every time step
            for n in range(1,self.nsteps+1):
                try:
                    self.step()
                    self.v += correction
                except FloatingPointError:
                    #
                    # something exploded
                    # cut time series to last saved solution and return
                    self.nout = self.ioutnum
                    self.vv.resize((self.nout+1,self.nx)) # nout+1 because the IC is in [0]
                    self.tt.resize(self.nout+1)          # nout+1 because the IC is in [0]
                    return -1
                if ( (self.iout>0) and (n%self.iout==0) ):
                    self.ioutnum += 1
                    self.vv[self.ioutnum,:] = self.v
                    self.tt[self.ioutnum]   = self.t

                #self.tqdm.update(1)
        #self.tqdm.close()

    def fou2real(self):
        #
        # Convert from spectral to physical space
        self.uu = np.real(ifft(self.vv))


    def compute_Ek(self):
        #
        # compute all forms of kinetic energy
        #
        # Kinetic energy as a function of wavenumber and time
        self.compute_Ek_kt()
        # Time-averaged energy spectrum as a function of wavenumber
        self.Ek_k = np.sum(self.Ek_kt, 0)/(self.ioutnum+1) # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0]
        # Total kinetic energy as a function of time
        self.Ek_t = np.sum(self.Ek_kt, 1)
		# Time-cumulative average as a function of wavenumber and time
        self.Ek_ktt = np.cumsum(self.Ek_kt, 0) / np.arange(1,self.ioutnum+2)[:,None] # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0] +1 more because we divide starting from 1, not zero
		# Time-cumulative average as a function of time
        self.Ek_tt = np.cumsum(self.Ek_t, 0) / np.arange(1,self.ioutnum+2)[:,None] # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0] +1 more because we divide starting from 1, not zero

    def compute_Ek_kt(self):
        try:
            self.Ek_kt = 1./2.*np.real( self.vv.conj()*self.vv / self.nx ) * self.dx
        except FloatingPointError:
            #
            # probable overflow because the simulation exploded, try removing the last solution
            problem=True
            remove=1
            self.Ek_kt = np.zeros([self.nout+1, self.nx]) + 1e-313
            while problem:
                try:
                    self.Ek_kt[0:self.nout+1-remove,:] = 1./2.*np.real( self.vv[0:self.nout+1-remove].conj()*self.vv[0:self.nout+1-remove] / self.nx ) * self.dx
                    problem=False
                except FloatingPointError:
                    remove+=1
                    problem=True
        return self.Ek_kt


    def space_filter(self, k_cut=2):
        #
        # spatially filter the time series
        self.uu_filt  = np.zeros([self.nout+1, self.nx])
        for n in range(self.nout+1):
            v_filt = np.copy(self.vv[n,:])    # copy vv[n,:] (otherwise python treats it as reference and overwrites vv on the next line)
            v_filt[np.abs(self.k)>=k_cut] = 0 # set to zero wavenumbers > k_cut
            self.uu_filt[n,:] = np.real(ifft(v_filt))
        #
        # compute u_resid
        self.uu_resid = self.uu - self.uu_filt


    def space_filter_int(self, k_cut=2, N_int=10):
        #
        # spatially filter the time series
        self.N_int        = N_int
        self.uu_filt      = np.zeros([self.nout+1, self.nx])
        self.uu_filt_int  = np.zeros([self.nout+1, self.N_int])
        self.x_int        = 2*pi*self.L*np.r_[0:self.N_int]/self.N_int
        for n in range(self.nout+1):
            v_filt = np.copy(self.vv[n,:])   # copy vv[n,:] (otherwise python treats it as reference and overwrites vv on the next line)
            v_filt[np.abs(self.k)>=k_cut] = 313e6
            v_filt_int = v_filt[v_filt != 313e6] * self.N_int/self.nx
            self.uu_filt_int[n,:] = np.real(ifft(v_filt_int))
            v_filt[np.abs(self.k)>=k_cut] = 0
            self.uu_filt[n,:] = np.real(ifft(v_filt))
        #
        # compute u_resid
        self.uu_resid = self.uu - self.uu_filt
