"""
This is a modified version of ns_2d.py from https://github.com/zongyi-li/fourier_neural_operator
"""

import torch

import math

from random_fields import GaussianRF

from timeit import default_timer

import scipy.io

import argparse

import os

from tqdm import tqdm

#w0: initial vorticity
#f: forcing term
#visc: viscosity (1/Re)
#T: final time
#delta_t: internal time-step for solve (descrease if blow-up)
#record_steps: number of in-time snapshots to record
def navier_stokes_2d(w0, f, domain_size, visc, T, delta_t=1e-4, record_steps=1):

    #Grid size - must be power of 2
    N = w0.size()[-1]

    #Maximum frequency
    k_max = math.floor(N/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)

    #Forcing to Fourier space
    f_h = torch.fft.rfft2(f)

    #If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    #Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)

    #Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2) / (domain_size ** 2)
    lap[0,0] = 1.0
    #Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    #Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    #Record counter
    c = 0
    #Physical time
    t = 0.0
    for j in range(steps):
        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        #Velocity field in x-direction = psi_y
        q = 2. * math.pi / domain_size * k_y * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N, N))

        #Velocity field in y-direction = -psi_x
        v = -2. * math.pi / domain_size * k_x * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N, N))

        #Partial x of vorticity
        w_x = 2. * math.pi / domain_size * k_x * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N, N))

        #Partial y of vorticity
        w_y = 2. * math.pi / domain_size * k_y * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N, N))

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q*w_x + v*w_y)

        #Dealias
        F_h = dealias* F_h

        #Crank-Nicolson update
        w_h = (-delta_t*F_h + delta_t*f_h + (1.0 - 0.5*delta_t*visc*lap)*w_h)/(1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            #Solution in physical space
            w = torch.fft.irfft2(w_h, s=(N, N))

            #Record solution and time
            sol[...,c] = w
            sol_t[c] = t

            c += 1


    return sol, sol_t

parser = argparse.ArgumentParser()
parser.add_argument("--nu", type=float, required=True)
parser.add_argument("--s", type=int, default=256)
parser.add_argument("--T", type=int, required=True, help="Time horizon")
parser.add_argument("--N", type=int, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--bsize", type=int, default=20)
parser.add_argument("--suffix", type=str, default=None)
parser.add_argument("--ntest", type=int, required=True, help="Number of superresolution examples")
parser.add_argument("--period", type=int, required=True, help="Period if sym is true")
parser.add_argument("--sym", action="store_true", default=True, help="Use a symmetric forcing term")
parser.add_argument("--domain_size", type=float, default=1)
args = parser.parse_args()

device = torch.device('cuda')

#Resolution
s = args.s # 256

#Number of solutions to generate
N = args.N # 20

#Set up 2d GRF with covariance parameters
GRF = GaussianRF(2, s, args.domain_size, alpha=2.5, tau=7, device=device)

#Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, args.domain_size, s+1, device=device)
t = t[0:-1]

X,Y = torch.meshgrid(t, t, indexing='ij')
if args.sym:
    f = 0.1 * (torch.cos(args.period * math.pi * X) + torch.cos(args.period * math.pi * Y))
else:
    f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

#Number of snapshots from solution
record_steps = args.T * 4 # 200

#Inputs
a = torch.zeros(N, s, s)
#Solutions
u = torch.zeros(N, s, s, record_steps)

#Solve equations in batches (order of magnitude speed-up)

#Batch size
bsize = args.bsize # 20

c = 0
t0 =default_timer()
for j in tqdm(range(N//bsize)):

    #Sample random feilds
    w0 = GRF.sample(bsize)

    #Solve NS
    sol, sol_t = navier_stokes_2d(w0, f, args.domain_size, args.nu, args.T, 1e-4, record_steps) # navier_stokes_2d(w0, f, 1e-3, 50.0, 1e-4, record_steps)

    a[c:(c+bsize),...] = w0
    u[c:(c+bsize),...] = sol

    c += bsize
    t1 = default_timer()
    print(j, c, t1-t0)

a_super = a[-args.ntest:]
u_super = u[-args.ntest:]

space_sub = s // 64
time_sub = 4

a = a[..., ::space_sub, ::space_sub]
u = u[..., ::space_sub, ::space_sub, ::time_sub]

if args.sym:
    data_name = f"ns_V{args.nu}_N{args.N}_T{args.T}_cos{args.period}{'_' + args.suffix if args.suffix is not None else ''}.mat"
else:
    data_name = f"ns_V{args.nu}_N{args.N}_T{args.T}_sin{'_' + args.suffix if args.suffix is not None else ''}.mat"
super_name = data_name[:-4] + "_super.mat"
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
save_dir = os.path.join(args.save_path, data_name)
super_dir = os.path.join(args.save_path, super_name)
scipy.io.savemat(save_dir, mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
scipy.io.savemat(super_dir, mdict={'a': a_super.cpu().numpy(), 'u': u_super.cpu().numpy(), 't': sol_t.cpu().numpy()})