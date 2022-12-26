
import torch
import numpy as np 
import os
import sys
import math

# x_q : query tensor 
# x   : boundaries tensor
# inspired from: https://github.com/sbarratt/torch_interpolations/blob/master/torch_interpolations/multilinear.py#L39
def get_interpolants_v1(x_q, x):
    imax = torch.bucketize(x_q, x)
    imax[imax >= x.shape[0]] = x.shape[0] - 1
    imin = (imax - 1).clamp(0, x.shape[0] - 1)

    ifrc = (x_q - x[imin]) / (x[imax] - x[imin] + 0.000001)
    ifrc[imax == imin] = 0.
    ifrc[ifrc < 0.0] = 0.

    return imin, imax, ifrc

def get_interpolants_v0(x_q, x, device):
    imin = torch.zeros(x_q.shape, dtype=torch.long).to(device)
    ifrc = torch.zeros(x_q.shape, dtype=torch.float32).to(device)
    N = x.shape[0]
    for i in range(N):
        if i==0:
            imin  = torch.where(x_q  <= x[i], torch.tensor(i, dtype=torch.long).to(device),  imin)
            ifrc  = torch.where(x_q  <= x[i], torch.tensor(0.).to(device), ifrc)

        if i==(N-1):
            imin  = torch.where(x[i] <= x_q,  torch.tensor(i, dtype=torch.long).to(device),  imin)
            ifrc  = torch.where(x[i] <= x_q,  torch.tensor(0.).to(device), ifrc)
        else:
            t = (x_q - x[i])/(x[i+1] - x[i])
            imin  = torch.where((x[i] <= x_q) & (x_q < x[i+1]), torch.tensor(i,dtype=torch.long).to(device), imin)
            ifrc  = torch.where((x[i] <= x_q) & (x_q < x[i+1]), t, ifrc)

    imax = torch.min(imin+1, torch.tensor(N-1, dtype=torch.long).to(device))

    return imin, imax, ifrc

def interp3(x, y, z, v, x_q, y_q, z_q):
    shp = x_q.shape
    x_q = x_q.flatten()
    y_q = y_q.flatten()
    z_q = z_q.flatten()

    imin, imax, ifrc = get_interpolants_v1(x_q, x)
    jmin, jmax, jfrc = get_interpolants_v1(y_q, y)
    kmin, kmax, kfrc = get_interpolants_v1(z_q, z)

    filtered = (
        ((v[jmin,imin,kmin] * (1.0-ifrc) + v[jmin,imax,kmin] * (ifrc)) * (1.0-jfrc) + 
         (v[jmax,imin,kmin] * (1.0-ifrc) + v[jmax,imax,kmin] * (ifrc)) *     (jfrc)) * (1.0 - kfrc) + 
        ((v[jmin,imin,kmax] * (1.0-ifrc) + v[jmin,imax,kmax] * (ifrc)) *     (1.0-jfrc) + 
         (v[jmax,imin,kmax] * (1.0-ifrc) + v[jmax,imax,kmax] * (ifrc)) *     (jfrc)) * (kfrc))

    return filtered.reshape(shp)

def interp1(x, v, x_q):
    shp = x_q.shape
    x_q = x_q.flatten()

    imin, imax, ifrc = get_interpolants_v1(x_q, x)

    filtered = v[imin] * (1.0-ifrc) + v[imax] * (ifrc) 

    return filtered.reshape(shp)


def test_interp3(device):
    x_q = torch.tensor([0.5, 1.9, 2.1], dtype=torch.float32).to(device)
    y_q = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32).to(device)
    z_q = torch.tensor([1.5, 2.0, 2.0], dtype=torch.float32).to(device)

    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to(device)
    y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to(device)
    z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).to(device)
    v = torch.tensor([
        [
        [10.0, 20.0, 30.0],
        [15.0, 30.0, 45.0],
        [20.0, 40.0, 60.0]],
        [
        [100.0, 200.0, 300.0],
        [150.0, 300.0, 450.0],
        [200.0, 400.0, 600.0]],
        [
        [1000.0, 2000.0, 3000.0],
        [1500.0, 3000.0, 4500.0],
        [2000.0, 4000.0, 6000.0]],
        ], dtype=torch.float32).to(device)

    print(x_q)
    print(x)
    print(v)
    print(interp3(x, y, z, v, x_q, y_q, z_q))


if __name__ == '__main__':
    test_interp3(torch.device('cpu'))
