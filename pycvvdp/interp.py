
import torch
import numpy as np 
import os
import sys
import math

def bucketize(tensor, bucket_boundaries):
    if tensor.device.type != 'mps':
        return torch.bucketize(tensor, bucket_boundaries)

    # MPS does not support bucketize yet
    result = torch.zeros_like(tensor, dtype=torch.int32)
    for boundary in bucket_boundaries:
        result += (tensor > boundary).int()
    assert (result.cpu() == torch.bucketize(tensor.cpu(), bucket_boundaries.cpu())).flatten().all()
    return result

# x_q : query tensor 
# x   : boundaries tensor
# inspired from: https://github.com/sbarratt/torch_interpolations/blob/master/torch_interpolations/multilinear.py#L39
def get_interpolants_v1(x_q, x):
    imax = bucketize(x_q, x)
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

# works only with uniformly sampled LUTs
def get_interpolants_quick(x_q, x):
    ind = ((x_q-x[0])/(x[-1]-x[0])*(x.numel()-1)).clamp(0,x.shape[0] - 1)
    ifrc = torch.frac(ind)
    imin = ind.to(dtype=torch.int32)
    imax = (imin+1).clamp(max=x.shape[0] - 1)    
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

# A quick interpolation for uniformly spaces samples
def interp1q(x, v, x_q):
    shp = x_q.shape
    x_q = x_q.flatten()

    imin, imax, ifrc = get_interpolants_quick(x_q, x)

    filtered = v[imin] * (1.0-ifrc) + v[imax] * (ifrc) 

    return filtered.reshape(shp)



# Performs 1-d interpolation on the 2nd dimension of tensor 'v' (hard coded as I do not know how to do it otherwise)
# This is equivalent to performing such interpolation multiple times, for each slice of other dimensions.
# x - tensor with the LUT x values, only one dimension can have size != 1
# v - tensor with the LUT v values (to be interpolated)
# x_q - tensor with the query x values, only one dimension can have size != 1
def interp1dim2(x, v, x_q):
    assert x.dim() == 1, "'x' must be a 1D vector"
    assert x_q.dim() == 1, "'x_q' must be a 1D vector"
    assert x.shape[0] == v.shape[1], "'x' must have the same number of elements as the second dimensiom of v"

    imin, imax, ifrc = get_interpolants_v1(x_q, x)
    sh = [1] * v.dim()
    sh[1] = ifrc.shape[0]
    ifrc = ifrc.view(sh)

    filtered = v[:,imin,...] * (1.0-ifrc) + v[:,imax,...] * (ifrc) 

    return filtered

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


def batch_interp1d(x, xp, fp):
    """
    Perform batch-wise linear interpolation.
    """
    # Ensure xp is increasing
    assert torch.all(xp[1:] >= xp[:-1]), "xp must be in increasing order"

    # Make tensors contiguous to avoid warnings and optimize performance
    x = x.contiguous()
    xp = xp.contiguous()
    fp = fp.contiguous()

    # Find indices of the closest points
    indices = torch.searchsorted(xp, x) - 1
    indices = torch.clamp(indices, 0, len(xp) - 2)

    # Gather the relevant points
    x0 = xp[indices]
    x1 = xp[indices + 1]
    y0 = fp[torch.arange(fp.shape[0]), indices]
    y1 = fp[torch.arange(fp.shape[0]), indices + 1]

    # Compute the slope
    slope = (y1 - y0) / (x1 - x0)

    # Compute the interpolated values
    return y0 + slope * (x - x0)


if __name__ == '__main__':
    test_interp3(torch.device('cpu'))
