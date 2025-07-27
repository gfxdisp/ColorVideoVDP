# Decimated Laplacian Pyramid
from functools import cache
import torch
import torch.nn.functional as Func
import numpy as np 
#import scipy.io as spio
#import os
#import sys
import math
#import torch.autograd.profiler as profiler

def ceildiv(a, b):
    return -(-a // b)

# Decimated Laplacian pyramid
class lpyr_dec():

    def __init__(self, W, H, ppd, device):
        self.device = device
        self.ppd = ppd
        self.min_freq = 0.2
        self.W = W
        self.H = H

        max_levels = int(np.floor(np.log2(min(self.H, self.W))))-1

        bands = np.concatenate([[1.0], np.power(2.0, -np.arange(0.0,14.0)) * 0.3228], 0) * self.ppd/2.0 

        # print(max_levels)
        # print(bands)
        # sys.exit(0)

        invalid_bands = np.array(np.nonzero(bands <= self.min_freq)) # we want to find first non0, length is index+1

        if invalid_bands.shape[-2] == 0:
            max_band = max_levels
        else:
            max_band = invalid_bands[0][0]

        # max_band+1 below converts index into count
        self.height = np.clip(max_band+1, 0, max_levels) # int(np.clip(max(np.ceil(np.log2(ppd)), 1.0)))
        self.band_freqs = np.array([1.0] + [0.3228 * 2.0 **(-f) for f in range(self.height)]) * self.ppd/2.0

        self.pyr_shape = self.height * [None] # shape (W,H) of each level of the pyramid
        self.pyr_ind = self.height * [None]   # index to the elements at each level

        cH = H
        cW = W
        for ll in range(self.height):
            self.pyr_shape[ll] = (cH, cW)
            cH = ceildiv(H,2)
            cW = ceildiv(W,2)

    def get_freqs(self):
        return self.band_freqs

    def get_band_count(self):
        return self.height+1

    def get_band(self, bands, band):
        if band == 0 or band == (len(bands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        return bands[band] * band_mul

    def set_band(self, bands, band, data):
        if band == 0 or band == (len(bands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        bands[band] = data / band_mul

    def get_gband(self, gbands, band):
        return gbands[band]

    # def get_gband_count(self):
    #     return self.height #len(gbands)

    # def clear(self):
    #     for pyramid in self.P:
    #         for level in pyramid:
    #             # print ("deleting " + str(level))
    #             del level

    def decompose(self, image): 
        # assert len(image.shape)==4, "NCHW (C==1) is expected, got " + str(image.shape)
        # assert image.shape[-2] == self.H
        # assert image.shape[-1] == self.W

        # self.image = image

        return self.laplacian_pyramid_dec(image, self.height+1)

    def reconstruct(self, bands):
        img = bands[-1]

        for i in reversed(range(0, len(bands)-1)):
            img = self.gausspyr_expand(img, [bands[i].shape[-2], bands[i].shape[-1]])
            img += bands[i]

        return img

    def laplacian_pyramid_dec(self, image, levels = -1, kernel_a = 0.4):
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return []

        lpyr = []
        for i in range(height-1):
            layer = gpyr[i] - self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
            lpyr.append(layer)

        lpyr.append(gpyr[height-1])

        # print("laplacian pyramid summary:")
        # print("self.height = %d" % self.height)
        # print("height      = %d" % height)
        # print("len(lpyr)   = %d" % len(lpyr))
        # print("len(gpyr)   = %d" % len(gpyr))
        # sys.exit(0)

        return lpyr, gpyr

    def interleave_zeros_and_pad(self, x, exp_size, dim):
        new_shape = [*x.shape]
        new_shape[dim] = exp_size[dim]+4
        z = torch.zeros( new_shape, dtype=x.dtype, device=x.device)
        odd_no = (exp_size[dim]%2)
        if dim==-2:
            z[...,2:-2:2,:] = x
            z[...,0,:] = x[...,0,:]
            z[...,-2+odd_no,:] = x[...,-1,:]
        elif dim==-1:
            z[...,:,2:-2:2] = x
            z[...,:,0] = x[...,:,0]
            z[...,:,-2+odd_no] = x[...,:,-1]
        else:
            assert False, "Wrong dimension"

        return z

    def gaussian_pyramid_dec(self, image, levels = -1, kernel_a = 0.4):

        default_levels = int(np.floor(np.log2(min(image.shape[-2], image.shape[-1]))))

        if levels == -1:
            levels = default_levels
        if levels > default_levels:
            raise Exception("Too many levels (%d) requested. Max is %d for %s" % (levels, default_levels, image.shape))

        res = [image]

        for i in range(1, levels):
            res.append(self.gausspyr_reduce(res[i-1], kernel_a))

        return res


    def sympad(self, x, padding, axis):
        if padding == 0:
            return x
        else:
            beg = torch.flip(torch.narrow(x, axis, 0,        padding), [axis])
            end = torch.flip(torch.narrow(x, axis, -padding, padding), [axis])

            return torch.cat((beg, x, end), axis)

    @cache
    def get_kernels( self, im, kernel_a = 0.4 ):

        # ch_dim = len(im.shape)-2
        # if hasattr(self, "K_horiz") and ch_dim==self.K_ch_dim:
        #     return self.K_vert, self.K_horiz

        K = torch.tensor([0.25 - kernel_a/2.0, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2.0], device=im.device, dtype=im.dtype)
        self.K_vert = torch.reshape(K, (1, 1, K.shape[0], 1))
        self.K_horiz = torch.reshape(K, (1, 1, 1, K.shape[0]))
        # self.K_ch_dim = ch_dim
        return self.K_vert, self.K_horiz
        

    def gausspyr_reduce(self, x, kernel_a = 0.4):

        K_vert, K_horiz = self.get_kernels( x, kernel_a )

        H, W = x.shape[-2], x.shape[-1]
        y_a = Func.conv2d(x.view(-1,1,H,W), K_vert, stride=(2,1), padding=(2,0)).view( x.shape[0:-2] + (-1,W) )
        # view(B,C,-1,W)

        # Symmetric padding 
        y_a[...,0,:] += x[...,0,:]*K_vert[...,1,0] + x[...,1,:]*K_vert[...,0,0]
        if (x.shape[-2] % 2)==1: # odd number of rows
            y_a[...,-1,:] += x[...,-1,:]*K_vert[...,3,0] + x[...,-2,:]*K_vert[...,4,0]
        else: # even number of rows
            y_a[...,-1,:] += x[...,-1,:]*K_vert[...,4,0]

        H = y_a.shape[-2]
        y = Func.conv2d(y_a.view(-1,1,H,W), K_horiz, stride=(1,2), padding=(0,2)).view( x.shape[0:-2] + (H,-1) )

        # Symmetric padding 
        y[...,:,0] += y_a[...,:,0]*K_horiz[...,0,1] + y_a[...,:,1]*K_horiz[...,0,0]
        if (x.shape[-2] % 2)==1: # odd number of columns
            y[...,:,-1] += y_a[...,:,-1]*K_horiz[...,0,3] + y_a[...,:,-2]*K_horiz[...,0,4]
        else: # even number of columns
            y[...,:,-1] += y_a[...,:,-1]*K_horiz[...,0,4] 

        return y

    def gausspyr_expand_pad(self, x, padding, axis):
        if padding == 0:
            return x
        else:
            beg = torch.narrow(x, axis, 0,        padding)
            end = torch.narrow(x, axis, -padding, padding)

            return torch.cat((beg, x, end), axis)

    # This function is (a bit) faster
    def gausspyr_expand(self, x, sz = None, kernel_a = 0.4):
        if sz is None:
            sz = [x.shape[-2]*2, x.shape[-1]*2]

        K_vert, K_horiz = self.get_kernels( x, kernel_a )

        y_a = self.interleave_zeros_and_pad(x, dim=-2, exp_size=sz)

        H, W = y_a.shape[-2], y_a.shape[-1]
        y_a = Func.conv2d(y_a.view(-1,1,H,W), K_vert*2).view( x.shape[0:-2] + (-1,W) )

        y   = self.interleave_zeros_and_pad(y_a, dim=-1, exp_size=sz)
        H, W = y.shape[-2], y.shape[-1]

        y   = Func.conv2d(y.view(-1,1,H,W), K_horiz*2).view( x.shape[0:-2] + (H,-1) )

        return y

    def interleave_zeros(self, x, dim):
        z = torch.zeros_like(x, device=self.device)
        if dim==2:
            return torch.cat([x,z],dim=3).view(x.shape[0], x.shape[1], 2*x.shape[2],x.shape[3])
        elif dim==3:
            return torch.cat([x.permute(0,1,3,2),z.permute(0,1,3,2)],dim=3).view(x.shape[0], x.shape[1], 2*x.shape[3],x.shape[2]).permute(0,1,3,2)



# Decimated Laplacian pyramid with a bit better interface - stores all bands within the object
class lpyr_dec_2(lpyr_dec):

    def __init__(self, W, H, ppd, device, keep_gaussian=False):
        self.device = device
        self.ppd = ppd
        self.min_freq = 0.2
        self.W = W
        self.H = H
        self.keep_gaussian=keep_gaussian

        max_levels = int(np.floor(np.log2(min(self.H, self.W))))-1

        bands = np.concatenate([[1.0], np.power(2.0, -np.arange(0.0,14.0)) * 0.3228], 0) * self.ppd/2.0 

        # print(max_levels)
        # print(bands)
        # sys.exit(0)

        invalid_bands = np.array(np.nonzero(bands <= self.min_freq)) # we want to find first non0, length is index+1

        if invalid_bands.shape[-2] == 0:
            max_band = max_levels
        else:
            max_band = invalid_bands[0][0]

        # max_band+1 below converts index into count
        self.height = np.clip(max_band+1, 0, max_levels) # int(np.clip(max(np.ceil(np.log2(ppd)), 1.0)))
        self.band_freqs = np.array([1.0] + [0.3228 * 2.0 **(-f) for f in range(self.height)]) * self.ppd/2.0

        self.pyr_shape = self.height * [None] # shape (W,H) of each level of the pyramid
        self.pyr_ind = self.height * [None]   # index to the elements at each level

        cH = H
        cW = W
        for ll in range(self.height):
            self.pyr_shape[ll] = (cH, cW)
            cH = ceildiv(H,2)
            cW = ceildiv(W,2)

        self.lbands = [None] * (self.height+1) # Laplacian pyramid bands
        if self.keep_gaussian:
            self.gbands = [None] * (self.height+1) # Gaussian pyramid bands

    def get_freqs(self):
        return self.band_freqs

    def get_band_count(self):
        return self.height+1

    def get_lband(self, band):
        if band == 0 or band == (len(self.lbands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        return self.lbands[band] * band_mul

    def set_lband(self, band, data):
        if band == 0 or band == (len(self.lbands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        self.lbands[band] = data / band_mul

    def get_gband(self, band):
        return self.gbands[band]

    # def clear(self):
    #     for pyramid in self.P:
    #         for level in pyramid:
    #             # print ("deleting " + str(level))
    #             del level

    def decompose(self, image): 
        return self.laplacian_pyramid_dec(image, self.height+1)

    def reconstruct(self):
        img = self.lbands[-1]

        for i in reversed(range(0, len(self.lbands)-1)):
            img = self.gausspyr_expand(img, [self.lbands[i].shape[-2], self.lbands[i].shape[-1]])
            img += self.lbands[i]

        return img

    def laplacian_pyramid_dec(self, image, levels = -1, kernel_a = 0.4):
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return

        lpyr = []
        for i in range(height-1):
            layer = gpyr[i] - self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
            lpyr.append(layer)

        lpyr.append(gpyr[height-1])
        self.lbands = lpyr

        if self.keep_gaussian:
            self.gbands = gpyr        

        return lpyr, gpyr

# This pyramid computes and stores contrast during decomposition, improving performance and reducing memory consumption
class weber_contrast_pyr(lpyr_dec):

    def __init__(self, W, H, ppd, device, contrast):
        super().__init__(W, H, ppd, device)
        self.contrast = contrast

    def decompose(self, image):
        levels = self.height+1
        kernel_a = 0.4
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return []

        lpyr = []
        L_bkg_pyr = []
        for i in range(height):
            is_baseband = (i==(height-1))

            if is_baseband:
                layer = gpyr[i]
                if self.contrast.endswith('ref'):
                    L_bkg = torch.clamp(gpyr[i][...,1:2,:,:,:], min=0.01)
                else:
                    # The sustained channels use the mean over the image as the background. Otherwise, they would be divided by itself and the contrast would be 1.
                    L_bkg = torch.mean(torch.clamp(gpyr[i][...,0:2,:,:,:], min=0.01), dim=[-1, -2], keepdim=True)
            else:
                glayer_ex = self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
                layer = gpyr[i] - glayer_ex 

                # Order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y
                # L_bkg is set to ref-sustained 
                if self.contrast == 'weber_g1_ref':
                    L_bkg = torch.clamp(glayer_ex[...,1:2,:,:,:], min=0.01)
                elif self.contrast == 'weber_g1':
                    L_bkg = torch.clamp(glayer_ex[...,0:2,:,:,:], min=0.01)
                elif self.contrast == 'weber_g0_ref':
                    L_bkg = torch.clamp(gpyr[i][...,1:2,:,:,:], min=0.01)
                else:
                    raise RuntimeError( f"Contrast {self.contrast} not supported")

            if L_bkg.shape[-4]==2: # If L_bkg NOT identical for the test and reference images
                contrast = torch.empty_like(layer)
                contrast[...,0::2,:,:,:] = torch.clamp(torch.div(layer[...,0::2,:,:,:], L_bkg[...,0:1,:,:,:]), max=1000.0)    
                contrast[...,1::2,:,:,:] = torch.clamp(torch.div(layer[...,1::2,:,:,:], L_bkg[...,1:2,:,:,:]), max=1000.0)    
            else:
                contrast = torch.clamp(torch.div(layer, L_bkg), max=1000.0)

            lpyr.append(contrast)
            L_bkg_pyr.append(torch.log10(L_bkg))

        # L_bkg_bb = gpyr[height-1][...,0:2,:,:,:]
        # lpyr.append(gpyr[height-1]) # Base band
        # L_bkg_pyr.append(L_bkg_bb) # Base band

        return lpyr, L_bkg_pyr


# This pyramid computes and stores contrast during decomposition, improving performance and reducing memory consumption
class log_contrast_pyr(lpyr_dec):

    def __init__(self, W, H, ppd, device, contrast):
        super().__init__(W, H, ppd, device)
        self.contrast = contrast

        # Assuming D65, there is a linear mapping from log10(L)+log10(M) to log10-luminance
        lms_d65 = [0.7347, 0.3163, 0.0208]
        self.a = 0.5
        self.b = math.log10(lms_d65[0]) - math.log10(lms_d65[1]) + math.log10(lms_d65[0]+lms_d65[1])

    def decompose(self, image):
        levels = self.height+1
        kernel_a = 0.4
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return []

        lpyr = []
        L_bkg_pyr = []
        for i in range(height):
            is_baseband = (i==(height-1))

            if is_baseband:
                contrast = gpyr[i]
                L_bkg = self.a * (gpyr[i][...,0:2,:,:,:] - self.b)
            else:
                glayer_ex = self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
                contrast = gpyr[i] - glayer_ex 

                # Order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y                
                # Mapping from log10(L) + log10(M) to log10(L+M)
                L_bkg = self.a * (glayer_ex[...,0:2,:,:,:] - self.b)

            lpyr.append(contrast)
            L_bkg_pyr.append(L_bkg)

        
        return lpyr, L_bkg_pyr


    # def gausspyr_expand(self, x, sz = None, kernel_a = 0.4):
    #     if sz is None:
    #         sz = [x.shape[-2]*2, x.shape[-1]*2]

    #     K_vert, K_horiz = self.get_kernels( x, kernel_a )

    #     y_a = self.interleave_zeros(x, dim=2)[...,0:sz[0],:]
    #     y_a = self.gausspyr_expand_pad(y_a, padding=2, axis=-2)
    #     y_a = Func.conv2d(y_a, K_vert*2)

    #     y   = self.interleave_zeros(y_a, dim=3)[...,:,0:sz[1]]
    #     y   = self.gausspyr_expand_pad(  y,   padding=2, axis=-1)
    #     y   = Func.conv2d(y, K_horiz*2)

    #     return y



# if __name__ == '__main__':

#     device = torch.device('cuda:0')

#     torch.set_printoptions(precision=2, sci_mode=False, linewidth=300)

#     image = torch.tensor([
#         [ 1,  2,  3,  4,  5,  6,  7,  8],
#         [11, 12, 13, 14, 15, 16, 17, 18],
#         [21, 22, 23, 24, 25, 26, 27, 28],
#         [31, 32, 33, 34, 35, 36, 37, 38],
#         [41, 42, 43, 44, 45, 46, 47, 48],
#         [51, 52, 53, 54, 55, 56, 57, 58],
#         [61, 62, 63, 64, 65, 66, 67, 68],
#         [71, 72, 73, 74, 75, 76, 77, 78],
#         ], dtype=torch.float32, device=device)

#     image = image.repeat((16, 16))
#     # image = torch.cat((image, image, image), axis = -1)
#     # image = torch.cat((image, image, image), axis = -2)

#     ppd = 50

#     im_tensor = image.view(1, 1, image.shape[-2], image.shape[-1])

#     lp = fvvdp_lpyr_dec_fast(im_tensor.shape[-2], im_tensor.shape[-1], ppd, device)
#     lp_old = fvvdp_lpyr_dec(im_tensor.shape[-2], im_tensor.shape[-1], ppd, device)

#     lpyr, gpyr = lp.decompose( im_tensor )
#     lpyr_2, gpyr_2 = lp_old.decompose( im_tensor )

#     for li in range(lp.get_band_count()):
#         E = Func.mse_loss(lp.get_band(lpyr, li), lp_old.get_band(lpyr_2, li))
#         print( "Level {}, MSE={}".format(li, E))

#     import torch.utils.benchmark as benchmark

#     t0 = benchmark.Timer(
#         stmt='lp.decompose( im_tensor )',
#         setup='',
#         globals={'im_tensor': im_tensor, 'lp': lp})

#     t1 = benchmark.Timer(
#         stmt='lp_old.decompose( im_tensor )',
#         setup='',
#         globals={'im_tensor': im_tensor, 'lp_old': lp_old})

#     print("New pyramid")
#     print(t0.timeit(30))

#     print("Old pyramid")
#     print(t1.timeit(30))

#     # print("----Gaussian----")
#     # for gi in range(lp.get_band_count()):
#     #     print(lp.get_gband(gpyr, gi))

#     # print("----Laplacian----")
#     # for li in range(lp.get_band_count()):
#     #     print(lp.get_band(lpyr, li))

