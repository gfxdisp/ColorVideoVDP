import os
import torch
import numpy as np
import json
import torch.nn.functional as Func
import math
from functools import cache

from pycvvdp.interp import interp1, interp1q
#from PIL import Image

from pycvvdp.third_party.loadmat import loadmat

def torch_gpu_mem_info():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    f = c-a  # free inside cache
    print("GPU mem used: %d M (cache %d M)" % (a/(1024*1024), c/(1024*1024)))

def json2dict(file):
    data = None
    if os.path.isfile(file):
        with open(file, "r") as json_file:
            data=json.load(json_file)
    else:
        raise RuntimeError( f"Error: Cannot find file {file}" )
    return data

def linear2srgb_torch(lin):
    lin = torch.clamp(lin, 0.0, 1.0)
    srgb = torch.where(lin > 0.0031308, (1.055 * (lin ** (1/2.4))) - 0.055, 12.92 * lin)
    return srgb

def srgb2linear_torch(srgb):
    srgb = torch.clamp(srgb, 0.0, 1.0)
    lin = torch.where(srgb > 0.04045, ((srgb + 0.055) / 1.055)**2.4, srgb/12.92)
    return lin

def img2np(img):
    return np.array(img, dtype="float32") * 1.0/255.0

# def np2img(nparr):
#     return Image.fromarray(np.clip(nparr * 255.0, 0.0, 255.0).astype('uint8'))

def l2rgb(x):
    return np.concatenate([x,x,x], -1)

def stack_horizontal(nparr):
    return np.concatenate([nparr[i] for i in range(len(nparr))], axis=-2)

def stack_vertical(nparr):
    return np.concatenate([nparr[i] for i in range(len(nparr))], axis=-3)


def load_mat_dict(filepath, data_label, device):
    # datapath = "D:\\work\\st_fov_metric"
    # filepath = os.path.join(datapath, rel_path)

    if not os.path.isfile(filepath):
        return None
    else:
        v = loadmat(filepath)
        if data_label in v:
            return v[data_label]
        else:
            print("Cannot find key %s, valid keys are %s" % (data_label, v.keys()))
            return None

def load_mat_tensor(filepath, data_label, device):
    # datapath = "D:\\work\\st_fov_metric"
    # filepath = os.path.join(datapath, rel_path)

    if not os.path.isfile(filepath):
        return None
    else:
        v = loadmat(filepath)
        if data_label in v:
            return torch.tensor(v[data_label], device=device)
        else:
            print("Cannot find key %s, valid keys are %s" % (data_label, v.keys()))
            return None


# args are indexed at 0
def fovdots_load_ref(content_id, device, data_res="full"):
    if data_res != "full":
        print("Note: Using data resolution %s" % (data_res))
    hwd_tensor = load_mat_tensor("D:\\work\\st_fov_metric\\data_vid_%s\\content_%d_ref.mat" % (data_res, content_id+1), "I_vid", device)
    dhw_tensor = hwd_tensor.permute(2,0,1)
    ncdhw_tensor = torch.unsqueeze(torch.unsqueeze(dhw_tensor, 0), 0)
    return ncdhw_tensor

# args are indexed at 0
def fovdots_load_condition(content_id, condition_id, device, data_res="full"):
    if data_res != "full":
        print("Note: Using data resolution %s" % (data_res))
    hwd_tensor = load_mat_tensor("D:\\work\\st_fov_metric\\data_vid_%s\\content_%d_condition_%d.mat" % (data_res, content_id+1, condition_id+1), "I_vid", device)
    dhw_tensor = hwd_tensor.permute(2,0,1)
    ncdhw_tensor = torch.unsqueeze(torch.unsqueeze(dhw_tensor, 0), 0)
    return ncdhw_tensor


class ImGaussFilt():
    def __init__(self, sigma, device):
        self.filter_size = 2 * int(np.ceil(2.0 * sigma)) + 1
        self.half_filter_size = (self.filter_size - 1)//2

        self.K = torch.zeros((1, 1, self.filter_size, self.filter_size), device=device)

        for ii in range(self.filter_size):
            for jj in range(self.filter_size):
                distsqr = float(ii - self.half_filter_size) ** 2 + float(jj - self.half_filter_size) ** 2
                self.K[0,0,jj,ii] = np.exp(-distsqr / (2.0 * sigma * sigma))

        self.K = self.K/self.K.sum()

    def run(self, img):
        
        if len(img.shape) == 2: img_4d = img.reshape((1,1,img.shape[0],img.shape[1]))
        else:                   img_4d = img

        pad = (
            self.half_filter_size,
            self.half_filter_size,
            self.half_filter_size,
            self.half_filter_size,)

        img_4d = Func.pad(img_4d, pad, mode='reflect')
        return Func.conv2d(img_4d, self.K)[0,0]


class config_files:
    # fvvdp_config_dir = None
    
    # @classmethod
    # def set_config_dir( cls, path ):
    #     cls.fvvdp_config_dir = path

    @classmethod
    def find(cls, fname, config_paths:list):

        if not isinstance( config_paths, list ):
            raise RuntimeError( "config_paths must be a list" )

        bname, ext = os.path.splitext(fname)
        # First check if the matching file name is in the config paths
        for cp in config_paths:
            if not (os.path.isfile(cp) or os.path.isdir(cp)):
                raise RuntimeError( f"config_path '{cp}' does not exist" )

            if os.path.isfile(cp) and os.path.basename(cp).startswith(bname):
                return cp

        # Then, check all directories
        for cp in config_paths:
            if os.path.isdir(cp):
                path = os.path.join( cp, fname )
                if os.path.isfile(path):
                    return path

        # Then, check CVVDP_PATH
        ev_config_dir = os.getenv("CVVDP_PATH")
        if not ev_config_dir is None:
            path = os.path.join( ev_config_dir, fname )
            if os.path.isfile(path):
                return path

        # Finally, check the default config dir
        path = os.path.join(os.path.dirname(__file__), "vvdp_data", fname)
        if os.path.isfile(path):
            return path

        raise RuntimeError( f"The configuration file {fname} not found" )


class PU():
    '''
    Transform absolute linear luminance values to/from the perceptually uniform space.
    This class is intended for adopting image quality metrics to HDR content.
    This is based on the new spatio-chromatic CSF from:
      Wuerger, S., Ashraf, M., Kim, M., Martinovic, J., P�rez-Ortiz, M., & Mantiuk, R. K. (2020).
      Spatio-chromatic contrast sensitivity under mesopic and photopic light levels.
      Journal of Vision, 20(4), 23. https://doi.org/10.1167/jov.20.4.23
    The implementation should work for both numpy arrays and torch tensors
    '''
    def __init__(self, L_min=0.005, L_max=10000, type='banding_glare'):
        self.L_min = L_min
        self.L_max = L_max

        if type == 'banding':
            self.p = [1.070275272, 0.4088273932, 0.153224308, 0.2520326168, 1.063512885, 1.14115047, 521.4527484]
        elif type == 'banding_glare':
            self.p = [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142]
        elif type == 'peaks':
            self.p = [1.043882782, 0.6459495343, 0.3194584211, 0.374025247, 1.114783422, 1.095360363, 384.9217577]
        elif type == 'peaks_glare':
            self.p = [816.885024, 1479.463946, 0.001253215609, 0.9329636822, 0.06746643971, 1.573435413, 419.6006374]
        else:
            raise ValueError(f'Unknown type: {type}')

        self.peak = self.p[6]*(((self.p[0] + self.p[1]*L_max**self.p[3])/(1 + self.p[2]*L_max**self.p[3]))**self.p[4] - self.p[5])

    def _encode_direct(self, Y):
        '''
        Convert from linear (optical) values Y to encoded (electronic) values V
        '''
        # epsilon = 1e-5
        # if (Y < (self.L_min - epsilon)).any() or (Y > (self.L_max + epsilon)).any():
        #     print( 'Values passed to encode are outside the valid range' )
        Y = Y.clip(self.L_min, self.L_max)
        Y_p = Y**self.p[3]
        V = self.p[6]*(((self.p[0] + self.p[1]*Y_p)/(1 + self.p[2]*Y_p))**self.p[4] - self.p[5])
        return V

    @cache
    def _get_encode_lut(self, device):
        Y_lut = torch.linspace(math.log10(self.L_min), math.log10(self.L_max), 2048, device=device)
        V_lut = self._encode_direct(10**Y_lut)
        return (Y_lut, V_lut)

    def encode(self, Y):
        '''
        Convert from linear (optical) values Y to encoded (electronic) values V
        '''
        if True or Y.numel()<100: # Interpolation seems slower than directly computing the values
            V = self._encode_direct(Y)
        else:
            (Y_lut, V_lut) = self._get_encode_lut(Y.device)
            V = interp1q( Y_lut, V_lut, torch.log10(Y.clamp(self.L_min, self.L_max)) )
        return V

    def decode(self, V):
        '''
        Convert from encoded (electronic) values V into linear (optical) values Y
        '''
        V_p = ((V/self.p[6] + self.p[5]).clip(min=0))**(1/self.p[4])
        Y = ((V_p - self.p[0]).clip(min=0)/(self.p[1] - self.p[2]*V_p))**(1/self.p[3])
        return Y
