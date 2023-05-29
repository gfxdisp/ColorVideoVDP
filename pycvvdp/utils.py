import os
import torch
import numpy as np
import json
import torch.nn.functional as Func
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
    def find(cls, fname, config_paths):

        bname, ext = os.path.splitext(fname)
        # First check if the matching file name is in the config paths
        for cp in config_paths:
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
            self.p = [1.063020987, 0.4200327408, 0.1666005322, 0.2817030548, 1.029472678, 1.119265011, 502.1303377]
        elif type == 'banding_glare':
            self.p = [234.0235618, 216.9339286, 0.0001091864237, 0.893206924, 0.06733984121, 1.444718567, 567.6315065];
        elif type == 'peaks':
            self.p = [1.057454135, 0.6234292574, 0.3060331179, 0.3702234502, 1.116868695, 1.109926637, 391.3707005];
        elif type == 'peaks_glare':
            self.p = [1.374063733, 0.3160810744, 0.1350497609, 0.510558148, 1.049265455, 1.404963498, 427.3579761];
        else:
            raise ValueError(f'Unknown type: {type}')

        self.peak = self.p[6]*(((self.p[0] + self.p[1]*L_max**self.p[3])/(1 + self.p[2]*L_max**self.p[3]))**self.p[4] - self.p[5])

    def encode(self, Y):
        '''
        Convert from linear (optical) values Y to encoded (electronic) values V
        '''
        # epsilon = 1e-5
        # if (Y < (self.L_min - epsilon)).any() or (Y > (self.L_max + epsilon)).any():
        #     print( 'Values passed to encode are outside the valid range' )

        Y = Y.clip(self.L_min, self.L_max)
        V = self.p[6]*(((self.p[0] + self.p[1]*Y**self.p[3])/(1 + self.p[2]*Y**self.p[3]))**self.p[4] - self.p[5])
        return V

    def decode(self, V):
        '''
        Convert from encoded (electronic) values V into linear (optical) values Y
        '''
        V_p = ((V/self.p[6] + self.p[5]).clip(min=0))**(1/self.p[4])
        Y = ((V_p - self.p[0]).clip(min=0)/(self.p[1] - self.p[2]*V_p))**(1/self.p[3])
        return Y
