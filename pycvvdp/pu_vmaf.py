import json
import os, os.path as osp
import torch

from pycvvdp.utils import PU
from pycvvdp.video_source import *
from pycvvdp.vq_metric import *

"""
PU21-VMAF metric. Usage is same as the FovVideoVDP metric (see pytorch_examples).
Required: ffmpeg compiled with libvmaf (https://github.com/Netflix/vmaf/blob/master/resource/doc/ffmpeg.md)
"""
class pu_vmaf(vq_metric):
    def __init__(self, ffmpeg_bin, cache_ref_loc='.', device=None):
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.pu = PU()
        assert osp.isdir(cache_ref_loc), f'Please create the directory to cache: {cache_ref_loc}'
        self.T_enc_path = osp.join(cache_ref_loc, 'temp_test.yuv')
        self.R_enc_path = osp.join(cache_ref_loc, 'temp_ref.yuv')
        self.output_file = osp.join(cache_ref_loc, 'vmaf_output.json')

        self.T_enc_file = open(self.T_enc_path,'w')
        self.R_enc_file = open(self.R_enc_path,'w')
        self.colorspace = 'RGB709'
        self.ffmpeg_bin = ffmpeg_bin

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    def predict_video_source(self, vid_source, frame_padding="replicate"):

        # T_vid and R_vid are the tensors of the size (1,1,N,H,W)
        # where:
        # N - the number of frames
        # H - height in pixels
        # W - width in pixels
        # Both images must contain linear absolute luminance values in cd/m^2
        # 
        # We assume the pytorch default NCDHW layout

        h, w, N_frames = vid_source.get_video_size()

        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.colorspace)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.colorspace)

            # Apply PU
            T_enc = self.pu.encode(T) / self.pu.encode(self.max_L)
            T_enc_np = T_enc.squeeze().permute(1,2,0).cpu().numpy()
            R_enc = self.pu.encode(R) / self.pu.encode(self.max_L)
            R_enc_np = R_enc.squeeze().permute(1,2,0).cpu().numpy()

            # Save the output as yuv file
            self.write_yuv_frame(T_enc_np, bit_depth=10, type='T')
            self.write_yuv_frame(R_enc_np, bit_depth=10, type='R')

        self.T_enc_file.close()
        self.R_enc_file.close()

        pix_fmt = 'yuv444p10le'
        ffmpeg_cmd = f'{self.ffmpeg_bin} -hide_banner -loglevel error ' \
                     f'-s {w}x{h} -pix_fmt {pix_fmt} -i {self.T_enc_path} ' \
                     f'-s {w}x{h} -pix_fmt {pix_fmt} -i {self.R_enc_path} ' \
                     f'-lavfi libvmaf=\"log_fmt=json:log_path={self.output_file}:n_threads=4\" -f null -'
        os.system(ffmpeg_cmd)

        with open(self.output_file) as f:
            results = json.load(f)
            quality = results['pooled_metrics']['vmaf']['mean']

        os.remove(self.T_enc_path)
        os.remove(self.T_enc_path)
        os.remove(self.output_file)
        return torch.tensor(quality), None

    def short_name(self):
        return 'PU21-VMAF'

    def set_display_model(self, display_photometry, display_geometry):
        self.max_L = display_photometry.get_peak_luminance()
        self.max_L = np.where( self.max_L < 300, self.max_L, 300)

    # This function takes into input an encoded RGB709 frame and saves it as a yuv file (it operates only on numpy arrays)
    def write_yuv_frame( self, RGB ,bit_depth=10,type = 'T'):
        _rgb2ycbcr_rec709 = np.array([[0.2126 , 0.7152 , 0.0722],\
        [-0.114572 , -0.385428 , 0.5],\
        [0.5 , -0.454153 , -0.045847]], dtype=np.float32)

        YUV = (np.reshape( RGB, (-1, 3), order='F' ) @ _rgb2ycbcr_rec709.transpose()).reshape( (RGB.shape), order='F' )

        YUV_fixed = self.float2fixed( YUV, bit_depth )

        Y = YUV_fixed[:,:,0]
        u = YUV_fixed[:,:,1]
        v = YUV_fixed[:,:,2]

        if type == 'T':
            Y.tofile(self.T_enc_file)
            u.tofile(self.T_enc_file)
            v.tofile(self.T_enc_file)
        elif type == 'R':
            Y.tofile(self.R_enc_file)
            u.tofile(self.R_enc_file)
            v.tofile(self.R_enc_file)
    
    # For now this code operates only on array vectors (Because there is no available torch.uint16)
    def float2fixed(self,YCbCr,nbit):

        offset = (2**(nbit-8))*16
        weight = (2**(nbit-8))*219
        max_lum = (2**nbit)-1

        if nbit<=8:
            dtype = np.uint8
        else:
            dtype = np.uint16
        
        Y = np.round(weight*YCbCr[:,:,0]+offset).clip(0,max_lum).astype(dtype)
        
        offset = (2**(nbit-8)) * 128
        weight = (2**(nbit-8)) * 224  
        
        U = np.round(weight*YCbCr[:,:,1]+offset).clip(0,max_lum).astype(dtype)
        V = np.round(weight*YCbCr[:,:,2]+offset).clip(0,max_lum).astype(dtype)
    
        return np.concatenate(  (Y[:,:,np.newaxis], U[:,:,np.newaxis], V[:,:,np.newaxis]), axis=2 )