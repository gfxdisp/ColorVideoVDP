# Dump intermediate channel data for debugging and visualization
import torch
import math
import os
import logging

# For debugging only
#from gfxdisp.pfs.pfs_torch import pfs_torch

from pycvvdp.video_writer import VideoWriter
from pycvvdp.lpyr_dec import lpyr_dec_2

DKLd65_to_RGB = ( (0.926502308187832,   0.960842501786725,   0.940315924461593),
   (6.448879567147620,  -2.074854167137361,   0.100486265553559),
   (0.181670434983238,  -0.190064026530768,   1.080345193424545) )

def dkld65_to_rgb( img ):
    M = torch.as_tensor( DKLd65_to_RGB, dtype=img.dtype, device=img.device).transpose(0,1)

    ABC = torch.empty_like(img)  # ABC represents any linear colour space
    # To avoid permute (slow), perform separate dot products
    for cc in range(3):
        ABC[...,cc,:,:,:] = torch.sum(img*(M[cc,:].view(1,3,1,1,1)), dim=-4, keepdim=True)
    return ABC

# Round an integer up to be divisible by 8
def ceil8(x):
    return int(math.ceil(x/8))*8

class DumpChannels:
    def __init__(self, dump_temp_ch=True, dump_lpyr=True, dump_diff=True, output_dir=None):
        self.vw_channels = None
        self.do_dump_temp_ch = dump_temp_ch
        self.do_dump_lpyr = dump_lpyr
        self.do_dump_diff = dump_diff
        self.output_dir = output_dir if output_dir else "."

    def open(self, fps):

        assert fps>0, "This feature currently works only on video"

        if self.do_dump_temp_ch:
            fname = os.path.join( self.output_dir, "temp_channels.mp4" )
            logging.info( f"Writing temporal channels to '{fname}'" )
            self.vw_channels = VideoWriter( fname, fps=fps, verbose=False )
        else:
            self.vw_channels = None

        self.max_V = None

        if self.do_dump_lpyr:
            fname = os.path.join( self.output_dir, "lpyr.mp4" )
            logging.info( f"Writing Laplacian pyramids to '{fname}'" )
            self.vw_lpyr = VideoWriter( fname, fps=fps )
        else:
            self.vw_lpyr = None

        if self.do_dump_diff:
            fname = os.path.join( self.output_dir, "diff.mp4" )
            logging.info( f"Writing visual differences to '{fname}'" )
            self.vw_diff = VideoWriter( fname, fps=fps )
            self.diff_pyr = None
        else:
            self.vw_diff = None


    def dump_temp_ch( self, R ):
        # Order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y
        # Images do not have the two last channels
        # R = torch.zeros((1, 8, cur_block_N_frames, height, width), device=self.device)

        if not self.do_dump_temp_ch:
            return

        # We dump only test at this moment
        white_dkl = torch.as_tensor( [1, 0.003775328226986, 0.010327227989383], device=R.device )
        ach_sust = R[0:1,0:1,...]
        ach_sust_rgb = dkld65_to_rgb( torch.cat( [ ach_sust, white_dkl[1].expand_as(ach_sust), white_dkl[2].expand_as(ach_sust) ], dim=1 ) )

        if not self.max_V:
            self.max_V = ach_sust_rgb.max()

        gray = white_dkl.view([1,3,1,1,1]) * (self.max_V/4)
        ach_trans = R[0:1,6:7,...]
        ach_trans_rgb = dkld65_to_rgb( torch.cat( [ ach_trans, white_dkl[1].expand_as(ach_sust), white_dkl[2].expand_as(ach_sust) ], dim=1 )+gray )
        rg = R[0:1,2:3,...]
        rg_rgb = dkld65_to_rgb( torch.cat( [ white_dkl[0].expand_as(rg), rg, white_dkl[2].expand_as(rg) ], dim=1 )+gray )
        yv = R[0:1,4:5,...]
        yv_rgb = dkld65_to_rgb( torch.cat( [ white_dkl[0].expand_as(yv), white_dkl[1].expand_as(yv), yv ], dim=1 )+gray )

        frame = torch.cat( [ torch.cat( [ach_sust_rgb, ach_trans_rgb], dim=-1 ), 
                  torch.cat( [rg_rgb, yv_rgb], dim=-1 ) ], dim=-2 )
        for ff in range(frame.shape[2]): # for each frame
            frame_de = ((frame[0,:,ff,...] / self.max_V) ** (1/2.2) * 255).clip(0, 255)
            self.vw_channels.write_frame_rgb( frame_de.permute((1,2,0)).to(torch.uint8).cpu().numpy() )

    def dump_lpyr( self, lpyr, bands ):
        # Dump the content of the Laplacian pyramid for a given channel
        # Order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y
        # Images do not have the two last channels
        # R = torch.zeros((1, 8, cur_block_N_frames, height, width), device=self.device)

        # We dump only test at this moment

        if not self.do_dump_lpyr:
            return

        b0 = lpyr.get_band(bands, 0)
        b0_sh = b0.shape
        width = ceil8((b0_sh[-1] + lpyr.get_band(bands, 1).shape[-1] + 1)*2)
        height = ceil8((b0_sh[-2]+1)*2)
        frames = b0_sh[1]
        lpv = torch.zeros( [3, frames, height, width], device=b0.device)

        white_dkl = torch.as_tensor( [1, 0.003775328226986, 0.010327227989383], device=b0.device )
        gray = white_dkl.view([3,1,1,1])
        gray[0] /= 2

        B = lpyr.get_band_count()
        CHs = [0, 6, 2, 4]
        for col in range(4):
            ch = CHs[col]
            pos = [int(col/2)*int(height/2), (col%2)*int(width/2)]
            for bb in range(B):
                band = lpyr.get_band(bands,bb)[ch:(ch+1),...]
                if ch in (0,1,6,7): # achromatic
                    band_col = dkld65_to_rgb( torch.cat( [ band+white_dkl[0]/2, white_dkl[1].expand_as(band), white_dkl[2].expand_as(band) ], dim=0 ) )
                elif ch in (2,3): # RG
                    band_col = dkld65_to_rgb( torch.cat( [ white_dkl[0].expand_as(band)/2, band+white_dkl[1], white_dkl[2].expand_as(band) ], dim=0 ) )
                elif ch in (4,5): # YV
                    band_col = dkld65_to_rgb( torch.cat( [ white_dkl[0].expand_as(band)/2, white_dkl[1].expand_as(band), band + white_dkl[2] ], dim=0 ) )
                lpv[:, :, pos[0]:(pos[0]+band.shape[-2]), pos[1]:(pos[1]+band.shape[-1])] = band_col
                if (bb % 2)==0:
                    pos[1] += band.shape[-1]+1
                else:
                    pos[0] += band.shape[-2]+1

        for ff in range(frames): # for each frame
            frame_de = ((lpv[:,ff,...]) ** (1/2.2) * 255).clip(0, 255)
            self.vw_lpyr.write_frame_rgb( frame_de.permute((1,2,0)).to(torch.uint8).cpu().numpy() )

    def set_diff_band( self, width, height, pix_per_deg, bb, band):
        if not self.do_dump_diff:
            return

        if not self.diff_pyr:
            self.diff_pyr = lpyr_dec_2(width, height, pix_per_deg, band.device)
        self.diff_pyr.set_lband(bb, band)


    def dump_diff( self ):
        # Dump the per-band differences between the test and reference images
        # Order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y
        # Images do not have the two last channels
        # R = torch.zeros((1, 8, cur_block_N_frames, height, width), device=self.device)

        # We dump only test at this moment

        if not self.do_dump_diff:
            return

        b0 = self.diff_pyr.get_lband(0)
        b0_sh = b0.shape
        width = ceil8((b0_sh[-1] + self.diff_pyr.get_lband(1).shape[-1] + 1)*2)
        height = ceil8((b0_sh[-2]+1)*2)
        frames = b0_sh[1]
        lpv = torch.zeros( [3, frames, height, width], device=b0.device)

        white_dkl = torch.as_tensor( [1, 0.003775328226986, 0.010327227989383], device=b0.device )

        B = self.diff_pyr.get_band_count()
        CHs = [0, 3, 1, 2]
        for col in range(4):
            ch = CHs[col]
            pos = [int(col/2)*int(height/2), (col%2)*int(width/2)]
            for bb in range(B):
                band = self.diff_pyr.get_lband(bb)[ch:(ch+1),...]
                band_col = (band/10).expand( (3,-1,-1,-1) )
                lpv[:, :, pos[0]:(pos[0]+band.shape[-2]), pos[1]:(pos[1]+band.shape[-1])] = band_col
                if (bb % 2)==0:
                    pos[1] += band.shape[-1]+1
                else:
                    pos[0] += band.shape[-2]+1

        for ff in range(frames): # for each frame
            frame_de = ((lpv[:,ff,...]) ** (1/2.2) * 255).clip(0, 255)
            self.vw_diff.write_frame_rgb( frame_de.permute((1,2,0)).to(torch.uint8).cpu().numpy() )

    def close(self):
        if self.vw_channels:
            self.vw_channels.close()
        if self.vw_lpyr:
            self.vw_lpyr.close()
        if self.vw_diff:
            self.vw_diff.close()
