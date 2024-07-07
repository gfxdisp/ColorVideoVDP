# Dump intermediate channel data for debugging and visualization
import torch

from pycvvdp.video_writer import VideoWriter

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

class DumpChannels:
    def __init__(self):
        self.vw_channels = None

    def open(self, fps):
        # self.vw_channels = VideoWriter( "temp_channels.mp4", fps=fps )
        self.max_V = None
        self.vw_lpyr = VideoWriter( "lpyr_rg.mp4", fps=fps )


    def dump_temp_ch( self, R ):
        # Order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y
        # Images do not have the two last channels
        # R = torch.zeros((1, 8, cur_block_N_frames, height, width), device=self.device)

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

    def dump_lpyr( self, lpyr, bands, ch ):
        # Dump the content of the Laplacian pyramid for a given channel
        # Order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y
        # Images do not have the two last channels
        # R = torch.zeros((1, 8, cur_block_N_frames, height, width), device=self.device)

        # We dump only test at this moment

        b0 = lpyr.get_band(bands, 0)
        b0_sh = b0.shape
        width = b0_sh[-1] + lpyr.get_band(bands,1).shape[-1] + 8
        height = b0_sh[-2]
        frames = b0_sh[1]
        lpv = torch.zeros( [3, frames, height, width], device=b0.device)

        white_dkl = torch.as_tensor( [1, 0.003775328226986, 0.010327227989383], device=b0.device )
        gray = white_dkl.view([3,1,1,1]) * (1/4)

        B = lpyr.get_band_count()
        pos = [0,0]
        for bb in range(B):
            band = lpyr.get_band(bands,bb)[ch:(ch+1),...]
            if ch in (0,1,6,7): # achromatic
                band_col = dkld65_to_rgb( torch.cat( [ band, white_dkl[1].expand_as(band), white_dkl[2].expand_as(band) ], dim=0 )+gray )
            elif ch in (2,3): # RG
                band_col = dkld65_to_rgb( torch.cat( [ white_dkl[0].expand_as(band), band, white_dkl[2].expand_as(band) ], dim=0 )+gray )
            elif ch in (4,5): # YV
                band_col = dkld65_to_rgb( torch.cat( [ white_dkl[0].expand_as(band), white_dkl[1].expand_as(band), yv ], dim=0 )+gray )
            lpv[:, :, pos[0]:(pos[0]+band.shape[-2]), pos[1]:(pos[1]+band.shape[-1])] = band_col
            if (bb % 2)==0:
                pos[1] += band.shape[-1]+1
            else:
                pos[0] += band.shape[-2]+1

        for ff in range(frames): # for each frame
            frame_de = ((lpv[:,ff,...]) ** (1/2.2) * 255).clip(0, 255)
            self.vw_lpyr.write_frame_rgb( frame_de.permute((1,2,0)).to(torch.uint8).cpu().numpy() )


    def close(self):
        if self.vw_channels:
            self.vw_channels.close()
