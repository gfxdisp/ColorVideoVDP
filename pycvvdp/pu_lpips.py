import torch

from pycvvdp.utils import PU
from pycvvdp.vq_metric import vq_metric

"""
PU21 + LPIPS
First install LPIPS with "pip install lpips"
"""
class pu_lpips(vq_metric):
    def __init__(self, device=None, net='vgg'):
        from lpips import LPIPS
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.pu = PU()
        self.lpips = LPIPS(net=net)
        self.lpips.to(self.device)
        self.colorspace = 'RGB2020'

    def predict_video_source(self, vid_source, frame_padding="replicate"):
        # T_vid and R_vid are the tensors of the size (1,3,N,H,W)
        # where:
        # N - the number of frames
        # H - height in pixels
        # W - width in pixels
        # Both images must contain linear absolute luminance values in cd/m^2
        # 
        # We assume the pytorch default NCDHW layout

        _, _, N_frames = vid_source.get_video_size()

        quality = 0.0
        for ff in range(N_frames):
            # TODO: Batched processing
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.colorspace)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.colorspace)

            # Apply PU and reshape to (1,C,H,W)
            # Input pixels shoulb lie in [-1,1]
            T_enc = self.pu.encode(T).squeeze(2) / 255. * 2 - 1
            R_enc = self.pu.encode(R).squeeze(2) / 255. * 2 - 1

            quality += self.lpips(T_enc, R_enc) / N_frames
        return quality, None

    def short_name(self):
        return 'PU21-LPIPS'
