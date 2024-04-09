import torch

from pycvvdp.video_source import *
from pycvvdp.vq_metric import *
from pycvvdp.video_writer import VideoWriter

try:
    # This may fail if OpenEXR is not installed. To install,
    # ubuntu: sudo apt install libopenexr-dev
    # mac: brew install openexr
    import pyexr
    pyexr_imported = True
except ImportError as e:
    # Imageio's imread is unreliable for OpenEXR images
    # See https://github.com/imageio/imageio/issues/517
    pyexr_imported = False


def tensor_to_numpy_image(T):
    return torch.squeeze(T.permute((3,4,1,0,2)), dim=(3,4)).cpu().numpy()

"""
A fake metric that writes the output of a display model to either HDR video or OpenEXR images/frames. This is useful for checking and debugging display models. 
"""
class dm_preview_metric(vq_metric):

    def __init__(self, output_exr=False, side_by_side=False, display_name="standard_4k", display_photometry=None, device=None):
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.output_exr = output_exr
        self.side_by_side = side_by_side
        self.set_display_model( display_name=display_name, display_photometry=display_photometry )        


    def predict_video_source(self, vid_source, frame_padding="replicate"):

        _, _, N_frames = vid_source.get_video_size()
        
        write_exr = self.output_exr or N_frames==1

        if write_exr:
            colorspace = 'RGB709'
        else:
            colorspace = 'RGB2020pq'
            fps = vid_source.get_frames_per_second()
            test_vw = VideoWriter(self.base_fname + "-test.mp4", hdr_mode=True, fps=fps, codec='h265')
            if not self.side_by_side:
                ref_vw = VideoWriter(self.base_fname + "-reference.mp4", hdr_mode=True, fps=fps, codec='h265')

        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=colorspace)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=colorspace)

            if self.side_by_side:
                concat_dim = -1 if T.shape[-1]<T.shape[-2] else -2
                T = torch.concatenate( (T, R), dim=concat_dim )

            frame_no = f"-{ff:04d}" if N_frames>1 else ""

            if write_exr:
                pyexr.write(self.base_fname + frame_no + "-test.exr", tensor_to_numpy_image(T))
                if not self.side_by_side:
                    pyexr.write(self.base_fname + frame_no + "-reference.exr", tensor_to_numpy_image(R))
            else:
                test_vw.write_frame_rgb(tensor_to_numpy_image(T))
                if not self.side_by_side:
                    ref_vw.write_frame_rgb(tensor_to_numpy_image(R))
        
        if not write_exr:
            test_vw.close()
            if not self.side_by_side:
                ref_vw.close()

        return torch.as_tensor(-1, device=self.device), None

    def short_name(self):
        return "dm-preview"

    def quality_unit(self):
        return ""

