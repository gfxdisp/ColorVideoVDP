import abc

from pycvvdp.video_source import *

# A base class for the video quality metrtics

class vq_metric:

    '''
    test_cont and reference_cont can be either numpy arrays or PyTorch tensors with images or video frames. 
        Depending on the display model (display_photometry), the pixel values should be either display encoded, or absolute linear.
        The two supported datatypes are float16 and uint8.
    dim_order - a string with the order of dimensions of test_cont and reference_cont. The individual characters denote
        B - batch
        C - colour channel
        F - frame
        H - height
        W - width
        Examples: "HW" - gray-scale image (column-major pixel order); "HWC" - colour image; "FCHW" - colour video
        The default order is "BCFHW". The processing can be a bit faster if data is provided in that order. 
    frame_padding - the metric requires at least 250ms of video for temporal processing. Because no previous frames exist in the
        first 250ms of video, the metric must pad those first frames. This options specifies the type of padding to use:
          'replicate' - replicate the first frame
          'circular'  - tile the video in the front, so that the last frame is used for frame 0.
          'pingpong'  - the video frames are mirrored so that frames -1, -2, ... correspond to frames 0, 1, ...
    '''
    def predict(self, test_cont, reference_cont, dim_order="BCFHW", frames_per_second=0, frame_padding="replicate"):

        test_vs = video_source_array( test_cont, reference_cont, frames_per_second, dim_order=dim_order, display_photometry=self.display_photometry, color_space_name=self.color_space )

        return self.predict_video_source(test_vs, frame_padding=frame_padding)

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    @abc.abstractmethod
    def predict_video_source(self, vid_source, frame_padding="replicate"):
        pass

    @abc.abstractmethod
    def short_name(self):
        pass

    @abc.abstractmethod
    def quality_unit(self):
        pass

    def get_info_string(self):
        return None

    def set_display_model(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[]):
        if display_photometry is None:
            self.display_photometry = vvdp_display_photometry.load(display_name, config_paths)
            self.display_name = display_name
        else:
            self.display_photometry = display_photometry
            self.display_name = "unspecified"

    '''
    Set the base name and path for any extra debuging info or outputs that a metric may produce. 
    '''
    def set_base_fname( self, base_fname ):
        self.base_fname = base_fname
