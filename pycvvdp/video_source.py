from abc import abstractmethod
import torch
import os
import numpy as np 
import logging
from torch.functional import Tensor
import pycvvdp.utils as utils
from pycvvdp.display_model import vvdp_display_photometry, vvdp_display_geometry, vvdp_display_photo_eotf

#from pycvvdp.colorspace import ColorTransform

"""
fvvdp_video_source_* objects are used to supply test/reference frames to FovVideoVDP. 
Those could be comming from memory or files. The subclasses of this abstract class implement
reading the frames and converting them to the approprtate format. 
"""
class video_source:
   
    # Return (height, width, frames) touple with the resolution and
    # the length of the video clip.
    @abstractmethod
    def get_video_size(self):
        pass

    # Return the frame rate of the video
    @abstractmethod
    def get_frames_per_second(self) -> int:
        pass
    
    # Get a test video frame in the selected colorspace. See display_model.py->linear_2_target_colourspace
    # for the list of available color spaces. 
    # You can also pass:
    # 'display_encoded_01', 'display_encoded_100nit' or 'display_encoded_dmax' for the method to return 
    #  display-encoded image (e.g. sRGB) with the values between 0 and 1. If the input source contains linear
    #  values (e.g. an HDR image) they will be PU-encoded:
    # 'display_encoded_01' when the input is 0.005 to 10000, the PU-encoded values are between 0-1
    # 'display_encoded_100nit' when the input is 100, the PU-encoded value is 1, the values can be >1
    # 'display_encoded_dmax' when the input is equal display peak luminance, the PU-encoded value is 1, the values can be >1
    @abstractmethod
    def get_test_frame( self, frame, device, colorspace ) -> Tensor:
        pass

    @abstractmethod
    def get_reference_frame( self, frame, device, colorspace ) -> Tensor:
        pass

    # Check whether pixel values are valid, display warning if it is not the case
    def check_if_valid( self, frame, target_colorspace ):

        if not hasattr( self, "warning_shown" ):
            self.warning_shown = False

        if not self.warning_shown and torch.isnan(frame).flatten().any():
            self.warning_shown = True
            logging.warning( 'Image contains one or more NaN values' )

        if not self.warning_shown and torch.isinf(frame).flatten().any():
            self.warning_shown = True
            logging.warning( 'Image contains one or more Inf values' )

        if not hasattr( self, "first_frame" ):
            self.first_frame = True

        if self.first_frame and not target_colorspace.startswith('display_encoded'):
            self.first_frame = False
            f_mean = torch.mean(frame[:,0,:,:,:])
            f_max = torch.max(frame[:,0,:,:,:])
            f_min = torch.min(frame[:,0,:,:,:])
            logging.debug( f"Content mean={f_mean}, max={f_max}, min={f_min}" )

            if not self.warning_shown and f_mean <= 1:
                logging.warning( 'The mean color value is less than 1 - the image may not be scaled in absolute photometric units!' )



"""
Function for changing the order of dimensions, for example, from "WHC" (width, height, colour) to "BCHW" (batch, colour, height, width)
If a dimension is missing in in_dims, it will be added as a singleton dimension
"""
def reshuffle_dims( T: Tensor, in_dims: str, out_dims: str ) -> Tensor:
    in_dims = in_dims.upper()
    out_dims = out_dims.upper()    

    assert len(in_dims) == T.dim(), "The in_dims string must have as many characters as there are dimensions in T"

    # Find intersection of two strings    
    inter_dims = ""
    for kk in range(len(out_dims)):
        if in_dims.find(out_dims[kk]) != -1:
            inter_dims += out_dims[kk]

    # First, squeeze out the dimensions that are missing in the output
    sq_dims = []
    new_in_dims = ""
    for kk in range(len(in_dims)):
        if inter_dims.find(in_dims[kk]) == -1: # The dimension is missing in the output
            sq_dims.append(kk)
            assert T.shape[kk] == 1, "Only the dimensions of size 1 can be skipped in the output"
        else:
            new_in_dims += in_dims[kk]
    in_dims = new_in_dims
    # For the compatibility with PyTorch pre 2.0, squeeze dims one by one
    sq_dims.sort(reverse=True)
    for kk in sq_dims:
        T = T.squeeze(dim=kk)

    # First, permute into the right order
    perm = [0] * len(inter_dims)
    for kk in range(len(inter_dims)):
        ind = in_dims.find(inter_dims[kk])
        assert ind != -1, 'Dimension "{}" missing in the target dimensions: "{}"'.format(in_dims[kk],out_dims)
        perm[kk] = ind                    
    T_p = T.permute(perm)

    # Add missing dimensions
    out_sh = [1] * len(out_dims)
    for kk in range(len(out_dims)):        
        ind = inter_dims.find(out_dims[kk])
        if ind != -1:
            out_sh[kk] = T_p.shape[ind]

    return T_p.reshape( out_sh )


def numpy2torch_frame(np_array, frame, device, dim_order="HWC" ):

    if isinstance( np_array, np.ndarray ):
        if np_array.dtype == np.uint16:
            # Torch does not natively support uint16. A workaround is to pack uint16 values into int16.
            # This will be efficiently transferred and unpacked on the GPU.
            # logging.info('Test has datatype uint16, packing into int16')
            np_array = np_array.astype(np.int16)
        torch_array = torch.tensor(np_array)
    else:
        torch_array = np_array # If it is already a tensor

    from_array = reshuffle_dims( torch_array, in_dims=dim_order, out_dims="BCFHW" )

    if from_array.dtype is torch.float32:
        frame = from_array[:,:,frame:(frame+1),:,:].to(device)
    elif from_array.dtype is torch.float16:
        frame = from_array[:,:,frame:(frame+1),:,:].to(device=device, dtype=torch.float32)
    elif from_array.dtype is torch.int16:
        # Use int16 to losslessly pack uint16 values
        # Unpack from int16 by bit masking as described in this thread:
        # https://stackoverflow.com/a/20766900
        # logging.info('Found int16 datatype, unpack into uint16')
        max_value = 2**16 - 1
        # Cast to int32 to store values >= 2**15
        frame_int32 = from_array[:,:,frame:(frame+1),:,:].to(device).to(torch.int32)
        frame_uint16 = frame_int32 & max_value
        # Finally convert to float in the range [0,1]
        frame = frame_uint16.to(torch.float32) / max_value
    elif from_array.dtype is torch.uint8:
        frame = from_array[:,:,frame:(frame+1),:,:].to(device).to(torch.float32)/255
    else:
        raise RuntimeError( f"Only uint8, uint16 and float32 is currently supported. {from_array.dtype} encountered." )
    return frame


"""
This video_source uses a photometric display model to convert input content (e.g. sRGB) to luminance maps. 
"""
class video_source_dm( video_source ):

    def __init__( self,  display_photometry='sdr_4k_30', config_paths=[] ):

#        self.color_trans = ColorTransform(color_space_name)

        if isinstance( display_photometry, str ):
            self.dm_photometry = vvdp_display_photometry.load(display_photometry, config_paths) 
        elif isinstance( display_photometry, vvdp_display_photometry ):
            self.dm_photometry = display_photometry
        else:
            raise RuntimeError( "display_model must be a string or fvvdp_display_photometry subclass" )

    def apply_dm_and_colour_transform(self, frame, target_colorspace):

        I = self.dm_photometry.source_2_target_colourspace(frame, target_colorspace)

        self.check_if_valid(I, target_colorspace)
        return I



"""
This video source supplies frames from either Pytorch tensors and Numpy arrays. It also applies a photometric display model.

A batch of videos should be stored as a tensor or numpy array. Ideally, the tensor should have the dimensions BCFHW (batch, colour, frame, height, width).If tensor is stored in another formay, you can pass the order of dimsions as "dim_order" parameter. If any dimension is missing, it will
be added as a singleton dimension. 

This class is for display-encoded (gamma-encoded) content that will be processed by a display model to produce linear  absolute luminance emitted from a display.
"""
class video_source_array( video_source_dm ):
               
    # test_video, reference video - tensor with test and reference video frames. See the class description above for the explanation of dimensions of those tensors.
    # fps - frames per second. Must be 0 for images
    # dim_order - a string with the order of the dimensions. 'BCFHW' is the default.
    # display_model - object that implements fvvdp_display_photometry
    #   class
    # color_space_name - name of the colour space (see
    #   fvvdp_data/color_spaces.json)
    def __init__( self, test_video, reference_video, fps, dim_order='BCFHW', display_photometry='sdr_4k_30', config_paths=[], ):

        super().__init__(display_photometry=display_photometry, config_paths=config_paths)        

        if test_video.shape != reference_video.shape:
            raise RuntimeError( 'Test and reference image/video tensors must be exactly the same shape' )
        
        if len(dim_order) != len(test_video.shape):
            raise RuntimeError( 'Input tensor much have exactly as many dimensions as there are characters in the "dims" parameter' )

        # Convert numpy arrays to tensors. Note that we do not upload to device or change dtype at this point (to save GPU memory)
        if isinstance( test_video, np.ndarray ):
            if test_video.dtype == np.uint16:
                # Torch does not natively support uint16. A workaround is to pack uint16 values into int16.
                # This will be efficiently transferred and unpacked on the GPU.
                # logging.info('Test has datatype uint16, packing into int16')
                test_video = test_video.astype(np.int16)
            test_video = torch.tensor(test_video)
        if isinstance( reference_video, np.ndarray ):
            if reference_video.dtype == np.uint16:
                # Torch does not natively support uint16. A workaround is to pack uint16 values into int16.
                # This will be efficiently transferred and unpacked on the GPU.
                # logging.info('Reference has datatype uint16, packing into int16')
                reference_video = reference_video.astype(np.int16)
            reference_video = torch.tensor(reference_video)

        # Change the order of dimension to match BFCHW - batch, frame, colour, height, width
        test_video = reshuffle_dims( test_video, in_dims=dim_order, out_dims="BCFHW" )
        reference_video = reshuffle_dims( reference_video, in_dims=dim_order, out_dims="BCFHW" )

        B, C, F, H, W = test_video.shape
        
        if fps==0 and F>1:
            raise RuntimeError( 'When passing video sequences, you must set ''frames_per_second'' parameter' )

        # if C == 4:
        #     logging.warning('Input media has 4 colour channels, ignoring the alpha channel to run cvvdp.')
        #     test_video = test_video[:,:3]
        #     reference_video = reference_video[:,:3]
        if C not in (1, 3):
            raise RuntimeError( 'The content must have either 1 or 3 colour channels.' )

        self.fps = fps
        self.is_video = (fps>0)
        self.is_color = (C==3)
        self.test_video = test_video
        self.reference_video = reference_video

    def get_frames_per_second(self):
        return self.fps
            
    # Return a [height width frames] vector with the resolution and
    # the number of frames in the video clip. [height width 1] is
    # returned for an image. 
    def get_video_size(self):

        sh = self.test_video.shape
        return (sh[3], sh[4], sh[2])
    
    # % Get a test video frame as a single-precision luminance map
    # % scaled in absolute inits of cd/m^2. 'frame' is the frame index,
    # % starting from 0. If use_gpu==true, the function should return a
    # % gpuArray.

    def get_test_frame( self, frame, device, colorspace ):
        return self._get_frame(self.test_video, frame, device, colorspace )

    def get_reference_frame( self, frame, device, colorspace ):
        return self._get_frame(self.reference_video, frame, device, colorspace )

    def _get_frame( self, from_array, frame, device, colorspace ):        
        # Determine the maximum value of the data type storing the
        # image/video

        if from_array.dtype is torch.float32:
            frame = from_array[:,:,frame:(frame+1),:,:].to(device)
        elif from_array.dtype is torch.float16:
            frame = from_array[:,:,frame:(frame+1),:,:].to(device=device, dtype=torch.float32)
        elif from_array.dtype is torch.int16:
            # Use int16 to losslessly pack uint16 values
            # Unpack from int16 by bit masking as described in this thread:
            # https://stackoverflow.com/a/20766900
            # logging.info('Found int16 datatype, unpack into uint16')
            max_value = 2**16 - 1
            # Cast to int32 to store values >= 2**15
            frame_int32 = from_array[:,:,frame:(frame+1),:,:].to(device).to(torch.int32)
            frame_uint16 = frame_int32 & max_value
            # Finally convert to float in the range [0,1]
            frame = frame_uint16.to(torch.float32) / max_value
        elif from_array.dtype is torch.uint8:
            frame = from_array[:,:,frame:(frame+1),:,:].to(device).to(torch.float32)/255
        else:
            raise RuntimeError( f"Only uint8, uint16 and float32 is currently supported. {from_array.dtype} encountered." )

        I = self.apply_dm_and_colour_transform(frame, colorspace)
        
        return I


class video_source_packed_array( video_source_dm ):
    def __init__(self, test_video, reference_video, fps, display_photometry='sdr_4k_30', config_paths=[], yuv=True, resize_mode='bilinear'):
        super().__init__(display_photometry, config_paths)

        self.fps = fps
        self.is_video = fps > 0
        self.test_video = test_video
        self.reference_video = reference_video
        self.yuv = yuv
        self.resize_mode = resize_mode

    def get_frames_per_second(self):
        return self.fps

    # Return a [height width frames] vector with the resolution and
    # the number of frames in the video clip. [height width 1] is
    # returned for an image.
    def get_video_size(self):
        n, _, _, _, _, h, w = map(int, self.test_video[:7])
        return h, w, n

    def get_test_frame(self, frame, device):
        return self._get_frame(self.test_video, frame, device)

    def get_reference_frame(self, frame, device):
        return self._get_frame(self.reference_video, frame, device)

    def _get_frame(self, from_array, idx, device):
        n, h, w, bit_depth, chroma_ss, resize_h, resize_w = map(int, from_array[:7])
        if self.yuv:
            y_pixels = h*w
            chroma_ss = str(chroma_ss)
            uv_shape = (h//2, w//2) if chroma_ss == '420' else (h, w)
            uv_pixels = uv_shape[0]*uv_shape[1]
            frame_pixels = y_pixels + 2*uv_pixels

            Y = from_array[7 + idx*frame_pixels                        : 7 + idx*frame_pixels + y_pixels]
            u = from_array[7 + idx*frame_pixels + y_pixels             : 7 + idx*frame_pixels + y_pixels + uv_pixels]
            v = from_array[7 + idx*frame_pixels + y_pixels + uv_pixels : 7 + idx*frame_pixels + y_pixels + 2*uv_pixels]

            Yuv_float = utils.fixed2float_upscale(Y, u, v, (h, w), uv_shape, bit_depth, chroma_ss, device)
            if self.color_space=='bt2020nc':
                # display-encoded (PQ) BT.2020 RGB image
                ycbcr2rgb = torch.tensor([[1, 0, 1.47460],
                                        [1, -0.16455, -0.57135],
                                        [1, 1.88140, 0]], device=device)
            else:
                # display-encoded (sRGB) BT.709 RGB image
                ycbcr2rgb = torch.tensor([[1, 0, 1.402],
                                        [1, -0.344136, -0.714136],
                                        [1, 1.772, 0]], device=device)
            RGB = Yuv_float @ ycbcr2rgb.transpose(1, 0)
            # Torch interpolate requires (B,C,H,W)
            RGB = RGB.permute(2,0,1).unsqueeze(0).clip(0, 1)
            if (resize_h != h) or (resize_w != w):
                RGB = torch.nn.functional.interpolate(RGB,
                                                      size=(resize_h, resize_w),
                                                      mode=self.resize_mode)

            # fvvdp required dims -> (B,C,N,H,W)
            frame = RGB.unsqueeze(2)
        else:
            frame_pixels = resize_h*resize_w*3
            if from_array.dtype == np.uint8:
                frame = torch.tensor(from_array[7 + idx*frame_pixels : 7 + (idx+1)*frame_pixels], dtype=torch.uint8)
                frame = frame.to(device).to(torch.float32)
                max_value = 2**8 - 1
            elif from_array.dtype == np.uint16:
                frame = utils.npuint16_to_torchfp32(from_array[7 + idx*frame_pixels : 7 + (idx+1)*frame_pixels], device)
                max_value = 2**16 - 1

            # fvvdp required dims -> (B,C,N,H,W)
            frame = frame.reshape(resize_h, resize_w, 3) / max_value
            frame = frame.permute(2,0,1).reshape(1, -1, 1, resize_h, resize_w)

        L = self.dm_photometry.forward( frame )

        if L.shape[1] == 3:
            # Convert to grayscale
            L = L[:,0:1,:,:,:]*self.color_to_luminance[0] + L[:,1:2,:,:,:]*self.color_to_luminance[1] + L[:,2:3,:,:,:]*self.color_to_luminance[2]

        return L
