from video_source import *
import re

import logging
from asyncio.log import logger

def decode_video_props( fname ):
    vprops = dict()
    vprops["width"]=1920
    vprops["height"]=1080

    vprops["fps"] = 24
    vprops["bit_depth"] = 8
    vprops["color_space"] = '709'
    vprops["chroma_ss"] = '420'

    bname = os.path.splitext(os.path.basename(fname))[0]
    fp = bname.split("_")

    res_match = re.compile( r'(\d+)x(\d+)p?(\d+)?' )

    for field in fp:

        if res_match.match( field ):
            nums = re.findall(r"\d+", field) 
            if len(nums)<2 or len(nums)>3:
                raise ValueError("Cannot decode the resolution")
            vprops["width"]=int(nums[0])
            vprops["height"]=int(nums[1])
            if len(nums)==3:
                vprops["fps"]=int(nums[2])
            continue

        if field.endswith("fps"):
            vprops["fps"] = float(field[:-3])
            continue

        if field=="444" or field=="420" or field=="422":
            vprops["chroma_ss"]=field
            continue

        if field=="10" or field=="10b" or field=="10bit":
            vprops["bit_depth"]=10
            continue

        if field=="8" or field=="8b" or field=="8bit":
            vprops["bit_depth"]=8
            continue

        if field=="2020" or field=="709":
            vprops["color_space"]=field
            continue

        if field=="bt709" or field=="sdr":
            vprops["color_space"]="709"
            continue

        if field=="ct2020" or field=="pq2020" or field=="hdr":
            vprops["color_space"]="2020"
            continue

    return vprops

# Create a filename which encodes the yuv header. It can be parsed with decode_video_props and pfstools.
def create_yuv_fname( basename, vprops ):
    width = vprops["width"]
    height = vprops["height"]
    bit_depth = vprops["bit_depth"]
    color_space = vprops["color_space"]
    chroma_ss = vprops["chroma_ss"]
    fps = vprops["fps"]
    fps = round(fps,3) if round(fps)!=fps else int(fps)  #do not use decimals if not needed
    yuv_name = f"{basename}_{width}x{height}_{bit_depth}b_{chroma_ss}_{color_space}_{fps}fps.yuv"
    return yuv_name


class YUVReader:

    def __init__(self, file_name):        
        self.file_name = file_name

        if not os.path.isfile(file_name):
            raise FileNotFoundError( "File {} not found".format(file_name) )

        vprops = decode_video_props(file_name)

        self.width = vprops["width"]
        self.height = vprops["height"]
        self.avg_fps = vprops["fps"]
        self.color_space = vprops["color_space"]
        self.chroma_ss = vprops["chroma_ss"]

        self.bit_depth = vprops["bit_depth"]
        self.frame_bytes = int(self.width*self.height)
        self.y_pixels = int(self.frame_bytes)
        self.y_shape = (vprops["height"], vprops["width"])

        if vprops["chroma_ss"]=="444":
            self.frame_bytes *= 3
            self.uv_pixels = self.y_pixels
            self.uv_shape = self.y_shape
        elif vprops["chroma_ss"]=="420": 
            self.frame_bytes = self.frame_bytes*3/2
            self.uv_pixels = int(self.y_pixels/4)
            self.uv_shape = (int(self.y_shape[0]/2), int(self.y_shape[1]/2))
        elif vprops["chroma_ss"]=="422": 
            self.frame_bytes = self.frame_bytes*2
            self.uv_pixels = int(self.y_pixels/2)
            self.uv_shape = (int(self.y_shape[0]), int(self.y_shape[1]/2))
        else:
            raise RuntimeError( f'Unsupported chroma subsampling {vprops["chroma_ss"]}' )

        self.frame_pixels = self.frame_bytes
        if vprops["bit_depth"]>8:
            self.frame_bytes *= 2
            self.dtype = np.uint16
        else:
            self.dtype = np.uint8

        self.frames = os.stat(file_name).st_size / self.frame_bytes
#        if math.ceil(self.frame_count)!=self.frame_count:
#            raise RuntimeError( ".yuv file does not seem to contain an integer number of frames" )

        self.frames = int(self.frames)

        self.mm = None

    def get_frame_count(self):
        return int(self.frames)
    
    def get_frame_yuv( self, frame_index ):

        if frame_index<0 or frame_index>=self.frames:
            raise RuntimeError( "The frame index is outside the range of available frames")

        if self.mm is None: # Mem-map as needed
            self.mm = np.memmap( self.file_name, self.dtype, mode="r")

        offset = int(frame_index*self.frame_pixels)
        Y = self.mm[offset:offset+self.y_pixels]
        u = self.mm[offset+self.y_pixels:offset+self.y_pixels+self.uv_pixels]
        v = self.mm[offset+self.y_pixels+self.uv_pixels:offset+self.y_pixels+2*self.uv_pixels]

        return (np.reshape(Y,self.y_shape,'C'),np.reshape(u,self.uv_shape,'C'),np.reshape(v,self.uv_shape,'C'))

    # Return RGB PyTorch tensor
    def get_frame_rgb_tensor( self, frame_index, device ):

        if frame_index<0 or frame_index>=self.frames:
            raise RuntimeError( "The frame index is outside the range of available frames")

        if self.mm is None: # Mem-map as needed
            self.mm = np.memmap( self.file_name, self.dtype, mode="r")

        offset = int(frame_index*self.frame_pixels)
        Y = self.mm[offset:offset+self.y_pixels]
        u = self.mm[offset+self.y_pixels:offset+self.y_pixels+self.uv_pixels]
        v = self.mm[offset+self.y_pixels+self.uv_pixels:offset+self.y_pixels+2*self.uv_pixels]

        Yuv_float = self._fixed2float_upscale(Y, u, v, device)

        if self.color_space=='2020':
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
        return RGB.clip(0, 1)


    def _np_to_torchfp32(self, X, device):
        if X.dtype == np.uint8:
            return torch.tensor(X, dtype=torch.uint8).to(device).to(torch.float32)
        elif X.dtype == np.uint16:
            return self._npuint16_to_torchfp32(X, device)

    # Torch does not natively support uint16. A workaround is to pack uint16 values into int16.
    # This will be efficiently transferred and unpacked on the GPU.
    # logging.info('Test has datatype uint16, packing into int16')
    def _npuint16_to_torchfp32(self, np_x_uint16, device):
        max_value = 2**16 - 1
        assert np_x_uint16.dtype == np.uint16
        np_x_int16 = torch.tensor(np_x_uint16.astype(np.int16), dtype=torch.int16)
        torch_x_int32 = np_x_int16.to(device).to(torch.int32)
        torch_x_uint16 = torch_x_int32 & max_value
        torch_x_fp32 = torch_x_uint16.to(torch.float32)
        return torch_x_fp32

    def _fixed2float_upscale(self, Y, u, v, device):
        offset = 16/219
        weight = 1/(2**(self.bit_depth-8)*219)
        Yuv = torch.empty(self.height, self.width, 3, device=device)

        Y = self._np_to_torchfp32(Y, device)
        Yuv[..., 0] = torch.clip(weight*Y - offset, 0, 1).reshape(self.height, self.width)

        offset = 128/224
        weight = 1/(2**(self.bit_depth-8)*224)

        uv = np.stack((u, v))
        uv = self._np_to_torchfp32(uv, device)
        uv = torch.clip(weight*uv - offset, -0.5, 0.5).reshape(1, 2, self.uv_shape[0], self.uv_shape[1])

        if self.chroma_ss=="420":
            # TODO: Replace with a proper filter.
            uv_upscaled = torch.nn.functional.interpolate(uv, scale_factor=2, mode='bilinear')
        if self.chroma_ss=="422":
            # TODO: Replace with a proper filter.
            uv_upscaled = torch.nn.functional.interpolate(uv, scale_factor=(1, 2), mode='bilinear')
        elif self.chroma_ss=="444":
            uv_upscaled = uv
        else:
            raise RuntimeError( f'Unsupported chroma subsampling {self.chroma_ss}' )

        Yuv[...,1:] = uv_upscaled.squeeze().permute(1,2,0)

        return Yuv

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.mm = None


"""
This class is compatible with ffmpeg video readers - supports resizing and can be used with video_source_video_file
"""
class video_reader_yuv(YUVReader):

    def __init__(self, vidfile, frames=-1, resize_fn=None, resize_height=-1, resize_width=-1, verbose=False):
        super().__init__(vidfile)
        self.src_width = self.width
        self.src_height = self.height
        self.in_pix_fmt = 'yuv' + self.chroma_ss + 'p'
        self.resize_fn=resize_fn
        self.resize_width = resize_width
        self.resize_height = resize_height        
        self.color_transfer = None
        if frames!=-1:
            self.frames = min(self.frames, frames)
        self.curr_frame = -1

    def get_frame(self):
        self.curr_frame += 1
        return self.curr_frame       

    def unpack(self, frame_index, device):
        RGB = self.get_frame_rgb_tensor(frame_index, device)

        if not self.resize_fn is None and (self.height != self.resize_height or self.width != self.resize_width):
            RGB = torch.nn.functional.interpolate(RGB.permute( (2,0,1) ).unsqueeze(0),
                                                size=(self.resize_height, self.resize_width),
                                                mode=self.resize_fn).squeeze(0).permute( (1,2,0) ).clip(0.,1.)
        return RGB


class video_source_yuv_file(video_source_dm):

    def __init__( self, test_fname, reference_fname, display_photometry='standard_4k', frames=-1, full_screen_resize=None, resize_resolution=None, retain_aspect_ratio=False, verbose=False ):

        self.reference_vidr = YUVReader(reference_fname)
        self.test_vidr = YUVReader(test_fname)
        self.total_frames = self.test_vidr.frames
        self.frames = self.total_frames if frames==-1 else min(self.total_frames, frames)
        self.offset = 0     # Offset for random access of a shorter subsequence

        self.full_screen_resize = full_screen_resize
        if retain_aspect_ratio:
            h, w = self.test_vidr.height, self.test_vidr.width
            if h / resize_resolution[1] * resize_resolution[0] <= w:
                # retain provided width: resize_resolution[0]
                resize_resolution = (resize_resolution[0], int(resize_resolution[0] / w * h))
            else:
                # retain provided height: resize_resolution[1]
                resize_resolution = (int(resize_resolution[1] / h * w), resize_resolution[1])

        self.resize_resolution = resize_resolution

        # if color_space_name=='auto':
        #     if self.test_vidr.color_space=='2020':
        #         color_space_name="BT.2020"
        #     else:
        #         color_space_name="sRGB"

        super().__init__(display_photometry=display_photometry)        

        for vr in [self.test_vidr, self.reference_vidr]:
            if vr == self.test_vidr:
                logging.debug(f"Test video '{test_fname}':")
            else:
                logging.debug(f"Reference video '{reference_fname}':")
            if full_screen_resize is None:
                rs_str = ""
            else:
                rs_str = f"->[{resize_resolution[0]}x{resize_resolution[1]}]"
            logging.debug(f"  [{vr.width}x{vr.height}]{rs_str}, colorspace: {vr.color_space}, EOTF: {self.dm_photometry.EOTF}, fps: {vr.avg_fps}, frames: {self.frames}" )

        
    # Return (height, width, frames) touple with the resolution and
    # the length of the video clip.
    def get_video_size(self):
        if not self.full_screen_resize is None:
            return [self.resize_resolution[1], self.resize_resolution[0], self.frames]
        else:
            return [self.test_vidr.height, self.test_vidr.width, self.frames]

    # Return the frame rate of the video
    def get_frames_per_second(self) -> int:
        return self.test_vidr.avg_fps
    
    # Get a pair of test and reference video frames as a single-precision luminance map
    # scaled in absolute inits of cd/m^2. 'frame' is the frame index,
    # starting from 0. 
    def get_test_frame( self, frame, device, colorspace="Y" ) -> Tensor:
        L = self._get_frame( self.test_vidr, frame, device, colorspace )
        return L

    def get_reference_frame( self, frame, device, colorspace="Y" ) -> Tensor:
        L = self._get_frame( self.reference_vidr, frame, device, colorspace )
        return L

    def _get_frame( self, vid_reader, frame, device, colorspace="Y" ):
        RGB = vid_reader.get_frame_rgb_tensor(self.offset + frame, device)
        RGB_bcfhw = reshuffle_dims( RGB, in_dims='HWC', out_dims="BCFHW" )

        if not self.full_screen_resize is None and (vid_reader.height != self.resize_resolution[1] or vid_reader.width != self.resize_resolution[0]):
            RGB_bcfhw = torch.nn.functional.interpolate(RGB_bcfhw.view(1,RGB_bcfhw.shape[1],RGB_bcfhw.shape[3],RGB_bcfhw.shape[4]),
                                                size=(self.resize_resolution[1], self.resize_resolution[0]),
                                                mode=self.full_screen_resize).view(1,RGB_bcfhw.shape[1],1,self.resize_resolution[1],self.resize_resolution[0]).clip(0.,1.)

        I = self.apply_dm_and_color_transform(RGB_bcfhw, colorspace)
        return I
    
    def set_offset( self, offset:int ):
        self.offset = offset

    # Depreciated
    def get_total_frames(self):
        return self.total_frames

    def set_num_frames(self, num_frames:int):
        if self.offset + num_frames > self.total_frames:
            logging.error(f'Cannot set num_frames={num_frames} because offset={self.offset} and total_frames={self.total_frames}. '
                          f'Clipping num_frames to {self.total_frames - self.offset}')
            num_frames = self.total_frames - self.offset
        self.frames = num_frames