from video_source import *
import re

import logging

def decode_video_props( fname ):
    vprops = dict()
    vprops["width"]=1920
    vprops["height"]=1080

    vprops["fps"] = 24
    vprops["bit_depth"] = 8
    vprops["color_space"] = '2020'
    vprops["chroma_ss"] = '420'

    bname = os.path.splitext(os.path.basename(fname))[0]
    fp = bname.split("_")

    res_match = re.compile( '(\d+)x(\d+)p?' )

    for field in fp:

        if res_match.match( field ):
            res = field.split( "x")
            if len(res) != 2:
                raise ValueError("Cannot decode the resolution")
            vprops["width"]=int(res[0])
            vprops["height"]=int(res[1])
            continue

        if field.endswith("fps"):
            vprops["fps"] = float(field[:-3])

        if field=="444" or field=="420":
            vprops["chroma_ss"]=field

        if field=="10" or field=="10b":
            vprops["bit_depth"]=10

        if field=="8" or field=="8b":
            vprops["bit_depth"]=8

        if field=="2020" or field=="709":
            vprops["color_space"]=field

        if field=="bt709":
            vprops["color_space"]="709"

        if field=="ct2020" or field=="pq2020":
            vprops["color_space"]="2020"

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
        self.fps = vprops["fps"]
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
        else: # Chroma sub-sampling
            self.frame_bytes = self.frame_bytes*3/2
            self.uv_pixels = int(self.y_pixels/4)
            self.uv_shape = (int(self.y_shape[0]/2), int(self.y_shape[1]/2))

        self.frame_pixels = self.frame_bytes
        if vprops["bit_depth"]>8:
            self.frame_bytes *= 2
            self.dtype = np.uint16
        else:
            self.dtype = np.uint8

        self.frame_count = os.stat(file_name).st_size / self.frame_bytes
#        if math.ceil(self.frame_count)!=self.frame_count:
#            raise RuntimeError( ".yuv file does not seem to contain an integer number of frames" )

        self.frame_count = int(self.frame_count)

        self.mm = None

    def get_frame_count(self):
        return int(self.frame_count)
    
    def get_frame_yuv( self, frame_index ):

        if frame_index<0 or frame_index>=self.frame_count:
            raise RuntimeError( "The frame index is outside the range of available frames")

        if self.mm is None: # Mem-map as needed
            self.mm = np.memmap( self.file_name, self.dtype, mode="r")

        offset = int(frame_index*self.frame_pixels)
        Y = self.mm[offset:offset+self.y_pixels]
        u = self.mm[offset+self.y_pixels:offset+self.y_pixels+self.uv_pixels]
        v = self.mm[offset+self.y_pixels+self.uv_pixels:offset+self.y_pixels+2*self.uv_pixels]

        return (np.reshape(Y,self.y_shape,'C'),np.reshape(u,self.uv_shape,'C'),np.reshape(v,self.uv_shape,'C'))

    # Return display-encoded (sRGB) BT.709 RGB image
    def get_frame_rgb_rec709( self, frame_index ):

        (Y,u,v) = self.get_frame_yuv(frame_index)

        YUV = fixed2float( np.concatenate( (Y[:,:,np.newaxis],\
            convert420to444(u)[:,:,np.newaxis],\
            convert420to444(v)[:,:,np.newaxis]), axis=2), self.bit_depth)

        RGB = (np.reshape( YUV, (self.y_pixels, 3), order='F' ) @ ycbcr2rgb_rec709.transpose()).reshape( (*self.y_shape, 3 ), order='F' )

        return RGB

    # Return display-encoded (PQ) BT.2020 RGB image
    def get_frame_rgb_rec2020( self, frame_index ):

        (Y,u,v) = self.get_frame_yuv(frame_index)

        YUV = fixed2float( np.concatenate( (Y[:,:,np.newaxis],\
            convert420to444(u)[:,:,np.newaxis],\
            convert420to444(v)[:,:,np.newaxis]), axis=2), self.bit_depth)

        RGB = (np.reshape( YUV, (self.y_pixels, 3), order='F' ) @ ycbcr2rgb_rec2020.transpose()).reshape( (*self.y_shape, 3 ), order='F' )

        return RGB

    # Return RGB PyTorch tensor
    def get_frame_rgb_tensor( self, frame_index, device ):

        if frame_index<0 or frame_index>=self.frame_count:
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
        else:
            uv_upscaled = uv

        Yuv[...,1:] = uv_upscaled.squeeze().permute(1,2,0)

        return Yuv

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.mm = None



class video_source_yuv_file(video_source_dm):

    def __init__( self, test_fname, reference_fname, display_photometry='standard_4k', frames=-1, full_screen_resize=None, resize_resolution=None, retain_aspect_ratio=False, verbose=False ):

        self.reference_vidr = YUVReader(reference_fname)
        self.test_vidr = YUVReader(test_fname)
        self.total_frames = self.test_vidr.frame_count
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
            logging.debug(f"  [{vr.width}x{vr.height}]{rs_str}, colorspace: {vr.color_space}, EOTF: {self.dm_photometry.EOTF}, fps: {vr.fps}, frames: {self.frames}" )

        
    # Return (height, width, frames) touple with the resolution and
    # the length of the video clip.
    def get_video_size(self):
        if not self.full_screen_resize is None:
            return [self.resize_resolution[1], self.resize_resolution[0], self.frames]
        else:
            return [self.test_vidr.height, self.test_vidr.width, self.frames]

    # Return the frame rate of the video
    def get_frames_per_second(self) -> int:
        return self.test_vidr.fps
    
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