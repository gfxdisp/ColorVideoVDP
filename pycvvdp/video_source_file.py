# Classes for reading images or videos from files so that they can be passed to FovVideoVDP frame-by-frame

import os
import imageio.v2 as io
import numpy as np
from torch.functional import Tensor
import torch
import ffmpeg
import re
import functools

import logging
from video_source import *

# for debugging only
# from gfxdisp.pfs import pfs
# from gfxdisp.pfs.pfs_torch import pfs_torch

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

def load_image_as_array(imgfile):
    ext = os.path.splitext(imgfile)[1].lower()
    if ext == '.exr':
        if not pyexr_imported:
            logging.error( "pyexr is needed to read OpenEXR files. Please follow the instriction in README.md to install it." )
            raise RuntimeError( "pyexr missing" )
        precisions = pyexr.open(imgfile).precisions
        assert precisions.count(precisions[0]) == len(precisions), 'All channels must have same precision'
        img = pyexr.read(imgfile, precision=precisions[0])
    else:
        # 16-bit PNG not supported by default
        lib = 'PNG-FI' if ext == '.png' else None
        try:
            img = io.imread(imgfile, format=lib)
        except RuntimeError:
            logging.warning('PNG-FI not found, downloading using imageio\'s script')
            import imageio
            imageio.plugins.freeimage.download()
            img = io.imread(imgfile, format=lib)

    if img.ndim==3 and img.shape[2]>3:
        logging.warning(f'Input image {imgfile} has more than 3 channels (alpha?). Ignoring the extra channels.')
        img = img[:,:,:3]

    # Expand the tensor to [H,W,1] if we have a one-channel image
    if img.ndim==2:
        img = img[:,:,np.newaxis]

    return img


class video_reader:

    def __init__(self, vidfile, frames=-1, resize_fn=None, resize_height=-1, resize_width=-1, verbose=False, fps=-1):
        try:
            probe = ffmpeg.probe(vidfile)
        except:
            raise RuntimeError("ffmpeg failed to open file \"" + vidfile + "\"")

        # select the first video stream
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

        self.file_name = vidfile
        self.width = int(video_stream['width']) 
        self.src_width = self.width
        self.height = int(video_stream['height'])
        self.src_height = self.height
        self.color_space = video_stream['color_space'] if ('color_space' in video_stream) else 'unknown'
        self.color_transfer = video_stream['color_transfer'] if ('color_transfer' in video_stream) else 'unknown'
        self.in_pix_fmt = video_stream['pix_fmt']

        avg_fps_num, avg_fps_denom = map(float, video_stream['r_frame_rate'].split('/'))
        self.avg_fps = avg_fps_num/avg_fps_denom if fps == -1 else fps
        try:
            num_frames = int(video_stream['nb_frames'])
        except KeyError:
            # Metadata may not contain total number of frames
            duration_text = video_stream['tags']['DURATION']
            hrs, mins, secs = map(float, duration_text.split(':'))
            duration = (hrs * 60 + mins) * 60 + secs

            num_frames = int(np.floor(duration * self.avg_fps))

        if frames==-1:
            self.frames = num_frames
        else:    
            # if num_frames < frames:
            #     err_str = 'Expecting {needed_frames} frames but only {available_frames} available in the file \"{file}\"'.format( needed_frames=frames, available_frames=num_frames, file=vidfile)
            #     raise RuntimeError( err_str )
            self.frames = min( num_frames, frames ) # Use at most as many frames as passed in "frames" argument

        self._setup_ffmpeg(vidfile, resize_fn, resize_height, resize_width, verbose)
        self.curr_frame = -1

    def _setup_ffmpeg(self, vidfile, resize_fn, resize_height, resize_width, verbose):
        if any(f'p{bit_depth}' in self.in_pix_fmt for bit_depth in [10, 12, 14, 16]): # >8 bit
            out_pix_fmt = 'rgb48le'
            self.bpp = 6 # bytes per pixel
            self.dtype = np.uint16
        else:
            out_pix_fmt='rgb24' # 8 bit
            self.bpp = 3 # bytes per pixel
            self.dtype = np.uint8

        stream = ffmpeg.input(vidfile)
        stream = ffmpeg.filter(stream, 'fps', fps=self.avg_fps)
        if (resize_fn is not None) and (resize_width!=self.width or resize_height!=self.height):
            resize_mode = resize_fn if resize_fn != 'nearest' else 'neighbor'
            stream = ffmpeg.filter(stream, 'scale', resize_width, resize_height, flags=resize_mode)
            self.width = resize_width
            self.height = resize_height

        self.frame_bytes = int(self.width * self.height * self.bpp)

        log_level = 'info' if verbose else 'quiet'
        stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt=out_pix_fmt).global_args( '-loglevel', log_level )
        #.global_args('-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda') - no effect on decoding speed
        self.process = ffmpeg.run_async(stream, pipe_stdout=True)

    def get_frame(self):
        in_bytes = self.process.stdout.read(self.frame_bytes )
        if not in_bytes or self.curr_frame == self.frames:
            return None
        in_frame = np.frombuffer(in_bytes, self.dtype)
        self.curr_frame += 1
        return in_frame       

    def unpack(self, frame_np, device):
        if self.dtype == np.uint8:
            assert frame_np.dtype == np.uint8
            frame_t_hwc = torch.tensor(frame_np, dtype=torch.uint8)
            max_value = 2**8 - 1
            frame_fp32 = frame_t_hwc.to(device).to(torch.float32)
        elif self.dtype == np.uint16:
            max_value = 2**16 - 1
            frame_fp32 = self._npuint16_to_torchfp32(frame_np, device)

        RGB = frame_fp32.reshape(self.height, self.width, 3) / max_value
        return RGB

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

    # Delete or close if program was interrupted
    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "process") and not self.process is None:
            self.process.stdout.close()
            self.process.kill() # We may wait forever if we do not read all the frames
            self.process = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()


'''
Decode frames to Yuv, perform upsampling and colour conversion with pytorch (on the GPU)
'''
class video_reader_yuv_pytorch(video_reader):
    def __init__(self, vidfile, frames=-1, resize_fn=None, resize_height=-1, resize_width=-1, verbose=False, fps=-1):
        super().__init__(vidfile, frames, resize_fn, resize_height, resize_width, verbose, fps)

        y_channel_pixels = int(self.width*self.height)
        self.y_pixels = y_channel_pixels
        self.y_shape = (self.height, self.width)

        if self.chroma_ss == "444":
            self.frame_bytes = y_channel_pixels*3
            self.uv_pixels = y_channel_pixels
            self.uv_shape = self.y_shape
        elif self.chroma_ss == "420":
            self.frame_bytes = y_channel_pixels*3//2
            self.uv_pixels = int(y_channel_pixels/4)
            self.uv_shape = (int(self.y_shape[0]/2), int(self.y_shape[1]/2))
        else:
            raise RuntimeError("Unrecognized chroma subsampling.")

        if self.bit_depth > 8:
            self.frame_bytes *= 2

    def _setup_ffmpeg(self, vidfile, resize_fn, resize_height, resize_width, verbose):

        # if not any(f'p{bit_depth}' in self.in_pix_fmt for bit_depth in [10, 12, 14, 16]): # 8 bit
        #     raise RuntimeError('GPU decoding not implemented for bit-depth 8')

        re_grp = re.search('p\d+', self.in_pix_fmt)
        self.bit_depth = 8 if re_grp is None else int(re_grp.group().strip('p'))


        self.chroma_ss = self.in_pix_fmt[3:6]
        if not self.chroma_ss in ['444', '420']: # TODO: implement and test 422
            raise RuntimeError(f"Unrecognized chroma subsampling {self.chroma_ss}")

        if self.bit_depth>8: 
            self.dtype = np.uint16
            out_pix_fmt = f'yuv{self.chroma_ss}p{self.bit_depth}le'
        else:
            self.dtype = np.uint8
            out_pix_fmt = f'yuv{self.chroma_ss}p'

        # Resize later on the GPU
        if resize_fn is not None:
            self.resize_fn = resize_fn
            self.resize_height = resize_height
            self.resize_width = resize_width

        stream = ffmpeg.input(vidfile)
        stream = ffmpeg.filter(stream, 'fps', fps=self.avg_fps)
        log_level = 'info' if verbose else 'quiet'
        stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt=out_pix_fmt).global_args( '-loglevel', log_level )
        self.process = ffmpeg.run_async(stream, pipe_stdout=True)

    def unpack(self, x, device):
        Y = x[:self.y_pixels]
        u = x[self.y_pixels:self.y_pixels+self.uv_pixels]
        v = x[self.y_pixels+self.uv_pixels:]

        Yuv_float = self._fixed2float_upscale(Y, u, v, device)

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
        if (hasattr(self, 'resize_fn')) and (self.resize_fn is not None) \
            and (self.height != self.resize_height or self.width != self.resize_width):
            RGB = torch.nn.functional.interpolate(RGB.permute(2,0,1)[None],
                                                  size=(self.resize_height, self.resize_width),
                                                  mode=self.resize_fn)
            RGB = RGB.squeeze().permute(1,2,0)

        return RGB.clip(0, 1)

    def _np_to_torchfp32(self, X, device):
        if X.dtype == np.uint8:
            return torch.tensor(X, dtype=torch.uint8).to(device).to(torch.float32)
        elif X.dtype == np.uint16:
            return self._npuint16_to_torchfp32(X, device)


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


'''
Use ffmpeg to read video frames, one by one.
'''
class video_source_video_file(video_source_dm):

    def __init__( self, test_fname, reference_fname, display_photometry='sdr_4k_30', config_paths=[], frames=-1, full_screen_resize=None, resize_resolution=None, ffmpeg_cc=False, verbose=False, match_fps=False ):

        fs_width = -1 if full_screen_resize is None else resize_resolution[0]
        fs_height = -1 if full_screen_resize is None else resize_resolution[1]
        reader_class = video_reader if ffmpeg_cc else video_reader_yuv_pytorch
        reader = functools.partial(reader_class, frames=frames, resize_fn=full_screen_resize, resize_width=fs_width, resize_height=fs_height, verbose=verbose)
        self.reference_vidr = reader(vidfile=reference_fname)
        fps = self.reference_vidr.avg_fps if match_fps else -1
        self.test_vidr = reader(vidfile=test_fname, fps=fps)

        self.frames = self.test_vidr.frames if frames==-1 else frames

        for vr in [self.test_vidr, self.reference_vidr]:
            if vr == self.test_vidr:
                logging.debug(f"Test video '{test_fname}':")
            else:
                logging.debug(f"Reference video '{reference_fname}':")
            if full_screen_resize is None:
                rs_str = ""
            else:
                rs_str = f"->[{resize_resolution[0]}x{resize_resolution[1]}]"
            logging.debug(f"  [{vr.src_width}x{vr.src_height}]{rs_str}, colorspace: {vr.color_space}, color transfer: {vr.color_transfer}, fps: {vr.avg_fps}, pixfmt: {vr.in_pix_fmt}, frames: {self.frames}" )

        # if color_space_name=='auto':
        #     if self.test_vidr.color_space=='bt2020nc':
        #         color_space_name="BT.2020"
        #     else:
        #         color_space_name="sRGB"

        if self.test_vidr.avg_fps != self.reference_vidr.avg_fps:
            logging.error(f"Test and reference videos have different frame rates: test is {self.test_vidr.avg_fps} fps, reference is {self.reference_vidr.avg_fps} fps." )
            raise RuntimeError( "Inconsistent frame rates" )


        super().__init__(display_photometry=display_photometry, config_paths=config_paths)

        if self.test_vidr.color_transfer=="smpte2084" and self.dm_photometry.EOTF!="PQ":
            logging.warning( f"Video color transfer function ({self.test_vidr.color_transfer}) inconsistent with EOTF of the display model ({self.dm_photometry.EOTF})" )

        # Resolutions may be different here because upscaling may happen on the GPU
        # if self.test_vidr.height != self.reference_vidr.height or self.test_vidr.width != self.reference_vidr.width:
        #     raise RuntimeError( f'Test and reference video sequences must have the same resolutions. Found: test {self.test_vidr.width}x{self.test_vidr.height}, reference {self.reference_vidr.width}x{self.reference_vidr.height}' )

        # self.last_test_frame = None
        # self.last_reference_frame = None
        
    # Return (height, width, frames) touple with the resolution and
    # the length of the video clip.
    def get_video_size(self):
        if hasattr(self.test_vidr, 'resize_fn') and self.test_vidr.resize_fn is not None:
            return (self.test_vidr.resize_height, self.test_vidr.resize_width, self.frames )
        else:
            return (self.test_vidr.height, self.test_vidr.width, self.frames )

    # Return the frame rate of the video
    def get_frames_per_second(self) -> int:
        return self.test_vidr.avg_fps

    # Get a pair of test and reference video frames as a single-precision luminance map
    # scaled in absolute inits of cd/m^2. 'frame' is the frame index,
    # starting from 0. 
    def get_test_frame( self, frame, device, colorspace="Y" ) -> Tensor:
        # if not self.last_test_frame is None and frame == self.last_test_frame[0]:
        #     return self.last_test_frame[1]
        L = self._get_frame( self.test_vidr, frame, device, colorspace )
        # self.last_test_frame = (frame,L)
        return L

    def get_reference_frame( self, frame, device, colorspace="Y" ) -> Tensor:
        # if not self.last_reference_frame is None and frame == self.last_reference_frame[0]:
        #     return self.last_reference_frame[1]
        L = self._get_frame( self.reference_vidr, frame, device, colorspace )
        # self.reference_test_frame = (frame,L)
        return L

    def _get_frame( self, vid_reader, frame, device, colorspace ):        

        if frame != (vid_reader.curr_frame+1):
            raise RuntimeError( 'Video can be currently only read frame-by-frame. Random access not implemented.' )

        frame_np = vid_reader.get_frame()

        if frame_np is None:
            raise RuntimeError( 'Could not read frame {}'.format(frame) )

        return self._prepare_frame(frame_np, device, vid_reader.unpack, colorspace)

    def _prepare_frame( self, frame_np, device, unpack_fn, colorspace="Y" ):
        frame_t_hwc = unpack_fn(frame_np, device)
        frame_t = reshuffle_dims( frame_t_hwc, in_dims='HWC', out_dims="BCFHW" )

        I = self.apply_dm_and_colour_transform(frame_t, colorspace)

        return I


'''
The same functionality as to fvvdp_video_source_video_file, but preloads all the frames and stores in the CPU memory - allows for random access.
'''
class video_source_video_file_preload(video_source_video_file):
    
    def _get_frame( self, vid_reader, frame, device, colorspace ):        

        if not hasattr( self, "frame_array_tst" ):

            # Preload on the first frame
            self.frame_array_tst = [None] * self.frames
            for ff in range(self.frames):
                frame_np = self.test_vidr.get_frame()
                self.frame_array_tst[ff] = frame_np
                if ff==0:
                    mb_used = self.frame_array_tst[0].size * self.frame_array_tst[0].itemsize * self.frames * 2 / 1e6
                    logging.debug( f"Allocating {mb_used}MB in the CPU memory to store videos ({self.frames} frames)." )


            self.frame_array_ref = [None] * self.frames
            for ff in range(self.frames):
                frame_np = self.reference_vidr.get_frame()
                self.frame_array_ref[ff] = frame_np


        if vid_reader is self.test_vidr:
            frame_np = self.frame_array_tst[frame]
        else:
            frame_np = self.frame_array_ref[frame]

        if frame_np is None:
            raise RuntimeError( 'Could not read frame {}'.format(frame) )

        return self._prepare_frame(frame_np, device, vid_reader.unpack, colorspace)


'''
Recognize whether the file is an image of video and wraps an appropriate video_source for the given content.
'''
class video_source_file(video_source):

    def __init__( self, test_fname, reference_fname, display_photometry='sdr_4k_30', config_paths=[], frames=-1, full_screen_resize=None, resize_resolution=None, preload=False, ffmpeg_cc=False, verbose=False ):
        # these extensions switch mode to images instead
        image_extensions = [".png", ".jpg", ".gif", ".bmp", ".jpeg", ".ppm", ".tiff", ".tif", ".dds", ".exr", ".hdr"]

        assert os.path.isfile(test_fname), f'Test file does not exists: "{test_fname}"'
        assert os.path.isfile(reference_fname), f'Reference file does not exists: "{reference_fname}"'

        extension = os.path.splitext(test_fname)[1].lower()
        if extension in image_extensions:
            assert os.path.splitext(reference_fname)[1].lower() in image_extensions, 'Test is an image, but reference is a video'
            # if color_space_name=='auto':
            #     color_space_name='sRGB' # TODO: detect the right colour space
            img_test = load_image_as_array(test_fname)
            img_reference = load_image_as_array(reference_fname)
            if not full_screen_resize is None:
                logging.error("full-screen-resize not implemented for images.")
                raise RuntimeError( "Not implemented" )
            self.vs = video_source_array( img_test, img_reference, 0, dim_order='HWC', display_photometry=display_photometry, config_paths=config_paths )

            hdr_extensions = [".exr", ".hdr"]
            if extension in hdr_extensions:
                if self.vs.dm_photometry.EOTF != "linear":
                    logging.warning('Use a display model with linear color space (EOTF="linear") for HDR images. Make sure that the pixel values are absolute.')
            else:
                if self.vs.dm_photometry.EOTF == "linear":
                    logging.warning('A display model with linear colour space should not be used with display-encoded SDR images.')

        else:
            assert os.path.splitext(reference_fname)[1].lower() not in image_extensions, 'Test is a video, but reference is an image'
            vs_class = video_source_video_file_preload if preload else video_source_video_file
            self.vs = vs_class( test_fname, reference_fname, 
                                display_photometry=display_photometry, 
                                config_paths=config_paths,
                                frames=frames, 
                                full_screen_resize=full_screen_resize, 
                                resize_resolution=resize_resolution, 
                                ffmpeg_cc=ffmpeg_cc, 
                                verbose=verbose )

    # Return (height, width, frames) touple with the resolution and
    # the length of the video clip.
    def get_video_size(self):
        return self.vs.get_video_size()

    # Return the frame rate of the video
    def get_frames_per_second(self) -> int:
        return self.vs.get_frames_per_second()
    
    # Get a pair of test and reference video frames as a single-precision luminance map
    # scaled in absolute inits of cd/m^2. 'frame' is the frame index,
    # starting from 0. 
    def get_test_frame( self, frame, device, colorspace="Y" ) -> Tensor:
        return self.vs.get_test_frame( frame, device, colorspace )

    def get_reference_frame( self, frame, device, colorspace="Y" ) -> Tensor:
        return self.vs.get_reference_frame( frame, device, colorspace )
