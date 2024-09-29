from abc import ABC, abstractmethod
import torch
import torch.nn.functional as Func
import numpy as np 
import math
import logging
import os

import pycvvdp.utils as utils

# I am unsure where it is comming from
# XYZ_to_LMS2006 = (
#   ( 0.185081950317403,   0.584085520715683,  -0.024070430377029 ),
#   (-0.134437959455888,   0.405757045863316,   0.035826598616685 ),
#   ( 0.000790172132780,  -0.000913083981083,   0.019850963261241 ) )

XYZ_to_LMS2006 = (
   ( 0.187596268556126,   0.585168649077728,  -0.026384263306304 ),
   (-0.133397430663221,   0.405505777260049,   0.034502127690364 ),
   (0.000244379021663,  -0.000542995890619,   0.019406849066323 ) )

LMS2006_to_DKLd65 = (
  (1.000000000000000,   1.000000000000000,                   0),
  (1.000000000000000,  -2.311130179947035,                   0),
  (-1.000000000000000,  -1.000000000000000,  50.977571328718781) )

XYZ_to_RGB2020 = (  (1.716502508360628, -0.355584689096764,  -0.253375213570850), \
                    (-0.666625609145029,   1.616446566522207,   0.015775479726511), \
                    (0.017655211703087,  -0.042810696059636,   0.942089263920533) )

XYZ_to_RGB709 = (   ( 3.2406, -1.5372, -0.4986), \
                    (-0.9689,  1.8758,  0.0415), \
                    (0.0557, -0.2040,  1.0570) )

def lms2006_to_dkld65( img ):
    M = torch.as_tensor( LMS2006_to_DKLd65, dtype=img.dtype, device=img.device)

    ABC = torch.empty_like(img)  # ABC represents any linear colour space
    # To avoid permute (slow), perform separate dot products
    for cc in range(3):
        ABC[...,cc,:,:,:] = torch.sum(img*(M[cc,:].view(1,3,1,1,1)), dim=-4, keepdim=True)
    return ABC

def lin2pq( L ):
    """ Convert from absolute linear values (between 0.005 and 10000) to PQ-encoded values V (between 0 and 1)
    """
    Lmax = 10000
    #Lmin = 0.005
    n    = 0.15930175781250000
    m    = 78.843750000000000
    c1   = 0.83593750000000000
    c2   = 18.851562500000000
    c3   = 18.687500000000000
    im_t = (torch.clip(L,0,Lmax)/Lmax) ** n
    V  = ((c2*im_t + c1) / (1+c3*im_t)) ** m
    return V

def pq2lin( V ):
    """ Convert from PQ-encoded values V (between 0 and 1) to absolute linear values (between 0.005 and 10000)
    """
    Lmax = 10000
    n    = 0.15930175781250000
    m    = 78.843750000000000
    c1   = 0.83593750000000000
    c2   = 18.851562500000000
    c3   = 18.687500000000000

    im_t = torch.pow(V,1/m)
    L = Lmax * torch.pow((im_t-c1).clamp(min=0)/(c2-c3*im_t), 1/n)
    return L

# Convert pixel values to linear RGB using sRGB non-linearity
# 
# L = srgb2lin( p )
#
# p - pixel values (between 0 and 1)
# L - relative linear RGB (or luminance), normalized to the range 0-1
def srgb2lin( p ):
    L = torch.where(p > 0.04045, ((p + 0.055) / 1.055)**2.4, p/12.92)
    return L


# Convert pixel values to linear using the Rec. 2100 HLG non-linearity
#
# rgb_d = hlg2lin( rgb, gamma )
#
# rgb   - pixel values (between 0 and 1)
# rgb_d - relative linear RGB (or luminance), normalized to the range 0-1
def hlg2lin( rgb, gamma ):
    # using formula from table 5 of
    # https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-1-201706-S!!PDF-E.pdf

    a = 0.17883277
    b = 1 - 4 * a
    c = 0.5 - a * math.log(4 * a)

    # inverse OETF
    rgb_s = torch.where(
        rgb <= 0.5,
        torch.pow(rgb, 2) / 3.0,
        (torch.exp((rgb - c) / a) + b) / 12.0
    )

    # apply OOTF
    Y_s = 0.2627 * rgb_s[:, 0] + 0.6780 * rgb_s[:, 1] + 0.0593 * rgb_s[:, 2]
    rgb_d = (Y_s ** (gamma - 1)).unsqueeze(1) * rgb_s

    return rgb_d

class vvdp_display_photometry:

    def __init__( self, source_colorspace='sRGB', config_paths=[] ):

        colorspaces_file = utils.config_files.find( "color_spaces.json", config_paths )
        colorspaces = utils.json2dict(colorspaces_file)

        if not source_colorspace in colorspaces:
            raise RuntimeError( "Unknown color space: \"" + source_colorspace + "\"" )

        if 'RGB2X' in colorspaces[source_colorspace]: # luminance will not have colour space primaries
            self.rgb2xyz_list = [colorspaces[source_colorspace]['RGB2X'], colorspaces[source_colorspace]['RGB2Y'], colorspaces[source_colorspace]['RGB2Z'] ]
        self.EOTF = colorspaces[source_colorspace]['EOTF']


    # Transforms gamma-encoded pixel values V, which must be in the range
    # 0-into absolute linear colorimetric values emitted from
    # the display.
    @abstractmethod
    def forward( self, V ):
        pass

    # Print the display specification    
    @abstractmethod
    def print( self ):
        pass

    # @classmethod
    # def default_model_file( cls ):
    #     return os.path.join(os.path.dirname(__file__), "fvvdp_data/display_models.json")

    @classmethod
    def list_displays( cls, config_paths ):
        models_file = utils.config_files.find( "display_models.json", config_paths )

        logging.info( f"JSON file with display models: {models_file}" )

        models = utils.json2dict(models_file)

        for display_name in models:
            dm = vvdp_display_photometry.load(display_name, config_paths)
            dm.print()

    @classmethod
    def load( cls, display_name, config_paths ):
        models_file = utils.config_files.find( "display_models.json", config_paths )

        models = utils.json2dict(models_file)

        if not display_name in models:
            logging.error(f"Display model: '{display_name}' not found in '{models_file}'")
            raise RuntimeError( 'Display model not found' )

        model = models[display_name]

        Y_peak = model["max_luminance"]

        if "colorspace" in model:
            colorspace = model["colorspace"]
        else:
            colorspace = 'sRGB'

        if "min_luminance" in model:
            contrast = Y_peak/model["min_luminance"]
        elif "contrast" in model:
            contrast = model["contrast"]
        else:
            contrast = 500

        # Ambient light
        if "E_ambient" in model:
            E_ambient = model["E_ambient"]
        else:
            E_ambient = 0

        # Reflectivity of the display panel
        if "k_refl" in model: 
            k_refl = model["k_refl"]
        else:
            k_refl = 0.005

        # Exposure
        if "exposure" in model:
            exposure = model["exposure"]
        else:
            exposure = 1

        obj = vvdp_display_photo_eotf( Y_peak, contrast=contrast, source_colorspace=colorspace, E_ambient=E_ambient, k_refl=k_refl, name=display_name, exposure=exposure, config_paths=config_paths)
        obj.full_name = model["name"]
        obj.short_name = display_name

        return obj

    # Transform content from its source colour space (typically display-encoded RGB) into 
    # the colorimetric values of light emmitted from the display and then into the target colour
    # space used by a metric.
    def source_2_target_colourspace(self, I_src, target_colorspace):        


        if target_colorspace in ['display_encoded_01', 'display_encoded_dmax', 'display_encoded_100nit']: # if a display-encoded frame is requested

            # Special case - if PQ, we still want to use PU21, as it should be marginally better
            if self.is_input_display_encoded() and not (isinstance( self, vvdp_display_photo_eotf) and self.EOTF == 'PQ'):
                I_target = I_src # no need to do anything
            else:
                # Otherwise, we need to PU-encode the frame
                if not hasattr( self, "PU" ):
                    self.PU = utils.PU()

                if target_colorspace == 'display_encoded_01':
                    PU_max = self.PU.encode(torch.as_tensor(10000.0))
                elif target_colorspace == 'display_encoded_100nit':
                    PU_max = self.PU.encode(torch.as_tensor(100.0)) # White diffuse of 100 nit will be mapped to 1
                else:
                    PU_max = self.PU.encode(torch.as_tensor(self.get_peak_luminance()))
                
                I_lin = self.forward( I_src )
                I_target = self.PU.encode(I_lin) / PU_max 
        else:
            # Apply forward display model to get absolute linear values
            I_lin = self.forward( I_src )

            is_color = (I_src.shape[-4]==3)
            if is_color:
                I_target = self.linear_2_target_colourspace(I_lin, target_colorspace)
            else:
                I_target = I_lin

        return I_target

    # Transform frame/image from native linear colour space to the target colour space.
    # Internal, do not use. 
    def linear_2_target_colourspace(self, RGB_lin, target_colorspace):        
        if hasattr(self, "rgb2xyz"):
            rgb2xyz = self.rgb2xyz
        else:
            rgb2xyz = torch.tensor( self.rgb2xyz_list, dtype=RGB_lin.dtype, device=RGB_lin.device )
            self.rgb2xyz = rgb2xyz

        if target_colorspace=="Y":
            return torch.sum(RGB_lin*(rgb2xyz[1,:].view(1,3,1,1,1)), dim=-4, keepdim=True)
        else:
            if target_colorspace=="XYZ":
                rgb2abc = rgb2xyz
            elif target_colorspace=="LMS2006":
                rgb2abc = torch.as_tensor( XYZ_to_LMS2006, dtype=RGB_lin.dtype, device=RGB_lin.device) @ rgb2xyz
            elif target_colorspace=="DKLd65":
                rgb2abc = torch.as_tensor( LMS2006_to_DKLd65, dtype=RGB_lin.dtype, device=RGB_lin.device) @ torch.as_tensor( XYZ_to_LMS2006, dtype=RGB_lin.dtype, device=RGB_lin.device) @ rgb2xyz
            elif target_colorspace=="RGB709":
                rgb2abc = torch.as_tensor( XYZ_to_RGB709, dtype=RGB_lin.dtype, device=RGB_lin.device) @ rgb2xyz
            elif target_colorspace=="RGB2020" or target_colorspace=="RGB2020pq":
                rgb2abc = torch.as_tensor( XYZ_to_RGB2020, dtype=RGB_lin.dtype, device=RGB_lin.device) @ rgb2xyz
            elif target_colorspace=="logLMS_DKLd65":
                rgb2abc = torch.as_tensor( XYZ_to_LMS2006, dtype=RGB_lin.dtype, device=RGB_lin.device) @ rgb2xyz
            else:
                raise RuntimeError( f"Unknown colorspace '{target_colorspace}'" )

            ABC = torch.empty_like(RGB_lin)  # ABC represents any linear colour space
            # To avoid permute (slow), perform separate dot products
            for cc in range(3):
                ABC[...,cc,:,:,:] = torch.sum(RGB_lin*(rgb2abc[cc,:].view(1,3,1,1,1)), dim=-4, keepdim=True)

            if target_colorspace=="logLMS_DKLd65":
                ABC = lms2006_to_dkld65( torch.log10(ABC) )
            elif target_colorspace=="RGB2020pq":
                ABC = lin2pq(ABC)

            return ABC

class vvdp_display_photo_eotf(vvdp_display_photometry): 
    # Display model with several EOTF, to simulate both SDR and HDR displays
    #
    # dm = vvdp_display_photo_eotf( Y_peak, contrast, EOTF, gamma, E_ambient, k_refl )
    #
    # Parameters (default value shown in []):
    # Y_peak - display peak luminance in cd/m^2 (nit), e.g. 200 for a typical
    #          office monitor, 1000 for an HDR display, ...
    # contrast - [1000] the contrast of the display. The value 1000 means
    #          1000:1
    # source_colorspace - color space from colorspaces.json. colorspace entry includes EOTF, 
    #          but it can be overriden using EOTF parameter.
    # EOTF - 'sRGB', 'PQ', 'linear' or a string with a numeric value, such as "2.2", for gamma 2.2. 
    #        This parameter will overwrite the EOTF attribute in the JSON file with corresponding 'source_colorspace'.
    # E_ambient - [0] ambient light illuminance in lux, e.g. 600 for bright
    #         office
    # k_refl - [0.005] reflectivity of the display screen
    # exposure - [1] exposure of the content. The colour in the linear colour space is multipled by this constant.
    #
    # For more details on the GOG display model, see:
    # https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2016perceptual_display.pdf
    #
    # Copyright (c) 2010-2022, Rafal Mantiuk
    def __init__( self, Y_peak, contrast = 1000, source_colorspace='sRGB', EOTF=None, E_ambient = 0, k_refl = 0.005, exposure=1, name=None, config_paths=[] ):
            
        super().__init__(source_colorspace=source_colorspace, config_paths=config_paths)
        if not EOTF is None: 
            self.EOTF = EOTF

        self.Y_peak = Y_peak            
        self.contrast = contrast
        self.E_ambient = E_ambient
        self.k_refl = k_refl
        self.name = name    
        self.exposure = exposure

    # Say whether the input frame is display-encoded. False if it is linear. 
    def is_input_display_encoded(self):
        # Is not display encoded if EOTF is "linear"
        return (self.EOTF!='linear')

    def __eq__(self, other): 
        if not isinstance(other, self.__class__):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.Y_peak == other.Y_peak \
            and self.contrast == other.contrast \
            and self.EOTF == other.EOTF \
            and self.E_ambient == other.E_ambient \
            and self.k_refl == other.k_refl \
            and self.exposure == other.exposure

    # Transforms display-encoded pixel values V, which must be in the range
    # 0-1 into absolute linear colorimetric values emitted from
    # the display.
    def forward( self, V ):
        
        if self.EOTF != 'linear' and ((V>1).flatten().any() or (V<0).flatten().any()):
            logging.warning("Pixel outside the valid range 0-1")
            V = V.clamp( 0., 1. )
            
        Y_black, Y_refl = self.get_black_level()
                
        if self.EOTF=='sRGB':
            if self.exposure == 1:
                L = (self.Y_peak-Y_black)*srgb2lin(V) + Y_black + Y_refl
            else:
                L = (self.Y_peak-Y_black)*(srgb2lin(V)*self.exposure).clip(0., 1.) + Y_black + Y_refl
        elif self.EOTF=='PQ':
            L = (pq2lin( V )*self.exposure).clip(0.005, self.Y_peak) + Y_black + Y_refl #TODO: soft clipping
        elif self.EOTF=='linear':
            L = (V*self.exposure).clip(max(0.005, Y_black), self.Y_peak) + Y_refl #TODO: soft clipping
        elif self.EOTF=='HLG':
            gamma = 1.2
            if self.Y_peak > 1000:
                # The correction term "- 0.07623 * math.log10(self.E_ambient / 5)" comes from BBC Research & Development White Paper WHP 369
                # https://downloads.bbc.co.uk/rd/pubs/whp/whp-pdf-files/WHP369.pdf
                gamma = 1.2 + 0.42 * math.log10(self.Y_peak / 1000) - 0.07623 * math.log10(self.E_ambient / 5)
            if self.exposure == 1:
                L = (self.Y_peak-Y_black)*hlg2lin(V, gamma) + Y_black + Y_refl
            else:
                L = (self.Y_peak-Y_black)*(hlg2lin(V, gamma)*self.exposure).clip(0., 1.) + Y_black + Y_refl
        elif self.EOTF[0].isnumeric(): # if the first char is numeric -> gamma
            gamma = float(self.EOTF)
            L = (self.Y_peak-Y_black)*(torch.pow(V, gamma)*self.exposure).clip(0., 1.) + Y_black + Y_refl
        else:
            raise RuntimeError( f"Unknown EOTF '{self.EOTF}'" )        
        return L
        

    def get_peak_luminance( self ):
        return self.Y_peak

    # Get the black level and the light reflected from the display
    def get_black_level( self ):
        Y_refl = self.E_ambient/math.pi*self.k_refl  # Reflected ambient light            
        Y_black = self.Y_peak/self.contrast

        return Y_black, Y_refl

    # Print the display specification    
    def print( self ):
        Y_black, Y_refl = self.get_black_level()
        
        logging.info( 'Photometric display model: {}'.format(self.name) )
        logging.info( '  Peak luminance: {} cd/m^2'.format(self.Y_peak) )
        logging.info( '  EOTF: {}'.format(self.EOTF) )
        logging.info( '  Contrast - theoretical: {}:1'.format( round(self.contrast) ) )
        logging.info( '  Contrast - effective: {}:1'.format( round(self.Y_peak/(Y_black+Y_refl)) ) )
        logging.info( '  Ambient light: {} lux'.format( self.E_ambient ) )
        logging.info( '  Display reflectivity: {}%'.format( self.k_refl*100 ) )
    


# Use this class to compute the effective resolution of a display in pixels
# per degree (ppd). The class accounts for the change in the projection
# when looking at large FOV displays (e.g. VR headsets) at certain
# eccentricity.
#
# The class is also useful for computing the size of a display in meters
# and visual degrees. Check 'display_size_m' and 'display_size_deg' class
# properties for that.
#
# R = fvvdp_display_geometry(resolution, distance_m=None, distance_display_heights=None, 
#                            fov_horizontal=None, fov_vertical=None, fov_diagonal=None, 
#                            diagonal_size_inches=None)
#
# resolution is the 2-element touple with the pixel resolution of the
# display: (horizontal_resolutution, vertical_resolutution)
# distance_m - viewing distance in meters
# distance_display_heights - viewing distance in the heights of a display
# fov_horizontal - horizontal field of view of the display in degrees
# fov_vertical - vertical field of view of the display in degrees
# fov_diagonal - diagonal field of view of the display in degrees
# diagonal_size_inches - display diagonal resolution in inches
#
# Examples:
# # HTC Pro
# # Note that the viewing distance must be specified even though the resolution
# # and 'fov_diagonal' are enough to find pix_per_deg.
# R = fvvdp_display_geometry( (1440, 1600), distance_m=3, fov_diagonal=110 )
# R.get_ppd( torch.tensor( [0, 10, 20, 30, 40])) # pix per deg at the given eccentricities
#
# # 30" 4K monitor seen from 0.6 meters
# R = fvvdp_display_geometry( (3840, 2160), diagonal_size_inches=30, distance_m=0.6 )
# R.get_ppd()
#
# # 47" SIM2 display seen from 3 display heights
# R = fvvdp_display_geometry( (1920, 1080), diagonal_size_inches=47, distance_display_heights=3 )
# R.get_ppd()
#
# Some information about the effective FOV of VR headsets
# http://www.sitesinvr.com/viewer/htcvive/index.html
class vvdp_display_geometry:

    def __init__(self, resolution, distance_m=None, distance_display_heights=None, fov_horizontal=None, fov_vertical=None, fov_diagonal=None, diagonal_size_inches=None, ppd=None) -> None:

        self.resolution = resolution
        
        ar = resolution[0]/resolution[1] # width/height
        
        if not ppd is None:
            self.fixed_ppd = ppd
            return
        else:
            self.fixed_ppd = None

        if not diagonal_size_inches is None:
            height_mm = math.sqrt( (diagonal_size_inches*25.4)**2 / (1+ar**2) )
            self.display_size_m = (ar*height_mm/1000, height_mm/1000)
                
        if (not distance_m is None) and (not distance_display_heights is None):
            raise RuntimeError( 'You can pass only one of: ''distance_m'', ''distance_display_heights''.' )
        
        if not distance_m is None:
            self.distance_m = distance_m;
        elif not distance_display_heights is None:
            if not hasattr( self, "display_size_m" ):
                raise RuntimeError( 'You need to specify display diagonal size ''diagonal_size_inches'' to specify viewing distance as ''distance_display_heights'' ' )
            self.distance_m = distance_display_heights * self.display_size_m[1]
        elif (not fov_horizontal is None) or (not fov_vertical is None) or (not fov_diagonal is None):
            # Default viewing distance for VR headsets
            self.distance_m = 3
        else:
            raise RuntimeError( 'Viewing distance must be specified as ''distance_m'' or ''distance_display_heights''.' )
        
        if ((not fov_horizontal is None) + (not fov_vertical is None) + (not fov_diagonal is None)) > 1:
            raise RuntimeError( 'You can pass only one of ''fov_horizontal'', ''fov_vertical'', ''fov_diagonal''. The other dimensions are inferred from the resolution assuming that the pixels are square.' )
        
        if not fov_horizontal is None:
            width_m = 2*math.tan( math.radians(fov_horizontal/2) )*self.distance_m
            self.display_size_m = (width_m, width_m/ar)
        elif not fov_vertical is None:
            height_m = 2*math.tan( math.radians(fov_vertical/2) )*self.distance_m
            self.display_size_m = (height_m*ar, height_m)
        elif not fov_diagonal is None:
            # Note that we cannot use Pythagorean theorem on degs -
            # we must operate on a distance measure
            # This is incorrect: height_deg = p.Results.fov_diagonal / sqrt( 1+ar^2 );
            
            distance_px = math.sqrt(self.resolution[0]**2 + self.resolution[1]**2) / (2.0 * math.tan( math.radians(fov_diagonal*0.5)) )
            height_deg = math.degrees(math.atan( self.resolution[1]/2 / distance_px ))*2
            
            height_m = 2*math.tan( math.radians(height_deg/2) )*self.distance_m
            self.display_size_m = (height_m*ar, height_m)
        
        self.display_size_deg = ( 2 * math.degrees(math.atan( self.display_size_m[0] / (2*self.distance_m) )), \
                                  2 * math.degrees(math.atan( self.display_size_m[1] / (2*self.distance_m) )) )

    def __eq__(self, other): 
        if not isinstance(other, self.__class__):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.resolution == other.resolution \
            and self.distance_m == other.distance_m \
            and self.display_size_m == other.display_size_m

    # Get the number of pixels per degree
    #
    # ppd = R.get_ppd()
    # ppd = R.get_ppd(eccentricity)
    #
    # eccentricity is the viewing angle from the center in degrees. If
    # not specified, the central ppd value (for 0 eccentricity) is
    # returned.
    def get_ppd(self, eccentricity = None):
        
        # if ~isempty( dr.fixed_ppd )
        #     ppd = dr.fixed_ppd;
        #     return;
        # end
        
        if not self.fixed_ppd is None:
            return self.fixed_ppd

        # pixel size in the centre of the display
        pix_deg = 2*math.degrees(math.atan( 0.5*self.display_size_m[0]/self.resolution[0]/self.distance_m ))
        
        base_ppd = 1/pix_deg
        
        if eccentricity is None:
            return base_ppd
        else:
            delta = pix_deg/2
            tan_delta = math.tan(math.radians(delta))
            tan_a = torch.tan( torch.deg2rad(eccentricity) )
            
            ppd = base_ppd * (torch.tan(torch.deg2rad(eccentricity+delta))-tan_a)/tan_delta
            return ppd


    # Convert pixel positions into eccentricities for the given
    # display
    #
    # resolution_pix - image resolution as [width height] in pix
    # x_pix, y_pix - pixel coordinates generated with meshgrid,
    #   pixels indexed from 0
    # gaze_pix - [x y] of the gaze position, in pixels
    def pix2eccentricity( self, resolution_pix, x_pix, y_pix, gaze_pix ):
                        
        if not self.fixed_ppd is None:
            ecc = torch.sqrt( (x_pix-gaze_pix[0])**2 + (y_pix-gaze_pix[1])**2 )/self.fixed_ppd
        else:
            # Position the image in the centre
            shift_to_centre = -resolution_pix/2
            x_pix_rel = x_pix+shift_to_centre[0]
            y_pix_rel = y_pix+shift_to_centre[1]
            
            x_m = x_pix_rel * self.display_size_m[0] / self.resolution[0]
            y_m = y_pix_rel * self.display_size_m[1] / self.resolution[1]
            
            device = x_pix.device

            gaze_m = (gaze_pix + shift_to_centre) * torch.tensor(self.display_size_m) / torch.tensor(self.resolution)
            gaze_deg = torch.rad2deg(torch.atan( gaze_m/self.distance_m ))
            
            ecc = torch.sqrt( (torch.rad2deg(torch.atan(x_m/self.distance_m))-gaze_deg[0])**2 + (torch.rad2deg(torch.atan(y_m/self.distance_m))-gaze_deg[1])**2 )
        
        return ecc
        
    def get_resolution_magnification( self, eccentricity ):
            # Get the relative magnification of the resolution due to
            # eccentricity.
            # 
            # M = R.get_resolution_magnification(eccentricity)
            # 
            # eccentricity is the viewing angle from the center to the fixation point in degrees.
            
            if not self.fixed_ppd is None:
                M = torch( (1), device=eccentricity.device )
            else:            
                eccentricity = torch.minimum( eccentricity, torch.tensor((89.9)) ) # To avoid singulatities
                
                # pixel size in the centre of the display
                pix_rad = 2*math.atan( 0.5*self.display_size_m[0]/self.resolution[0]/self.distance_m )
                
                delta = pix_rad/2
                tan_delta = math.tan(delta)
                tan_a = torch.tan( torch.deg2rad(eccentricity) )
                
                M = (torch.tan(torch.deg2rad(eccentricity)+delta)-tan_a)/tan_delta

            return M

    def print(self):
        logging.info( 'Geometric display model:' )
        if hasattr( self, "fixed_ppd" ):
            logging.info( '  Fixed pixels-per-degree: {}'.format(self.fixed_ppd) )
        else:
            logging.info( '  Resolution: {w} x {h} pixels'.format( w=self.resolution[0], h=self.resolution[1] ) )
            logging.info( '  Display size: {w:.1f} x {h:.1f} cm'.format( w=self.display_size_m[0]*100, h=self.display_size_m[1]*100) )
            logging.info( '  Display size: {w:.2f} x {h:.2f} deg'.format( w=self.display_size_deg[0], h=self.display_size_deg[1] ) )
            logging.info( '  Viewing distance: {d:.3f} m'.format(d=self.distance_m) )
            logging.info( '  Pixels-per-degree (center): {ppd:.2f}'.format(ppd=self.get_ppd()) )

    @classmethod
    def load( cls, display_name, config_paths=[] ):

        models_file = utils.config_files.find( "display_models.json", config_paths )
        models = utils.json2dict(models_file)

        if not display_name in models:
            logging.error(f"Display model: '{display_name}' not found in '{models_file}'")
            raise RuntimeError( 'Display model not found' )

        model = models[display_name]

        assert "resolution" in model

        inches_to_meters = 0.0254

        W, H = model["resolution"]

        if "pixels_per_degree" in model:
            obj = vvdp_display_geometry( (W, H), ppd=model["pixels_per_degree"])
        else:
            if "fov_diagonal" in model: fov_diagonal = model["fov_diagonal"]
            else:                       fov_diagonal = None

            if   "viewing_distance_meters" in model: distance_m = model["viewing_distance_meters"]
            elif "viewing_distance_inches" in model: distance_m = model["viewing_distance_inches"] * inches_to_meters
            else:                                    distance_m = None

            if   "diagonal_size_meters" in model: diag_size_inch = model["diagonal_size_meters"] / inches_to_meters
            elif "diagonal_size_inches" in model: diag_size_inch = model["diagonal_size_inches"] 
            else:                                 diag_size_inch = None

            obj = vvdp_display_geometry( (W, H), distance_m=distance_m, fov_diagonal=fov_diagonal, diagonal_size_inches=diag_size_inch)
        return obj

