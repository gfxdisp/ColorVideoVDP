# Colour space transformations
import torch
import pycvvdp.utils as utils

XYZ_to_LMS2006 = (
  ( 0.185081950317403,   0.584085520715683,  -0.024070430377029 ),
  (-0.134437959455888,   0.405757045863316,   0.035826598616685 ),
  ( 0.000790172132780,  -0.000913083981083,   0.019850963261241 ) )

LMS2006_to_DKLd65 = (
  (1.000000000000000,   1.000000000000000,                   0),
  (1.000000000000000,  -2.311130179947035,                   0),
  (-1.000000000000000,  -1.000000000000000,  50.977571328718781) )

class ColorTransform:

    def __init__( self, color_space_name='sRGB' ):

        colorspaces_file = utils.config_files.find( "color_spaces.json" )
        colorspaces = utils.json2dict(colorspaces_file)

        if not color_space_name in colorspaces:
            raise RuntimeError( "Unknown color space: \"" + color_space_name + "\"" )

        self.rgb2xyz_list = [colorspaces[color_space_name]['RGB2X'], colorspaces[color_space_name]['RGB2Y'], colorspaces[color_space_name]['RGB2Z'] ]

    def rgb2colourspace(self, RGB_lin, colorspace):        

        if hasattr(self, "rgb2xyz"):
            rgb2xyz = self.rgb2xyz
        else:
            rgb2xyz = torch.tensor( self.rgb2xyz_list, dtype=RGB_lin.dtype, device=RGB_lin.device )
            self.rgb2xyz = rgb2xyz

        if colorspace=="Y":
            return torch.sum(RGB_lin*(rgb2xyz[1,:].view(1,3,1,1,1)), dim=-4, keepdim=True)
        else:
            if colorspace=="XYZ":
                rgb2abc = rgb2xyz
            elif colorspace=="LMS2006":
                rgb2abc = torch.as_tensor( XYZ_to_LMS2006, dtype=RGB_lin.dtype, device=RGB_lin.device) @ rgb2xyz
            elif colorspace=="DKLd65":
                rgb2abc = torch.as_tensor( LMS2006_to_DKLd65, dtype=RGB_lin.dtype, device=RGB_lin.device) @ torch.as_tensor( XYZ_to_LMS2006, dtype=RGB_lin.dtype, device=RGB_lin.device) @ rgb2xyz
            else:
                raise RuntimeError( f"Unknown colorspace '{colorspace}'" )

            ABC = torch.empty_like(RGB_lin)  # ABC represents any linear colour space
            # To avoid permute (slow), perform separate dot products
            for cc in range(3):
                ABC[...,cc,:,:,:] = torch.sum(RGB_lin*(rgb2abc[cc,:].view(1,3,1,1,1)), dim=-4, keepdim=True)
            return ABC


