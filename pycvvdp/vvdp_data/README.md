# ColourVideoVDP configuration files

This directory contains configurations files and look-up tables used by ColourVideoVDP.

Editing those files in the installation directory is not recommended. Instead, create a separate directory with the files that should be altered and then pass `--config-dir <your_directory>` when invoking `cvvdp` from the command line. You can also set an environment variable `CVVDP_PATH` with such a directory. When searching for the configuration files, ColorVideoVDP will check the directories in the following order: 
* The directory passed as `--config-dir <your_directory>`
* The directory pointed by `CVVDP_PATH`
* The installation directory (`pyvvdp/vvdp_data`)

## cvvdp_parameters.json

This file contains the parameters from ColourVideoVDP calibration. 

## display_models.json

Contains a list of available display models. Each display models specifies:
* Display geometry: its dimensions and resolution
* Display photometric characteristic: its peak luminance, black level, reflectivity
* Input color space and EOTF: name listed in colorspaces.json. Each colorspace specifies its primaries and an EOTF used to tranform display-encoded values into linear colour.

Each entry contains the following fields:

* `name` - descriptive name
* `resolution` [res_x, res_y] - display resolution in pixels
* `colorspace` - the name of a color space, listed in colorspaces.json - see below 
* `viewing_distance_meters` - viewing distance in meters
* `diagonal_size_inches` - viewing distance in inches
* `max_luminance` - the maximum (peak) luminance in cd/m^2 (or nit)
* `contrast` - contrast. for 1000:1 contrast, put there 1000
* `E_ambient` - the amount of ambient light in lux
* `source` - comment, typically URL to the source from which the information comes from

## colorspaces.json

List of available color spaces and their EOTFs. 

Each entry contains the following fields:

* `EOTF` - Non-linearity used to transform display-encoded pixel values (in the input) to linear values - absolute or relative. Select from 'sRGB', 'PQ', or a string with numerical value (e.g. 2.2) for gamma. 
* `whitepoint` - currently unused
* `RGB2X`, `RGB2Y`, `RGB2Z` - three vectors with a 3x3 matric converting from the native source/display color space to CIE XYZ 1931
* `XYZ2R`, `XYZ2G`, `XYZ2B` - unused

The selected color space specifies color space of the input image/video and the color space of the display. It is currently not possible to have input pixels in different color space than the one used by the display. 
