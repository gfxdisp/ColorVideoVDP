import numpy as np
from scipy.ndimage import gaussian_filter


# Add 0 mean Gaussian noise
# std: Standard deviation in normalized units
# static: Set to True if same noise should be added to all frames
# peak: Intensity of brightest pixel
def imnoise(clean, std, static=False, peak=None):
    dtype = clean.dtype

    if peak is None:
        peak = 1 if dtype.kind == 'f' else np.iinfo(dtype).max

    if static:
        # Constant noise for all frames
        h, w, c, N = clean.shape    # axis=-1 is frame axis
        noise = np.repeat((np.random.randn(h, w, c, 1)*std), N, axis=-1)
    else:
        noise = np.random.randn(*clean.shape)*std
    noisy = clean.astype(np.float32)/peak + noise
    noisy = (noisy.clip(0, 1)*peak).astype(dtype)
    return noisy


# Blur RGB image by applying 2d Gaussian kernel
def imgaussblur(clean, sigmas):
    if clean.ndim == 3:    # Handle single input image
        clean = clean[...,np.newaxis]

    if np.isscalar(sigmas):
        sigmas = np.repeat(sigmas, clean.shape[-1])
    assert sigmas.shape[0] == clean.shape[-1]

    blur = np.zeros_like(clean)
    for ff, sigma in enumerate(sigmas): # for each frame
        for cc in range(3):              # for each color
            blur[...,cc,ff] = gaussian_filter(clean[...,cc,ff], sigma,
                                              mode='nearest', truncate=2.0)

    return blur.squeeze()


# Convert array of images to different datatypes
uint16to8 = lambda imgs: (np.floor(im/256).astype(np.uint8) for im in imgs)
# uint16toint16 = lambda imgs: (im.astype(np.int16) for im in imgs)
# uint16tofp32 = lambda imgs: (im.astype(np.float32)/(2**16 - 1) for im in imgs)

# Below are the functions for colour space transforms 

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
    im_t = np.power(np.clip(L,0,Lmax)/Lmax,n)
    V  = np.power((c2*im_t + c1) / (1+c3*im_t), m)
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

    im_t = np.power(np.maximum(V,0),1/m)
    L = Lmax * np.power(np.maximum(im_t-c1,0)/(c2-c3*im_t), 1/n)        
    return L

def srgb2lin( p ):
    t = 0.04045
    a = 0.055
    p = p.clip(0,1)
    L = np.where( p<=t, p/12.92, ((p+a)/(1+a))**2.4 )
    return L

def lin2srgb( L ):
    t = 0.0031308
    a = 0.055
    L = L.clip(0,1)
    p = np.where( L<=t, L*12.92, (1+a)*(L)**(1/2.4) - a )
    return p

__xyz2lms = np.array( [ [0.3592, 0.6976, -0.0358],\
                        [-0.1922, 1.1004, 0.0755], \
                        [0.0070, 0.0749,  0.8434] ] )

__lms2ICtCp = np.array( [ [0.5000, 0.5000, 0.0000], \
                        [1.6137, -3.3234, 1.7097], \
                        [4.3780, -4.2455, -0.1325] ] )

__ICtCp2lms = np.array( [ [1,  0.0086,   0.1110], \
                          [1, -0.0086,  -0.1110], \
                          [1,  0.5600,  -0.3206] ] )
__lms2xyz = np.array( [ [2.0702,  -1.3265,   0.2066], \
                        [0.3650,   0.6805,  -0.0454], \
                        [-0.0496,  -0.0494,   1.1880] ] )

def xyz2itp(xyz):    
    itp = im_ctrans(lin2pq(im_ctrans(xyz,M=__xyz2lms)),M=__lms2ICtCp)
    return itp 

def lms2itp(lms):
    itp = im_ctrans(lin2pq(lms),M=__lms2ICtCp)
    return itp 

def itp2lms(itp):
    lms = pq2lin(im_ctrans(itp,M=__ICtCp2lms))
    return lms


def xyz2Yxy(col_vec):
    assert(col_vec.shape[1]==3)
    sum = np.sum(col_vec,axis=1)
    return np.stack( (col_vec[:,1], col_vec[:,0]/sum, col_vec[:,1]/sum), axis=1)

def Yxy2xyz(col_vec):
    assert(col_vec.shape[1]==3)
    return np.stack( (col_vec[:,0]*col_vec[:,1]/col_vec[:,2], \
                      col_vec[:,0], \
                      col_vec[:,0]/col_vec[:,2]*(1-col_vec[:,1]-col_vec[:,2])), axis=1)

def im2colvec(im):
    """ Convert an image ([height width 3] array) into a colour vector ([height*width 3] array)
    """
    if im.ndim==2 and im.shape[1]==3: # Aleady a colour vector
        return im

    assert(im.shape[2]==3)
    npix = im.shape[0]*im.shape[1]
    return im.reshape( (npix, 3), order='F' )    

def colvec2im(colvec, shape):
    """ Convert a colour vector ([height*width 3] array) into an image ([height width 3] array) 
    """
    if colvec.ndim==3 and colvec.shape[2]==3: # Already an image
        return colvec

    assert(colvec.shape[1]==3)
    return col_vec.reshape( shape, order='F' )


__rgb2020_2xyz = np.array( [ [0.6370, 0.1446, 0.1689], \
                    [0.2627, 0.6780, 0.0593], \
                    [0.0000, 0.0281, 1.0610] ] )


__rgb709_2xyz = np.array( [ [0.4124, 0.3576, 0.1805], \
                        [0.2126, 0.7152, 0.0722], \
                        [0.0193, 0.1192, 0.9505] ] )

__xyz2rgb2020 = np.array( [ [ 1.716502508360628, -0.355584689096764,  -0.253375213570850], \
                        [-0.666625609145029,   1.616446566522207,   0.015775479726511], \
                        [0.017655211703087,  -0.042810696059636,   0.942089263920533] ] )

__xyz2rgb709 = np.array( [ [3.2406, -1.5372, -0.4986], \
                        [-0.9689,  1.8758,  0.0415], \
                        [0.0557, -0.2040,  1.0570] ] )

# Get colour transform from "fromCS" to "toSC". CIE XYZ 1931 is used as an intermediate colour space
def get_cform( fromCS, toCS ):

    # Get the transform from 'fromCS' to XYZ
    if fromCS=="rgb2020":
        in2xyz = __rgb2020_2xyz
    elif fromCS=="rgb709":
        in2xyz = __rgb709_2xyz
    elif fromCS=="xyz":
        in2xyz = np.eye(3,3)
    else:
        assert( False ) # Not recognized colour space

    if toCS=="rgb2020":
        xyz2out = __xyz2rgb2020
    elif toCS=="rgb709":
        xyz2out = __xyz2rgb709
    elif toCS=="lms":
        xyz2out = __xyz2lms
    elif toCS=="xyz":
        xyz2out = np.eye(3,3)
    else:
        assert( False ) # Not recognized colour space

    return xyz2out @ in2xyz

# Recipes for converting from a given colour space to CIE XYZ 1931
# First value - non-linear conversion function (or None), second - colour conversion matrix 
__to_xyz_cforms = { 
    "rgb2020" : (None, __rgb2020_2xyz),
    "rgb709" : (None, __rgb709_2xyz),
    "xyz" : (None, np.eye(3,3)),
    "pq_rgb" : (pq2lin, __rgb2020_2xyz),
    "srgb" : (srgb2lin, __rgb709_2xyz),
    "Yxy" : (Yxy2xyz, np.eye(3,3)),
    "itp" : (itp2lms, __lms2xyz)
}

# Recipes for converting from CIE XYZ 1931 to a given colour space 
# First value - colour transform matrix, second column - non-linear conversion function (or None)
__from_xyz_cforms = { 
    "rgb2020" : (__xyz2rgb2020, None),
    "rgb709" : (__xyz2rgb709, None),
    "xyz" : (np.eye(3,3), None),
    "pq_rgb" : (__xyz2rgb2020, lin2pq),
    "srgb" : (__xyz2rgb709, lin2srgb),
    "Yxy" : (np.eye(3,3), xyz2Yxy),
    "itp" : (__xyz2lms, lms2itp)
}

def im_ctrans( im, fromCS=None, toCS=None, M=None, exposure=1 ):
    """Transform an image or a colour vector from one colour space into another
    Parameters:
    in - either an image as (width, height, 3) array or (n, 3) colour vector
    fromCS, toCS - strings with the name of the input and output colour spaces. 
                   Linear colour spaces: rgb709, rgb2020, xyz, 
                   Non-linear colour spaces: pq_rgb (BT.2020), srgb (BT.709), Yxy
    M - if fromCS and toCS are not specified, you must pass the colour transformation matrix as M 
        (default is None)
    exposure - The colour values are multiplied (in linear space) by the value of the `exposure`. 
               Default is 1. This parameter is useful when converting between relative and absolute 
               colour spaces, for example:

               im_ctrans(im, "srgb", "pq_rgb", exposure=100)

               will map peak white (1,1,1) in sRGB to (100,100,100) or 100 cd/m^2 D65 in BT.2020. 

    Returns:
    An image or colour vector in the new colour space.
    """

    col_vec = im2colvec(im)

    if fromCS:
        assert fromCS in __to_xyz_cforms, "Unknown colour space"
        nl_func, in2xyz = __to_xyz_cforms[fromCS]        
        if nl_func:
            col_vec = nl_func(col_vec)
        
    if toCS:
        assert toCS in __from_xyz_cforms, "Unknown colour space"
        xyz2out, to_nl_func = __from_xyz_cforms[toCS]
    else:
        to_nl_func = None

    if M is None:
        M = xyz2out @ in2xyz

    col_vec_out = col_vec @ (M.transpose().astype(col_vec.dtype) * exposure)

    if to_nl_func: # Non-linearity, if needed
        col_vec_out = to_nl_func(col_vec_out)

    if im.ndim==3: # an image
        im_out = col_vec_out.reshape( im.shape, order='F' )
    else:
        im_out = col_vec_out

    return im_out
