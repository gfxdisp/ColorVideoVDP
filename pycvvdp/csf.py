import torch
import pycvvdp.utils as utils

# from interp import interp1 # deprecated

def linear_interp(x, xp, fp):
    """
    Perform linear interpolation for the given input of PyTorch tensors
    """
    # Get the indices where xp[j] <= x < xp[j+1]
    idx = torch.searchsorted(xp, x, right=True) - 1
    idx = torch.clamp(idx, 0, len(xp) - 2)  # Clamping to avoid index out of bounds

    # Slope for each segment
    slope = (fp[idx + 1] - fp[idx]) / (xp[idx + 1] - xp[idx])

    # Linear interpolation
    return fp[idx] + slope * (x - xp[idx])

def batch_interp1d(x, xp, fp):
    """
    Perform batch-wise linear interpolation.
    """
    # Ensure xp is increasing
    assert torch.all(xp[1:] >= xp[:-1]), "xp must be in increasing order"

    # Make tensors contiguous to avoid warnings and optimize performance
    x = x.contiguous()
    xp = xp.contiguous()
    fp = fp.contiguous()

    # Find indices of the closest points
    indices = torch.searchsorted(xp, x) - 1
    indices = torch.clamp(indices, 0, len(xp) - 2)

    # Gather the relevant points
    x0 = xp[indices]
    x1 = xp[indices + 1]
    y0 = fp[torch.arange(fp.shape[0]), indices]
    y1 = fp[torch.arange(fp.shape[0]), indices + 1]

    # Compute the slope
    slope = (y1 - y0) / (x1 - x0)

    # Compute the interpolated values
    return y0 + slope * (x - x0)

class castleCSF:

    def __init__(self, csf_version, device, config_paths=[]):
        self.device = device
        csf_lut_file = utils.config_files.find( f"csf_lut_{csf_version}.json", config_paths )
        csf_lut = utils.json2dict(csf_lut_file)

        self.log_L_bkg = torch.log10( torch.as_tensor(csf_lut["L_bkg"], device=device) )
        self.log_rho = torch.log10( torch.as_tensor(csf_lut["rho"], device=device) )
        self.omega = csf_lut["omega"]

        self.logS = []
        for oo in range(2): # For each temp frequency
            self.logS.append([])
            ch_num = 3 if oo==0 else 1
            for cc in range(ch_num):
                field_name = f"o{self.omega[oo]}_c{cc+1}"
                self.logS[oo].append( torch.as_tensor(csf_lut[field_name], device=device) )

        self.logS_rho = {}


    def sensitivity(self, rho, omega, logL_bkg, cc, sigma):
        # rho - spatial frequency
        # omega - temporal frequency
        # L_bkg - background luminance
        # sigma - radius of spatial integration (Gaussian envelope)

        # Which LUT to use
        oo = 0 if omega==0 else 1
        logS = self.logS[oo][cc]

        # First interpolate between spatial frequencies rho
        rho_str = f"o{oo}_c{cc}_rho{rho}"
        if rho_str in self.logS_rho: # Check if it is cached
            logS_r = self.logS_rho[rho_str]
        else:
            N = self.log_L_bkg.numel()
            logS_r = batch_interp1d(torch.log10(torch.as_tensor(rho, device=self.device, dtype=torch.float32)).expand(N), self.log_rho, logS)
            self.logS_rho[rho_str] = logS_r

        # Then, interpolate across luminance levels    
        S = 10**linear_interp( logL_bkg, self.log_L_bkg, logS_r )        

        return S

    def update_device( self, device ):
        self.device = device
        self.log_L_bkg = self.log_L_bkg.to(device)
        self.log_rho = self.log_rho.to(device)

        for oo in range(2): # For each temp frequency
            ch_num = 3 if oo==0 else 1
            for cc in range(ch_num):
                self.logS[oo][cc] = self.logS[oo][cc].to(device)
