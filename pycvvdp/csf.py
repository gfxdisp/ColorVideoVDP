import torch
import pycvvdp.utils as utils

from interp import interp1q, batch_interp1d

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
            logS_r = torch.empty((N), device=self.device)
            logS_r = batch_interp1d(torch.log10(torch.as_tensor(rho, device=self.device, dtype=torch.float32)).expand(N), self.log_rho, logS)
            self.logS_rho[rho_str] = logS_r        

        # Then, interpolate across luminance levels    
        S = 10**interp1q( self.log_L_bkg, logS_r, logL_bkg )

        return S

    def update_device( self, device ):
        self.device = device
        self.log_L_bkg = self.log_L_bkg.to(device)
        self.log_rho = self.log_rho.to(device)

        for oo in range(2): # For each temp frequency
            ch_num = 3 if oo==0 else 1
            for cc in range(ch_num):
                self.logS[oo][cc] = self.logS[oo][cc].to(device)
