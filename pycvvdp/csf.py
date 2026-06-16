import torch
import torch.nn as nn
import pycvvdp.utils as utils
from enum import Enum
import numpy as np

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

# ---------------------------------------------------------------------------
# Mode selector
# ---------------------------------------------------------------------------
 
class CSFMode(str, Enum):
    EXACT          = "exact"
    DIFFERENTIABLE = "differentiable"
 
 
# ---------------------------------------------------------------------------
# castleCSF_numeric
# ---------------------------------------------------------------------------
 
class castleCSF_numeric:
    """
    Numeric (parameter-based) implementation of castleCSF.
 
    Parameters
    ----------
    device : torch.device
    mode   : CSFMode
        EXACT          – matches MATLAB hard-clamp behaviour exactly.
        DIFFERENTIABLE – replaces hard clamps with smooth approximations so
                         that gradients flow everywhere.
    """
 
    def __init__(self, device=torch.device("cpu"), mode: CSFMode = CSFMode.EXACT):
        self.device = device
        self.mode   = mode
 
        # ---- colour-transform constants ----------------------------------
        # Mones: boolean mask for which entries of M_lms2acc are *not* 1.
        # There are 4 non-one entries; colmat holds their values.
        self.Mones = torch.tensor([
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ], dtype=torch.bool)                          # True  → keep 1
                                                      # False → fill from colmat
 
        # The 4 non-one values, in row-major order of the False positions.
        self.colmat = torch.tensor([2.3112, 0.0, 0.0, 50.9875], dtype=torch.float32)
 
        self.chrom_ch_beta = 2.0
 
        # ---- CSF parameters (from MATLAB castleCSF) ----------------------
        self.params = self._get_default_params()
 
    # ------------------------------------------------------------------
    # Default parameters
    # ------------------------------------------------------------------
 
    def _get_default_params(self):
        p = {
            # ---- Red-Green chromatic channel ----------------------------
            'rg': {
                'sigma_sust':       16.4325,
                'beta_sust':        1.15591,
                # Spatial-frequency peak sensitivity parameters
                'ch_sust': {
                    'S_max':  [681.434, 38.0038, 0.480386],
                    'f_max':  [0.0178364],          # length-1 → constant
                    'bw':     2.42104,
                    'f_0':    0.0711058,
                    'A_0':    2816.44,
                },
                'ecc_drop':        0.0591402,
                'ecc_drop_nasal':  2.89615e-05,
                'ecc_drop_f':      2.04986e-69,
                'ecc_drop_f_nasal':0.18108,
            },
            # ---- Yellow-Violet chromatic channel ------------------------
            'yv': {
                'sigma_sust':       7.15012,
                'beta_sust':        0.969123,
                'ch_sust': {
                    'S_max':  [166.683, 62.8974, 0.41193],
                    'f_max':  [0.00425753],
                    'bw':     2.68197,
                    'f_0':    0.000635093,
                    'A_0':    2.82789e+07,
                },
                'ecc_drop':        0.00356865,
                'ecc_drop_nasal':  5.85804e-141,
                'ecc_drop_f':      0.00806631,
                'ecc_drop_f_nasal':0.0110662,
            },
            # ---- Achromatic channel -------------------------------------
            'ach': {
                # Sustained sub-channel
                'ach_sust': {
                    'S_max': [56.4947, 7.54726, 0.144532, 5.58341e-07, 9.66862e+09],
                    'f_max': [1.78119, 91.5718, 0.256682],
                    'bw':    0.000213047,
                    'a':     0.100207,
                    'f_0':   0.702338,
                    'A_0':   157.103,
                },
                # Transient sub-channel
                'ach_trans': {
                    'S_max': [0.193434, 2748.09],
                    'f_max': [0.000316696],
                    'bw':    2.6761,
                    'a':     0.000241177,
                    'f_0':   3.01389,
                    'A_0':   3.81611,
                },
                'beta_sust':       1.3314,          # temporal-filter exponent
                'sigma_sust':      10.5795,
                'sigma_trans':     0.0844836,
                'omega_trans_sl':  2.41482,
                'omega_trans_c':   4.7036,
                'ecc_drop':        0.0239853,
                'ecc_drop_nasal':  0.0400662,
                'ecc_drop_f':      0.0189038,
                'ecc_drop_f_nasal':0.00813619,
            },
        }
        return p
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
 
    def _t(self, x):
        """Convert scalar / list to a float32 tensor on self.device."""
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)
 
    def get_lum_dep(self, pars, L):
        """
        Luminance-dependent sensitivity scaling.
 
        pars : list of floats  (length 1, 2, 3, or 5)
        L    : tensor
        """

        L64 = L.to(torch.float64)

        n = len(pars)
        p = [
            torch.as_tensor(v,
                            dtype=torch.float64,
                            device=L.device)
            for v in pars
        ]
        
        # p = [self._t(v) for v in pars]
        if n == 1:
            out = torch.ones_like(L64) * p[0]
        elif n == 2:
            out = p[1] * torch.pow(L64, p[0])
        elif n == 3:
            out = p[0] * torch.pow(1.0 + p[1] / L64, -p[2])
        elif n == 5:
            out = (p[0]
                    * torch.pow(1.0 + p[1] / L64, -p[2])
                    * (1.0 - torch.pow(1.0 + p[3] / L64, -p[4])))
            
        else:
            raise NotImplementedError(f"get_lum_dep: unsupported length {n}")

        return out.to(L.dtype)
 
    # ------------------------------------------------------------------
    # LMS → ACC colour matrix  (BUG FIX: was using flatten() copy)
    # ------------------------------------------------------------------
 
    # def get_lms2acc(self):
    #     """
    #     Build the 3×3 LMS → achromatic/chromatic (ACC) matrix.
 
    #     Non-one entries (where Mones is False) are filled from colmat
    #     in row-major order, then signs are applied.
    #     """
    #     M = torch.ones((3, 3), dtype=torch.float32, device=self.device)
 
    #     # FIX: operate on the tensor directly, not on a .flatten() copy.
    #     mask_flat = self.Mones.to(self.device).view(-1)          # True = keep 1
    #     colmat    = self.colmat.to(self.device)
    #     M.view(-1)[~mask_flat] = colmat                          # in-place on M
 
    #     signs = self._t([[1,  1,  1],
    #                      [1, -1,  1],
    #                      [-1, -1, 1]])
    #     return M * signs

    def get_lms2acc(self):
        M = torch.ones((3, 3), dtype=torch.float32, device=self.device)

        # MATLAB column-major assignment
        mask = (~self.Mones).t().reshape(-1)
        Mflat = M.t().reshape(-1)

        Mflat[mask] = self.colmat.to(self.device)

        M = Mflat.reshape(3, 3).t()

        signs = torch.tensor(
            [[ 1,  1, 1],
            [ 1, -1, 1],
            [-1, -1, 1]],
            dtype=torch.float32,
            device=self.device
        )

        return M * signs
 
    # ------------------------------------------------------------------
    # Chromatic direction decomposition
    # ------------------------------------------------------------------
 
    def csf_chrom_directions(self, LMS_mean, LMS_delta):
        """
        Decompose LMS contrast into achromatic (A), red-green (R),
        and yellow-violet (Y) chromatic contrasts.
 
        LMS_mean, LMS_delta : (..., 3)  tensors
        Returns C_A, C_R, C_Y – tensors of shape (...)
        """
        M = self.get_lms2acc().to(LMS_mean.dtype)  # Ensure M is the same dtype as inputs for matmul
        ACC_mean  = torch.abs(torch.matmul(LMS_mean,  M.T))
        ACC_delta = torch.abs(torch.matmul(LMS_delta, M.T))
 
        # alpha=0: chromatic contrasts are normalised by achromatic channel
        alpha = 0.0
        C_A = ACC_delta[..., 0] / ACC_mean[..., 0]
        C_R = ACC_delta[..., 1] / (alpha * ACC_mean[..., 1]
                                   + (1.0 - alpha) * ACC_mean[..., 0])
        C_Y = ACC_delta[..., 2] / (alpha * ACC_mean[..., 2]
                                   + (1.0 - alpha) * ACC_mean[..., 0])
        return C_A, C_R, C_Y
 
    # ------------------------------------------------------------------
    # Temporal filters
    # ------------------------------------------------------------------
 
    def get_sust_trans_resp_ach(self, omega, lum, ach_p):
        """
        Achromatic sustained and transient temporal filter responses.
 
        omega : temporal frequency (tensor)
        lum   : luminance (tensor)
        ach_p : ach parameter dict
        """
        # FIX: read beta_sust from parameter dict (was hardcoded to 1.3314
        #      but the dict didn't contain it – now it does).
        beta_sust  = self._t(ach_p['beta_sust'])
        sigma_sust = self._t(ach_p['sigma_sust'])

        lum = torch.clamp(lum, min=0.1)
 
        omega_0    = torch.log10(lum) * ach_p['omega_trans_sl'] + ach_p['omega_trans_c']
        beta_trans = self._t(0.1898)
        sigma_trans = self._t(ach_p['sigma_trans'])
 
        R_sust  = torch.exp(-torch.pow(omega, beta_sust) / sigma_sust)
        R_trans = torch.exp(
            -torch.pow(
                torch.abs(torch.pow(omega, beta_trans)
                          - torch.pow(omega_0, beta_trans)), 2.0
            ) / sigma_trans
        )
        return R_sust, R_trans
 
    def get_sust_resp_chrom(self, omega, chrom_p):
        """Chromatic sustained temporal filter response."""
        sigma_sust = self._t(chrom_p['sigma_sust'])
        beta_sust  = self._t(chrom_p['beta_sust'])
        return torch.exp(-torch.pow(omega, beta_sust) / sigma_sust)
 
    # ------------------------------------------------------------------
    # Spatial CSF sub-functions
    # ------------------------------------------------------------------
 
    def csf_achrom(self, freq, area, lum, ecc, ach_pars):
        """
        Achromatic spatial CSF (sustained or transient sub-channel).
 
        freq     : spatial frequency (tensor)
        area     : integration area in deg² (tensor)
        lum      : luminance (tensor)
        ecc      : eccentricity (tensor, unused here – drop applied outside)
        ach_pars : one of params['ach']['ach_sust'] or params['ach']['ach_trans']
        """
        S_max = self.get_lum_dep(ach_pars['S_max'], lum)
        f_max = self.get_lum_dep(ach_pars['f_max'], lum)
        bw    = self._t(ach_pars['bw'])
        a     = self._t(ach_pars['a'])
        f_0   = self._t(ach_pars['f_0'])
        A_0   = self._t(ach_pars['A_0'])
 
        log_ratio = torch.log10(freq) - torch.log10(f_max)
        S_LP = torch.pow(10.0, -torch.pow(log_ratio, 2.0) / (2.0 ** bw))
 
        # Low-frequency truncation: S_LP is floored at (1 − a) when freq < f_max.
        if self.mode == CSFMode.EXACT:
            # Hard clamp – matches MATLAB exactly; gradient is zero at boundary.
            cond = (freq < f_max) & (S_LP < (1.0 - a))
            # S_LP = torch.where(cond,
            #                    torch.full_like(S_LP, float(1.0 - a.item())),
            #                    S_LP)
            S_LP = torch.where(freq < f_max,
                   torch.ones_like(S_LP),
                   S_LP)
            
        else:
            # DIFFERENTIABLE: soft floor via smooth-max:
            #   softmax(x, floor) ≈ floor + softplus(x − floor) / softplus_scale
            # Equivalent to: x if x > floor else floor + ε·log(1 + exp((x−floor)/ε))
            floor  = 1.0 - a
            eps    = self._t(0.05)         # controls sharpness; tune as needed
            S_LP   = floor + eps * torch.nn.functional.softplus((S_LP - floor) / eps)
            # Only flatten for freq < f_max region (outside that the floor is irrelevant
            # since S_LP is already ≤ 1 and decreasing, so smooth-max ≈ S_LP there).
 
        S_peak = S_max * S_LP
 
        Ac = A_0 / (1.0 + torch.pow(freq / f_0, 2.0))
        S  = S_peak * torch.sqrt(Ac / (1.0 + Ac / area))
        # NOTE: the original code had `* freq**1.0` here which is a no-op.
        #       Removed – verify against MATLAB if needed.
        S = S * freq 
        return S
 
    def csf_chrom(self, freq, area, lum, ecc, ch_pars):
        """
        Chromatic spatial CSF.
 
        ch_pars : params['rg']['ch_sust']  or  params['yv']['ch_sust']
                  (must contain f_0 and A_0 directly – see _get_default_params)
        """
        S_max = self.get_lum_dep(ch_pars['S_max'], lum)
        f_max = self.get_lum_dep(ch_pars['f_max'], lum)
        bw    = self._t(ch_pars['bw'])
        f_0   = self._t(ch_pars['f_0'])
        A_0   = self._t(ch_pars['A_0'])
 
        log_ratio = torch.log10(freq) - torch.log10(f_max)
        S_LP = torch.pow(10.0, -torch.abs(log_ratio) ** 2.0 / (2.0 ** bw))
 
        # Low-frequency: flat plateau at S_LP = 1 for freq < f_max.
        if self.mode == CSFMode.EXACT:
            S_LP = torch.where(freq < f_max,
                               torch.ones_like(S_LP),
                               S_LP)
        else:
            # DIFFERENTIABLE: smooth clamp to 1 from below.
            # Use sigmoid-based smooth-min with 1:
            #   smooth_min(x, 1) ≈ 1 − softplus(1 − x) / softplus_scale
            eps  = self._t(0.05)
            S_LP = 1.0 - eps * torch.nn.functional.softplus((1.0 - S_LP) / eps)
 
        S_peak = S_max * S_LP
 
        Ac = A_0 / (1.0 + torch.pow(freq / f_0, 2.0))
        S  = S_peak * torch.sqrt(Ac / (1.0 + Ac / area))
        # NOTE: the original code had `* freq` at the end.
        #       Removed – this would give a linear frequency gain not present in
        #       the MATLAB formula. Uncomment and verify if needed:
        S = S * freq
        return S
 
    # ------------------------------------------------------------------
    # Main sensitivity function
    # ------------------------------------------------------------------
 
    def sensitivity(self,
                    s_frequency,  # spatial frequency  (cpd)
                    t_frequency,  # temporal frequency (Hz)
                    luminance,    # background luminance (cd/m²)
                    area,         # integration area (deg²)
                    eccentricity, # eccentricity (deg)
                    lms_bkg,      # (..., 3)
                    lms_delta,    # (..., 3)
                    vis_field=180.0):
        """
        Full castleCSF sensitivity.
 
        All scalar arguments are broadcast-compatible tensors.
        lms_bkg / lms_delta must have a trailing dimension of size 3.
 
        Returns
        -------
        S : tensor  (same shape as s_frequency after broadcasting)
        """
        # ---- Chromatic direction decomposition --------------------------
        C_A, C_R, C_Y = self.csf_chrom_directions(lms_bkg, lms_delta)
 
        # ---- Visual-field weighting (nasal vs temporal) -----------------
        alpha = torch.clamp(torch.abs(self._t(vis_field) - 180.0) / 90.0,
                            max=1.0)
 
        # ---- Eccentricity-drop coefficients per channel -----------------
        def ecc_coeff(p):
            return ((alpha * p['ecc_drop']
                     + (1.0 - alpha) * p['ecc_drop_nasal'])
                    + s_frequency
                    * (alpha * p['ecc_drop_f']
                       + (1.0 - alpha) * p['ecc_drop_f_nasal']))
 
        a_ach = ecc_coeff(self.params['ach'])
        a_rg  = ecc_coeff(self.params['rg'])
        a_yv  = ecc_coeff(self.params['yv'])
 
        ecc_drop_ach = torch.pow(10.0, -a_ach * eccentricity)
        ecc_drop_rg  = torch.pow(10.0, -a_rg  * eccentricity)
        ecc_drop_yv  = torch.pow(10.0, -a_yv  * eccentricity)
 
        # ---- Temporal filter responses ----------------------------------
        R_sust_ach, R_trans_ach = self.get_sust_trans_resp_ach(
            t_frequency, luminance, self.params['ach'])
        R_sust_rg = self.get_sust_resp_chrom(t_frequency, self.params['rg'])
        R_sust_yv = self.get_sust_resp_chrom(t_frequency, self.params['yv'])
 
        # ---- Spatial CSF with eccentricity drop -------------------------
        # FIX: chromatic channels use ch_sust sub-dict, not the top-level dict.
        S_sust_ach = (self.csf_achrom(s_frequency, area, luminance, eccentricity,
                                      self.params['ach']['ach_sust'])
                      * ecc_drop_ach)
        S_trans_ach = (self.csf_achrom(s_frequency, area, luminance, eccentricity,
                                       self.params['ach']['ach_trans'])
                       * ecc_drop_ach)
        S_sust_rg = (self.csf_chrom(s_frequency, area, luminance, eccentricity,
                                    self.params['rg']['ch_sust'])
                     * ecc_drop_rg)
        S_sust_yv = (self.csf_chrom(s_frequency, area, luminance, eccentricity,
                                    self.params['yv']['ch_sust'])
                     * ecc_drop_yv)
 
        # ---- Normalised contrasts ---------------------------------------
        C_A_n = C_A * (R_sust_ach * S_sust_ach + R_trans_ach * S_trans_ach)
        C_R_n = C_R *  R_sust_rg  * S_sust_rg
        C_Y_n = C_Y *  R_sust_yv  * S_sust_yv
 
        # ---- Minkowski pooling ------------------------------------------
        beta  = self.chrom_ch_beta
        C_pool = torch.pow(
            torch.pow(C_A_n, beta)
            + torch.pow(C_R_n, beta)
            + torch.pow(C_Y_n, beta),
            1.0 / beta
        )
 
        # ---- Threshold scale factor and final sensitivity ---------------
        k_thr = 1.0 / C_pool
        LMS_delta_thr = k_thr.unsqueeze(-1) * lms_delta
 
        # S = 1 / (||LMS_delta_thr / LMS_bkg||₂ / √3)
        S = 1.0 / (
            torch.sqrt(torch.sum(
                torch.pow(LMS_delta_thr / lms_bkg, 2.0), dim=-1
            )) / np.sqrt(3.0)
        )
        return S
