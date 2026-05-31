# This examples shows how the visibiliy of distortions changes with display brightness. 
# As the peak luminance of the simulated display is increased, the distortion (noise) 
# becomes more visible and the quality is decreased.

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import examples.ex_utils as utils

import pycvvdp

def display_brightness_example(metric_class = pycvvdp.cvvdp, device=None):

    I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
    std = np.sqrt(0.1)
    I_test_noise = utils.imnoise(I_ref, std)

    # Measure quality on displays of different brightness
    disp_peaks = np.logspace(np.log10(1), np.log10(1000), 5)

    # Display parameters
    contrast = 1000   # Display contrast 1000:1
    EOTF = "2.2"       # Standard gamma-encoding
    E_ambient = 100   # Ambient light = 100 lux
    k_refl = 0.005    # Reflectivity of the display

    metric = metric_class(display_name='standard_4k', heatmap='none', device=device )

    res = []
    Q_JOD = []
    for dd, Y_peak in enumerate(disp_peaks):
        disp_photo = pycvvdp.vvdp_display_photo_eotf(Y_peak=Y_peak, contrast=contrast, EOTF=EOTF, E_ambient=E_ambient, k_refl=k_refl)
        metric.set_display_model(display_photometry=disp_photo)

        start = time.time()
        q, stats = metric.predict(I_test_noise, I_ref, dim_order="HWC")
        end = time.time()
        tst_time = end-start

        Q_JOD.append(q.cpu())
        res.append( (f"{Y_peak:.2f} cd/m^2", q, tst_time) )

    fig, ax = plt.subplots()
    ax.plot(disp_peaks, Q_JOD, '-o')
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle='--')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    ax.set_xlabel('Display peak luminance [cd/m^2]')
    ax.set_ylabel('Quality [JOD]')

    return res

if __name__ == '__main__':
    display_brightness_example( )
    plt.show()
