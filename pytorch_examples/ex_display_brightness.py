import os
import numpy as np
import matplotlib.pyplot as plt
import ex_utils as utils

import pycvvdp

I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
std = np.sqrt(0.001)
I_test_noise = utils.imnoise(I_ref, std)

# Measure quality on displays of different brightness
disp_peaks = np.logspace(np.log10(1), np.log10(1000), 5)

# Display parameters
contrast = 1000   # Display contrast 1000:1
gamma = 2.2       # Standard gamma-encoding
E_ambient = 100   # Ambient light = 100 lux
k_refl = 0.005    # Reflectivity of the display

Q_JOD = []
for dd, Y_peak in enumerate(disp_peaks):
    disp_photo = pycvvdp.vvdp_display_photo_eotf(Y_peak=Y_peak, contrast=contrast, EOTF='sRGB', gamma=gamma, E_ambient=E_ambient, k_refl=k_refl)
    cv = pycvvdp.cvvdp(display_name='standard_4k', display_photometry=disp_photo, heatmap='threshold')
    
    q, stats = cv.predict(I_test_noise, I_ref, dim_order="HWC")
    Q_JOD.append(q.cpu())

plt.plot(disp_peaks, Q_JOD, '-o')
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle='--')
plt.xscale('log')
plt.xlabel('Display peak luminance [cd/m^2]')
plt.ylabel('Quality [JOD]')

plt.show()
