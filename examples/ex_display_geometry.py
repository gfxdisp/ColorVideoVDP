# This example shows how the quality predictions change depending on the viewing distance. 
# As the viewing distances increases, the noise becomes less visible and the quality increases. 

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import numpy as np
import ex_utils as utils
import matplotlib.pyplot as plt

import pycvvdp

I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
std = np.sqrt(0.005)
I_test_noise = utils.imnoise(I_ref, std)

# Measure quality at several viewing distances
distances = np.linspace(0.5, 2, 5)

metric = pycvvdp.cvvdp(display_name='standard_4k', heatmap='threshold')

Q_JOD = []
for dd, dist in enumerate(distances):
    # 4K, 30 inch display, seen at different viewing distances
    disp_geo = pycvvdp.vvdp_display_geometry((3840, 2160), diagonal_size_inches=30, distance_m=dist)
    metric.set_display_model(display_geometry=disp_geo)

    q, stats = metric.predict(I_test_noise, I_ref, dim_order="HWC")
    Q_JOD.append(q.cpu())

plt.plot(distances, Q_JOD, '-o')
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle='--')
plt.xlabel('Viewing distance [m]')
plt.ylabel('Quality [JOD]')

# plt.savefig('results.png')
plt.show()
