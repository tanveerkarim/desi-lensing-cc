import os

import numpy as np
import matplotlib.pyplot as plt
import plotparams
plotparams.buba()
    
fs_sky = np.load("/mnt/gosling1/boryanah/light_cone_catalog/sky_coverage/fs_sky.npy")
zs_mt = np.load("/mnt/gosling1/boryanah/light_cone_catalog/sky_coverage/zs_mt.npy")
full_sky = 360**2/np.pi
z_max = 2.4587894699413746

print(fs_sky)
print(zs_mt)
quit()

plt.figure(figsize=(9,7))
# loop over all redshifts
for i in range(len(zs_mt)):
    # current redshift
    z = zs_mt[i]
    f_sky = fs_sky[i]
    if f_sky == 0.: continue
    plt.axvline(x=z, color='lightgray', lw=1.5, ls='--')


plt.axhline(y=full_sky/8., color='r', lw=1.5, ls='-')
plt.axvline(x=z_max, color='g', ls='-')

plt.plot(zs_mt[fs_sky > 0.], fs_sky[fs_sky > 0.])
plt.xlabel(r"$z$")
plt.ylabel(r"${\rm Sky \ coverage \ [deg^2]}$")
plt.xlim([0., 3.])
plt.gca().minorticks_on()
#plt.gca().tick_params(axis='x', which='minor', bottom=True)
plt.savefig("sky_coverage.png")
plt.show()
