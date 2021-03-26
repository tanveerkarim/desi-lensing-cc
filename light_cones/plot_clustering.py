import numpy as np
import matplotlib.pyplot as plt


wp = np.load("data/wp.npy")
wp_down = np.load("data/wp_down.npy")
wp_sf = np.load("data/wp_sandy.npy")
wp_s = np.load("/home/boryanah/repos/abacusutils/scripts/hod/data/wp.npy")
bc_s = np.load("/home/boryanah/repos/abacusutils/scripts/hod/data/bin_cents.npy")
wp_b = np.load("data/wp_buba.npy")
bin_cents = np.load("data/rbin_cents.npy")

#plt.plot(bin_cents, wp*bin_cents, label='light cone mock')
plt.plot(bin_cents, wp_sf*bin_cents, label='annulus snapshot')
plt.plot(bc_s, wp_s*bc_s, label='full snapshot')
#plt.plot(bin_cents, wp_down*bin_cents, label='light cone downsampled')
plt.plot(bin_cents, wp_b*bin_cents, label='light cone downsampled after cleaning')
plt.xscale('log')
plt.legend()
plt.savefig("figs/wp.png")
plt.show()
