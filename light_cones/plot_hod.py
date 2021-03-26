import numpy as np
import matplotlib.pyplot as plt

m_binc = np.load("data/m_binc.npy")

hist = np.load("data/hist_sandy.npy")
hist_norm = np.load("data/hist_norm_sandy.npy")

plt.plot(m_binc, hist/hist_norm, c='b', label="snapshot")

hist = np.load("data/hist_buba.npy")
hist_norm = np.load("data/hist_norm_buba.npy")

plt.plot(m_binc, hist/hist_norm, c='magenta', label="light cones, z = 0.8")

hist_cent = np.load("data/hist_cent.npy")
hist_sats = np.load("data/hist_sats.npy")
hist_norm = np.load("data/hist_norm.npy")

#plt.plot(m_binc, hist_cent/hist_norm, 'k--', label="light cones")
#plt.plot(m_binc, hist_sats/hist_norm, 'k')

bin_cents = np.load("/home/boryanah/repos/abacus_lc_cat/hod/data/bin_cents.npy")
hod_cen = np.load("/home/boryanah/repos/abacus_lc_cat/hod/data/hist_cen.npy")
hod_sat = np.load("/home/boryanah/repos/abacus_lc_cat/hod/data/hist_sat.npy")

plt.plot(bin_cents, hod_cen, 'r--', label="analytic")
plt.plot(bin_cents, hod_sat, 'r')

plt.legend()

plt.ylim([1.e-7, 10.])

plt.xscale('log')
plt.yscale('log')
plt.savefig("figs/hod.png")
plt.show()
