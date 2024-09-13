import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.CMS)

hist_counts = np.loadtxt("hist.log", dtype=float, delimiter=",")

c1 = hist_counts[:,0]
c2 = hist_counts[:,1]
bins = np.linspace(0, 1, len(hist_counts) + 1)

indices = np.argsort(c1/c2)

c1_str = r"$\sum g_1=$" + f"{c1.sum():.2f}"
c2_str = r"$\sum g_4=$" + f"{c2.sum():.2f}"

plt.gca().axhline(0.0015, lw=2, color='k', label="Stat Limit", linestyle=(0, (5, 10)))
hep.histplot([c1[indices], c2[indices]], bins, label=[c1_str, c2_str])
plt.xlabel("Dummy Index [sorted g1/g4 ascending]")
plt.ylabel("MultiDimensional Count / Bin")
plt.legend()
plt.tight_layout()
plt.savefig('test_split.png')
