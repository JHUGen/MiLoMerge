import numpy as np
import MiLoMerge
import tqdm
import cProfile

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt


testing1 = np.hstack(
    (
        np.random.normal(3, 0.5, (200000,1)), 
        np.random.choice([-1, 1], size=(200000,1))*np.sinh(np.random.rand(200000,1)*2*np.pi)
    ),
)
testing1 = testing1[
    (testing1[:,0] > 0) & (testing1[:,0] < 10) & (testing1[:,1] > -1) & (testing1[:,1] < 1)
]


testing2 = np.hstack(
    (
        np.abs(np.random.normal(0, 3, (1000000,1))),
        np.cos(np.random.rand(1000000,1)*2*np.pi)
    )
)
testing2 = testing2[
    (testing2[:,0] > 0) & (testing2[:,0] < 10) & (testing2[:,1] > -1) & (testing2[:,1] < 1)
]

bins = np.linspace(0, 10, 11)
bins2 = np.linspace(-1,1, 11)

signal, *_ = np.histogram2d(testing1[:,0], testing1[:,1], (bins, bins2))
signal_sum = np.sum(signal)
signal = signal/signal_sum
bkg, *_ = np.histogram2d(testing2[:,0], testing2[:,1], (bins, bins2))
bkg_sum = np.sum(bkg)
bkg = bkg/bkg_sum

map_at=(
        1800,
        1700,
        1600,
        1500,
        1300,
        1000,
        900,
        800,
        700,
        500,
        350,
        238,
        80,
        90,
        70,
        60, 
        50,
        40,
        30,
        20,
        10,
        7,
        5,
        3,
        2
    )

merger = MiLoMerge.MergerNonlocal(
    (
        bins,
        bins2
    ),
    signal.copy(),
    bkg.copy(),
    map_at=map_at
)
# pr = cProfile.Profile()
# pr.enable()
new_bins = merger.run(3)
# pr.disable()
# pr.print_stats()
print(new_bins)

x = []
y = []

for i in tqdm.tqdm((None, ) + map_at):
    if i is None:
        _, _, score = MiLoMerge.ROC_curve(signal.ravel(), bkg.ravel())
        x.append(len(signal.ravel()))
        y.append(score - 0.5)
    elif i > len(signal.ravel()):
        continue
    else:
        signal_counts = MiLoMerge.place_array_nonlocal(i, testing1)
        signal_counts, _ = np.histogram(signal_counts, np.arange(i))
        signal_counts = signal_counts/signal_counts.sum()
        background_counts = MiLoMerge.place_array_nonlocal(i, testing2)
        background_counts, _ = np.histogram(background_counts, np.arange(i))
        background_counts = background_counts/background_counts.sum()
        
        _, _, score = MiLoMerge.ROC_curve(signal_counts, background_counts)
        x.append(i)
        y.append(score - 0.5)

x = np.array(x)
y = np.array(y)


y = y[np.argsort(x)]
x = x[np.argsort(x)]

plt.plot(x, y)
plt.savefig("test")

# signalNew, _ = np.histogram(testing1, new_bins)
# bkgNew, _ = np.histogram(testing2, new_bins)
# hep.histplot([signalNew, bkgNew], new_bins);
# ROC_curves.ROC_score(signalNew.astype(float), bkgNew.astype(float))

