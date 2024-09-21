import sys
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
sys.path.append("../")
sys.path.append("../../")
import bin_splitter as bs
import ROC_curves as ROC
import tqdm

N=1000000
names = ["MZ1", "MZ2", "costheta1d", "costheta2d", "Phid"]
data = uproot.open("../../../test_data/all_reweighted_2e2mu.root")['eventTree'].arrays(
    names + ["p_Gen_GG_SIG_ghg2_1_ghz1_1_JHUGen", "p_Gen_GG_SIG_ghg2_1_ghz4_1_JHUGen"],
    entry_stop=N
)

def processCutsFromLog(file):
    binCounts = [
        [],
        []
    ]
    with open(file) as f:
        for line in tqdm.tqdm(f, total=N):
            cut = bs.decode(names, line.strip())
            for entry in names:
                cut = cut.replace(entry, f"data.{entry}")
            b = eval(cut)


            binCounts[0].append(len(data.p_Gen_GG_SIG_ghg2_1_ghz4_1_JHUGen[b]))

            binCounts[1].append(
                ak.sum(
                    (data.p_Gen_GG_SIG_ghg2_1_ghz4_1_JHUGen/data.p_Gen_GG_SIG_ghg2_1_ghz1_1_JHUGen)[b]
                )
            )

    binCounts = np.array(binCounts)
    print(binCounts)
    x, y, score, err = ROC.ROC_curve(*binCounts)
    return x,y,score,err

x,y,score,err = processCutsFromLog("../cuts.log")
plt.plot(x,y,label=f"{1 - score:.2f} $\pm$ {err:.2e}")

x,y,score,err = processCutsFromLog("../cuts_new.log")
plt.plot(x,y,label=f"{1 - score:.2f} $\pm$ {err:.2e}")

plt.ylim(0,1)
plt.xlim(0,1)
plt.gca().plot([0, 1], [0, 1], transform=plt.gca().transAxes)
plt.legend()
plt.savefig("ROC_split.png")