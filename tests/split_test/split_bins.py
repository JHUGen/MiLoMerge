import sys, os
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
sys.path.append(os.path.abspath("../../optimized_binning"))
import bin_splitter as bs

hep.style.use(hep.style.CMS)

data = uproot.open(
    "all_reweighted_2e2mu.root"
)['eventTree'].arrays(
    ["MZ1", "MZ2", "costheta1d", "costheta2d", "Phid", "p_Gen_GG_SIG_ghg2_1_ghz1_1_JHUGen", "p_Gen_GG_SIG_ghg2_1_ghz4_1_JHUGen"], 
    library='np', entry_stop=100_000)

h1 = data["p_Gen_GG_SIG_ghg2_1_ghz1_1_JHUGen"]
h2 = data["p_Gen_GG_SIG_ghg2_1_ghz4_1_JHUGen"]

data_input = [
    {"MZ1":data["MZ1"], "MZ2":data["MZ2"], "Costheta1":data["costheta1d"], "Costheta2":data["costheta2d"], "Phi":data["Phid"]},
    {"MZ1":data["MZ1"], "MZ2":data["MZ2"], "Costheta1":data["costheta1d"], "Costheta2":data["costheta2d"], "Phi":data["Phid"]}
]
w = [
    h1.copy()/h1,
    h2.copy()/h1
]

splitter = bs.Splitter(*data_input, weights=w)
splitter.split(100, stat_limit=0.0015)
with open("cuts_made.log", "w+") as f:
    for node in splitter.decision_tree.get_leaves():
        f.write(f"{node.get_all_cuts_as_str()}\n")
    f.write("\n")

counts, bins = splitter.get_histogram()
np.savetxt("hist.log", counts, delimiter=", ")

