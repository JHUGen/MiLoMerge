import timeit


setup_code = """
import sys
import uproot
import numpy as np
sys.path.append('brunelle_merger/')
import brunelle_merger as bm
data = uproot.open('test_data/data.root')
import multidimensionaldiscriminant.optimizeroc as optimizeroc

branches = ['Z1Mass', 'Z2Mass', 'helphi', 'helcosthetaZ1', 'helcosthetaZ2']
sm_data = data['sm'].arrays(branches, library='np')
ps_data = data['ps'].arrays(branches, library='np')

counts_sm = [None]*5
counts_ps = [None]*5
edges = [None]*5    

ranges = [
    None,
    None,
    [-3.14, 3.14],
    [-1,1],
    [-1,1]
]


for n, i in enumerate(branches):
    counts_sm[n], edges[n] = np.histogram(sm_data[i], 100, range=ranges[n], density=True)
    counts_ps[n], _ = np.histogram(ps_data[i], edges[n], density=True)

x = counts_sm[0]
xp = counts_ps[0]

dim_bins = bm.Grim_Brunelle_merger(edges[0], x.copy(), xp.copy(), stats_check=False, subtraction_metric=True)
dim_bins_nonlocal = bm.Grim_Brunelle_nonlocal(edges[0], x.copy(), xp.copy(), stats_check=False, subtraction_metric=True)


hists = {
        "0+": dim_bins.merged_counts.copy()[0],
        "0-": dim_bins.merged_counts.copy()[1]
        }
rocareacoeffdict = {("0+", "0-"): 1}
optimizer = optimizeroc.OptimizeRoc(hists, rocareacoeffdict, mergemultiplebins=True, smallchangetolerance=1e-5)
"""

local_main = """
dim_bins.run(5)
"""
nonlocal_main="""
dim_bins_nonlocal.run(5)
"""
heshy_main="""
optimizer.run(resultfilename="output.pkl", rawfilename="output_raw.pkl")
"""

# exec(setup_code)

print("Local:", timeit.timeit(stmt=local_main, setup=setup_code, number=10000))
print("Non-Local:", timeit.timeit(stmt=nonlocal_main, setup=setup_code, number=10000))
print("Heshy:", timeit.timeit(stmt=heshy_main, setup=setup_code, number=10000))

