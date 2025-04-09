import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import matplotlib.patches as mpatches
from starforge_mult_search.analysis.analyze_stack import subtract_path

from starforge_mult_search.analysis.analyze_multiples_part2 import get_bound_snaps
from starforge_mult_search.analysis.figures.figure_preamble import *
import cgs_const as cgs


def get_spin_angle(tmp_spin1, tmp_spin2):
    return np.dot(tmp_spin1, tmp_spin2) / np.linalg.norm(tmp_spin1) / np.linalg.norm(tmp_spin2)

def spin_angle_from_lookup(spin_lookup, tmp_id1, tmp_id2, sel_bin_snap):
    tmp_spin1 = spin_lookup[str(tmp_id1)][sel_bin_snap][1:]
    tmp_spin2 = spin_lookup[str(tmp_id2)][sel_bin_snap][1:]
    ##Spin vectors
    spin_misalign = get_spin_angle(tmp_spin1, tmp_spin2)
    return spin_misalign

def get_soft_time(id1, id2):
    delta = subtract_path(path_lookup[f"{id1}"][:, pxcol:pzcol + 1], path_lookup[f"{id2}"][:, pxcol:pzcol + 1])
    delta = np.sum(delta * delta, axis=1)**.5 * cgs.pc / cgs.au
    idx_soft = np.where(delta < 40)[0]
    if len(idx_soft) == 0:
        return np.inf
    else:
        return idx_soft[0]

##NOT normal survival filter -- for now just look at things that survive as binaries.
final_bin_filter = (my_data["final_bound_snaps_norm"]==1)
quasi_filter = my_data["quasi_filter"]
bfb_filter = my_data["same_sys_at_fst"].astype(bool)
pmult_before_bin = np.load("pmult_before_bin.npz")["pmult_filt"]
exchange_filter = ~pmult_before_bin

bprops = []
my_data["soft_time"] = np.zeros(len(my_data["bin_ids"]))
for ii, row in enumerate(my_data["bin_ids"]):
    bin_list = list(row)
    id1 = bin_list[0]
    id2 = bin_list[1]
    sys1_info = lookup_dict[id1]
    sys2_info = lookup_dict[id2]
    ##Snaps are all the snapshots the stars are in the same system -- not the snapshots they are in the same binary.
    b1, b2, snaps = get_bound_snaps(sys1_info, sys2_info)
    sel_bin_snap = int(my_data["final_bound_snaps"][ii])
    spin_ang = [spin_angle_from_lookup(spin_lookup, id1, id2, ss) for ss in b1[:, LOOKUP_SNAP].astype(int)]
    sim_end_snap = lookup_dict[id1][0, -1]
    ##Softening filter??
    ##get_first_soft_time--can be done separately...
    q_no_halo =  np.min((b1[:, LOOKUP_M] / b2[:, LOOKUP_M], b2[:, LOOKUP_M] / b1[:, LOOKUP_M]), axis=0)
    my_data["soft_time"][ii] = get_soft_time(id1, id2)
    ##Also have toggle for halo masses(!)
    ## Mass ratio without halos -- controlled by button. Need new column or can be reconstructed from masses...
    # bprops.append([b1[:, LOOKUP_SMA], b1[:, LOOKUP_ECC], np.min((b1[:, LOOKUP_Q], b2[:, LOOKUP_Q]), axis=0),
    #                b1[:, LOOKUP_SNAP] - b1[0,LOOKUP_SNAP], b1[:, LOOKUP_SNAP] / sim_end_snap, b1[:, LOOKUP_SNAP], np.ones(len(b1)) * ii, spin_ang,
    #                np.min((b1[:, LOOKUP_M] / b2[:, LOOKUP_M], b2[:, LOOKUP_M] / b1[:, LOOKUP_M]), axis=0)])
    bprops.append([b1[:, LOOKUP_SMA], b1[:, LOOKUP_ECC], np.min((b1[:, LOOKUP_Q], b2[:, LOOKUP_Q]), axis=0),
                   b1[:, LOOKUP_SNAP] - b1[0,LOOKUP_SNAP], b1[:, LOOKUP_SNAP] / sim_end_snap,  np.ones(len(b1)) * ii, b1[:, LOOKUP_SNAP], spin_ang, q_no_halo])

import pandas as pd
bprops = pd.concat([pd.DataFrame(row).T for row in bprops], ignore_index=0)
bprops.rename(columns={0:"sma", 1:"e", 2:"q", 3:"delay", 4:"snap_norm", 5: "bin_idx", 6:"snap", 7:"spin_ang", 8:"q_no_halo"}, inplace=True)
bprops.set_index("bin_idx", inplace=True)

qfilter = pd.DataFrame(quasi_filter)
qfilter.index.name = "bin_idx"
qfilter.rename(columns={0:"quasi_filter"}, inplace=True)

efilter = pd.DataFrame(exchange_filter)
efilter.index.name = "bin_idx"
efilter.rename(columns={0:"exchange_filter"}, inplace=True)

bfilter = pd.DataFrame(bfb_filter)
bfilter.index.name = "bin_idx"
bfilter.rename(columns={0:"bfb_filter"}, inplace=True)

mfilter = pd.DataFrame(my_data["mfinal_primary"])
mfilter.index.name = "bin_idx"
mfilter.rename(columns={0:"mfinal_primary"}, inplace=True)

##TO DO: ADD THIS FILTER
soft_time = pd.DataFrame(my_data["soft_time"])
soft_time.index.name = "bin_idx"
soft_time.rename(columns={0:"soft_time"}, inplace=True)

seeds_lookup = pd.DataFrame(seeds_lookup)
seeds_lookup.index.name = "bin_idx"
seeds_lookup.rename(columns={0:"seeds_lookup"}, inplace=True)


bprops = pd.merge(bprops, qfilter, on="bin_idx", how="outer")
bprops = pd.merge(bprops, efilter, on="bin_idx", how="outer")
bprops = pd.merge(bprops, bfilter, on="bin_idx", how="outer")
bprops = pd.merge(bprops, mfilter, on="bin_idx", how="outer")
bprops = pd.merge(bprops, soft_time, on="bin_idx", how="outer")
bprops = pd.merge(bprops, seeds_lookup, on="bin_idx", how="outer")

bprops.to_parquet("bprops.pq")