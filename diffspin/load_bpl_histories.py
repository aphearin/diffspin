"""
"""
import numpy as np
import os

BEBOP = "/lcrc/project/halotools/BolshoiPlanck/FULL_TREES/LOGMPCUT_TRUNKS"
TASSO = "/Users/aphearin/work/DATA/SIMS/BPl/full_trees"
BPL_LOGMP_CUT = 10.0


def load_histories(drn, colname):
    halos = np.load(os.path.join(drn, "bpl_cens_trunks_{}.npy".format(colname)))
    log_mahs = np.load(os.path.join(drn, "bpl_cens_trunks_mahs.npy"))["log_mah"]
    log_mahs = np.maximum.accumulate(log_mahs, axis=1)
    t_bpl = np.load(os.path.join(drn, "bpl_cosmic_time.npy"))
    return halos["halo_id"], halos[colname], log_mahs, t_bpl, BPL_LOGMP_CUT
