"""Step 0: fetch LePhare auxiliary data (filters, SEDs, extinction, opacity) for the COSMOS2020 config
into the project-local LEPHAREDIR. COSMOS2020 catalog download is done with curl (see README)."""
import lephare as lp
from lephare import data_retrieval as dr

cfg = dict(lp.default_cosmos_config)
extra = ["opa/OPACITY.dat"] + [f"opa/tau{i:02d}.out" for i in range(0, 81)]
extra += ["vega/a0v.sed", "vega/BD+17o4708.sed", "vega/SunLCB.sed", "vega/VegaLCB.sed", "vega/a0v_n.sed"]
extra += [f"filt/lsst/total_{b}.pb" for b in ["u", "g", "r", "i", "z", "y3"]]
dr.get_auxiliary_data(keymap=cfg, additional_files=extra)
print("done; LEPHAREDIR =", lp.LEPHAREDIR)
