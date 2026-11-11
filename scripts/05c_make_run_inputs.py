"""Build the LePhare input tables (fixed random subsets, so runs are reproducible and can be split into parts).

data_run : COSMOS2020 clean area (FLAG_COMBINED==0): all objects with spec-z + 40k random i<25 objects without
           (the latter to compare our rerun with the official lp_zBEST on the photometric sample).
sim_main : 300k random Cardinal galaxies with noisy i_HSC<25 (10 deg^2 patch).
sim_diag : 40k random subset of sim_main; also written as a noiseless copy (fluxes = true fluxes, same errors).
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from bands import BANDS

PROJ = os.environ["PROJ"]
rng = np.random.default_rng(42)
OUT = f"{PROJ}/work/run_inputs"; os.makedirs(OUT, exist_ok=True)
imag = lambda f: np.where(f > 0, 23.9 - 2.5 * np.log10(np.clip(f, 1e-30, None)), 99.0)

d = pd.read_parquet(f"{PROJ}/data/cosmos2020/cosmos2020_classic_30band.parquet")
d = d[d.FLAG_COMBINED == 0].copy()
d["i_mag"] = imag(d.f_i.values)
spec = d.z_spec.notna()
phot = d[~spec & (d.i_mag < 25)].sample(40000, random_state=1)
dr = pd.concat([d[spec], phot]).reset_index(drop=True)
dr["has_spec"] = dr.z_spec.notna()
dr.to_parquet(f"{OUT}/data_run.parquet", index=False)
print("data_run", len(dr), "spec", dr.has_spec.sum())

s = pd.read_parquet(f"{PROJ}/work/cardinal/cardinal_patch_cosmos30_obs.parquet")
s["i_mag"] = imag(s.f_i.values)
sm = s[s.i_mag < 25].sample(300000, random_state=2).reset_index(drop=True)
sm.to_parquet(f"{OUT}/sim_main.parquet", index=False)
sd = sm.sample(40000, random_state=3).reset_index(drop=True)
sd.to_parquet(f"{OUT}/sim_diag.parquet", index=False)
nl = sd.copy()
for k in BANDS:
    nl[f"f_{k}"] = np.where(np.isfinite(sd[f"f_{k}"]), sd[f"ftrue_{k}"], np.nan)
nl.to_parquet(f"{OUT}/sim_diag_noiseless.parquet", index=False)
print("sim_main", len(sm), "twin", sm.twin.sum(), "| sim_diag", len(sd))
