"""Step 1: COSMOS2020 CLASSIC -> 30-band (28 measured; no IRAC ch3/ch4 in CLASSIC) total, MW-corrected fluxes [uJy]
following IRSA misc/COSMOS2020_prepare_apertures.txt (2" apertures + total_off2, A_lambda/E(B-V)), joined to the
multimatch spec-z (instrument type 's', homogenised flag >= 3)."""
import numpy as np, pandas as pd, fitsio, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from bands import BANDS

PROJ = os.environ["PROJ"]
FITS = sys.argv[1] if len(sys.argv) > 1 else f"{PROJ}/data/cosmos2020/COSMOS2020_CLASSIC_R1_v2.2_p3.fits"
MM = "/global/cfs/cdirs/desc-pz/users/tqzhang/rail/projects/multimatch/work/cosmos"
NOAPER = {"IRAC_CH1", "IRAC_CH2"}
META = ["ID", "ALPHA_J2000", "DELTA_J2000", "FLAG_HSC", "FLAG_SUPCAM", "FLAG_UVISTA", "FLAG_UDEEP", "FLAG_COMBINED",
        "EBV_MW", "total_off2", "lp_zBEST", "lp_type", "lp_zPDF", "lp_zPDF_l68", "lp_zPDF_u68", "lp_zMinChi2",
        "lp_chi2_best", "lp_NbFilt", "lp_model", "lp_dust", "lp_MNUV", "lp_MR", "lp_MJ", "lp_mass_med",
        "ez_z_phot", "ez_restU", "ez_restV", "ez_restJ"]

h = fitsio.FITS(FITS)[1]
have = set(h.get_colnames())
cols = list(META)
for key, (filt, pre, _) in BANDS.items():
    if pre in NOAPER:
        cols += [f"{pre}_FLUX", f"{pre}_FLUXERR"]
    elif f"{pre}_FLUX_APER2" in have:
        cols += [f"{pre}_FLUX_APER2", f"{pre}_FLUXERR_APER2"]
d = h.read(columns=cols)
out = pd.DataFrame({c: d[c].astype(np.float64) if d[c].dtype.kind == "f" else d[c].astype(d[c].dtype.newbyteorder("="))  for c in META})
off = d["total_off2"].astype(np.float64); ebv = d["EBV_MW"].astype(np.float64)
for key, (filt, pre, aebv) in BANDS.items():
    if pre in NOAPER:
        f, e = d[f"{pre}_FLUX"].astype(np.float64), d[f"{pre}_FLUXERR"].astype(np.float64)
        scale = np.ones_like(f)
    elif f"{pre}_FLUX_APER2" in have:
        f, e = d[f"{pre}_FLUX_APER2"].astype(np.float64), d[f"{pre}_FLUXERR_APER2"].astype(np.float64)
        scale = np.where(np.isfinite(off), 10 ** (-0.4 * off), 1.0)        # aperture -> total
    else:
        out[f"f_{key}"] = np.nan; out[f"ferr_{key}"] = np.nan
        print("missing band", key, pre); continue
    scale = scale * 10 ** (0.4 * aebv * np.nan_to_num(ebv))              # MW de-reddening
    bad = ~np.isfinite(f) | ~np.isfinite(e) | (e <= 0)
    out[f"f_{key}"] = np.where(bad, np.nan, f * scale)
    out[f"ferr_{key}"] = np.where(bad, np.nan, e * scale)

# spec-z truth from multimatch
r = pd.read_parquet(f"{MM}/compilation/refz_341rule_cosmos.parquet")
r = r[(r.instrument_type_homogenized == "s") & (r.z_flag_homogenized >= 3)]
src = pd.read_parquet(f"{MM}/multimatch_cosmos_sources.parquet", columns=["object_id", "survey", "orig_id"])
src = src[src.survey == "COSMOS2020_CLASSIC"]
m = r.drop(columns=["orig_id"]).merge(src[["object_id", "orig_id"]], on="object_id")
m["ID"] = m.orig_id.astype(np.int64)
m = m.rename(columns={"z": "z_spec", "z_flag_homogenized": "z_spec_flag", "survey": "z_spec_survey"})
out["ID"] = out.ID.astype(np.int64)
out = out.merge(m[["ID", "z_spec", "z_spec_flag", "z_spec_survey"]], on="ID", how="left")
print("N total", len(out), "| FLAG_COMBINED==0", (out.FLAG_COMBINED == 0).sum(),
      "| with spec-z", out.z_spec.notna().sum(), "| spec-z & clean", (out.z_spec.notna() & (out.FLAG_COMBINED == 0)).sum())
out.to_parquet(f"{PROJ}/data/cosmos2020/cosmos2020_classic_30band.parquet", index=False)
