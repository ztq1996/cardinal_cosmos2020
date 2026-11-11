"""Step 4: add COSMOS2020-like noise to the Cardinal noiseless fluxes.
Empirical, per band and per UltraVISTA region (FLAG_UDEEP): every sim object gets the flux error of a random real
COSMOS2020 object (FLAG_COMBINED==0) with similar flux (0.1 mag bins); faint/negative fluxes use the sky-limited
error distribution of the faintest bins. Noisy flux = true flux + N(0, err). IRAC ch3/ch4 are absent in CLASSIC,
so they are set to NaN in the sim too."""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from bands import BANDS

PROJ = os.environ["PROJ"]
rng = np.random.default_rng(20261110)
data = pd.read_parquet(f"{PROJ}/data/cosmos2020/cosmos2020_classic_30band.parquet")
data = data[data.FLAG_COMBINED == 0]
truth = pd.read_parquet(f"{PROJ}/work/cardinal/cardinal_patch_cosmos30_truemag.parquet")
patch = pd.read_parquet(f"{PROJ}/work/cardinal/cardinal_patch.parquet",
                        columns=["galaxy_id", "ra", "dec", "true_redshift", "redshift", "twin", "mag_i_lsst"])
assert (truth.galaxy_id.values == patch.galaxy_id.values).all()

# UltraVISTA ultra-deep area fraction from bright, complete objects
fi = data.f_i.values
bright = data[(fi > 0) & (23.9 - 2.5 * np.log10(np.clip(fi, 1e-30, None)) < 23)]
f_ud = (bright.FLAG_UDEEP == 1).mean()
print(f"ultra-deep fraction (by i<23 counts): {f_ud:.3f}")
out = patch.copy()
out["udeep"] = (rng.random(len(out)) < f_ud).astype(np.int8)

MBINS = np.arange(14.0, 32.01, 0.1)
NOISE_META = {}
for key in BANDS:
    ftrue = 10 ** (-0.4 * (truth[f"mag_{key}"].values - 23.9))           # uJy
    out[f"ftrue_{key}"] = ftrue
    if data[f"f_{key}"].notna().sum() == 0:
        out[f"f_{key}"] = np.nan
        out[f"ferr_{key}"] = np.nan
        print(f"{key:6s}: not in CLASSIC -> NaN")
        continue
    err = np.full(len(out), np.nan)
    for reg in (0, 1):
        dd = data[(data.FLAG_UDEEP == reg) & data[f"f_{key}"].notna()]
        flux_d, err_d = dd[f"f_{key}"].values, dd[f"ferr_{key}"].values
        mag_d = np.where(flux_d > 0, 23.9 - 2.5 * np.log10(np.clip(flux_d, 1e-30, None)), 99.0)
        bin_d = np.digitize(mag_d, MBINS)
        pools = {b: err_d[bin_d == b] for b in np.unique(bin_d)}
        good = [b for b in pools if len(pools[b]) >= 50 and b < len(MBINS)]
        faint_ref, bright_ref = max(good), min(good)
        sel = np.where(out.udeep.values == reg)[0]
        mag_s = 23.9 - 2.5 * np.log10(ftrue[sel])
        bin_s = np.clip(np.digitize(mag_s, MBINS), bright_ref, faint_ref)
        for b in np.unique(bin_s):
            idx = sel[bin_s == b]
            pool = pools.get(b)
            if pool is None or len(pool) < 50:
                pool = pools[faint_ref]
            err[idx] = rng.choice(pool, size=len(idx))
    # extra (unreported) noise, e.g. IRAC confusion/deblending residuals: from the negative-flux half of the data,
    # sigma_neg = median(|f| : f<0)/0.6745 (a lower bound on the true noise); sigma_extra^2 = sigma_neg^2 - err_neg^2
    fd_all, ed_all = data[f"f_{key}"].values, data[f"ferr_{key}"].values
    neg = np.isfinite(fd_all) & (fd_all < 0)
    sig_neg = np.median(-fd_all[neg]) / 0.6745
    sig_extra = np.sqrt(max(0.0, sig_neg ** 2 - np.median(ed_all[neg]) ** 2)) if neg.sum() > 10000 else 0.0
    out[f"ferr_{key}"] = err                                              # reported error (as in the catalogue)
    out[f"f_{key}"] = ftrue + rng.normal(size=len(out)) * np.sqrt(err ** 2 + sig_extra ** 2)
    sky = np.median(err[ftrue < np.percentile(ftrue, 5)])
    print(f"{key:6s}: reported sky err {sky:.4f} uJy (3sig {23.9 - 2.5 * np.log10(3 * sky):.2f}); "
          f"extra noise {sig_extra:.4f} uJy -> true 3sig {23.9 - 2.5 * np.log10(3 * np.hypot(sky, sig_extra)):.2f}")
    NOISE_META[key] = dict(sky_err=sky, sig_extra=sig_extra)
pd.DataFrame(NOISE_META).T.to_csv(f"{PROJ}/work/cardinal/noise_model_summary.csv")
out.to_parquet(f"{PROJ}/work/cardinal/cardinal_patch_cosmos30_obs.parquet", index=False)
print("written", len(out))
