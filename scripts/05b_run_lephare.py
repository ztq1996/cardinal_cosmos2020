"""Step 5: run LePhare (COSMOS2020 config) on a prepared flux table (data or sim).

  python 05b_run_lephare.py --kind data --input data/cosmos2020/cosmos2020_classic_30band.parquet \
      --query "FLAG_COMBINED==0 and i_mag<25" --zcol z_spec --offsets auto --tag data_i25

Input table columns: f_<band>, ferr_<band> [uJy] for every key in bands.BANDS (NaN = missing).
--offsets: 'zero', 'auto' (AUTO_ADAPT on objects with finite zcol & i_mag<ADAPT max, max --nadapt objects),
           or a path to a .npy with per-band offsets.
Output: work/lephare_runs/<tag>.parquet with LePhare point outputs, PDF quantiles and PIT at zcol, plus
work/lephare_runs/<tag>_offsets.npy."""
import argparse, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from bands import BANDS
from lephare_config import cosmos_config, OUTKEYS
import lephare as lp

PROJ = os.environ["PROJ"]
p = argparse.ArgumentParser()
p.add_argument("--kind", required=True, choices=["data", "sim"])
p.add_argument("--input", required=True)
p.add_argument("--query", default=None)
p.add_argument("--zcol", default=None)
p.add_argument("--idcol", default="ID")
p.add_argument("--offsets", default="zero")
p.add_argument("--nadapt", type=int, default=30000)
p.add_argument("--fluxprefix", default="f_")
p.add_argument("--tag", required=True)
p.add_argument("--chunk", type=int, default=50000)
p.add_argument("--nmax", type=int, default=0)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--nproc", type=int, default=1)
p.add_argument("--reverse", action="store_true", help="process chunks last-first (to share a run between two nodes)")
p.add_argument("--offsets_only", action="store_true")
a = p.parse_args()

outdir = f"{PROJ}/work/lephare_runs"; os.makedirs(outdir, exist_ok=True)
df = pd.read_parquet(a.input if a.input.startswith("/") else f"{PROJ}/{a.input}")
fi = df[f"{a.fluxprefix}i"].values
df["i_mag"] = np.where(fi > 0, 23.9 - 2.5 * np.log10(np.clip(fi, 1e-30, None)), 99.0)
if a.query:
    df = df.query(a.query)
if a.nmax and len(df) > a.nmax:
    df = df.sample(a.nmax, random_state=a.seed)
df = df.reset_index(drop=True)
print(f"[{a.tag}] {len(df)} objects", flush=True)

keys = list(BANDS)
F = np.stack([df[f"{a.fluxprefix}{k}"].values for k in keys], axis=1).astype(np.float64)
E = np.stack([df[f"ferr_{k}"].values for k in keys], axis=1).astype(np.float64)
bad = ~np.isfinite(F) | ~np.isfinite(E) | (E <= 0)
F *= 1e-29; E *= 1e-29                      # uJy -> erg/s/cm^2/Hz (LePhare INP_TYPE F convention)
F[bad] = -99.0; E[bad] = -99.0
zs = df[a.zcol].values.astype(np.float64) if a.zcol else np.full(len(df), -99.0)
zs = np.where(np.isfinite(zs), zs, -99.0)
ids = df[a.idcol].astype(str).values

cfg = lp.all_types_to_keymap(cosmos_config(a.kind))
photz = lp.PhotoZ(cfg)

def make_sources(idx):
    srcs = []
    for j in idx:
        s = lp.onesource(int(j), photz.gridz)
        s.readsource(ids[j], F[j], E[j], 0, zs[j], "")
        photz.prep_data(s)
        srcs.append(s)
    return srcs

# --- zero-point offsets
offfile = f"{outdir}/{a.tag}_offsets.npy"
if a.offsets == "zero":
    a0 = np.zeros(len(keys))
elif a.offsets == "auto" and os.path.exists(offfile):
    a0 = np.load(offfile)
    print("re-using offsets: " + " ".join(f"{k}:{v:+.3f}" for k, v in zip(keys, a0)), flush=True)
elif a.offsets == "auto":
    lo, hi = [float(x) for x in cfg["ADAPT_LIM"].value.split(",")]
    cand = np.where((zs > 0) & (df.i_mag.values > lo) & (df.i_mag.values < hi))[0]
    rng = np.random.default_rng(a.seed)
    if len(cand) > a.nadapt:
        cand = rng.choice(cand, a.nadapt, replace=False)
    print(f"AUTO_ADAPT on {len(cand)} objects", flush=True)
    t0 = time.time()
    cfg_ad = lp.all_types_to_keymap({**cosmos_config(a.kind), "AUTO_ADAPT": "YES"})   # PhotoZ reads it at init
    photz_ad = lp.PhotoZ(cfg_ad)
    srcs_ad = []
    for j in cand:
        s = lp.onesource(int(j), photz_ad.gridz)
        s.readsource(ids[j], F[j], E[j], 0, zs[j], "")
        photz_ad.prep_data(s)
        srcs_ad.append(s)
    a0 = np.array(photz_ad.compute_offsets(srcs_ad))
    del photz_ad, srcs_ad
    print(f"offsets ({time.time()-t0:.0f}s): " + " ".join(f"{k}:{v:+.3f}" for k, v in zip(keys, a0)), flush=True)
else:
    a0 = np.load(a.offsets)
np.save(offfile, a0)
if a.offsets_only:
    sys.exit(0)

# --- main run in chunks; with --nproc>1 the chunks are processed by forked workers that share the
# already-loaded library (run with OMP_NUM_THREADS=1: libgomp is not fork-safe once its thread pool exists)
PDFKEY = photz.pdfdict["BAY_ZG"]
QS = [0.025, 0.16, 0.5, 0.84, 0.975]
chunkdir = f"{outdir}/{a.tag}_chunks"; os.makedirs(chunkdir, exist_ok=True)
starts = list(range(0, len(df), a.chunk))
todo = [c0 for c0 in starts if not os.path.exists(f"{chunkdir}/chunk_{c0:08d}.parquet")]
if a.reverse:
    todo = todo[::-1]


def run_chunk(c0):
    fn = f"{chunkdir}/chunk_{c0:08d}.parquet"
    if os.path.exists(fn):          # done meanwhile by another node
        return 0
    idx = np.arange(c0, min(c0 + a.chunk, len(df)))
    srcs = make_sources(idx)
    photz.run_photoz(srcs, list(a0))
    t = photz.build_output_tables(srcs, para_out=cfg["PARA_OUT"].value)
    tab = t[[n for n in t.colnames if len(t[n].shape) <= 1]].to_pandas()
    zg = np.array(srcs[0].pdfmap[PDFKEY].xaxis)
    P = np.clip(np.array([s.pdfmap[PDFKEY].vPDF for s in srcs]), 0, None)
    norm = P.sum(1, keepdims=True); norm[norm == 0] = 1
    C = np.cumsum(P / norm, axis=1)
    for q in QS:
        tab[f"zpdf_q{int(q*1000):03d}"] = zg[np.clip((C < q).sum(1), 0, len(zg) - 1)]
    zt = zs[idx]
    tab["PIT"] = np.where(zt >= 0, np.array([np.interp(z, zg, c) for z, c in zip(zt, C)]), np.nan)
    tab["row"] = idx
    tab.to_parquet(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)
    return len(idx)


t0 = time.time(); ndone = 0
print(f"[{a.tag}] {len(todo)}/{len(starts)} chunks to run with {a.nproc} process(es)", flush=True)
if a.nproc > 1:
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.nproc) as pool:
        for k, n in enumerate(pool.imap_unordered(run_chunk, todo)):
            ndone += n
            if k % max(1, len(todo) // 20) == 0 or k == len(todo) - 1:
                el = time.time() - t0
                print(f"  {k+1}/{len(todo)} chunks, {ndone} objects in {el:.0f}s, {ndone/el:.1f} obj/s", flush=True)
else:
    for c0 in todo:
        ndone += run_chunk(c0); el = time.time() - t0
        print(f"  chunk {c0}: {ndone} objects in {el:.0f}s, {ndone/el:.1f} obj/s", flush=True)

# merge (only once every chunk exists; another node may still be working on the same run)
fns = [f"{chunkdir}/chunk_{c0:08d}.parquet" for c0 in starts]
missing = sum(not os.path.exists(f) for f in fns)
if missing:
    print(f"[{a.tag}] {missing} chunks still missing; not merging", flush=True)
    sys.exit(0)
res = pd.concat([pd.read_parquet(f) for f in fns], ignore_index=True)
meta = df.drop(columns=[c for c in df.columns if c.startswith(("f_", "ferr_", "ftrue_"))])
res = pd.concat([meta.iloc[res.row.values].reset_index(drop=True), res.drop(columns=["row"])], axis=1)
res.to_parquet(f"{outdir}/{a.tag}.parquet", index=False)
print(f"[{a.tag}] merged {len(res)} rows in {time.time()-t0:.0f}s", flush=True)
