"""Step 3: noiseless COSMOS2020-band (+LSST) photometry for the Cardinal patch from its kcorrect SEDs.
SED_g = COEFFS[sedid] . basis; band flux is linear in COEFFS, so we tabulate basis-template band fluxes on a
fine z grid (with Madau95 IGM) and interpolate. Mags are normalised to the catalogue mag_i_lsst."""
import numpy as np, pandas as pd, fitsio, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from bands import BANDS, LSST

PROJ = os.environ["PROJ"]; LDIR = os.environ["LEPHAREDIR"]
TRAIN = "/global/cfs/cdirs/des/jderose/SkyFactory-config/Addgals/sdss_vagc_kcorr_r177_z_005_02_wdens.fits"
BASIS = "/global/cfs/projectdirs/desc-pz/Cardinal/basis_seds/component_{}.dat"
ZGRID = np.arange(0.0, 2.401, 0.002)

def madau_tau(lobs, z):
    """Madau (1995) IGM optical depth, lobs [A] (array), scalar z."""
    tau = np.zeros_like(lobs)
    for lj, A in zip([1215.67, 1025.72, 972.54, 949.74], [3.6e-3, 1.7e-3, 1.2e-3, 9.3e-4]):
        m = lobs < lj * (1 + z)
        tau[m] += A * (lobs[m] / lj) ** 3.46
    m = lobs < 911.75 * (1 + z)
    xc, xe = lobs[m] / 911.75, 1 + z
    tau[m] += (0.25 * xc**3 * (xe**0.46 - xc**0.46) + 9.4 * xc**1.5 * (xe**0.18 - xc**0.18)
               - 0.7 * xc**3 * (xc**-1.32 - xe**-1.32) - 0.023 * (xe**1.68 - xc**1.68))
    return tau

def load_filter(rel):
    d = np.loadtxt(f"{LDIR}/filt/{rel}", comments="#")
    d = d[np.argsort(d[:, 0])]
    lo, hi = d[d[:, 1] > 0, 0].min(), d[d[:, 1] > 0, 0].max()
    lam = np.linspace(lo, hi, 4000)
    return lam, np.clip(np.interp(lam, d[:, 0], d[:, 1]), 0, None)

def basis_table(filters):
    B = [np.loadtxt(BASIS.format(k)) for k in range(1, 6)]
    wl = B[0][:, 0]; S = np.array([b[:, 1] for b in B])           # (5, nwl) f_lambda rest
    tab = np.zeros((len(filters), 5, len(ZGRID)))
    for ib, (lam, T) in enumerate(filters):
        norm = np.trapezoid(T / lam, lam) * 2.998e18
        for iz, z in enumerate(ZGRID):
            lr = lam / (1 + z)
            att = np.exp(-madau_tau(lam, z)) if z > 0 else 1.0
            for k in range(5):
                f = np.interp(lr, wl, S[k], left=0, right=0) * att
                tab[ib, k, iz] = np.trapezoid(f * T * lam, lam) / norm   # ~ f_nu (arbitrary norm)
    return tab

def main():
    names = list(BANDS) + [f"lsst_{b}" for b in LSST]
    rels = [v[0] for v in BANDS.values()] + [v[0] for v in LSST.values()]
    cache = f"{PROJ}/work/cardinal/basis_band_table.npz"
    if os.path.exists(cache):
        tab = np.load(cache)["tab"]
    else:
        tab = basis_table([load_filter(r) for r in rels]); np.savez(cache, tab=tab, names=names, z=ZGRID)
    coeffs = fitsio.read(TRAIN, columns=["COEFFS"])["COEFFS"].astype(np.float64)
    df = pd.read_parquet(f"{PROJ}/work/cardinal/cardinal_patch.parquet")
    c = coeffs[df.sedid.values]                                     # (N,5)
    z = df.true_redshift.values
    iz = np.clip(np.searchsorted(ZGRID, z) - 1, 0, len(ZGRID) - 2)
    w = (z - ZGRID[iz]) / (ZGRID[iz + 1] - ZGRID[iz])
    out = pd.DataFrame({"galaxy_id": df.galaxy_id.values})
    flux = {}
    for ib, n in enumerate(names):
        F = tab[ib][:, iz] * (1 - w) + tab[ib][:, iz + 1] * w        # (5,N)
        flux[n] = np.einsum("nk,kn->n", c, F)
    m_i_sed = -2.5 * np.log10(flux["lsst_i"])
    off = df.mag_i_lsst.values - m_i_sed
    for n in names:
        with np.errstate(divide="ignore"):
            out[f"mag_{n}"] = np.where(flux[n] > 0, -2.5 * np.log10(flux[n]) + off, 99.0)
    out.to_parquet(f"{PROJ}/work/cardinal/cardinal_patch_cosmos30_truemag.parquet", index=False)
    # validation against catalogue LSST mags
    for b in "ugrzy":
        d = out[f"mag_lsst_{b}"] - df[f"mag_{b}_lsst"]
        print(f"LSST {b}: synth-cat median {np.median(d):+.4f}, 16/84 {np.percentile(d,16):+.4f} {np.percentile(d,84):+.4f}, frac|d|>0.05 {(abs(d)>0.05).mean():.3f}")
    print(out.describe().T[["mean", "min", "50%", "max"]].to_string())

if __name__ == "__main__":
    main()
