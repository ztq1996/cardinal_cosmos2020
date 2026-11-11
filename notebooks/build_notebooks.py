"""Generate the analysis notebooks: python notebooks/build_notebooks.py [01 02 03] (default: all; this overwrites
executed outputs), then execute with jupyter nbconvert --execute --inplace."""
import nbformat as nbf
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = {"name": "rail-2.0-dev", "display_name": "rail-2.0-dev", "language": "python"}
SETUP = """%matplotlib inline
import os, sys
os.environ['PROJ'] = os.path.abspath('..')   # notebooks are executed from notebooks/
sys.path.insert(0, os.environ['PROJ'] + '/scripts')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from analysis_utils import *
plt.rcParams.update({'figure.dpi': 90, 'font.size': 9})
pd.set_option('display.width', 200)"""


def nb(cells, path):
    """Write notebook `path`; with command-line args, only the notebooks whose name contains one of them."""
    if len(sys.argv) > 1 and not any(a in path for a in sys.argv[1:]):
        return
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = KERNEL
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    nbf.write(n, f"{HERE}/{path}")


# =====================================================================================================
nb([
"""md:# 01 — Inputs: Cardinal → COSMOS2020-like 30-band photometry

This notebook shows how the mock is built and how its photometry compares with the real catalogue.
1. **SEDs**: Cardinal `sedid` → SDSS-VAGC kcorrect `COEFFS` × 5 basis templates. Check: synthetic LSST mags vs the catalogue columns.
2. **Noise**: per band, each simulated galaxy takes the reported error of a random real COSMOS2020 object with the same flux, in the same UltraVISTA ultra-deep/deep region. Extra unreported noise, measured from the real negative-flux tail, is added to the flux realisation; in practice this only affects IRAC.
3. **Photometric sample**: number counts, colour–colour densities and redshift distributions, sim vs data.

Samples: COSMOS2020 CLASSIC clean area (`FLAG_COMBINED==0`). Cardinal: 10.07 deg² patch from triofile 101 (RA 126.7–132.1, Dec 40.5–43.0), including a 2.01 deg² “COSMOS twin” sub-box.""",
SETUP,
"""data = pd.read_parquet(f'{PROJ}/data/cosmos2020/cosmos2020_classic_30band.parquet')
data = data[data.FLAG_COMBINED == 0].copy()
sim = pd.read_parquet(f'{PROJ}/work/cardinal/cardinal_patch_cosmos30_obs.parquet')
truth = pd.read_parquet(f'{PROJ}/work/cardinal/cardinal_patch_cosmos30_truemag.parquet')
patch = pd.read_parquet(f'{PROJ}/work/cardinal/cardinal_patch.parquet')
for df in (data, sim):
    add_mags(df); df['i_mag'] = mag(df.f_i)
# areas: sim from the box; data from occupied healpix cells (nside 4096, ~0.74 arcmin^2) of the clean catalogue
import healpy as hp
pix = hp.ang2pix(4096, data.ALPHA_J2000.values, data.DELTA_J2000.values, lonlat=True)
A_pix = len(np.unique(pix)) * hp.nside2pixarea(4096, degrees=True)
A_data = 1.27   # Weaver+22: effective area of FLAG_COMBINED==0 (the occupied-pixel count above over-counts edge pixels)
A_sim = 10.07
print(f'data clean area {A_data} deg2 (occupied-pixel upper bound {A_pix:.2f}), N={len(data)};  sim area {A_sim} deg2, N={len(sim)}')
for c in (24, 25):
    print(f'  i<{c}: data {(data.i_mag < c).sum()/A_data:,.0f} /deg2 (galaxies {((data.i_mag < c) & (data.lp_type == 0)).sum()/A_data:,.0f}),  sim {(sim.i_mag < c).sum()/A_sim:,.0f} /deg2')""",
"""md:## 1. SED reconstruction check
Synthetic LSST magnitudes from the reconstructed SEDs, normalised to the catalogue `mag_i_lsst`, compared with the other catalogue LSST columns. u, g, r and z agree to ≲0.01 mag. y is offset in a redshift-dependent way: Cardinal used a different LSST y curve, and the 4000 Å break crosses the y band at z≈1.5. The COSMOS bands are computed with their own filter curves, so the y offset does not propagate.
At z>1.8 the synthetic u band is fainter than Cardinal's by up to 0.2 mag. Our photometry includes Madau (1995) IGM absorption (the Lyα forest enters u at z≳1.8); the Cardinal catalogue magnitudes evidently do not.""",
"""zb = np.arange(0, 2.41, 0.1); zc = 0.5*(zb[1:]+zb[:-1])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for b in 'ugrzy':
    d = truth[f'mag_lsst_{b}'] - patch[f'mag_{b}_lsst']
    idx = np.digitize(patch.true_redshift, zb) - 1
    q = lambda k, p: np.percentile(d[idx == k], p) if (idx == k).sum() > 10 else np.nan
    med = [q(k, 50) for k in range(len(zc))]; p16 = [q(k, 16) for k in range(len(zc))]; p84 = [q(k, 84) for k in range(len(zc))]
    ax[0].plot(zc, med, label=f'LSST {b}'); ax[0].fill_between(zc, p16, p84, alpha=0.15)
    ax[1].hist(d.clip(-0.3, 0.3), bins=120, histtype='step', label=b, density=True)
ax[0].axhline(0, c='k', lw=0.5); ax[0].set_xlabel('true z'); ax[0].set_ylabel('synthetic - catalogue [mag]'); ax[0].legend(); ax[0].set_ylim(-0.3, 0.1)
ax[1].set_xlabel('synthetic - catalogue [mag]'); ax[1].set_yscale('log'); ax[1].legend()
plt.show()""",
"""md:## 2. Noise model
Left: median reported flux error vs magnitude in each band, for data and sim (UltraVISTA ultra-deep and deep stripes are pooled). Right: the error-calibration test on real data. For objects with negative measured flux, flux/err should be the negative half of a unit Gaussian if the errors are right. IRAC ch1/ch2 fail this badly: deblending/confusion residuals are not in the reported errors. That is why ≈0.1 µJy of extra noise is added to the sim IRAC fluxes.""",
"""nm = pd.read_csv(f'{PROJ}/work/cardinal/noise_model_summary.csv', index_col=0)
nm['3sig_reported'] = 23.9 - 2.5*np.log10(3*nm.sky_err); nm['3sig_true'] = 23.9 - 2.5*np.log10(3*np.hypot(nm.sky_err, nm.sig_extra))
display(nm.round(4))
show = ['u', 'g', 'i', 'z', 'Ks', 'IB427', 'IA679', 'NB816', 'ch1']
mb = np.arange(18, 30.01, 0.25); mc = 0.5*(mb[1:]+mb[:-1])
fig, axs = plt.subplots(2, len(show)//2+1, figsize=(18, 7)); axs = axs.flat
for k, b in enumerate(show):
    for df, lab in ((data, 'data'), (sim, 'sim')):
        m = mag(df[f'f_{b}']); e = df[f'ferr_{b}']
        idx = np.digitize(m, mb) - 1
        med = [np.nanmedian(e[idx == j]) if (idx == j).sum() > 30 else np.nan for j in range(len(mc))]
        axs[k].plot(mc, med, color=COLORS[lab], label=lab)
    axs[k].set_yscale('log'); axs[k].set_title(b); axs[k].set_xlabel('mag (measured)')
axs[0].set_ylabel('median reported err [uJy]'); axs[0].legend()
ax = axs[len(show)]
for b in ['i', 'Ks', 'IB427', 'NB816', 'ch1', 'ch2']:
    x = (data[f'f_{b}']/data[f'ferr_{b}']).values; x = x[np.isfinite(x) & (x < 0)]
    ax.hist(-x, bins=np.logspace(-2, 2, 60), histtype='step', density=True, label=f'{b} (N={len(x)})')
xx = np.logspace(-2, 2, 200); ax.plot(xx, 2*np.exp(-xx**2/2)/np.sqrt(2*np.pi), 'k--', label='half-normal, sigma=1')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-4, 5); ax.set_xlabel('-flux/err (objects with flux<0)'); ax.legend(fontsize=6)
for a in list(axs)[len(show)+1:]: a.axis('off')
plt.tight_layout(); plt.show()""",
"""md:### Signal-to-noise vs magnitude (measured S/N, i<26)
The sim should follow the data in every band. In the IRAC bands the sim's *reported* S/N matches the data by construction, while its *true* scatter includes the extra noise.""",
"""fig, axs = plt.subplots(2, 5, figsize=(18, 6)); axs = axs.flat
for k, b in enumerate(['u', 'g', 'r', 'i', 'y', 'J', 'Ks', 'IA527', 'NB118', 'ch1']):
    for df, lab in ((data, 'data'), (sim, 'sim')):
        s = df[df.i_mag < 26].sample(min(60000, (df.i_mag < 26).sum()), random_state=1)
        axs[k].scatter(mag(s[f'f_{b}']), s[f'f_{b}']/s[f'ferr_{b}'], s=0.3, alpha=0.2, color=COLORS[lab], rasterized=True, label=lab)
    axs[k].set_yscale('log'); axs[k].set_ylim(0.3, 3000); axs[k].set_xlim(18, 29); axs[k].set_title(b); axs[k].axhline(5, c='k', lw=0.5)
axs[0].legend(markerscale=20); plt.tight_layout(); plt.show()""",
"""md:## 3. Number counts
Differential counts per deg² from the measured total magnitudes. The data also contain stars (no stars in the sim), which matter at bright magnitudes. The mock counts turn over at i≈25.5 (Cardinal's galaxy population is incomplete fainter than this), so the photo-z analysis uses i<25. In the NIR/IRAC the mock is brighter at fixed number density, i.e. its galaxies are redder in i−Ks.""",
"""bins = np.arange(16, 28.01, 0.25); bc = 0.5*(bins[1:]+bins[:-1])
fig, axs = plt.subplots(1, 4, figsize=(18, 4))
for ax, b in zip(axs, ['i', 'g', 'Ks', 'ch1']):
    for df, A, lab in ((data, A_data, 'data'), (sim, A_sim, 'sim')):
        h = np.histogram(mag(df[f'f_{b}']), bins)[0] / A / 0.25
        ax.plot(bc, h, color=COLORS[lab], label=lab)
    ds = data[data.lp_type == 0]
    ax.plot(bc, np.histogram(mag(ds[f'f_{b}']), bins)[0]/A_data/0.25, color=COLORS['data'], ls=':', label='data, lp_type==0 (galaxies)')
    ax.set_yscale('log'); ax.set_xlabel(f'{b} [AB]'); ax.set_ylabel('N / deg$^2$ / mag'); ax.set_title(b)
axs[0].legend(); plt.tight_layout(); plt.show()""",
"""md:## 4. Colour distributions (i < 25)
Contours enclose 50/80/95% of the density. Galaxy colours in the mock come from a library of low-z SDSS galaxies placed on a light cone, so differences here are the main limitation of the mock.""",
"""pairs = [('g_i', 'i_Ks', 'g-i', 'i-Ks', (-0.5, 3.5), (-1, 4.5)), ('u_g', 'g_i', 'u-g', 'g-i', (-0.5, 2.5), (-0.5, 3.5)),
         ('r_z', 'z_J', 'r-z', 'z-J', (-0.5, 2.5), (-0.5, 2.5)), ('i_Ks', 'Ks_ch1', 'i-Ks', 'Ks-ch1', (-1, 4.5), (-1.5, 2.5))]
dsel, ssel = data[(data.i_mag < 25) & (data.lp_type == 0)], sim[sim.i_mag < 25]
fig, axs = plt.subplots(2, 4, figsize=(18, 8))
for k, (x, y, xl, yl, xr, yr) in enumerate(pairs):
    for r, (df, lab) in enumerate(((dsel, 'data'), (ssel, 'sim'))):
        axs[r, k].hist2d(df[x], df[y], bins=100, range=[xr, yr], cmap='Greys', norm=LogNorm(), rasterized=True)
        axs[r, k].set_title(f'{LABELS[lab] if lab=="sim" else "COSMOS2020 (all, galaxies)"}', fontsize=9)
        axs[r, k].set_xlabel(xl); axs[r, k].set_ylabel(yl)
plt.tight_layout(); plt.show()
fig, axs = plt.subplots(1, 4, figsize=(18, 4))
for k, (x, y, xl, yl, xr, yr) in enumerate(pairs):
    for df, lab in ((dsel, 'data'), (ssel, 'sim')):
        H, xe, ye = np.histogram2d(df[x], df[y], bins=60, range=[xr, yr], density=True)
        Hs = np.sort(H.ravel())[::-1]; c = np.cumsum(Hs)/Hs.sum()
        lev = sorted(Hs[np.searchsorted(c, q)] for q in (0.95, 0.8, 0.5))
        axs[k].contour(0.5*(xe[1:]+xe[:-1]), 0.5*(ye[1:]+ye[:-1]), H.T, levels=lev, colors=COLORS[lab])
        axs[k].plot([], [], color=COLORS[lab], label=lab)
    axs[k].set_xlabel(xl); axs[k].set_ylabel(yl)
axs[0].legend(); plt.tight_layout(); plt.show()""",
"""md:### 1-D colour distributions in i-magnitude brackets""",
"""brk = [(18, 22), (22, 23), (23, 24), (24, 25)]
fig, axs = plt.subplots(3, 4, figsize=(18, 9))
for r, (c, rng_) in enumerate([('g_i', (-0.5, 3.5)), ('i_Ks', (-1, 4.5)), ('u_g', (-0.5, 2.5))]):
    for k, (lo, hi) in enumerate(brk):
        for df, lab in ((data[data.lp_type == 0], 'data'), (sim, 'sim')):
            v = df[(df.i_mag >= lo) & (df.i_mag < hi)][c]
            axs[r, k].hist(v, bins=80, range=rng_, histtype='step', density=True, color=COLORS[lab], label=lab)
        axs[r, k].set_title(f'{lo}<=i<{hi}'); axs[r, k].set_xlabel(c.replace('_', '-'))
axs[0, 0].legend(); plt.tight_layout(); plt.show()""",
"""md:## 5. Redshift distributions
Sim: true z. Data: official COSMOS2020 `lp_zBEST` for the photometric sample, and the spec-z sample (which is biased to bright galaxies and particular populations).""",
"""brk = [(18, 23), (23, 24), (24, 25)]
fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for ax, (lo, hi) in zip(axs, brk):
    d = data[(data.i_mag >= lo) & (data.i_mag < hi) & (data.lp_type == 0)]
    s = sim[(sim.i_mag >= lo) & (sim.i_mag < hi)]
    ax.hist(s.true_redshift, bins=80, range=(0, 4), histtype='step', density=True, color=COLORS['sim'], label='sim true z')
    ax.hist(d.lp_zBEST, bins=80, range=(0, 4), histtype='step', density=True, color=COLORS['data'], label='data lp_zBEST')
    ax.hist(d.z_spec.dropna(), bins=80, range=(0, 4), histtype='stepfilled', alpha=0.25, density=True, color='C0', label='data spec-z')
    ax.set_title(f'{lo}<=i<{hi};  data frac with lp_zBEST>2.3: {(d.lp_zBEST > 2.3).mean():.2f}'); ax.set_xlabel('z')
axs[0].legend(); plt.tight_layout(); plt.show()""",
"""md:## Key findings
1. **SEDs.** Cardinal's `sedid` → VAGC kcorrect `COEFFS` reproduces the catalogue LSST colours to ≲0.01 mag (u, g, r, z). The y band offset comes from a different y curve, and the z>1.8 u offset from our IGM treatment.
2. **Noise.** The empirical error model reproduces the reported COSMOS2020 errors in every band. The real IRAC (CLASSIC) errors are ~8× too small for faint sources (negative-flux test), so the mock IRAC fluxes carry an extra ≈0.10 µJy of noise (true 3σ ≈25.2). One known imperfection: in sparsely populated *bright* bins (e.g. u<21) the mock takes sky-limited errors. This is irrelevant for photo-z because LePhare's ERR_SCALE floor (0.02–0.05 mag) dominates there.
3. **Counts.** At i<25 the mock has 141k gal/deg² vs 167k for COSMOS2020 galaxies (−16%). The mock is incomplete beyond i≈25.5.
4. **Colours.** The mock covers the main COSMOS colour loci but with different weights: more red g−i at faint magnitudes, a redder i−Ks, a narrower blue cloud, and sharp, template-like features in Ks−ch1. The mock also has no galaxies above z≈2.3, whereas 17% of COSMOS2020 galaxies at 24<i<25 are at z>2.3.""",
], "01_inputs_photometry.ipynb")

# =====================================================================================================
nb([
"""md:# 02 — Photo-z performance of the COSMOS2020-like mock vs real COSMOS2020

The COSMOS2020 LePhare configuration (`lephare.default_cosmos_config`: 33 COSMOS templates, 4 extinction laws, emission lines, z = 0–7) is run on
* **COSMOS2020 CLASSIC** (clean area, total MW-corrected fluxes): all spec-z objects plus 40k random i<25 objects. Zero-point offsets come from AUTO_ADAPT on spec-z with i<23.
* **Cardinal mock**: 300k random galaxies with i_HSC<25 from the 10 deg² patch, no zero-point offsets.

The same 28 bands are used in both (CLASSIC has no IRAC ch3/ch4). The point estimate is `Z_MED` (PDF median), the analogue of COSMOS2020 `lp_zBEST`.
Metrics use Δz/(1+z): bias = median, σ_NMAD = 1.4826·MAD, η = fraction with |Δz|/(1+z) > 0.15.

**Truth**: sim = true z (all galaxies). Data = multimatch spec-z (instrument type `s`, flag ≥ 3), restricted to official `lp_type==0` (galaxies) and 0.01 < z < 2.2 because the Cardinal light cone ends at z≈2.3.
**The goal is the true performance of the mock itself.** The data curves are a reference; they come from a selected spec-z sample. **"Spec-z weighted" mock**: each mock galaxy is weighted by N_spec/N_mock in its (i, g−i, i−Ks) cell (0.5 × 0.25 × 0.25 mag), so the weighted mock has the spec-z sample's magnitude–colour distribution. **Every data–mock comparison uses this weighted mock**, because it is the apples-to-apples comparison. The unweighted mock is the mock's true performance for an i<25 sample. It is shown alongside as a reference (global table, z–z panels, the (i, z) grids, the 1-D trends, the 3-curve figure in section 3 and PIT); the difference panels are always weighted mock − data.""",
SETUP,
"""D = load_data(); S = load_sim('sim_main')
Ds = spec_sample(D)
S = S[S.ztrue < ZMAX].copy()
S['w_spec'] = spec_weights(S, Ds)
print(f'data: {len(D)} run, {D.ztrue.notna().sum()} with spec-z, {len(Ds)} in truth sample (galaxies, 0.01<z<{ZMAX})')
print(f'sim : {len(S)} galaxies (z<{ZMAX});  {(S.w_spec > 0).mean():.1%} lie in (i,g-i,i-Ks) cells occupied by the spec-z sample')
print('data spec-z with z>2.2 (excluded):', ((D.ztrue >= ZMAX) & (D.lp_type == 0)).sum())""",
"""md:## Global summary""",
"""rows = {
  'data spec-z: LePhare rerun (Z_MED)': df_metrics(Ds),
  'data spec-z: official lp_zBEST': metrics(Ds.lp_zBEST, Ds.ztrue),
  'sim: all i<25': df_metrics(S),
  'sim: spec-z weighted': df_metrics(S, 'w_spec'),
  'sim: i<23': df_metrics(S[S.i_mag < 23]),
  'data spec-z: i<23': df_metrics(Ds[Ds.i_mag < 23]),
  'sim: COSMOS twin (2 deg2) i<25': df_metrics(S[S.twin]),
}
summ = pd.DataFrame(rows).T[['N', 'Neff', 'bias', 'nmad', 'eta', 'sig68', 'cov68']]
summ.round(4)""",
"""md:### Our LePhare rerun vs the official COSMOS2020 photo-z
A pipeline check on all run objects (galaxies), including those without spec-z. Our rerun uses 28 bands (the official run also had Subaru broad bands and GALEX) and our own zero-points, so close but not identical agreement is expected.""",
"""g = D[D.lp_type == 0]
off = np.load(f'{RUNS}/data_run_offsets.npy')
m = metrics(g.zp, g.lp_zBEST)
print(f"rerun Z_MED vs official lp_zBEST, all {len(g)} galaxies: bias {m['bias']:+.4f}  nmad {m['nmad']:.4f}  eta {m['eta']:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
zz_panel(ax[0], g.lp_zBEST, g.zp, title='data: official lp_zBEST (x) vs our rerun (y)', zmax=6)
ax[0].set_xlabel('official lp_zBEST'); ax[0].set_ylabel('rerun Z_MED')
ax[1].bar(range(len(off)), off); ax[1].set_xticks(range(len(off))); ax[1].set_xticklabels(list(BANDS), rotation=90)
ax[1].set_ylabel('zero-point offset [mag]'); ax[1].set_title('AUTO_ADAPT offsets (data, spec-z i<23)')
plt.tight_layout(); plt.show()""",
"""md:## 1. z_true vs z_phot""",
"""fig, axs = plt.subplots(1, 4, figsize=(20, 5))
zz_panel(axs[0], Ds.ztrue, Ds.zp, title='COSMOS2020 spec-z: LePhare rerun')
zz_panel(axs[1], Ds.ztrue, Ds.lp_zBEST, title='COSMOS2020 spec-z: official lp_zBEST')
zz_panel(axs[2], S.ztrue, S.zp, title='Cardinal mock: all galaxies i<25')
zz_panel(axs[3], S.ztrue, S.zp, w=S.w_spec, title='Cardinal mock: spec-z weighted')
plt.tight_layout(); plt.show()""",
"""md:### z_true vs z_phot in i-magnitude brackets
Rows: COSMOS2020 spec-z, spec-z-weighted mock (the comparison), full mock (reference).""",
"""fig, axs = plt.subplots(3, 4, figsize=(20, 13.5))
for k, (lo, hi) in enumerate([(16, 22), (22, 23), (23, 24), (24, 25)]):
    d = Ds[(Ds.i_mag >= lo) & (Ds.i_mag < hi)]; s = S[(S.i_mag >= lo) & (S.i_mag < hi)]
    zz_panel(axs[0, k], d.ztrue, d.zp, title=f'COSMOS2020 spec-z, {lo}<=i<{hi}')
    zz_panel(axs[1, k], s.ztrue, s.zp, w=s.w_spec, title=f'mock, spec-z weighted, {lo}<=i<{hi}')
    zz_panel(axs[2, k], s.ztrue, s.zp, title=f'mock, all, {lo}<=i<{hi}')
plt.tight_layout(); plt.show()""",
"""md:## 2. Bias, scatter and outlier rate on the (i-mag, z_true) grid
Columns: COSMOS2020 spec-z | spec-z-weighted mock | full mock (reference) | weighted mock − data. Cells need N (N_eff for the weighted mock) ≥ 20.""",
"""ib = np.arange(18, 25.01, 0.5); zb = np.arange(0, ZMAX + 0.001, 0.2)
G = {'data': grid2d(Ds, 'i_mag', ib, 'ztrue', zb), 'sim_w': grid2d(S, 'i_mag', ib, 'ztrue', zb, w='w_spec'),
     'sim': grid2d(S, 'i_mag', ib, 'ztrue', zb)}
fig = grid_figure(G, ['data', 'sim_w', 'sim'], ib, zb, 'i_HSC', 'z_true', diff=('sim_w', 'data')); plt.show()""",
"""fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, n in zip(axs, ['data', 'sim']):
    im = ax.imshow(np.log10(G[n]['N']), origin='lower', extent=[ib[0], ib[-1], zb[0], zb[-1]], aspect='auto'); plt.colorbar(im, ax=ax)
    ax.set_title(f'log10 N per cell: {LABELS[n]}'); ax.set_xlabel('i_HSC'); ax.set_ylabel('z_true')
plt.tight_layout(); plt.show()""",
"""md:### Same grid in (i-mag, z_phot)
Binning in photo-z, the quantity available in practice, and measuring the scatter of z_true around it.""",
"""Gp = {'data': grid2d(Ds, 'i_mag', ib, 'zp', zb), 'sim_w': grid2d(S, 'i_mag', ib, 'zp', zb, w='w_spec'),
      'sim': grid2d(S, 'i_mag', ib, 'zp', zb)}
fig = grid_figure(Gp, ['data', 'sim_w', 'sim'], ib, zb, 'i_HSC', 'z_phot (Z_MED)', diff=('sim_w', 'data')); plt.show()""",
"""md:### 1-D: metrics vs i-mag and vs z_true""",
"""fig, axs = plt.subplots(2, 4, figsize=(20, 8))
samples = {'data': (Ds, None), 'sim': (S, None), 'sim_w': (S, 'w_spec')}
for r, (col, bins, lab) in enumerate([('i_mag', np.arange(17, 25.01, 0.5), 'i_HSC'), ('ztrue', np.arange(0, ZMAX + 0.01, 0.1), 'z_true')]):
    for name, (df, w) in samples.items():
        t = binned(df, col, bins, w)
        for k, m in enumerate(['bias', 'nmad', 'eta', 'cov68']):
            axs[r, k].errorbar(t.x, t[m], yerr=t.get(f'{m}_err'), fmt='o-', ms=3, color=COLORS[name], label=LABELS[name])
            axs[r, k].set_xlabel(lab); axs[r, k].set_ylabel(MLAB[m]); axs[r, k].set_ylim(*MLIM[m])
    axs[r, 0].axhline(0, c='k', lw=0.5); axs[r, 3].axhline(0.68, c='k', lw=0.5)
axs[0, 0].legend(fontsize=7); plt.tight_layout(); plt.show()""",
"""md:## 3. Bias, scatter and outlier rate vs colour, in i-magnitude brackets
Error bars are approximate statistical errors (bias: 1.25σ/√N, σ_NMAD: σ/√2N, η: binomial; N → N_eff for the weighted mock). Points need N_eff ≥ 15.

### All three samples (g−i)
COSMOS2020 spec-z vs the full mock vs the spec-z-weighted mock. The difference between the two mock curves shows how much the spec-z selection (within a colour bin, mostly through the other colour and the magnitude distribution inside the bracket) changes the mock's metrics.""",
"""brackets = [(16, 22), (22, 23), (23, 24), (24, 25)]
samples_3 = {'data': (Ds, None), 'sim': (S, None), 'sim_w': (S, 'w_spec')}
fig = curves_figure(samples_3, 'g_i', np.arange(-0.25, 3.01, 0.25), brackets, 'i_mag', 'i', 'g - i',
                    suptitle='metrics vs g - i in i-mag brackets: data, mock (all), mock (spec-z weighted)'); plt.show()""",
"""md:### COSMOS2020 spec-z vs spec-z-weighted mock, four colours""",
"""samples_w = {'data': (Ds, None), 'sim_w': (S, 'w_spec')}
for col, cb, lab in [('g_i', np.arange(-0.25, 3.01, 0.25), 'g - i'), ('i_Ks', np.arange(-0.75, 4.01, 0.25), 'i - Ks'),
                     ('u_g', np.arange(-0.25, 2.51, 0.25), 'u - g'), ('z_J', np.arange(-0.25, 2.51, 0.2), 'z - J')]:
    fig = curves_figure(samples_w, col, cb, brackets, 'i_mag', 'i', lab,
                        suptitle=f'metrics vs {lab} in i-mag brackets: data vs spec-z-weighted mock'); plt.show()""",
"""md:## 4. Metrics vs colour at fixed magnitude, in z_true brackets
For each i-magnitude bracket, metrics vs colour in four true-redshift brackets: COSMOS2020 spec-z vs the spec-z-weighted mock. Bracketing on z_true for the mock and spec-z for the data shows how much of the colour dependence is redshift dependence.""",
"""zbr = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, ZMAX)]
for (lo, hi) in [(16, 23), (23, 24), (24, 25)]:
    for col, cb, lab in [('g_i', np.arange(-0.25, 3.01, 0.25), 'g - i'), ('i_Ks', np.arange(-0.75, 4.01, 0.25), 'i - Ks')]:
        sel = lambda g, lo=lo, hi=hi: (g.i_mag >= lo) & (g.i_mag < hi)
        fig = curves_figure(samples_w, col, cb, zbr, 'ztrue', 'z_true', lab, extra_sel=sel,
                            suptitle=f'{lo} <= i < {hi}: metrics vs {lab} in z_true brackets'); plt.show()""",
"""md:### Same, bracketing on z_phot (what an analysis can actually select on)""",
"""for (lo, hi) in [(16, 23), (23, 24), (24, 25)]:
    sel = lambda g, lo=lo, hi=hi: (g.i_mag >= lo) & (g.i_mag < hi)
    fig = curves_figure(samples_w, 'g_i', np.arange(-0.25, 3.01, 0.25), zbr, 'zp', 'z_phot', 'g - i',
                        extra_sel=sel, suptitle=f'{lo} <= i < {hi}: metrics vs g-i in z_phot brackets'); plt.show()""",
"""md:## 5. 2-D colour–colour maps (g−i vs i−Ks) of the metrics, per i bracket
COSMOS2020 spec-z vs the spec-z-weighted mock; the last column is weighted mock − data. Cells need N_eff ≥ 15.""",
"""gb = np.arange(-0.5, 3.51, 0.25); kb = np.arange(-1.0, 4.51, 0.25)
for (lo, hi) in [(16, 23), (23, 24), (24, 25)]:
    d = Ds[(Ds.i_mag >= lo) & (Ds.i_mag < hi)]; s = S[(S.i_mag >= lo) & (S.i_mag < hi)]
    Gc = {'data': grid2d(d, 'g_i', gb, 'i_Ks', kb, nmin=15), 'sim_w': grid2d(s, 'g_i', gb, 'i_Ks', kb, w='w_spec', nmin=15)}
    fig = grid_figure(Gc, ['data', 'sim_w'], gb, kb, 'g - i', 'i - Ks', diff=('sim_w', 'data')); fig.suptitle(f'{lo} <= i < {hi}', y=1.01); plt.show()""",
"""md:## 6. PDF calibration: PIT and 68% interval width""",
"""fig, axs = plt.subplots(1, 3, figsize=(17, 4))
for name, (df, w) in {'data': (Ds, None), 'sim': (S, None), 'sim_w': (S, 'w_spec')}.items():
    axs[0].hist(df.PIT, bins=50, range=(0, 1), histtype='step', density=True, weights=None if w is None else df[w], color=COLORS[name], label=LABELS[name])
    t = binned(df, 'i_mag', np.arange(17, 25.01, 0.5), w)
    axs[1].plot(t.x, t.sig68, 'o-', color=COLORS[name]); axs[2].plot(t.x, t.cov68, 'o-', color=COLORS[name])
axs[0].axhline(1, c='k', lw=0.5); axs[0].set_xlabel('PIT'); axs[0].legend(fontsize=7)
axs[1].set_xlabel('i'); axs[1].set_ylabel('median (q84-q16)/2/(1+z)'); axs[2].set_xlabel('i'); axs[2].set_ylabel('68% coverage'); axs[2].axhline(0.68, c='k', lw=0.5)
plt.tight_layout(); plt.show()""",
"""md:## Key findings (run of 2026-11-10; 300k mock galaxies vs 33k COSMOS2020 spec-z galaxies, z<2.2, i<25)
All data–mock statements use the **spec-z-weighted mock** (apples-to-apples); "full mock" is its unweighted, true i<25 performance.

1. **Pipeline check.** Our LePhare rerun reproduces the official COSMOS2020 photo-z: σ_NMAD 0.0119 vs 0.0107 and η 3.7% vs 3.6% against spec-z. Rerun vs official on all galaxies: σ_NMAD 0.007.
2. **Scatter vs magnitude matches to ~10%.** σ_NMAD in i = <22 / 22–23 / 23–24 / 24–25: weighted mock 0.009 / 0.012 / 0.019 / 0.033 vs data 0.009 / 0.011 / 0.018 / 0.031.
3. **Outlier rates vs magnitude are comparable once the selection is matched.** η is 0.3 / 2.8 / 9.2 / 11.5% for the weighted mock vs 0.6 / 2.1 / 6.4 / 13.9% for the data. The full mock has much lower η (0.2 / 0.5 / 1.1 / 4.1%) because the spec-z sample over-represents blue galaxies (14% have g−i<0.5, vs 6% of the full mock), which are outlier-prone in both. So the low η of the full mock reflects its galaxy population, not better photometry.
4. **But the redshift dependence differs.** The weighted mock matches the data at z<0.5 (σ_NMAD 0.009 vs 0.009, η 3.7% vs 3.8%). It is worse at 0.5<z<1 (σ_NMAD 0.016 vs 0.010, η 5.1% vs 2.3%) and has fewer outliers at z>1 (η 1.8% vs 4.8% at 1–1.5, 3.2% vs 8.1% at 1.5–2.2). In the (i, z) grid the mock's excess outliers sit at 23<i<25, 0.5<z<1.
5. **Bias.** The data are biased low (median −0.002 to −0.005, and −0.011 at 1<z<1.5). The weighted mock is ≈0 overall but biased *high* at 1<z<1.5 (+0.003) and at faint 0.75<z<1.5, so in that regime the sign differs.
6. **Colour dependence.** For g−i ≳ 0.75 the weighted mock tracks the data's bias and σ_NMAD in every magnitude bracket. Its failures are (a) very blue galaxies (g−i<0.5 at i>22: bias up to +0.03, η 15–40%, σ_NMAD up to 0.08, all far above the data) and (b) a bump in bias and η at i−Ks≈1 for i>22 that the data do not show. Conversely, the data's scatter grows towards red i−Ks (>2) at faint magnitudes and the mock's does not.
7. **PDFs are over-confident in both** (68% coverage ≈0.35–0.45). Part of this is the 0.01 redshift grid: bright objects' PDFs are narrower than one grid step, which produces the PIT spikes at 0 and 1.""",
], "02_photoz_performance.ipynb")

# =====================================================================================================
nb([
"""md:# 03 — Diagnostics: where does the mock's photo-z error come from?

* **Noiseless vs noisy mock** (same 40k galaxies): the noiseless run isolates the *template-mismatch floor*. This is the error from fitting COSMOS templates to kcorrect SEDs with perfect photometry.
* **Zero-point offsets**: AUTO_ADAPT on the mock measures the systematic mismatch between the mock SED library and the COSMOS templates, band by band. Compared with the data offsets, which also absorb real calibration errors. The `sim_adapt` run applies the mock-derived offsets.
* **Fit quality**: χ²/N_band and best-fit template / E(B−V) distributions, data vs mock.
* **SOM**: a 20×20 SOM trained on the data's broad-band colours (u…Ks). Mock galaxies are mapped onto it to compare cell occupation, redshift per cell, and per-cell photo-z metrics.""",
SETUP,
"""D = load_data(); Ds = spec_sample(D)
S = load_sim('sim_main'); S = S[S.ztrue < ZMAX].copy()
S['w_spec'] = spec_weights(S, Ds)          # weights making the mock match the spec-z sample in (i, g-i, i-Ks)
N0 = load_sim('sim_noiseless'); A = load_sim('sim_adapt')
N0 = N0[N0.ztrue < ZMAX]; A = A[A.ztrue < ZMAX]
Sd = S[S.galaxy_id.isin(A.galaxy_id)]
print(len(Ds), len(S), len(N0), len(A), len(Sd))""",
"""md:## 1. Noiseless / noisy / noisy + mock zero-points (same galaxies)
This is a mock-internal comparison on the 40k diagnostic subset. It stays **unweighted**: spec-z weighting would leave only ~3k effective galaxies. The data spec-z curve is shown for orientation only.""",
"""rows = {'noiseless (template-mismatch floor)': df_metrics(N0), 'noisy, no offsets (= sim_main)': df_metrics(Sd), 'noisy + mock AUTO_ADAPT offsets': df_metrics(A)}
display(pd.DataFrame(rows).T[['N', 'bias', 'nmad', 'eta', 'sig68', 'cov68']].round(4))
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
zz_panel(axs[0], N0.ztrue, N0.zp, title='mock noiseless'); zz_panel(axs[1], Sd.ztrue, Sd.zp, title='mock noisy'); zz_panel(axs[2], A.ztrue, A.zp, title='mock noisy + mock offsets')
plt.tight_layout(); plt.show()""",
"""fig, axs = plt.subplots(2, 3, figsize=(17, 8))
for r, (col, bins, lab) in enumerate([('i_mag', np.arange(17, 25.01, 0.5), 'i'), ('ztrue', np.arange(0, ZMAX + 0.01, 0.1), 'z_true')]):
    for name, df, c in [('noiseless', N0, 'k'), ('noisy', Sd, 'C1'), ('noisy+offsets', A, 'C3'), ('data spec-z', Ds, 'C0')]:
        t = binned(df, col, bins)
        for k, m in enumerate(['bias', 'nmad', 'eta']):
            axs[r, k].errorbar(t.x, t[m], yerr=t[f'{m}_err'], fmt='o-', ms=3, color=c, label=name)
            axs[r, k].set_ylim(*MLIM[m]); axs[r, k].set_xlabel(lab); axs[r, k].set_ylabel(MLAB[m])
axs[0, 0].legend(fontsize=7); plt.tight_layout(); plt.show()""",
"""md:### Template-mismatch floor vs colour and redshift (noiseless mock)""",
"""fig = curves_figure({'sim': (N0, None)}, 'g_i', np.arange(-0.25, 3.01, 0.25), [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, ZMAX)], 'ztrue', 'z_true', 'g - i',
                    suptitle='noiseless mock: metrics vs g-i in z_true brackets'); plt.show()""",
"""md:## 2. Zero-point offsets: data vs mock""",
"""od = np.load(f'{RUNS}/data_run_offsets.npy'); os_ = np.load(f'{RUNS}/sim_adapt_offsets.npy')
x = np.arange(len(od))
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(x - 0.2, od, 0.4, label='data (spec-z, i<23)'); ax.bar(x + 0.2, os_, 0.4, label='mock (true z, i<23)')
ax.set_xticks(x); ax.set_xticklabels(list(BANDS), rotation=90); ax.axhline(0, c='k', lw=0.5); ax.set_ylabel('offset [mag]'); ax.legend()
plt.show()
pd.DataFrame({'data': od, 'mock': os_}, index=list(BANDS)).T.round(3)""",
"""md:## 3. Fit quality and best-fit templates
Data spec-z vs the **spec-z-weighted** mock. The noiseless mock (40k subset) is shown unweighted for reference.""",
"""fig, axs = plt.subplots(1, 4, figsize=(20, 4))
for name, df, w, c in [('data spec-z', Ds, None, 'C0'), ('mock (spec-z weighted)', S, S.w_spec, 'C2'), ('mock noiseless (all)', N0, None, 'k')]:
    kw = dict(histtype='step', density=True, color=c, weights=w)
    axs[0].hist(np.log10((df.CHI_BEST / df.NBAND_USED).clip(1e-3)), bins=80, range=(-2.5, 2.5), label=name, **kw)
    axs[1].hist(df.MOD_BEST, bins=np.arange(0.5, 34.5), **kw)
    axs[2].hist(df.EBV_BEST, bins=np.arange(-0.025, 0.55, 0.05), **kw)
    axs[3].hist(df.EXTLAW_BEST, bins=np.arange(-0.5, 4.5), **kw)
axs[0].set_xlabel('log10 chi2_best / N_band'); axs[0].legend(); axs[1].set_xlabel('best template (1-33: Ell..SB)'); axs[2].set_xlabel('E(B-V)'); axs[3].set_xlabel('extinction law index')
plt.tight_layout(); plt.show()
import analysis_utils as au
ibins = np.arange(17, 25.01, 0.5)
for name, df, w, c in [('data spec-z', Ds, np.ones(len(Ds)), 'C0'), ('mock (spec-z weighted)', S, S.w_spec.values, 'C2')]:
    x = (df.CHI_BEST / df.NBAND_USED).values; b = np.digitize(df.i_mag, ibins) - 1
    med = [au._wmed(x[(b == k) & (w > 0)], w[(b == k) & (w > 0)]) if ((b == k) & (w > 0)).sum() > 10 else np.nan for k in range(len(ibins) - 1)]
    plt.plot(0.5 * (ibins[1:] + ibins[:-1]), med, 'o-', color=c, label=name)
plt.yscale('log'); plt.xlabel('i'); plt.ylabel('median chi2/N_band'); plt.legend(); plt.show()""",
"""md:## 4. SOM comparison
SOM trained on the data broad-band colours (all clean i<25 objects in the data run); mock galaxies mapped onto the same cells. Per-cell photo-z metrics and median redshifts compare data spec-z with the **spec-z-weighted** mock (N_eff ≥ 10 per cell). The occupation map compares the full photometric samples, where spec-z selection does not apply.""",
"""from minisom import MiniSom
import analysis_utils as au
SB = ['u', 'g', 'r', 'i', 'z', 'y', 'Y', 'J', 'H', 'Ks']
def cols(df): return np.stack([df[f'm_{a}'] - df[f'm_{b}'] for a, b in zip(SB[:-1], SB[1:])], axis=1)
Dtr = D[(D.i_mag < 25)]
Xd = cols(Dtr); ok = np.isfinite(Xd).all(1); Xd = Xd[ok]; mu, sd = Xd.mean(0), Xd.std(0)
som = MiniSom(20, 20, Xd.shape[1], sigma=2.0, learning_rate=0.5, random_seed=3)
som.pca_weights_init((Xd - mu) / sd); som.train_random((Xd - mu) / sd, 200000)
W = som.get_weights().reshape(-1, Xd.shape[1])
def cell(df):
    X = (cols(df) - mu) / sd; X = np.nan_to_num(X)
    return np.concatenate([((X[i:i+50000, None, :] - W[None]) ** 2).sum(-1).argmin(1) for i in range(0, len(X), 50000)])
Ds['som'] = cell(Ds); S['som'] = cell(S); Dtr = Dtr.assign(som=cell(Dtr))
def per_cell(df, w=None):
    out = {}
    for c, g in df.groupby('som'):
        out[c] = df_metrics(g, w)
    return pd.DataFrame(out).T.reindex(range(400))
cd, cs = per_cell(Ds), per_cell(S, 'w_spec')
occ_d = Dtr.som.value_counts(normalize=True).reindex(range(400)).fillna(0); occ_s = S.som.value_counts(normalize=True).reindex(range(400)).fillna(0)
fig, axs = plt.subplots(3, 4, figsize=(20, 14))
for r, m in enumerate(['bias', 'nmad', 'eta']):
    lo, hi = MLIM[m]
    for c, (t, name) in enumerate([(cd, 'data spec-z'), (cs, 'mock (spec-z weighted)')]):
        v = t[m].values.astype(float).reshape(20, 20)
        im = axs[r, c].imshow(v, vmin=lo, vmax=hi, cmap='RdBu_r' if m == 'bias' else 'viridis'); plt.colorbar(im, ax=axs[r, c]); axs[r, c].set_title(f'{m}: {name}')
    ok = cd[m].notna() & cs[m].notna()
    axs[r, 2].scatter(cd[m][ok], cs[m][ok], s=5 + 3000 * occ_d[ok]); axs[r, 2].plot([lo, hi], [lo, hi], 'k--'); axs[r, 2].set_xlim(lo, hi); axs[r, 2].set_ylim(lo, hi)
    axs[r, 2].set_xlabel(f'{m} data'); axs[r, 2].set_ylabel(f'{m} mock (spec-z weighted)'); axs[r, 2].set_title('per cell (size ~ data occupation)')
im = axs[0, 3].imshow(np.log10((occ_s + 1e-5) / (occ_d + 1e-5)).values.reshape(20, 20), cmap='RdBu_r', vmin=-1, vmax=1); plt.colorbar(im, ax=axs[0, 3]); axs[0, 3].set_title('log10 occupation mock/data')
zd = Ds.groupby('som').ztrue.median().reindex(range(400)).values.reshape(20, 20); zs = pd.Series({c: au._wmed(g.ztrue.values, g.w_spec.values) for c, g in S[S.w_spec > 0].groupby('som')}).reindex(range(400)).values.reshape(20, 20)
im = axs[1, 3].imshow(zd, cmap='magma', vmin=0, vmax=2.2); plt.colorbar(im, ax=axs[1, 3]); axs[1, 3].set_title('median spec-z per cell (data)')
im = axs[2, 3].imshow(zs - zd, cmap='RdBu_r', vmin=-0.5, vmax=0.5); plt.colorbar(im, ax=axs[2, 3]); axs[2, 3].set_title('median z_true(weighted mock) - z_spec(data)')
plt.tight_layout(); plt.show()
print('fraction of mock galaxies in cells with <0.01% data occupation:', occ_s[occ_d < 1e-4].sum().round(3))""",
"""md:## Key findings (run of 2026-11-10)
1. **The mock's photo-z scatter is set by template mismatch, not by photometric noise.** On the same 40k galaxies, the noiseless run has σ_NMAD 0.019 (η 1.2%) vs 0.022 (η 2.3%) with noise. Up to i≈23 the noiseless and noisy curves coincide. Noise matters only at i>23: σ_NMAD 0.027→0.034 and η 2%→5% at 24<i<25.
2. **At fixed true redshift the mock is worse than the data** at 0.4<z<1.0. σ_NMAD is ≈0.02 for the full mock and 0.016 spec-z weighted, vs ≈0.010 for the data. The COSMOS templates fit the mock's kcorrect SEDs less well than they fit real galaxies, even though χ²/N_band is lower in the mock. The per-SOM-cell comparison shows the same thing: mock σ_NMAD sits above the data wherever the data are very precise.
3. **Zero-point offsets.** Mock AUTO_ADAPT offsets are ≲0.02 mag in all optical/NIR bands, apart from IRAC ch1 (−0.07) and ch2 (−0.03); applying them changes nothing. The data offsets are larger (up to −0.16 in IA679, −0.07 in r) because they also absorb real calibration and template errors.
4. **Best-fit templates.** Mock galaxies pile onto a few COSMOS templates (7, 22, 25) and mostly the default extinction law. Real galaxies spread over more templates (especially the bluer 27–31) and use the bump extinction laws much more. This is the limited SED diversity of the SDSS-based library.
5. **SOM, per cell (data vs spec-z-weighted mock).** Where both are populated, σ_NMAD scatters around the 1:1 line with the mock somewhat above it in the most precise cells. Per-cell η is mostly lower in the mock, with a few cells (blue edge of the map) far higher. The mock leaves parts of the data colour space under-populated (occupation ratio <0.1), and in those cells its median z differs strongly from the data.""",
], "03_diagnostics.ipynb")
print("notebooks written")
