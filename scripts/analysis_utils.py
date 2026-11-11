"""Shared loading / metric / plotting helpers for the cardinal_cosmos2020 notebooks."""
import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PROJ = os.environ.setdefault("PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{PROJ}/scripts")
from bands import BANDS  # noqa: E402

RUNS = f"{PROJ}/work/lephare_runs"
INPUTS = f"{PROJ}/work/run_inputs"
ZPHOT = "Z_MED"          # LePhare PDF median, the analogue of COSMOS2020 lp_zBEST (= lp_zPDF for galaxies)
ZMAX = 2.2               # Cardinal light cone ends at z~2.3
COLORS = {"data": "C0", "sim": "C1", "sim_w": "C2"}
LABELS = {"data": "COSMOS2020 (spec-z)", "sim": "Cardinal mock (all)", "sim_w": "Cardinal mock (spec-z weighted)"}


def mag(f, e=None):
    """AB mag from uJy; if errors given, floor flux at 1 sigma (finite colours for faint/negative fluxes)."""
    f = np.asarray(f, float)
    if e is not None:
        f = np.fmax(f, np.asarray(e, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, 23.9 - 2.5 * np.log10(f), np.nan)


def add_mags(df, bands=("u", "g", "r", "i", "z", "y", "Y", "J", "H", "Ks", "ch1", "ch2")):
    for b in bands:
        df[f"m_{b}"] = mag(df[f"f_{b}"], df[f"ferr_{b}"])
    df["g_i"] = df.m_g - df.m_i
    df["i_Ks"] = df.m_i - df.m_Ks
    df["u_g"] = df.m_u - df.m_g
    df["r_z"] = df.m_r - df.m_z
    df["z_J"] = df.m_z - df.m_J
    df["Ks_ch1"] = df.m_Ks - df.m_ch1
    return df


def load_run(tag, idcol):
    """LePhare output joined with the photometry of its input table."""
    out = pd.read_parquet(f"{RUNS}/{tag}.parquet")
    inp_name = {"data_run": "data_run", "sim_main": "sim_main", "sim_adapt": "sim_diag",
                "sim_noiseless": "sim_diag_noiseless"}[tag]
    inp = pd.read_parquet(f"{INPUTS}/{inp_name}.parquet")
    fcols = [c for c in inp.columns if c.startswith(("f_", "ferr_"))]
    out = out.merge(inp[[idcol] + fcols], on=idcol, how="left", validate="one_to_one")
    out = add_mags(out)
    out["zp"] = out[ZPHOT]
    out["sig68"] = (out.zpdf_q840 - out.zpdf_q160) / 2
    return out


def load_data():
    d = load_run("data_run", "ID")
    d["ztrue"] = d.z_spec
    return d


def load_sim(tag="sim_main"):
    s = load_run(tag, "galaxy_id")
    s["ztrue"] = s.true_redshift
    return s


def spec_sample(d, zmax=ZMAX, galaxies_only=True):
    """COSMOS2020 objects with spec-z as truth: 0.01<z<zmax, official LePhare type galaxy (lp_type==0)."""
    m = d.ztrue.notna() & (d.ztrue > 0.01) & (d.ztrue < zmax)
    if galaxies_only:
        m &= d.lp_type == 0
    return d[m].copy()


def spec_weights(sim, spec, cols=("i_mag", "g_i", "i_Ks"),
                 edges=(np.arange(16, 25.01, 0.5), np.arange(-1, 4.01, 0.25), np.arange(-2, 5.01, 0.25))):
    """Weights making the sim match the spec-z sample's distribution in (i, g-i, i-Ks)."""
    X = lambda df: np.c_[[np.clip(df[c].values, e[0], e[-1] - 1e-6) for c, e in zip(cols, edges)]].T
    hd = np.histogramdd(X(spec), bins=edges)[0]
    hs = np.histogramdd(X(sim), bins=edges)[0]
    r = np.where(hs > 0, hd / np.where(hs > 0, hs, 1), 0.0)
    ix = [np.clip(np.digitize(sim[c].values, e) - 1, 0, len(e) - 2) for c, e in zip(cols, edges)]
    w = r[tuple(ix)]
    return w * len(sim) / max(w.sum(), 1)


# ------------------------------------------------------------------ metrics
def _wmed(x, w):
    o = np.argsort(x); c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, 0.5 * c[-1])]


def metrics(zp, zt, w=None, sig68=None, pit=None, nmin=10):
    zp, zt = np.asarray(zp, float), np.asarray(zt, float)
    w = np.ones_like(zt) if w is None else np.asarray(w, float)
    ok = np.isfinite(zp) & np.isfinite(zt) & (zp >= 0) & (w > 0)
    zp, zt, w = zp[ok], zt[ok], w[ok]
    nan = dict(N=len(zt), Neff=np.nan, bias=np.nan, nmad=np.nan, eta=np.nan, bias_err=np.nan, nmad_err=np.nan,
               eta_err=np.nan, sig68=np.nan, cov68=np.nan)
    if len(zt) < nmin:
        return nan
    dz = (zp - zt) / (1 + zt)
    neff = w.sum() ** 2 / (w ** 2).sum()
    b = _wmed(dz, w); s = 1.4826 * _wmed(np.abs(dz - b), w); e = np.average(np.abs(dz) > 0.15, weights=w)
    r = dict(N=len(zt), Neff=neff, bias=b, nmad=s, eta=e, bias_err=1.2533 * s / np.sqrt(neff),
             nmad_err=s / np.sqrt(2 * neff), eta_err=np.sqrt(max(e * (1 - e), 1 / neff) / neff),
             sig68=np.nan, cov68=np.nan)
    if sig68 is not None:
        r["sig68"] = _wmed(np.asarray(sig68, float)[ok] / (1 + zt), w)
    if pit is not None:
        p = np.asarray(pit, float)[ok]
        r["cov68"] = np.average((p > 0.16) & (p < 0.84), weights=w)
    return r


def df_metrics(df, w=None, **kw):
    return metrics(df.zp, df.ztrue, None if w is None else df[w], df.sig68, df.PIT, **kw)


def binned(df, col, bins, w=None, nmin=10):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = df[(df[col] >= lo) & (df[col] < hi)]
        rows.append(dict(lo=lo, hi=hi, x=0.5 * (lo + hi), **df_metrics(g, w, nmin=nmin)))
    return pd.DataFrame(rows)


def grid2d(df, xcol, xbins, ycol, ybins, w=None, nmin=20):
    res = {k: np.full((len(ybins) - 1, len(xbins) - 1), np.nan) for k in ["bias", "nmad", "eta", "N", "cov68"]}
    ix = np.digitize(df[xcol], xbins) - 1; iy = np.digitize(df[ycol], ybins) - 1
    for (i, j), g in df.groupby([ix, iy]):
        if 0 <= i < len(xbins) - 1 and 0 <= j < len(ybins) - 1:
            m = df_metrics(g, w, nmin=nmin)
            for k in res:
                res[k][j, i] = m[k]
    return res


# ------------------------------------------------------------------ plots
MLIM = {"bias": (-0.04, 0.04), "nmad": (0, 0.08), "eta": (0, 0.3), "cov68": (0.2, 0.9)}
MLAB = {"bias": r"bias = median $\Delta z/(1+z)$", "nmad": r"$\sigma_{\rm NMAD}$",
        "eta": r"outlier rate $\eta$ ($|\Delta z|/(1+z)>0.15$)", "cov68": "68% PDF coverage"}


def zz_panel(ax, zt, zp, w=None, title="", zmax=ZMAX, stats=True):
    ax.hist2d(zt, zp, bins=[np.linspace(0, zmax, 111), np.linspace(0, zmax + 1, 161)], weights=w,
              cmap="Greys", norm=LogNorm(), rasterized=True)
    x = np.linspace(0, zmax, 10)
    ax.plot(x, x, "r-", lw=0.6)
    for s in (1, -1):
        ax.plot(x, x + s * 0.15 * (1 + x), "r--", lw=0.5)
    ax.set_xlabel(r"$z_{\rm true}$"); ax.set_ylabel(r"$z_{\rm phot}$"); ax.set_title(title, fontsize=10)
    if stats:
        m = metrics(zp, zt, w)
        ax.text(0.03, 0.97, f"N={m['N']}\nbias={m['bias']:+.4f}\n$\\sigma_{{NMAD}}$={m['nmad']:.4f}\n$\\eta$={m['eta']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9, bbox=dict(fc="w", alpha=0.8))


def grid_figure(grids, names, xbins, ybins, xlabel, ylabel, metrics_=("bias", "nmad", "eta"), diff=("sim", "data"),
                annot=True):
    """grids: dict name -> grid2d result. One row per metric; one column per sample (+ difference)."""
    ncol = len(names) + (1 if diff else 0)
    nx, ny = len(xbins) - 1, len(ybins) - 1
    pw, ph = max(5.0, 0.42 * nx + 1.2), max(4.0, 0.36 * ny + 0.8)          # ~0.42" per cell: room for 3 chars
    fig, axs = plt.subplots(len(metrics_), ncol, figsize=(pw * ncol, ph * len(metrics_)), squeeze=False)
    ext = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
    scale = {"bias": 1e3, "nmad": 1e3, "eta": 1e2, "cov68": 1e2}            # annotate as compact integers
    unit = {"bias": r"[$\times10^{-3}$]", "nmad": r"[$\times10^{-3}$]", "eta": "[%]", "cov68": "[%]"}

    def cbfmt(m):
        from matplotlib.ticker import FuncFormatter
        return FuncFormatter(lambda v, _: f"{v * scale[m]:g}") if annot else None

    def annotate(ax, img, cmap, norm, m):
        if not annot:
            return
        for (j, i), v in np.ndenumerate(img):
            if np.isfinite(v):
                rgb = cmap(norm(v))[:3]
                txt_c = "k" if (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) > 0.5 else "w"
                ax.text(0.5 * (xbins[i] + xbins[i + 1]), 0.5 * (ybins[j] + ybins[j + 1]), f"{v * scale[m]:.0f}",
                        ha="center", va="center", fontsize=6.5, color=txt_c)

    for r, m in enumerate(metrics_):
        lo, hi = MLIM[m]
        for c, n in enumerate(names):
            ax = axs[r, c]
            cmap = plt.get_cmap("RdBu_r" if m == "bias" else "viridis")
            im = ax.imshow(grids[n][m], origin="lower", extent=ext, aspect="auto", vmin=lo, vmax=hi, cmap=cmap)
            plt.colorbar(im, ax=ax, format=cbfmt(m))
            ax.set_title(f"{LABELS.get(n, n)}: {m} {unit[m] if annot else ''}", fontsize=10)
            annotate(ax, grids[n][m], cmap, im.norm, m)
        if diff:
            a, b = diff
            dlim = max(abs(lo), abs(hi)) / (1 if m == "bias" else 2)
            d = grids[a][m] - grids[b][m]
            cmap = plt.get_cmap("RdBu_r")
            im = axs[r, -1].imshow(d, origin="lower", extent=ext, aspect="auto", vmin=-dlim, vmax=dlim, cmap=cmap)
            plt.colorbar(im, ax=axs[r, -1], format=cbfmt(m))
            axs[r, -1].set_title(f"{m}: {a} - {b} {unit[m] if annot else ''}", fontsize=10)
            annotate(axs[r, -1], d, cmap, im.norm, m)
    for ax in axs.flat:
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def curves_figure(samples, col, cbins, brackets, bracket_col, bracket_label, color_label,
                  metrics_=("bias", "nmad", "eta"), nmin=15, extra_sel=None, suptitle=None):
    """samples: dict name -> (df, weight col or None). Columns = brackets in bracket_col; rows = metrics."""
    fig, axs = plt.subplots(len(metrics_), len(brackets), figsize=(4.2 * len(brackets), 3.1 * len(metrics_)),
                            sharex=True, squeeze=False)
    for c, (lo, hi) in enumerate(brackets):
        for name, (df, w) in samples.items():
            g = df[(df[bracket_col] >= lo) & (df[bracket_col] < hi)]
            if extra_sel is not None:
                g = g[extra_sel(g)]
            r = binned(g, col, cbins, w, nmin=nmin)
            for k, m in enumerate(metrics_):
                axs[k, c].errorbar(r.x, r[m], yerr=r[f"{m}_err"], fmt="o-", ms=3, lw=1, capsize=0,
                                   color=COLORS.get(name), label=f"{LABELS.get(name, name)}")
        for k, m in enumerate(metrics_):
            axs[k, c].set_ylim(*MLIM[m])
            if m == "bias":
                axs[k, c].axhline(0, c="k", lw=0.5)
            if c == 0:
                axs[k, c].set_ylabel(MLAB[m], fontsize=9)
        axs[0, c].set_title(f"{lo} <= {bracket_label} < {hi}", fontsize=10)
        axs[-1, c].set_xlabel(color_label)
    axs[0, 0].legend(fontsize=7)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig
