# A COSMOS2020-like 30-band mock from Cardinal, with LePhare photo-z

This repository builds a COSMOS2020-like many-band photometric mock from the
[Cardinal](https://arxiv.org/abs/2105.13547) simulation. It runs the COSMOS2020 LePhare photo-z configuration on both
the mock and the real COSMOS2020 CLASSIC catalogue, and measures the mock's photo-z performance (bias, σ_NMAD and
outlier rate vs magnitude, redshift and colour), with real spectroscopic redshifts as the data reference.

> **Data release:** I will release this COSMOS2020-like mock as part of the Cardinal data products soon.
> Until then, the code here reproduces it from the public/NERSC inputs listed below.

## Results
The executed notebooks (with all figures) are in `notebooks/`. Each ends with a "Key findings" cell.

| notebook | content |
|---|---|
| `01_inputs_photometry.ipynb` | SED-reconstruction check, noise model, number counts, colour–colour and n(z) comparisons, mock vs COSMOS2020 |
| `02_photoz_performance.ipynb` | z_true vs z_phot; bias / σ_NMAD / η on the (i, z) grid; vs colour in i-mag brackets; vs colour in z brackets at fixed i; colour–colour maps; PIT |
| `03_diagnostics.ipynb` | noiseless vs noisy mock (template-mismatch floor), zero-point offsets, χ² and best-fit templates, SOM cell comparison |

Headline numbers (i<25, z<2.2; data = 33k COSMOS2020 galaxies with spec-z; mock = 300k galaxies vs true z):

| i range | σ_NMAD mock / data | outlier rate η mock / data |
|---|---|---|
| < 22 | 0.010 / 0.009 | 0.2% / 0.6% |
| 22–23 | 0.015 / 0.011 | 0.5% / 2.1% |
| 23–24 | 0.020 / 0.018 | 1.1% / 6.4% |
| 24–25 | 0.031 / 0.031 | 4.1% / 13.9% |

Our LePhare rerun on the real data reproduces the official COSMOS2020 photo-z: σ_NMAD 0.0119 vs 0.0107 and η 3.7% vs
3.6% against spec-z. The mock's scatter is dominated by the mismatch between its (SDSS kcorrect) SEDs and the COSMOS
templates rather than by noise, and it under-produces faint catastrophic outliers. See the notebooks for the colour
dependence.

## How the mock is built
1. **Galaxies.** A 10.07 deg² patch of Cardinal (`Chinchilla-3-triofile.101`, RA 126.7–132.1, Dec 40.5–43.0; 3.0M
   galaxies to i≈27), with a 2.01 deg² "COSMOS twin" sub-box flagged.
2. **SEDs.** Each galaxy's `sedid` indexes the Addgals training set
   (`sdss_vagc_kcorr_r177_z_005_02_wdens.fits`). Its 5 kcorrect `COEFFS` multiply the basis templates in
   `Cardinal/basis_seds/`. This reproduces Cardinal's own LSST colours to ≲0.01 mag.
3. **Photometry.** The SEDs are redshifted to `true_redshift`, attenuated by the Madau (1995) IGM, and integrated through the 30
   COSMOS2020 filters of `lephare.default_cosmos_config`: CFHT u*/u; HSC grizy; UltraVISTA YJHKs and NB118;
   12 Subaru intermediate bands; NB711, NB816; IRAC ch1–4. They are normalised to Cardinal's `mag_i_lsst`.
4. **Noise.** For each band and UltraVISTA ultra-deep/deep region, each galaxy gets the reported error of a random real
   COSMOS2020 object of the same flux. Unreported noise, measured from the real negative-flux distribution, is added
   to the flux realisation; it is ≈0.1 µJy in IRAC and ≈0 elsewhere.
5. **Photo-z.** LePhare 0.3 with the COSMOS2020 setup (33 COSMOS templates, 4 extinction laws with
   `MOD_EXTINC 18,26,26,33,26,33,26,33`, emission lines, z=0–7). The same 28 bands are used for data and mock, because
   COSMOS2020 CLASSIC has no IRAC ch3/ch4. The data run uses AUTO_ADAPT zero-points from spec-z; the mock run uses
   none, plus diagnostic runs with mock-derived zero-points and with noiseless fluxes.

## Reproducing

### Requirements
* **NERSC (Perlmutter)** access to the inputs:
  * Cardinal: `/global/cfs/cdirs/lsst/groups/PZ/Cardinal/{parquet_files,basis_seds}`
  * Addgals training set: `/global/cfs/cdirs/des/jderose/SkyFactory-config/Addgals/sdss_vagc_kcorr_r177_z_005_02_wdens.fits`
  * Spec-z crossmatch (multimatch, COSMOS field): `/global/cfs/cdirs/desc-pz/users/tqzhang/rail/projects/multimatch/work/cosmos/`
    (`compilation/refz_341rule_cosmos.parquet`, `multimatch_cosmos_sources.parquet`)
* **COSMOS2020 CLASSIC** from IRSA (4.2 GB): https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/
* **Python environment** with `lephare==0.3`, `numpy`, `pandas`, `pyarrow`, `fitsio`, `astropy`, `healpy`,
  `minisom`, `matplotlib`, `jupyter`. At NERSC the DESC-PZ env works out of the box:
  `/global/cfs/cdirs/desc-pz/users/echarles/conda/envs/rail-2.0-dev` (jupyter kernel `rail-2.0-dev`).

Large inputs and all intermediate products go to `data/` and `work/` inside the repository. Both are git-ignored and
need ~15 GB.

### Steps
```bash
git clone git@github.com:ztq1996/cardinal_cosmos2020.git && cd cardinal_cosmos2020
source scripts/env.sh                      # sets PROJ and a repo-local LEPHAREDIR / LEPHAREWORK

# 0. inputs: LePhare filters/SEDs + COSMOS2020 CLASSIC catalogue
python scripts/00_get_inputs.py
mkdir -p data/cosmos2020 && curl -o data/cosmos2020/COSMOS2020_CLASSIC_R1_v2.2_p3.fits.gz \
    https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/COSMOS2020_CLASSIC_R1_v2.2_p3.fits.gz
gunzip -c data/cosmos2020/COSMOS2020_CLASSIC_R1_v2.2_p3.fits.gz > $PSCRATCH/COSMOS2020_CLASSIC_R1_v2.2_p3.fits

# 1. real data: total, MW-corrected 30-band fluxes + multimatch spec-z
python scripts/01_cosmos2020_prepare.py $PSCRATCH/COSMOS2020_CLASSIC_R1_v2.2_p3.fits
# 2-4. mock: Cardinal patch -> SED photometry -> COSMOS2020-like noise
python scripts/02_cardinal_patch.py
python scripts/03_sed_photometry.py
python scripts/04_noise_model.py
# 5. LePhare: libraries, run inputs, runs (on compute nodes)
python scripts/05a_build_lephare_lib.py data sim
python scripts/05c_make_run_inputs.py
salloc -q interactive -C cpu -N 1 -t 01:00:00 -A <account> srun -n 1 -c 256 --cpu-bind=none bash scripts/run_node.sh A
salloc -q interactive -C cpu -N 1 -t 01:30:00 -A <account> srun -n 1 -c 256 --cpu-bind=none bash scripts/run_node.sh B
# 6. notebooks
cd notebooks && python build_notebooks.py && for nb in 01_inputs_photometry 02_photoz_performance 03_diagnostics; do
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=rail-2.0-dev $nb.ipynb; done
```
Node `A` runs the data (with AUTO_ADAPT offsets), the noiseless-mock and the mock-zero-point runs, about 45 min.
Node `B` runs the 300k-galaxy mock, about 70 min. Runs are checkpointed per chunk, so rerunning the same command resumes a
run that hit the time limit (`bash scripts/run_node.sh adapt` resumes only the mock-zero-point run).
Random seeds are fixed, so the noise realisation and subsamples are reproducible.

### Repository layout
| path | content |
|---|---|
| `scripts/bands.py` | the 30 bands: LePhare filter file, COSMOS2020 column, A_λ/E(B−V) |
| `scripts/lephare_config.py` | COSMOS2020 LePhare configuration (`data` / `sim` variants) |
| `scripts/00_…05c_*.py` | pipeline steps above |
| `scripts/run_node.sh` | runs LePhare on one CPU node (one process, 120 forked workers) |
| `scripts/analysis_utils.py` | loading, metrics (bias, σ_NMAD, η, coverage), spec-z weighting, plotting |
| `notebooks/build_notebooks.py` | generates the notebooks (`python build_notebooks.py 02` rebuilds only notebook 02) |

## Notes and caveats
* **LePhare at scale.** `run_photoz` scales poorly with OpenMP threads (~0.25 s/object/core). The runner therefore loads
  the 850 MB library once and forks single-threaded workers, reaching ~75 objects/s per node (memory-bandwidth bound).
  Fluxes are passed in erg/s/cm²/Hz, and `AUTO_ADAPT=YES` must be set in the config that builds the `PhotoZ` object that
  computes offsets.
* **Filter calibration.** The mock uses `FILTER_CALIB=0` for all bands, because synthetic fluxes are exact AB integrals.
  The data keep the COSMOS IRAC convention (`FILTER_CALIB=1`).
* **Mock coverage.** Cardinal has no galaxies above z≈2.3 and is incomplete beyond i≈25.5, so analyses use i<25 and z<2.2.
  The data truth sample is spec-z objects with official `lp_type==0` (galaxies).
* **Spec-z weighting.** "Spec-z weighted" mock results reweight the mock to the spec-z sample's distribution in
  (i, g−i, i−Ks), to mimic its selection in those three quantities only.
* **PDF calibration.** The LePhare z-grid step (0.01) is comparable to the bright-galaxy scatter, so PIT/coverage at
  i<22 are partly grid-limited.
* **Bright-end errors.** In sparsely populated bright magnitude bins (e.g. u<21), the noise model assigns sky-limited
  errors. This has no effect on photo-z, because LePhare's ERR_SCALE floor dominates there.
