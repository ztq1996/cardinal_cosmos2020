"""Step 2: cut a contiguous Cardinal patch (~10 deg^2) plus a ~2 deg^2 'COSMOS twin' sub-box from one triofile."""
import numpy as np, pyarrow.parquet as pq, pandas as pd, os, sys

PROJ = os.environ.get("PROJ", ".")
FILE = "/global/cfs/projectdirs/desc-pz/Cardinal/parquet_files/Chinchilla-3-triofile.101.parquet"
BIG = dict(ra=(126.7, 132.1), dec=(40.5, 43.0))
TWIN = dict(ra=(128.0, 129.9), dec=(41.0, 42.42))
COLS = ["galaxy_id", "ra", "dec", "redshift", "true_redshift", "sedid", "size",
        "mag_u_lsst", "mag_g_lsst", "mag_r_lsst", "mag_i_lsst", "mag_z_lsst", "mag_y_lsst",
        "Roman_Y106", "Roman_J129", "Roman_H158", "Roman_F184", "Roman_K213",
        "Euclid_Y", "Euclid_J", "Euclid_H", "WISE_W1", "WISE_W2"]

def area(b):
    return np.deg2rad(b["ra"][1] - b["ra"][0]) * (np.sin(np.deg2rad(b["dec"][1])) - np.sin(np.deg2rad(b["dec"][0]))) * (180 / np.pi) ** 2

def inbox(df, b):
    return (df.ra > b["ra"][0]) & (df.ra < b["ra"][1]) & (df.dec > b["dec"][0]) & (df.dec < b["dec"][1])

f = pq.ParquetFile(FILE)
out = []
for i in range(f.metadata.num_row_groups):
    t = f.read_row_group(i, columns=COLS).to_pandas()
    t = t[inbox(t, BIG)]
    out.append(t)
    print(i, len(t), t.true_redshift.min(), t.true_redshift.max(), flush=True)
df = pd.concat(out, ignore_index=True)
df["galaxy_id"] = df.galaxy_id.astype(np.int64)
df["twin"] = inbox(df, TWIN).values
print(f"big area {area(BIG):.2f} deg2, N={len(df)}; twin area {area(TWIN):.2f} deg2, N={df.twin.sum()}")
os.makedirs(f"{PROJ}/work/cardinal", exist_ok=True)
df.to_parquet(f"{PROJ}/work/cardinal/cardinal_patch.parquet", index=False)
