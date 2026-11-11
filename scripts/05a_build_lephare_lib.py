"""Build the LePhare filter + template magnitude libraries for kind in {data, sim}."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import lephare as lp
from lephare_config import cosmos_config, QSO_OVERRIDE

for kind in sys.argv[1:] or ["data", "sim"]:
    t0 = time.time()
    cfg = cosmos_config(kind)
    lp.prepare(cfg, star_config={}, gal_config={}, qso_config=QSO_OVERRIDE)
    print(kind, "library built in", round(time.time() - t0), "s", flush=True)
