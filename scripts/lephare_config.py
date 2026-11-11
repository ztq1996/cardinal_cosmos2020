"""COSMOS2020 LePhare configuration (lephare.default_cosmos_config + the COSMOS2020 extinction setup from
lephare-data/examples/README_short). kind='data': IRAC calibrated as nu*f_nu=const (FILTER_CALIB=1) as in the
real catalogue; kind='sim': synthetic fluxes are exact AB integrals, so FILTER_CALIB=0 for all bands."""
import os
import lephare as lp

PROJ = os.environ["PROJ"]
OUTKEYS = ["IDENT", "Z_BEST", "Z_BEST68_LOW", "Z_BEST68_HIGH", "Z_MED", "Z_MED68_LOW", "Z_MED68_HIGH", "Z_MODE",
           "CHI_BEST", "MOD_BEST", "EXTLAW_BEST", "EBV_BEST", "Z_SEC", "CHI_SEC", "NBAND_USED", "CONTEXT", "ZSPEC",
           "CHI_STAR", "MOD_STAR", "ZQ_BEST", "CHI_QSO"]

def write_output_para(path=f"{PROJ}/work/lephare_output.para"):
    with open(path, "w") as f:
        f.write("\n".join(OUTKEYS) + "\n")
    return path

def cosmos_config(kind="data"):
    c = dict(lp.default_cosmos_config)
    n = len(c["FILTER_LIST"].split(","))
    c["MOD_EXTINC"] = "18,26,26,33,26,33,26,33"
    c["FILTER_FILE"] = f"filter_cosmos_{kind}"
    if kind == "sim":
        c["FILTER_CALIB"] = ",".join(["0"] * n)
    c["STAR_LIB_OUT"] = f"STAR_COSMOS_{kind}"
    c["QSO_LIB_OUT"] = f"QSO_COSMOS_{kind}"
    c["GAL_LIB_OUT"] = f"CE_COSMOS_{kind}"
    c["ZPHOTLIB"] = f"CE_COSMOS_{kind},STAR_COSMOS_{kind},QSO_COSMOS_{kind}"
    c["PARA_OUT"] = write_output_para()
    c["PDZ_TYPE"] = "BAY_ZG"
    c["VERBOSE"] = "NO"
    c["APPLY_MW_EXTINCTION"] = "NO"   # MW correction applied to the data fluxes beforehand
    return c

QSO_OVERRIDE = {"MOD_EXTINC": "0,1000", "EB_V": "0.,0.1,0.2,0.3", "EXTINC_LAW": "SB_calzetti.dat"}
