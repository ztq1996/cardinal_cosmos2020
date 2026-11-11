#!/bin/bash
# Run LePhare on one Perlmutter CPU node (inside salloc/sbatch).
# lephare's run_photoz scales poorly with OpenMP threads (~0.25 s/object/core), so the main runs use one Python
# process that loads the library once and forks NP single-threaded workers (05b --nproc). Offsets (AUTO_ADAPT)
# are computed beforehand in a separate multi-threaded call.
#   bash scripts/run_node.sh A   -> data run (+AUTO_ADAPT offsets), sim noiseless, sim with sim-derived offsets
#   bash scripts/run_node.sh B   -> sim main run (300k)
#   bash scripts/run_node.sh adapt -> only the sim_adapt run (resume)
# Runs are chunk-checkpointed: rerunning resumes.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd $PROJ
NP=${NP:-120}
PY=${PY:-$(command -v python)}   # python of the rail-2.0-dev env (lephare 0.3)
R="$PY -u scripts/05b_run_lephare.py"
QUIET='removed to improve|Keyword|Reading library|^#|^ *$|^Number|^ *[A-Za-z_0-9]+ +: '

run () {   # run <tag> <args...>
  local tag=$1; shift
  if [[ "$*" == *"--offsets auto"* ]]; then
    OMP_NUM_THREADS=64 $R "$@" --tag $tag --offsets_only 2>&1 | grep --line-buffered -Ev "$QUIET"
  fi
  OMP_NUM_THREADS=1 $R "$@" --tag $tag --nproc $NP 2>&1 | grep --line-buffered -Ev "$QUIET"
}
case "$1" in
  A)
    run data_run --kind data --input work/run_inputs/data_run.parquet --zcol z_spec --idcol ID --offsets auto --nadapt 15000 --chunk 200
    run sim_noiseless --kind sim --input work/run_inputs/sim_diag_noiseless.parquet --zcol true_redshift --idcol galaxy_id --offsets zero --chunk 200
    run sim_adapt --kind sim --input work/run_inputs/sim_diag.parquet --zcol true_redshift --idcol galaxy_id --offsets auto --nadapt 15000 --chunk 200
    ;;
  adapt)
    run sim_adapt --kind sim --input work/run_inputs/sim_diag.parquet --zcol true_redshift --idcol galaxy_id --offsets auto --nadapt 15000 --chunk 200
    ;;
  B)
    run sim_main --kind sim --input work/run_inputs/sim_main.parquet --zcol true_redshift --idcol galaxy_id --offsets zero --chunk 500
    ;;
esac
echo "node $1 finished"
