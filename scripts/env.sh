# source this before running any project script:  source scripts/env.sh
# PROJ = repository root (derived from this file's location); LePhare data/work dirs are kept inside the repo (git-ignored).
export PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export LEPHAREDIR=$PROJ/work/lephare_dir
export LEPHAREWORK=$PROJ/work/lephare_work
mkdir -p $LEPHAREDIR $LEPHAREWORK
