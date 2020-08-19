#!/bin/bash
#
# install_with_fresh_env_from_github_clone.sh

unset EIGENPATH
unset BOOSTMATHPATH
unset PYTHONPATH

DATESTAMP=`date '+%Y%m%d_%H%M%S'`
ENV_NAME="bnpy_py27_$DATESTAMP"

TMP_DIR="/tmp/bnpy_py27_$DATESTAMP/"

conda create python=2.7 --name $ENV_NAME --yes

source activate $ENV_NAME

mkdir -p $TMP_DIR/
pushd $TMP_DIR/

git clone https://github.com/bnpy/bnpy ./
pip install -e . --no-cache-dir 
#pip install seaborn --no-cache-dir

#pushd ./docs/
#make html



