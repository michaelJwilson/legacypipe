desiconda_version=20190311-1.2.7-img

module use /global/common/software/desi/$NERSC_HOST/desiconda/$desiconda_version/modulefiles

module load desiconda

export PYTHONPATH=/global/homes/m/mjwilson/LEGACY/legacypipe/py/:$PYTHONPATH
