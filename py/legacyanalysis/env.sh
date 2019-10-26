desiconda_version=20190311-1.2.7-img

module use /global/common/software/desi/$NERSC_HOST/desiconda/$desiconda_version/modulefiles

module load desiconda


export LEGACY_SURVEY_DIR=/project/projectdirs/cosmo/data/legacysurvey/dr8/
export PYTHONPATH=/global/homes/m/mjwilson/LEGACY/legacypipe/py/:$PYTHONPATH

export DUST_DIR=/global/project/projectdirs/cosmo/data/dust/v0_1

export UNWISE_COADDS_DIR=/global/projecta/projectdirs/cosmo/work/wise/outputs/merge/neo5/fulldepth:/global/project/projectdirs/cosmo/data/unwise/allwise/unwise-coadds/fulldepth

export UNWISE_COADDS_TIMERESOLVED_DIR=/global/projecta/projectdirs/cosmo/work/wise/outputs/merge/neo5

export GAIA_CAT_DIR=/global/project/projectdirs/cosmo/work/gaia/chunks-gaia-dr2-astrom-2/
export GAIA_CAT_VER=2

export TYCHO2_KD_DIR=/global/project/projectdirs/cosmo/staging/tycho2
export LARGEGALAXIES_CAT=/global/project/projectdirs/cosmo/staging/largegalaxies/v3.0/LSLGA-V3.0.kd.fits

export PS1CAT_DIR=/global/project/projectdirs/cosmo/work/ps1/cats/chunks-qz-star-v3/

export UNWISE_PSF_DIR=/global/project/projectdirs/cosmo/staging/unwise_psf/dr9
export PYTHONPATH=$UNWISE_PSF_DIR/py/unwise_psf:$PYTHONPATH
