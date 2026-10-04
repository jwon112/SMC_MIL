#!/usr/bin/env bash
# Open a persistent interactive shell with the existing SMC environment.
# No packages installed, mounts changed, jobs started/stopped, or files deleted.
case $- in
    *i*) ;;
    *) exec bash --rcfile /home/jupyter/image_team/projects/SMC_MIL/smc.sh -i ;;
esac

if [ -r /home/jupyter/image_team/miniconda3/etc/profile.d/conda.sh ]; then
    . /home/jupyter/image_team/miniconda3/etc/profile.d/conda.sh
    if ! conda activate clam_latest; then
        echo 'WARNING: clam_latest activation failed. Do not install/recreate it yet.'
    fi
else
    echo 'WARNING: the existing conda installation is not visible in this container.'
fi

if ! cd /home/jupyter/image_team/projects/SMC_MIL; then
    echo 'WARNING: SMC_MIL project directory is not available here.'
fi

echo '--- SMC terminal environment ---'
hostname
pwd
printf 'CONDA_DEFAULT_ENV: %s\n' "${CONDA_DEFAULT_ENV:-not activated}"
if command -v python >/dev/null 2>&1; then
    python -c 'import sys; print("Python:", sys.executable); import h5py; print("h5py OK")'
else
    echo 'WARNING: python is not available in this shell.'
fi

if [ -d /home/jupyter/data/image_team/pathomics_all_wsi_v1/full ]; then
    echo 'Existing pathomics result directory is visible.'
    echo 'You can now run: bash eta.txt'
else
    echo 'WARNING: existing pathomics results are still NOT visible in this container.'
    echo 'Conda activation does not restore the old container or its data mount.'
    echo 'Do not relaunch extraction or create replacement output folders yet.'
fi
echo 'Interactive shell is ready. Type exit to return to the previous shell.'
