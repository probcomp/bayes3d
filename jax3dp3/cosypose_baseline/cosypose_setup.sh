#!/usr/bin/env bash -l

__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup

# git clone --recurse-submodules https://github.com/Simple-Robotics/cosypose.git
cd cosypose
conda env create -n cosypose --file environment.yaml
conda activate cosypose
git lfs pull
python setup.py install
mkdir local_data

echo "Downloading data..."
python -m cosypose.scripts.download --bop_dataset=ycbv
python -m cosypose.scripts.download --bop_extra_files=ycbv
python -m cosypose.scripts.download --urdf_models=ycbv

echo "Downloading models..."
python -m cosypose.scripts.download --model=detector-bop-ycbv-synt+real--292971
python -m cosypose.scripts.download --model=coarse-bop-ycbv-synt+real--822463
python -m cosypose.scripts.download --model=refiner-bop-ycbv-synt+real--631598
