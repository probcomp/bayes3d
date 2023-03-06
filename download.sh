#!/bin/bash

export BOP_SITE=https://bop.felk.cvut.cz/media/data/bop_datasets
mkdir -p assets/bop
mkdir -p assets/ycb_video_models

echo "Downloading ycb bop"
wget $BOP_SITE/ycbv_base.zip -P assets/bop
wget $BOP_SITE/ycbv_models.zip -P assets/bop
wget $BOP_SITE/ycbv_test_bop19.zip -P assets/bop

echo "Unpacking ycb bop"
unzip assets/bop/ycbv_base.zip -d assets/bop
unzip assets/bop/ycbv_models.zip -d assets/bop/ycbv
unzip assets/bop/ycbv_test_bop19.zip -d assets/bop/ycbv

echo "Downloading ycb video models"
file_id="1gmcDD-5bkJfcMKLZb3zGgH_HUFbulQWu"
file_name="ycb_video_models.zip"
gdown --id "${file_id}" -O "${file_name}"
unzip "${file_name}" -d assets/ycb_video_models
rm "${file_name}"