export BOP_SITE=https://bop.felk.cvut.cz/media/data/bop_datasets
mkdir -p assets/bop
wget $BOP_SITE/ycbv_base.zip -P assets/bop
wget $BOP_SITE/ycbv_models.zip -P assets/bop

unzip assets/bop/ycbv_base.zip
unzip assets/bop/ycbv_models.zip -d assets/bop/ycbv