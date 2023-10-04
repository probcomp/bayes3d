############
# Build a Docker container image for Bayes3D
############

SCRIPT=$(realpath "$0")
DOCKERPATH=$(dirname "$SCRIPT")
BAYES3DPATH=$(dirname "$DOCKERPATH")

cd ${BAYES3DPATH}
git clone git+ssh://git@github.com/probcomp/genjax.git  # temp clone for dependency resolution
docker build -t bayes3d:latest ${BAYES3DPATH}  # append --no-cache after changes to dockerfile
rm -rf genjax
