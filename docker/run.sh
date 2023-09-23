############
# Open an interactive Docker container shell for Bayes3D
############

SCRIPT=$(realpath "$0")
DOCKERPATH=$(dirname "$SCRIPT")
BAYES3DPATH=$(dirname "$DOCKERPATH")

docker run --runtime=nvidia -it --gpus all --rm --ipc=host -v $(dirname "$BAYES3DPATH"):/workspace bayes3d:latest  # mount the directory that contains Bayes3D into container