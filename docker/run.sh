############
# Open an interactive Docker container shell for Bayes3D
############

SCRIPT=$(realpath "$0")
DOCKERPATH=$(dirname "$SCRIPT")
BAYES3DPATH=$(dirname "$DOCKERPATH")
echo "Mounting $BAYES3DPATH into /workspace/bayes3d"
docker run --runtime=nvidia -it -p 8888:8888 --gpus all --rm --ipc=host -v $(dirname "$BAYES3DPATH"):/workspace/ bayes3d:latest  # mount the directory that contains Bayes3D into container