docker run \
    --name foldinfer-2 \
    -dit \
    --gpus all \
    --net=host --ipc=host \
    -v $PWD:/workspace \
    --runtime=nvidia \
    nvcr.io/nvidia/pytorch:24.12-py3