# Installation

```bash
VLLM_USE_PRECOMPILED=1 pip install -e third_party/vllm # This step might take a while
# TODO: No sure why `torch-2.6.0-cp312-cp312-manylinux1_x86_64.whl` is downloaded again even inside docker container `nvcr.io/nvidia/pytorch:24.12-py3`

```

## Error

```bash
ImportError: .../flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda9SetDeviceEi
```

```bash
pip uninstall flash-attn

# Collect all version info
pip show torch
nvcc -V
python --version

# Pick the corresponding release from https://github.com/Dao-AILab/flash-attention/releases (abiFALSE)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

pip install workspace/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

