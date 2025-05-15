# Installation

```bash
# https://github.com/vllm-project/vllm/issues/12577#issuecomment-2757027368
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/15/77/7beca2061aadfdfd2d81411102e6445b459bcfedfc46671d4712de6a00fb/vllm-0.8.0-cp38-abi3-manylinux1_x86_64.whl

VLLM_USE_PRECOMPILED=1 pip install -e third_party/vllm # This step might take a while
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

# Run

Can run on 4 RTX4090(24GB).

```bash
srun --nodes=1 --gpus=4 --mail-type=ALL --pty bash

python workspace/prof.py
```

