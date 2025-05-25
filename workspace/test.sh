#!/bin/bash
nsys profile \
        --trace=cuda,nvtx --output=moe_profile_nvtx \
        --force-overwrite=true \
        --sample=none \
        --trace-fork-before-exec=true \ # evil
        python prof_dp.py \
        --model="../scripts/DeepSeek-V3" \
        --dp-size=2 \
        --tp-size=2
