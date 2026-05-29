#!/bin/bash
export ASCEND_GLOBAL_LOG_LEVEL=1
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# torchrun --nproc_per_node=8 dlrm_msprof.py \
#     --mini-batch-size=320 \
#     --use-gpu \
#     --inference-only \
#     --arch-embedding-size="10000-10000-10000-10000-10000-10000-10000-10000" \
#     --arch-sparse-feature-size=64000 \
#     --data-size=3200 \
#     --arch-mlp-bot="600-500-400-300-64000" \
#     --arch-mlp-top="640-600-500-320-1" \
#     --dist-backend="hccl" \
#     --num-indices-per-lookup-fixed=True \
#     --num-indices-per-lookup=10000 \
#     --num-workers=8 \
#     --profiler-path="./profiling_test_16_4"
# torchrun --nproc_per_node=8 dlrm_msprof.py \
#     --mini-batch-size=1024 \
#     --arch-sparse-feature-size=128 \
#     --arch-embedding-size="1000-1000-1000-1000-1000-1000-1000-1000" \
#     --arch-mlp-bot="512-256-128" \
#     --arch-mlp-top="1024-1024-1" \
#     --arch-interaction-op="dot" \
#     --data-generation="random" \
#     --data-size=32000 \
#     --loss-function="bce" \
#     --use-gpu \
#     --dist-backend="hccl" \
#     --num-workers=8 \
#     --inference-only \
#     --load-model="./output/dlrm_model.pth"
# torchrun --nproc_per_node=8 dlrm_msprof.py \
#     --mini-batch-size=32 \
#     --use-gpu \
#     --hbm-ratio=0.005 \
#     --use-mtrain \
#     --inference-only \
#     --arch-embedding-size="10000000-10000000-10000000-10000000-10000000-10000000-10000000-10000000" \
#     --arch-sparse-feature-size=64000 \
#     --data-size=320 \
#     --arch-mlp-bot="600-500-400-300-64000" \
#     --arch-mlp-top="640-600-500-320-1" \
#     --dist-backend="hccl" \
#     --num-indices-per-lookup-fixed=True \
#     --num-indices-per-lookup=100 \
#     --num-workers=8 \
#     --profiler-path="./profiling_test_17"
torchrun --nproc_per_node=8 dlrm_msprof.py \
    --mini-batch-size=32 \
    --use-gpu \
    --use-mtrain \
    --inference-only \
    --arch-embedding-size="100000-100000-100000-100000-100000-100000-100000-100000" \
    --arch-sparse-feature-size=64000 \
    --data-size=320 \
    --arch-mlp-bot="600-500-400-300-64000" \
    --arch-mlp-top="640-600-500-320-1" \
    --dist-backend="hccl" \
    --num-indices-per-lookup-fixed=True \
    --num-indices-per-lookup=100 \
    --num-workers=8 \
    --profiler-path="./profiling_test_17"
# torchrun --nproc_per_node=8 dlrm_msprof.py \
#     --mini-batch-size=32 \
#     --use-gpu \
#     --hbm-ratio=0.1 \
#     --dram-cache-ratio=0.9 \
#     --use-mtrain \
#     --inference-only \
#     --arch-embedding-size="50000-50000-50000-50000-50000-50000-50000-50000" \
#     --arch-sparse-feature-size=64000 \
#     --data-size=320 \
#     --arch-mlp-bot="600-500-400-300-64000" \
#     --arch-mlp-top="640-600-500-320-1" \
#     --dist-backend="hccl" \
#     --num-indices-per-lookup-fixed=True \
#     --num-indices-per-lookup=100 \
#     --num-workers=8 \
#     --profiler-path="./profiling_test_17"
# torchrun dlrm_msprof.py \
#     --mini-batch-size=32 \
#     --inference-only \
#     --arch-embedding-size="1000-1000-1000-1000-1000-1000-1000-1000" \
#     --arch-sparse-feature-size=64000 \
#     --data-size=320 \
#     --arch-mlp-bot="600-500-400-300-64000" \
#     --arch-mlp-top="640-600-500-320-1" \
#     --num-indices-per-lookup-fixed=True \
#     --num-indices-per-lookup=100 \
#     --profiler-path="./profiling_test_17"
# torchrun --nproc_per_node=8 dlrm_msprof.py \
#     --mini-batch-size=32 \
#     --use-gpu \
#     --inference-only \
#     --arch-embedding-size="800000-800000-800000-800000-800000-800000-800000-800000" \
#     --arch-sparse-feature-size=64000 \
#     --data-size=320 \
#     --arch-mlp-bot="600-500-400-300-64000" \
#     --arch-mlp-top="640-600-500-320-1" \
#     --dist-backend="hccl" \
#     --num-indices-per-lookup-fixed=True \
#     --num-indices-per-lookup=100 \
#     --num-workers=8 \
#     --profiler-path="./profiling_test_64GB"

# torchrun --nproc_per_node=8 dlrm_msprof.py \
#     --mini-batch-size=32 \
#     --arch-embedding-size="100-100-100-100-100-100-100-100" \
#     --arch-sparse-feature-size=640 \
#     --data-size=320 \
#     --inference-only \
#     --use-gpu \
#     --arch-mlp-bot="600-500-400-300-640" \
#     --arch-mlp-top="640-600-500-320-1" \
#     --dist-backend="hccl" \
#     --num-indices-per-lookup-fixed=True \
#     --num-indices-per-lookup=100 \
#     --num-workers=8 \
#     --profiler-path="./profiling_test_17"
# 运行分布式训练
# torchrun --nproc_per_node=8 dlrm_msprof.py \
#     --mini-batch-size=128 \
#     --use-gpu \
#     --data-generation=dataset \
#     --data-set=kaggle \
#     --raw-data-file="/data/wpr/dlrm_ascend-main/train.txt" \
#     --processed-data-file="/data/wpr/dlrm_ascend-main/processed_data.npz" \
#     --arch-embedding-size="100000-50000-10000" \
#     --arch-sparse-feature-size=16 \
#     --arch-mlp-bot="13-512-256-64" \
#     --arch-mlp-top="512-256-1" \
#     --dist-backend="hccl" \
#     --num-indices-per-lookup-fixed=True \
#     --num-indices-per-lookup=10 \
#     --num-workers=8 \
#     --nepochs=10 \
#     --test-freq=1000 \
#     --save-model="./saved_model.pth" \
#     --profiler-path="./profiling_test_criteo"
# torchrun --nproc_per_node=8 dlrm_msprof.py \
#     --data-generation=internal \
#     --data-size=1000000 \
#     --mini-batch-size=128 \
#     --arch-embedding-size="100000-50000-10000" \
#     --arch-sparse-feature-size=16 \
#     --arch-mlp-bot="13-512-256-64" \
#     --arch-mlp-top="512-256-1" \
#     --num-indices-per-lookup=10 \
#     --num-indices-per-lookup-fixed=True \
#     --rand-data-dist=uniform \
#     --nepochs=10 \
#     --test-freq=1000 \
#     --use-gpu \
#     --num-workers=8 \
#     --profiler-path="./profiling_test_16_4"