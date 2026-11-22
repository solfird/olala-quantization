utc=2026-11-20T20:10:31Z
arm=bf16
MAX_JOBS=6
container_cpus=6
container_mem_gb=57
command: vllm serve /workspace/models/olala-bf16 --trust-remote-code --served-model-name olala --mamba-ssm-cache-dtype float32 --reasoning-parser olala --tool-call-parser olala --enable-auto-tool-choice --max-model-len 32688 --max-num-seqs 256 --host 127.0.0.1 --port 8000
CUDA_HOME=/usr/local/cuda
CUDA_VERSION=13.0.3
TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0
VLLM_BUILD_COMMIT=unknown
VLLM_BUILD_PIPELINE=local
VLLM_BUILD_URL=
VLLM_ENABLE_CUDA_COMPATIBILITY=0
VLLM_IMAGE_TAG=local/vllm-openai:dev
VLLM_USAGE_SOURCE=production-docker-image
