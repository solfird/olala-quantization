utc=2026-11-20T15:31:47Z
arm=fp8pc
MAX_JOBS=10
container_cpus=10
container_mem_gb=57
command: vllm serve /workspace/models/olala-bf16 --trust-remote-code --served-model-name olala --mamba-ssm-cache-dtype float32 --reasoning-parser olala --tool-call-parser olala --enable-auto-tool-choice --max-model-len 32688 --max-num-seqs 256 --quantization fp8_per_channel --quantization-config '{"ignore":["model.layers.1.mixer_proj","model.layers.7.mixer_proj","model.layers.23.mixer_proj","model.layers.0.mlp.*","model.layers.1.mlp.*","model.layers.5.mlp.*","model.layers.19.mlp.*"]}' --host 127.0.0.1 --port 8000
CUDA_HOME=/usr/local/cuda
CUDA_VERSION=13.0.3
TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0
VLLM_BUILD_COMMIT=unknown
VLLM_BUILD_PIPELINE=local
VLLM_BUILD_URL=
VLLM_ENABLE_CUDA_COMPATIBILITY=0
VLLM_IMAGE_TAG=local/vllm-openai:dev
VLLM_USAGE_SOURCE=production-docker-image
