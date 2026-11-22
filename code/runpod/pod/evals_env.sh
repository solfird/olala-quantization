#!/usr/bin/env bash
# Shared settings of setup_evals_env.sh and run_accuracy.sh (the accuracy harness on a pod). Source it; do not run it.
# EVALS_DIR is the evals/ folder of the current bundle (/workspace/bundle/evals; OLALA_EVALS_DIR replaces it, the tests
# use it). The Python environment lives OUTSIDE the bundle (/workspace/venv-evals) so that a new bundle upload does not
# throw it away: `uv sync` reconciles it with the new uv.lock. The Hugging Face cache is /workspace/hf_cache: the
# datasets are fetched once by setup_evals_env.sh and a later run does not race on downloads.
# A detached shell (pod_ssh.sh detach) has a bare PATH: uv installed with --user lives in ~/.local/bin.

# shellcheck disable=SC2034  # a library: the variables are used by the scripts that source it
_evals_env_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
. "$_evals_env_here/common.sh"
EVALS_DIR=${OLALA_EVALS_DIR:-$OLALA_WORK/bundle/evals}
export HF_HOME=${HF_HOME:-$OLALA_WORK/hf_cache}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$OLALA_WORK/venv-evals}
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
# The committed uv.lock, Python 3.12 (evals/pyproject.toml: requires-python >=3.12,<3.13). --frozen never re-locks.
UV_RUN_ARGS="run --frozen --python 3.12"
