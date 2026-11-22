#!/usr/bin/env bash
# Runs ON the pod: prepare the accuracy harness (evals/ of the bundle) so that run_accuracy.sh starts at once.
# Usage: setup_evals_env.sh [--dry-run]
# Steps, idempotent (a second run changes nothing and costs seconds):
#   1 uv: used if present, otherwise `python3 -m pip install --user uv==VERSION` from PyPI (pinned; OLALA_UV_VERSION
#     replaces the version). No `curl | sh`: an installer script fetched at run time is third-party code we do not read.
#   2 uv sync --frozen --python 3.12 in the bundle's evals/: the committed uv.lock decides every version. The
#     environment is /workspace/venv-evals, outside the bundle, so a later bundle upload keeps it.
#   3 prefetch_evals_data.py: MGSM, EvalPlus HumanEval+ and MBPP+, GSM8K, IFEval and AIME 2025 into /workspace/hf_cache
#     (item counts checked) and the NLTK tokenizers of the IFEval scorer into ~/.cache/nltk_data: a later run does not
#     race on downloads.
#   4 the hermetic test suite of evals/ (no GPU, no network, no Docker): its result is printed.
# Every step logs to /workspace/logs/setup/evals-env-<UTC>.log (stdout and stderr of the tools included). The last
# line is EVALS_ENV_DONE or EVALS_ENV_FAILED: <reason> (exit 1). Safe to start detached with pod_ssh.sh detach.
# Needs the network for steps 1 to 3 (PyPI, GitHub for the pinned instruction-following-eval, the Hub, the NLTK host).
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=evals_env.sh
. "$here/evals_env.sh"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN   # no token of any kind lives on a pod: every dataset used here is public

dry=0
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) dry=1; shift ;;
    *) die "unknown argument $1" ;;
  esac
done
uv_version=${OLALA_UV_VERSION:-0.13.7}
stamp=$(utc_stamp)
log=$LOG_ROOT/setup/evals-env-$stamp.log

step() { echo "== $(utc_iso) $*"; }
failed() { echo "EVALS_ENV_FAILED: $*"; }

if [ $dry -eq 1 ]; then
  echo "setup_evals_env.sh (dry run: nothing is run, no file is written)"
  echo "log: $log"
  echo "evals dir: $EVALS_DIR (environment $UV_PROJECT_ENVIRONMENT, Hugging Face cache $HF_HOME)"
  echo "+ 1 command -v uv || python3 -m pip install --user --break-system-packages uv==$uv_version"
  echo "+ 2 cd $EVALS_DIR && uv sync --frozen --python 3.12"
  echo "+ 3 cd $EVALS_DIR && uv $UV_RUN_ARGS python $here/prefetch_evals_data.py --evals-dir $EVALS_DIR"
  echo "+     fetches MGSM, EvalPlus HumanEval+ and MBPP+, GSM8K, IFEval, AIME 2025 and the NLTK punkt data"
  echo "+ 4 cd $EVALS_DIR && uv $UV_RUN_ARGS pytest -q   (the hermetic suite; its tail is printed)"
  exit 0
fi

main() {
  echo "setup_evals_env $(utc_iso): $EVALS_DIR, log $log"
  step "0 check the bundle"
  if [ ! -f "$EVALS_DIR/pyproject.toml" ] || [ ! -f "$EVALS_DIR/uv.lock" ]; then
    failed "$EVALS_DIR has no pyproject.toml and uv.lock: upload the bundle first (pod_setup.sh NAME --bundle-only)"
    return 1
  fi

  step "1 uv"
  if command -v uv > /dev/null 2>&1; then
    echo "uv present: $(uv --version)"
  else
    echo "uv missing: installing uv==$uv_version from PyPI for this user"
    python3 -m pip install --user --break-system-packages "uv==$uv_version" || { failed "could not install uv"; return 1; }
    command -v uv > /dev/null 2>&1 || { failed "uv is still not on the PATH after the install (looked in $HOME/.local/bin)"; return 1; }
    echo "uv installed: $(uv --version)"
  fi

  step "2 uv sync --frozen --python 3.12 (the committed uv.lock; environment $UV_PROJECT_ENVIRONMENT)"
  (cd "$EVALS_DIR" && uv sync --frozen --python 3.12) || { failed "uv sync failed"; return 1; }

  step "3 prefetch the datasets (HF cache $HF_HOME) and the NLTK data"
  # shellcheck disable=SC2086  # UV_RUN_ARGS is a list of words
  (cd "$EVALS_DIR" && uv $UV_RUN_ARGS python "$here/prefetch_evals_data.py" --evals-dir "$EVALS_DIR") \
    || { failed "the dataset prefetch failed (a missing, short or unreachable dataset): see above"; return 1; }

  step "4 the hermetic test suite of evals/"
  # shellcheck disable=SC2086
  (cd "$EVALS_DIR" && set -o pipefail && uv $UV_RUN_ARGS pytest -q 2>&1 | tail -n 15) \
    || { failed "the hermetic test suite failed: see above"; return 1; }

  echo "EVALS_ENV_DONE"
}

mkdir -p "$LOG_ROOT/setup" "$HF_HOME" || die "cannot create $LOG_ROOT/setup"
main 2>&1 | tee -a "$log"
exit "${PIPESTATUS[0]}"
