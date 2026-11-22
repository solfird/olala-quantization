#!/usr/bin/env bash
# Shared settings and helpers of the pod-side scripts (serve_olala.sh, stop_server.sh, smoke.sh,
# gpu_snapshot.sh, setup_remote.sh). Source it; do not run it.
# Layout on the pod: /workspace/models (weights), /workspace/logs/ARM (server logs, smoke.json, gpu logs),
# /workspace/run (pid files of the one server), /workspace/results, /workspace/bundle (these scripts).
# OLALA_WORK replaces /workspace (the tests use it).

# shellcheck disable=SC2034  # a library: the variables are used by the scripts that source it
OLALA_WORK=${OLALA_WORK:-/workspace}
LOG_ROOT=$OLALA_WORK/logs
RUN_DIR=$OLALA_WORK/run
MODEL_ROOT=$OLALA_WORK/models
POD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# An SSH session does not inherit the container's environment: start_sshd.sh saved an allowlisted copy.
# shellcheck disable=SC1091
[ -f /etc/olala_env.sh ] && . /etc/olala_env.sh
# No token of any kind lives on a pod: the model repository is public.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

say() { echo "$*"; }
die() { echo "error: $*" >&2; exit 1; }
utc_stamp() { date -u +%Y%m%dT%H%M%SZ; }
utc_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }
# alive PID: running, and not a zombie (a dead server whose parent has not reaped it still answers kill -0)
alive() {
  [ -n "${1:-}" ] && kill -0 "$1" 2> /dev/null || return 1
  case $(ps -o stat= -p "$1" 2> /dev/null) in Z*|"") return 1 ;; esac
  return 0
}

# Limits of the container, not of the host (2026-11-05: 128 cores visible, 27 allowed, 62 GB: 18 compile
# jobs were killed). cgroup v2 first, then v1. OLALA_NPROC and CGROUP_DIR exist for the tests.
CGROUP_DIR=${CGROUP_DIR:-/sys/fs/cgroup}
container_cpus() {
  local quota period n
  if [ -r "$CGROUP_DIR/cpu.max" ]; then
    read -r quota period < "$CGROUP_DIR/cpu.max"
  elif [ -r "$CGROUP_DIR/cpu/cpu.cfs_quota_us" ]; then
    quota=$(cat "$CGROUP_DIR/cpu/cpu.cfs_quota_us"); period=$(cat "$CGROUP_DIR/cpu/cpu.cfs_period_us")
    [ "$quota" -gt 0 ] 2> /dev/null || quota=max
  else
    quota=max
  fi
  if [ "${quota:-max}" = max ]; then
    echo "${OLALA_NPROC:-$(nproc 2> /dev/null || echo 1)}"
  else
    n=$(( quota / period )); [ "$n" -ge 1 ] || n=1
    echo "$n"
  fi
}
container_mem_gb() {
  local bytes=""
  if [ -r "$CGROUP_DIR/memory.max" ]; then bytes=$(cat "$CGROUP_DIR/memory.max")
  elif [ -r "$CGROUP_DIR/memory/memory.limit_in_bytes" ]; then bytes=$(cat "$CGROUP_DIR/memory/memory.limit_in_bytes"); fi
  case $bytes in ''|max|*[!0-9]*) bytes="" ;; esac
  if [ -z "$bytes" ] || [ "${#bytes}" -gt 15 ]; then   # unlimited: the host's memory
    bytes=$(awk '/^MemTotal:/{print $2 * 1024}' /proc/meminfo 2> /dev/null)
    bytes=${bytes:-17179869184}
  fi
  echo $(( bytes / 1073741824 ))
}
# Compile jobs for the JIT build of the fork's kernels: one per CPU of the container, at most one per 4 GB.
max_jobs() {
  local cpus mem jobs
  cpus=$(container_cpus); mem=$(container_mem_gb)
  jobs=$(( mem / 4 )); [ "$cpus" -lt "$jobs" ] && jobs=$cpus
  [ "$jobs" -ge 1 ] || jobs=1
  echo "$jobs"
}
