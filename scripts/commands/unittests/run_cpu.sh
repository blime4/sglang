#!/bin/bash

# Run sglang's CPU UT suite (base-a/b/c-test-cpu) inside the runtime container.
# CPU-only tests (no GPU/model) — broadens DLIN CI coverage to sglang's base UTs.
cd ${REPO_PATH}/scripts/run && bash ${REPO_PATH}/scripts/run/docker_run.sh cpu_unittests unittests
