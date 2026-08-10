#!/bin/bash
# Disciplined pre-run reset for DLIN sglang/vLLM tests (avoids the cache-corruption +
# leaked-shm crashes). Call BEFORE every engine launch. Safe: only touches MY processes
# (model-path match) + cards 24-27 + restores the known-good triton cache.
#
# Cache self-healing (Plan 1, 2026-07-28):
# - Restores the known-good ~/.triton/cache from backup (if backup exists).
# - Sets cache READ-ONLY (chmod a-w) so a crash can't write a corrupt dl_chunk entry.
# - To add new shapes: chmod -R u+w ~/.triton/cache && <warmup> && chmod -R a-w ~/.triton/cache
# - To regenerate backup: chmod -R u+w ~/.triton/cache && <full warmup> && cp -a ~/.triton/cache ~/.triton/cache.good_backup && chmod -R a-w ~/.triton/cache
set +e
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/sglang
PASS="$(cat /home/shaobo.xie/.claude/.dl_sudo_pass 2>/dev/null)"
# 1. kill MY sglang/vllm procs + any hung schedulers
pkill -9 -f "Qwen3.6-35B-A3B-FP8" 2>/dev/null
for p in $(dlsmi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -iE "sglang::schedul" | cut -d, -f1); do kill -9 "$p" 2>/dev/null; done
sleep 3
# 2. reset cards 24-27 (clears leaked VRAM + hung D-state)
for i in 24 25 26 27; do echo "$PASS" | sudo -S dlsmi -r -i "$i" >/dev/null 2>&1; done
sleep 6
# 3. restore known-good triton cache (dl_chunk binary) + write-protect
if [ -d ~/.triton/cache.good_backup ]; then
  rm -rf ~/.triton/cache && cp -a ~/.triton/cache.good_backup ~/.triton/cache
fi
chmod -R a-w ~/.triton/cache 2>/dev/null   # read-only: crash can't corrupt it
# 4. report
CACHE_PERMS=$(stat -c '%A' ~/.triton/cache 2>/dev/null | head -c 3)
echo "cards: $(dlsmi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | sed -n '25,28p' | tr '\n' ' ')"
echo "hung: $(dlsmi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -c sglang::schedul)  cache: $(du -sh ~/.triton/cache 2>/dev/null|cut -f1) perms: ${CACHE_PERMS}(read-only)"
