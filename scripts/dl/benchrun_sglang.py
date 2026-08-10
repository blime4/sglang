#!/usr/bin/env python3
# DL: benchrun_sglang — config-driven serving benchmark harness for sglang on DLIN.
#
# Port of vLLM's DLIN `benchrun_serving.py` (the serving path behind `vllm bench run`),
# keeping the SAME config schema (server_params / client_params / fixed_params) and the
# SAME output artifacts so vLLM-vs-sglang results are directly comparable:
#   - benchrun_result.json      (dict keyed by full_key_name)
#   - per-case client/server log dirs  ({output_dir}/benchrun_{model}/{client|server}/<params>/)
#   - GitHub-markdown summary table     (to_markdown(tablefmt="github"), first 18 cols)
#   - profiler_serving_data.json        (profiler_dir_batch_{N})
#
# Only TWO engine hooks differ from the vLLM original (everything else is reused verbatim):
#   1. Server launch  -> python -m sglang.launch_server (+ DLIN env/flag defaults)
#   2. Bench client   -> python -m sglang.bench_serving  (instead of `vllm bench serve`)
#
# sglang's bench_serving prints the same "Serving Benchmark Result" block as vLLM's, so the
# stdout-parsing / record / json / summary code is identical.
#
# Usage:
#   python3 scripts/dl/benchrun_sglang.py                 # write config_serving.json template (DLIN defaults)
#   python3 scripts/dl/benchrun_sglang.py config.json     # run serving sweep
#   python3 scripts/dl/benchrun_sglang.py config.json serving   # explicit mode (latency = phase 2)

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from itertools import product
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# DLIN defaults baked into every server launch + the config template.
# Verified working set from the team's run_sglang.sh and scripts/dl/run_qwen35_35b.py.
# Env vars consumed in python/sglang/srt/layers/quantization/fp8.py (SGLANG_DL_MOE_*).
# ---------------------------------------------------------------------------
DLIN_SERVER_ENV = {
    # bf16-bmm MoE path (commit a2048d00fc; fp8.py:1962). Robust default — only needs
    # gptq_dlblas_gemmex (FP8 linear), not the fused-MoE op. Gives ~13-17 tok/s.
    "SGLANG_DL_MOE_MAX_BF16_M": "128",
    # Offline tokenizer/model load (DLIN hosts have no HF network; avoids httpx
    # "client has been closed" in the bench client's tokenizer load). See run_sglang.sh phase_bench.
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
# SGLANG_DL_MOE_FUSED=1 (the 2x-vLLM fused-MoE fast path) is OPT-IN: it needs the
# invoke_fused_moe_opt op in _dl_C, which not every vLLM _dl_C.so ships. To enable it,
# export SGLANG_DL_MOE_FUSED=1 (and SGLANG_DL_MOE_FUSED_MAX_M=128) in your shell before
# running — _configure_environment_variables uses setdefault, so user-set values win.

SGLANG_SERVER_MODULE = "sglang.launch_server"
SGLANG_BENCH_MODULE = "sglang.bench_serving"
# Proven on DLIN (run_sglang.sh phase_bench): sglang-oai hits /v1/completions; the native
# 'sglang' backend also works but the team validated sglang-oai. random-ids avoids the
# ShareGPT download that 'random' triggers (fails offline).
SGLANG_BENCH_BACKEND = "sglang-oai"

# vLLM-only server keys that have no direct sglang equivalent (or a different name) and
# would make sglang.launch_server reject the argv. Dropped with a warning so a vLLM
# config_serving.json can be reused for sglang with minimal edits.
SGLANG_UNSUPPORTED_SERVER_KEYS = {
    "max_num_batched_tokens",      # sglang uses mem_fraction_static / --max-running-requests
    "gpu_memory_utilization",      # sglang: mem_fraction_static
    "max_num_seqs",
    "enforce_eager",               # sglang: disable_cuda_graph / cuda_graph_backend_*=disabled
    "enable_prefix_caching",
    "distributed_executor_backend",
    "mm_processor_cache_gb",
}


def warn_missing_hta_dependency():
    """HTA (holistic-trace-analysis) is a vLLM-bundmarked extra; not shipped with sglang.
    The harness still writes profiler_serving_data.json + traces for manual dlPTI analysis."""
    print("[INFO] HTA auto-analysis is not bundled with sglang; profiler traces are still "
          "captured for manual dlPTI inspection.")


def append_to_json_file(data, filename):
    """Append data to JSON file (dict keyed by full_key_name)."""
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        if not isinstance(existing_data, dict):
            existing_data = {}

        if isinstance(data, pd.DataFrame):
            data_to_append = data.to_dict("records")
        else:
            data_to_append = data if isinstance(data, list) else [data]

        for item in data_to_append:
            key = item.get("full_key_name")
            if key:
                existing_data[key] = item

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Data successfully appended to {filename}")
    except Exception as e:
        print(f"[ERROR] Error appending data to JSON file {filename}: {str(e)}")
        raise


def append_to_profile_json_file(data, filename):
    """Append profiler data to JSON file."""
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        for key, value in data.items():
            existing_data[key] = value

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Profiler data successfully appended to {filename}")
    except Exception as e:
        print(f"[ERROR] Error appending profiler data to JSON file {filename}: {str(e)}")
        raise


def init_serving_config(model_path="/xxx"):
    """Write a config_serving.json template with DLIN-proven defaults.

    Same 3-block schema as vLLM's benchrun serving config; server_params use sglang-native
    keys so the generic snake->kebab mapper produces valid sglang.launch_server flags.
    """
    default_config = {
        "server_params": {
            "tensor_parallel_size": 1,
            "attention_backend": "fa3",          # DLIN-proven (scripts/dl/run_qwen35_35b.py)
            "page_size": 16,
            "dtype": "bfloat16",
            "mem_fraction_static": 0.85,
            "load_format": "auto",
            "trust_remote_code": "",
        },
        "client_params": {
            # 'random-ids' is offline-safe (no ShareGPT download); 'random' needs network.
            "dataset_name": "random-ids",
            "random_input_len": [1024],
            "random_output_len": [128],
            "num_prompts": [8],
            "temperature": 0,
            # DLIN FA2 crashes once >~2 prefills overlap (jit_kernel/flash_attention.py
            # q.reshape); keep concurrency low until that bug is fixed.
            "max_concurrency": [1],
            "profile": [False],
        },
        "fixed_params": {
            "mode": "serving",
            "model": model_path,
            "port": 30000,
            "time_out": 600,
            "output_file": "benchrun_result.json",
            "output_dir": ".",
            "ci_log_format": False,
            "output_json_format": False,
        },
        "config_path": "config_serving.json",
    }

    config_path = "config_serving.json"
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)
        print(f"[OK] Serving benchmark configuration template generated: {config_path}")
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if model_path != "/xxx":
            existing.setdefault("fixed_params", {})["model"] = model_path
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=4, ensure_ascii=False)
            print(f"[OK] Updated model path in existing {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f), config_path


def load_config(config_file):
    """Load JSON configuration file."""
    try:
        with open(config_file, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"[ERROR] Error loading config file {config_file}: {str(e)}")
        raise


class ParameterGenerator:
    """Parameter generator and command builder (serving mode)."""

    def __init__(self, config_path: str, model_name: str = ""):
        self.config = self._load_config(config_path)
        self.param_combinations = self._generate_serving_combinations()
        self.model_name = model_name

    def _load_config(self, path: str) -> dict:
        try:
            with open(Path(path).expanduser(), "r") as f:
                config = json.load(f)
            serving_keys = {"server_params", "client_params", "fixed_params"}
            if not serving_keys.issubset(config.keys()):
                raise ValueError("Configuration missing required fields for serving mode "
                                 "(server_params / client_params / fixed_params)")
            return config
        except Exception as e:
            print(f"[ERROR] Error loading config from {path}: {str(e)}")
            raise

    def _generate_serving_combinations(self) -> list:
        """Cartesian product over client_params list-values only; server_params held constant."""
        try:
            client_dynamic = {
                k: v if isinstance(v, list) else [v]
                for k, v in self.config["client_params"].items()
            }
            valid = []
            for vals in product(*client_dynamic.values()):
                client_params = dict(zip(client_dynamic.keys(), vals))
                valid.append({
                    **self.config["server_params"],
                    **client_params,
                    **self.config["fixed_params"],
                })
            return valid
        except Exception as e:
            print(f"[ERROR] Error generating parameter combinations: {str(e)}")
            raise

    @staticmethod
    def _remove_trailing_slash(s: str) -> str:
        return s[:-1] if s.endswith("/") else s

    def build_serving_commands(self):
        """Returns (server_commands, client_commands, benchrun_run_config).

        Each command is [args_list, cmd_args_dict]. snake_case keys are mapped to
        --kebab-case flags; sglang.launch_server / sglang.bench_serving both accept these.
        """
        server_commands, client_commands = [], []
        benchrun_run_config = {}

        for params in self.param_combinations:
            server_cmd_args, client_cmd_args, cmd_args_dict = [], [], {}

            if self.model_name != "" and "model" in params:
                params["model"] = self.model_name

            model_name = self._remove_trailing_slash(params["model"])
            cmd_args_dict["model"] = model_name

            # ---- server command ----
            server_cmd_args.append(f"--model {model_name}")
            server_cmd_args.append(f"--port {params['port']}")

            server_params = {k: v for k, v in params.items()
                             if k in self.config.get("server_params", {})}
            for key, value in server_params.items():
                if key == "model":
                    continue
                if key in SGLANG_UNSUPPORTED_SERVER_KEYS:
                    print(f"[WARN] Dropping vLLM-only server param '{key}' (no sglang equivalent).")
                    continue
                arg_name = f"--{key.replace('_', '-')}"
                if value is None:
                    continue
                elif isinstance(value, bool):
                    if value:
                        server_cmd_args.append(arg_name)
                        cmd_args_dict[key] = "true"
                elif isinstance(value, (dict, list)):
                    server_cmd_args.append(f"{arg_name} '{json.dumps(value).replace(' ', '')}'")
                    cmd_args_dict[key] = value
                else:
                    server_cmd_args.append(f"{arg_name} {value}")
                    cmd_args_dict[key] = value

            # ---- client command ----
            client_cmd_args.append(f"--model {model_name}")
            client_cmd_args.append(f"--port {params['port']}")

            client_params = {k: v for k, v in params.items()
                             if k in self.config.get("client_params", {})}
            for key, value in client_params.items():
                arg_name = f"--{key.replace('_', '-')}"
                if value is None:
                    continue
                elif isinstance(value, bool):
                    if value:
                        client_cmd_args.append(arg_name)
                        cmd_args_dict[key] = "true"
                elif isinstance(value, (dict, list)):
                    client_cmd_args.append(f"{arg_name} '{json.dumps(value).replace(' ', '')}'")
                    cmd_args_dict[key] = value
                else:
                    client_cmd_args.append(f"{arg_name} {value}")
                    cmd_args_dict[key] = value

            if not benchrun_run_config:
                benchrun_run_config = {
                    "model_name": model_name,
                    "output_file": params.get("output_file", "benchrun_serving_result.json"),
                    "ci_log_format": params.get("ci_log_format", False),
                    "output_json_format": params.get("output_json_format", False),
                    "output_dir": params.get("output_dir", "."),
                    "time_out": params.get("time_out", 600),
                }

            server_commands.append([server_cmd_args, cmd_args_dict])
            client_commands.append([client_cmd_args, cmd_args_dict])

        return server_commands, client_commands, benchrun_run_config


def execute_command(command: str, env: dict = None) -> dict:
    """Execute a shell command, stream merged stdout/stderr, return captured lines + code."""
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, **(env or {})},
    )

    full_output = []
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            sys.stdout.write(output)
            sys.stdout.flush()
            full_output.append(output.strip())

    return_code = process.poll()
    return {
        "status": "success" if return_code == 0 else "error",
        "return_code": return_code,
        "output": full_output,
    }


def _port_from_server_args(server_cmd_args) -> int:
    """Extract --port value from the list of '--key value' strings."""
    for param in server_cmd_args:
        if param.startswith("--port"):
            if " " in param:
                return int(param.split(" ", 1)[1].strip())
    return 30000


# ---------------------------------------------------------------------------
# SWAP 1: sglang server launch (replaces vLLM's start_vllm_server).
# ---------------------------------------------------------------------------
def start_sglang_server(server_cmd_args, log_path=None, time_out=600, enable_profiler=False):
    """Launch `python -m sglang.launch_server` in its own process group, tee stdout to
    log_path, and wait for readiness via the /health endpoint. Returns the Popen or None."""
    try:
        args = ["python3", "-u", "-m", SGLANG_SERVER_MODULE]

        # server_cmd_args is a list of "--key value" / bare "--flag" strings.
        for param in server_cmd_args:
            if not param or param.isspace():
                continue
            if " " in param:
                key, value = param.split(" ", 1)
                if key and not key.isspace():
                    if not value or value.isspace():
                        args.append(key)            # bare flag (e.g. --trust-remote-code)
                    else:
                        args.extend([key, value])
            else:
                args.append(param)

        print(f"🚀 Starting sglang server with command: {' '.join(args)}")

        env = os.environ.copy()
        _configure_environment_variables(env, enable_profiler, log_path)

        if log_path:
            log_dir = os.path.dirname(log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            print(f"📝 Server log will be saved to: {log_path}")

        popen_kwargs = {}
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True  # killable as a group on teardown
        elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        server_process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=env,
            **popen_kwargs,
        )

        # Daemon thread: tee server output to screen + log file.
        if log_path:
            import threading
            log_file = open(log_path, "w", encoding="utf-8")

            def read_server_output():
                try:
                    while True:
                        line = server_process.stdout.readline()
                        if not line and server_process.poll() is not None:
                            break
                        if line:
                            print(line, end="")
                            log_file.write(line)
                            log_file.flush()
                except Exception as e:
                    print(f"❌ Error reading server output: {str(e)}")
                finally:
                    log_file.close()

            threading.Thread(target=read_server_output, daemon=True,
                             name="ServerOutputReader").start()

        port = _port_from_server_args(server_cmd_args)
        if _wait_for_sglang_startup(server_process, port, time_out, log_path):
            print("✅ sglang server started successfully")
            return server_process

        return None
    except Exception as e:
        print(f"❌ Exception starting sglang server: {str(e)}")
        return None


def _configure_environment_variables(env, enable_profiler=False, log_path=None):
    """Inject DLIN env defaults + profiler env for the sglang server process."""
    for var_name in ("CUDA_VISIBLE_DEVICES",):
        var_value = os.environ.get(var_name)
        if var_value is not None:
            env[var_name] = var_value

    # DLIN proven defaults (see DLIN_SERVER_ENV).
    for k, v in DLIN_SERVER_ENV.items():
        env.setdefault(k, v)

    if enable_profiler:
        # sglang: server dumps torch profiler traces to SGLANG_TORCH_PROFILER_DIR; the client
        # --profile flag triggers /start_profile + /stop_profile. DLPTI_AUTO_LOAD hooks the
        # DLIN runtime (works on sglang processes too; torch.profiler crashes on DLIN, dlPTI
        # is the working path per project memory).
        env["DLPTI_AUTO_LOAD"] = "1"
        if log_path:
            env["SGLANG_TORCH_PROFILER_DIR"] = os.path.dirname(log_path)
        print("Setting DLPTI_AUTO_LOAD=1 / SGLANG_TORCH_PROFILER_DIR for profile batch")
    else:
        env.pop("DLPTI_AUTO_LOAD", None)


def _wait_for_sglang_startup(server_process, port, timeout, log_path,
                             host="127.0.0.1", ready_banner="The server is fired up and ready to roll!"):
    """Wait until the server is ready. Primary probe: HTTP GET /health (sglang-native).
    Banner string in the log is a secondary signal. Crash -> False; timeout -> False."""
    start_time = time.time()
    url = f"http://{host}:{port}/health"

    while time.time() - start_time < timeout:
        if server_process.poll() is not None:
            print("sglang server process terminated unexpectedly")
            return False
        # Primary: health endpoint.
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print("sglang server /health returned 200")
                    return True
        except Exception:
            pass
        # Secondary: readiness banner already streamed to the log.
        if log_path and os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    if ready_banner in f.read():
                        print("sglang server readiness banner detected")
                        return True
            except Exception:
                pass
        time.sleep(1)

    print(f"Timeout waiting for sglang server to start (port {port}, {timeout}s)")
    return False


def stop_sglang_server(server_process):
    """Stop the sglang server process group (SIGTERM, then SIGKILL after 20s)."""
    if server_process and server_process.poll() is None:
        print("Stopping sglang server...")
        try:
            if os.name != "nt":
                os.killpg(server_process.pid, signal.SIGTERM)
            else:
                server_process.terminate()
        except ProcessLookupError:
            pass
        try:
            server_process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                if os.name != "nt":
                    os.killpg(server_process.pid, signal.SIGKILL)
                else:
                    server_process.kill()
            except ProcessLookupError:
                pass
            server_process.wait()
        print("[OK] sglang server stopped")


def split_client_server_batches(server_commands, client_commands):
    """Split cases into non-profile (first) and profile (second) batches; one server per batch."""
    batches = []
    grouped_cases = {False: [], True: []}

    for server_cmd, client_cmd in zip(server_commands, client_commands):
        profile_enabled = client_cmd[1].get("profile") == "true"
        grouped_cases[profile_enabled].append((server_cmd, client_cmd))

    for profile_enabled in (False, True):
        cases = grouped_cases[profile_enabled]
        if not cases:
            continue
        batches.append({
            "profile_enabled": profile_enabled,
            "server_cmd": cases[0][0],
            "client_cmds": [case[1] for case in cases],
        })
    return batches


def convert_benchmark_args_to_filename(params, mode="client"):
    """Build the on-disk case-dir leaf name from a fixed common_key_map subset."""
    common_key_map = {
        "tensor_parallel_size": "tp",
        "batch_size": "batch",
        "enforce_eager": "eager",
        "num_iters_warmup": "iters_warmup",
        "num_iters": "iters",
        "max_model_len": "model_len",
        "request_rate": "qps",
        "num_prompts": "prompts",
        "max_num_seqs": "max_seqs",
        "max_num_batched_tokens": "batch_tokens",
        "gpu_memory_utilization": "gpu_mem",
        "enable_prefix_caching": "prefix_cache",
        "profile": "profile",
    }
    exclude_keys = [
        "model", "distributed_executor_backend", "load_format", "compilation_config",
        "cuda_graph_sizes", "host", "port", "swap_space", "disable_log_requests",
        "download_dir", "backend", "dataset_name", "dataset_path", "ignore_eos",
        "disable_tqdm", "percentile_metrics", "goodput", "sonnet_prefix_len", "mode",
        "output_file", "output_dir", "ci_log_format", "output_json_format", "base_url",
        "hf_subset", "endpoint", "trust_remote_code",
    ]

    mapped_params = {}
    for k, v in params.items():
        if k in exclude_keys:
            continue
        if k in common_key_map:
            mk = common_key_map[k]
            mapped_params[mk] = "_".join(map(str, v)) if isinstance(v, list) else v

    model_name = params["model"].split("/")[-1] if "/" in params["model"] else params["model"]

    filename = "_".join(f"{k}_{v}" for k, v in mapped_params.items())
    return filename, model_name


def create_benchrun_log_path(model_name, filename_part, output_dir, mode="bench",
                             prefix="", suffix=""):
    """Create {output_dir}/benchrun_{model}/{mode}/{filename}/ and return (log_path, dir_path, case_name)."""
    if not mode:
        mode = "default"

    print(f"{mode.capitalize()} benchmark configuration:", filename_part)
    print("Model name:", model_name)

    if not os.path.exists(output_dir):
        print(f"[ERROR] Output directory {output_dir} does not exist, please check docker mount!")
        return None, None, None

    components = []
    if prefix:
        components.append(prefix)
    if mode and mode != "default":
        components.append(mode)
    components.append(filename_part)
    if suffix:
        components.append(suffix)
    filename = "_".join(components)

    dir_components = [output_dir, f"benchrun_{model_name}"]
    if mode and mode != "default":
        dir_components.append(mode)
    dir_components.append(filename)
    dir_path = os.path.join(*dir_components)

    try:
        os.makedirs(dir_path, exist_ok=True)
        print(f"[OK] Directory {dir_path} created successfully")
    except OSError as e:
        print(f"[ERROR] Error creating directory {dir_path}: {str(e)}")
        return None, None, None

    log_name_components = []
    if prefix:
        log_name_components.append(prefix)
    if mode and mode != "default":
        log_name_components.append(mode)
    log_name_components.append("benchmarks")
    if suffix:
        log_name_components.append(suffix)
    log_name_base = "_".join(log_name_components) + ".log"
    log_path = os.path.join(dir_path, log_name_base)

    if os.path.exists(log_path):
        suffix_num = 1
        while os.path.exists(log_path):
            log_path = os.path.join(dir_path, log_name_base.replace(".log", f"_{suffix_num}.log"))
            suffix_num += 1

    case_name_components = []
    if prefix:
        case_name_components.append(prefix)
    if mode and mode != "default":
        case_name_components.append(mode)
    case_name_components.append(model_name)
    case_name_components.append(filename_part)
    if suffix:
        case_name_components.append(suffix)
    case_name = "_".join(case_name_components)

    return log_path, dir_path, case_name


def create_serving_model_log_path(params, output_dir, mode="client"):
    """Wrapper: extract model name + filename part, then create the benchrun log path."""
    model_name = params["model"].split("/")[-1] if "/" in params["model"] else params["model"]
    filename_part, _ = convert_benchmark_args_to_filename(params, mode=mode)
    return create_benchrun_log_path(model_name, filename_part, output_dir, mode=mode)


# ---------------------------------------------------------------------------
# SWAP 2: sglang bench client (replaces `vllm bench serve ...`).
# ---------------------------------------------------------------------------
def execute_client_command(client_cmd_args, benchrun_run_config, path, idx):
    """Run `python -m sglang.bench_serving --backend sglang ...` for one case, parse the
    Serving Benchmark Result block into a record dict. Returns {status, data, case_name}."""
    def timeout_handler(signum, frame):
        print(f"[ERROR] Client command {idx} timed out")
        raise TimeoutError(f"Client command {idx} execution timed out")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)  # 10-minute wall clock (matches vLLM contract)

    try:
        cmd_args_list = client_cmd_args[0]
        cmd_args_dict = client_cmd_args[1]

        cmd = " ".join(cmd_args_list)
        print(f"[INFO] Client command: {cmd}")

        log_path, dir_path, case_name = create_serving_model_log_path(
            cmd_args_dict, benchrun_run_config["output_dir"], mode="client")
        if not log_path:
            print(f"[ERROR] Failed to create client log directory for combination {idx}")
            return {"status": "error", "data": None, "case_name": None}

        cmd += f" | tee {log_path}"

        # sglang bench client. --tokenizer pinned to the model path (proven on DLIN);
        # offline env so the tokenizer loads from the local model dir.
        model_path = cmd_args_dict.get("model", "")
        command = (f"python3 -m {SGLANG_BENCH_MODULE} --backend {SGLANG_BENCH_BACKEND} "
                   f"--host 127.0.0.1 --tokenizer {model_path} --disable-tqdm {cmd}")
        result = execute_command(command, env={
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

        if result["return_code"] == 0:
            print("[OK] Execution successful")
            pd_data = {}
            params_str = "_".join(f"{k}_{v}" for k, v in cmd_args_dict.items())
            pd_data["model_config_case"] = [case_name]
            pd_data["full_key_name"] = [params_str]
            pd_data.update(cmd_args_dict)

            # Parse only the "Serving Benchmark Result" block (sglang's stdout is noisier
            # than vLLM's — tqdm/startup lines would otherwise leak into the record).
            in_block = False
            for temp in result["output"]:
                if "Serving Benchmark Result" in temp:
                    in_block = True
                    continue
                if not in_block:
                    continue
                if temp.strip() and set(temp.strip()) <= {"="}:  # closing border
                    break
                if ":" in temp:
                    parts = temp.split(":")
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        value = ":".join(parts[1:]).strip()
                        key = key.replace(" ", "_").replace("(", "").replace(")", "").lower()
                        pd_data[key] = [value]

            return {"status": "success", "data": pd_data, "case_name": case_name}
        else:
            print(f"[ERROR] Execution failed: {result['status']}")
            return {"status": "error", "data": None, "case_name": case_name}
    finally:
        signal.alarm(0)


def execute_server_command(server_params, idx, time_out=600, enable_profiler=False):
    """Start one sglang server for a batch; return (process, dir_path) or None."""
    print(f"Starting server for combination {idx}")
    server_cmd_args = server_params[0]
    cmd_args_dict = server_params[1]
    output_dir = cmd_args_dict.get("output_dir", ".")

    log_path, dir_path, case_name = create_serving_model_log_path(
        cmd_args_dict, output_dir, mode="server")
    if not log_path:
        print(f"Failed to create server log directory for combination {idx}")
        return None

    server_process = start_sglang_server(server_cmd_args, log_path, time_out,
                                         enable_profiler=enable_profiler)
    if not server_process:
        print(f"Failed to start server for combination {idx} ERROR")
        if log_path and os.path.exists(log_path):
            print(f"\nServer log content from {log_path}:")
            print("-" * 50)
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    print(f.read())
            except Exception as log_error:
                print(f"Failed to read server log: {str(log_error)} ERROR")
            print("-" * 50)
        return None

    print(f"Server started successfully for combination {idx}")
    return server_process, dir_path


def load_json_to_dict(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Error loading JSON file {filename}: {str(e)}")
        raise


def add_ttft_tpot_aliases(data):
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, dict):
                if "mean_ttft_ms" in value and "ttft" not in value:
                    value["ttft"] = value["mean_ttft_ms"]
                if "mean_tpot_ms" in value and "tpot" not in value:
                    value["tpot"] = value["mean_tpot_ms"]
    return data


def traverse_json(data, first_level=False):
    if isinstance(data, dict):
        for key, value in data.items():
            if first_level:
                case_name = value.get("model_config_case", key) if isinstance(value, dict) else key
                print(f"CASE_NAME:{case_name}")
            else:
                print(f"{key}:{value}")
            traverse_json(value, first_level=False)
            if first_level:
                print("-" * 20)
    elif isinstance(data, list):
        for item in data:
            traverse_json(item, first_level=False)


def copy_file(source_path, destination_path, rename=""):
    if not os.path.exists(source_path):
        print(f"[ERROR] Source file not found: {source_path}")
        return
    try:
        os.makedirs(destination_path, exist_ok=True)
    except OSError as exc:
        print(f"[ERROR] Failed to create destination directory {destination_path}: {exc}")
        return
    target_name = rename or os.path.basename(source_path)
    target_path = os.path.join(destination_path, target_name)
    try:
        shutil.copy(source_path, target_path)
    except OSError as exc:
        print(f"[ERROR] Failed to copy {source_path} to {target_path}: {exc}")


def run_hta_analysis(profiler_json_path, path):
    """No-op for sglang: HTA is a vLLM-bundmarked extra. Profiler traces are still saved
    under each batch's server dir for manual dlPTI inspection (see the dl-profile skill)."""
    print("[INFO] HTA auto-analysis skipped (not bundled with sglang). "
          f"Profiler data: {profiler_json_path}")


def benchrun_serving(config_path):
    """Run the full serving sweep: per batch -> start sglang server -> run client cases ->
    collect records -> emit benchrun_result.json + summary table (+ profiler json)."""
    benchrun_result_json = None
    profiler_path_json = None
    try:
        warn_missing_hta_dependency()

        path = os.path.dirname(os.path.abspath(__file__))
        print("path", path)

        generator = ParameterGenerator(config_path)
        server_commands, client_commands, benchrun_run_config = generator.build_serving_commands()

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        columns = [
            "model_config_case", "model", "successful_requests", "benchmark_duration_s",
            "total_input_tokens", "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
            "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "median_itl_ms",
            "p99_itl_ms", "total_generated_tokens", "request_throughput_req/s",
            "output_token_throughput_tok/s", "total_token_throughput_tok/s",
            "max_num_batched_tokens", "tensor_parallel_size", "load_format", "trust_remote_code",
            "dataset_name", "random_input_len", "random_output_len", "num_prompts",
        ]
        result_df = pd.DataFrame(columns=columns)

        benchrun_result_json = f"{path}/benchrun_result.json"
        profiler_path_json = f"{path}/profiler_serving_data.json"

        if os.path.exists(benchrun_result_json):
            os.remove(benchrun_result_json)
            print(f"{benchrun_result_json} history has been deleted.")
        if os.path.exists(profiler_path_json):
            os.remove(profiler_path_json)
            print(f"{profiler_path_json} history has been deleted.")

        profiler_data = {}

        if not server_commands:
            print("No server commands available")
            sys.exit(1)
        client_batches = split_client_server_batches(server_commands, client_commands)
        global_case_idx = 1

        for batch_idx, batch in enumerate(client_batches, 1):
            batch_label = "profile" if batch["profile_enabled"] else "standard"
            print(f"Starting {batch_label} server batch {batch_idx}/{len(client_batches)}...")

            result = execute_server_command(
                batch["server_cmd"], batch_idx, benchrun_run_config["time_out"],
                enable_profiler=batch["profile_enabled"])
            if not result:
                print(f"Failed to start {batch_label} server batch")
                sys.exit(1)
            server_process, server_dir_path = result

            if batch["profile_enabled"]:
                profiler_data[f"profiler_dir_batch_{batch_idx}"] = server_dir_path

            try:
                for client_cmd in batch["client_cmds"]:
                    print(f"Executing client combination {global_case_idx}/{len(client_commands)}")
                    client_result = execute_client_command(
                        client_cmd, benchrun_run_config, path, global_case_idx)
                    if client_result["status"] == "success":
                        new_pd = pd.DataFrame(client_result["data"])
                        result_df = pd.concat([result_df, new_pd], ignore_index=True)
                        append_to_json_file(new_pd, benchrun_result_json)
                    global_case_idx += 1
            finally:
                stop_sglang_server(server_process)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("benchrun result: ", benchrun_result_json)
        print("hta result: ", profiler_path_json)
        sys.exit(2)

    model_name = (benchrun_run_config["model_name"].split("/")[-1]
                  if "/" in benchrun_run_config["model_name"]
                  else benchrun_run_config["model_name"])

    if profiler_data:
        append_to_profile_json_file(profiler_data, profiler_path_json)
        if os.path.exists(profiler_path_json) and os.path.getsize(profiler_path_json) > 0:
            copy_file(profiler_path_json,
                      f"{benchrun_run_config['output_dir']}/benchrun_{model_name}",
                      benchrun_run_config["output_file"])
        else:
            print("Profiler data file is empty or does not exist. Skipping file copy.")
        run_hta_analysis(profiler_path_json, path)

    column_mapping = {
        "model_config_case": "case", "model": "model",
        "max_num_batched_tokens": "batch_tokens", "tensor_parallel_size": "tp_size",
        "load_format": "load_fmt", "trust_remote_code": "trust_code",
        "dataset_name": "dataset", "random_input_len": "input_len",
        "random_output_len": "output_len", "num_prompts": "num_prompts",
        "successful_requests": "success_req", "benchmark_duration_s": "duration_s",
        "total_input_tokens": "input_tokens", "total_generated_tokens": "gen_tokens",
        "request_throughput_req/s": "req_throughput",
        "output_token_throughput_tok/s": "output_throughput",
        "total_token_throughput_tok/s": "total_throughput",
        "mean_ttft_ms": "mean_ttft", "median_ttft_ms": "median_ttft", "p99_ttft_ms": "p99_ttft",
        "mean_tpot_ms": "mean_tpot", "median_tpot_ms": "median_tpot", "p99_tpot_ms": "p99_tpot",
        "mean_itl_ms": "mean_itl", "median_itl_ms": "median_itl", "p99_itl_ms": "p99_itl",
    }
    existing_columns = [col for col in column_mapping.keys() if col in result_df.columns]
    if existing_columns:
        result_df = result_df.rename(columns={col: column_mapping[col] for col in existing_columns})
    print("Benchmarks Perf:")
    try:
        print(result_df.iloc[:, :18].to_markdown(tablefmt="github", index=False))
    except ImportError:
        # tabulate optional — fall back to a plain table so a successful run still prints.
        print("[WARN] 'tabulate' not installed; falling back to plain text. "
              "For exact vLLM markdown parity: pip install tabulate")
        print(result_df.iloc[:, :18].to_string(index=False))

    if os.path.exists(benchrun_result_json):
        copy_file(benchrun_result_json,
                  f"{benchrun_run_config['output_dir']}/benchrun_{model_name}",
                  "benchrun_result.json")

    if benchrun_run_config["ci_log_format"] is True:
        if os.path.exists(benchrun_result_json):
            data = load_json_to_dict(benchrun_result_json)
            data = add_ttft_tpot_aliases(data)
            traverse_json(data, True)

    if benchrun_run_config["output_json_format"] is True:
        if os.path.exists(benchrun_result_json):
            data = load_json_to_dict(benchrun_result_json)
            print(benchrun_result_json)
            print(data)

    return path


def main():
    args = sys.argv[1:]
    if len(args) == 0:
        init_serving_config()
        print("\nUsage:")
        print("  python3 scripts/dl/benchrun_sglang.py                  # write config_serving.json template")
        print("  python3 scripts/dl/benchrun_sglang.py config.json      # run serving sweep")
        print("Edit 'model' in config_serving.json, then run again with the config path.")
        return

    config_path = args[0]
    mode = args[1] if len(args) > 1 else None

    config = load_config(config_path)
    if mode is None:
        mode = config.get("fixed_params", {}).get("mode", "serving")

    if mode == "serving":
        benchrun_serving(config_path)
    elif mode == "latency":
        print("[ERROR] latency mode is not implemented in v1 (serving-only). "
              "Phase 2 will add the sglang.Engine offline path + QA Result seam.")
        sys.exit(2)
    else:
        print(f"[ERROR] Unknown mode '{mode}'. Expected 'serving' or 'latency'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
