"""Resolve SPEED-Bench settings without inheriting serving benchmark defaults."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from . import UPSTREAM_COMMIT

DOMAINS = {
    "coding",
    "humanities",
    "math",
    "qa",
    "rag",
    "reasoning",
    "stem",
    "writing",
    "multilingual",
    "summarization",
    "roleplay",
}
SUPPORTED = {
    "model",
    "tasks",
    "backend",
    "tp_size",
    "dp_size",
    "pp_size",
    "temperature",
    "top_p",
    "top_k",
    "tokens_to_generate",
    "chat_template_kwargs",
    "chat_template",
    "trust_remote_code",
    "extra_server_args",
    "max_model_len",
    "gpu_memory_utilization",
    "config",
}


def resolve(params, explicit):
    unsupported = explicit - SUPPORTED
    if unsupported:
        raise ValueError("SPEED-Bench does not support: " + ", ".join(sorted(unsupported)))
    match = re.fullmatch(r"speed-bench(?::([1-9]\d*))?", params["tasks"].strip())
    if not match:
        raise ValueError("SPEED-Bench requires --tasks speed-bench[:N] alone")
    if params["backend"] != "vllm" or params.get("dp_size") not in (None, 1):
        raise ValueError("SPEED-Bench requires one vLLM engine (DP=1)")
    parallel = {key: 1 if params.get(key) is None else params[key] for key in ("tp_size", "pp_size")}
    if any(type(value) is not int or value < 1 for value in parallel.values()):
        raise ValueError("TP and PP must be positive integers")
    data = {}
    if params.get("config"):
        import yaml

        data = yaml.safe_load(Path(params["config"]).read_text()) or {}
    allowed = {
        "max_model_len",
        "gpu_memory_utilization",
        "extra_server_args",
        "inference",
        "chat_template_kwargs",
    }
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError(f"SPEED-Bench YAML supports only {sorted(allowed)}")
    inference = data.get("inference", {})
    if not isinstance(inference, dict) or set(inference) - {
        "temperature",
        "top_p",
        "top_k",
        "tokens_to_generate",
    }:
        raise ValueError("Unsupported SPEED-Bench inference settings")

    def option(key, default, values=data):
        return params[key] if key in explicit else values.get(key, default)

    generation = {
        k: option(k, v, inference)
        for k, v in {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "tokens_to_generate": 4096,
        }.items()
    }
    if (
        generation["temperature"] not in (0, 1)
        or generation["top_p"] != 1
        or generation["top_k"] != -1
        or generation["tokens_to_generate"] != 4096
    ):
        raise ValueError("SPEED-Bench uses temperature 0 or 1, top_p=1, top_k=-1 and max_tokens=4096")
    generation = {k: (float(v) if k in {"temperature", "top_p"} else int(v)) for k, v in generation.items()}
    ctk = option("chat_template_kwargs", None)
    generation["chat_template_kwargs"] = ctk or {}
    if not isinstance(generation["chat_template_kwargs"], dict):
        raise TypeError("chat_template_kwargs must be a mapping")
    generation["ignore_eos"] = False
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "model": params["model"],
        **parallel,
        "repeats": int(match[1] or 1),
        "generation": generation,
        "concurrency": 32,
        "seeds": list(range(int(match[1] or 1))),
        "extra_server_args": option("extra_server_args", ""),
        "max_model_len": option("max_model_len", None),
        "gpu_memory_utilization": option("gpu_memory_utilization", None),
        "trust_remote_code": params["trust_remote_code"],
        "chat_template": params["chat_template"],
    }


def load_dataset(path, *, full=True):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "turns" not in row:
            messages = row["messages"]
            if any(m["role"] not in {"user", "system"} for m in messages):
                raise ValueError("SPEED-Bench input must contain only system/user prompts")
            systems = [m["content"] for m in messages if m["role"] == "system"]
            if len(systems) > 1 or systems and messages[0]["role"] != "system":
                raise ValueError("Expected at most one initial system prompt")
            row = {
                "question_id": row["question_id"],
                "category": row["category"],
                "turns": [m["content"] for m in messages if m["role"] == "user"],
                "system_prompt": systems[0] if systems else None,
            }
        if not row.get("turns") or not all(isinstance(t, str) and t for t in row["turns"]):
            raise ValueError("Empty or invalid conversation")
        rows.append({k: row.get(k) for k in ("question_id", "category", "turns", "system_prompt")})
    if len({str(r["question_id"]) for r in rows}) != len(rows) or any(r["question_id"] is None for r in rows):
        raise ValueError("Duplicate or missing question IDs")
    if full and (
        Counter(r["category"] for r in rows) != Counter({k: 80 for k in DOMAINS})
        or sum(len(r["turns"]) for r in rows) != 1138
    ):
        raise ValueError("Expected Qualitative: 880 conversations, 11 domains, 1,138 turns")
    return rows


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def engine_options(settings):
    import argparse
    import shlex

    parser = argparse.ArgumentParser(exit_on_error=False, allow_abbrev=False)
    parser.add_argument("--attention-backend")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--speculative-config", type=json.loads)
    parser.add_argument("--compilation-config", type=json.loads)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--tokenizer")
    parser.add_argument("--enable-expert-parallel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    options, unknown = parser.parse_known_args(shlex.split(settings["extra_server_args"]))
    if unknown:
        raise ValueError("Unsupported SPEED-Bench engine arguments: " + " ".join(unknown))
    values = vars(options)
    for key, expected in {
        "max_num_seqs": 32,
        "enable_prefix_caching": False,
        "stream_interval": 1,
        "seed": 0,
    }.items():
        if values[key] != expected:
            raise ValueError(f"SPEED-Bench requires {key}={expected}")
    for key in ("max_model_len", "gpu_memory_utilization"):
        if settings[key] is not None:
            values[key] = settings[key]
    values.update(
        model=settings["model"],
        tensor_parallel_size=settings.get("tp_size", 1),
        pipeline_parallel_size=settings.get("pp_size", 1),
        data_parallel_size=1,
        trust_remote_code=settings["trust_remote_code"],
        skip_tokenizer_init=False,
    )
    spec = values.get("speculative_config")
    if spec is not None and not isinstance(spec, dict):
        raise ValueError("speculative_config must be a JSON object")
    if spec == {}:
        values.pop("speculative_config")
    if spec:
        k = spec.get("num_speculative_tokens")
        if (
            type(k) is not int
            or k < 1
            or spec.get("enable_adaptive_verification")
            or any("dynamic" in key for key in spec)
        ):
            raise ValueError(
                "SPEED-Bench requires a positive fixed num_speculative_tokens; adaptive/dynamic K is unsupported"
            )
    return {k: v for k, v in values.items() if v is not None}


def configure(settings, dataset_path):
    import importlib.metadata

    rows = load_dataset(dataset_path)
    settings["dataset_sha256"] = digest(rows)
    upstream = Path(__file__).resolve().parent
    settings["upstream_sha256"] = digest(
        {str(p.relative_to(upstream.parent)): p.read_text() for p in sorted(upstream.rglob("*.py"))}
    )
    engine = engine_options(settings)
    settings["engine"] = engine
    settings["vllm_version"] = importlib.metadata.version("vllm")
    spec = engine.get("speculative_config") or {}
    settings["speculative_tokens"] = spec.get("num_speculative_tokens", 0)
    settings["measurement_mode"] = "speculative" if spec else "baseline"
    # Model metadata is inexpensive to fingerprint; full weight hashes are not implied.
    settings["model_metadata"] = {}
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    ):
        path = Path(settings["model"]) / name
        if path.is_file():
            import hashlib

            settings["model_metadata"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if settings["chat_template"]:
        settings["chat_template_sha256"] = digest(Path(settings["chat_template"]).read_text())
    return rows
